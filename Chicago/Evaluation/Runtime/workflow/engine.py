"""Dependency-aware manuscript tasks and the independent computation worker."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

from .common import ROOT, path, tool, group_lock, RUNTIME_PATHS, intermediate
from .products import network_file, evaluation_file, table_path, figure_path, map_folder
from .progress import parse_progress
from .runtime_language import english_text, install_english_output
from .python_runtime import python_environment
from .host_identity import host_id

CHICAGO = ('ahmed', 'biagioni', 'davies', 'edelkamp', 'karagiorgou', 'proposed')
MAINLING = ('proposed', 'edelkamp', 'biagioni', 'karagiorgou')
ABLATIONS = ('full', 'wo_activity', 'wo_uturn', 'wo_evidence_topology')
NAMES = {'ahmed':'Ahmed', 'biagioni':'Biagioni', 'davies':'Davies', 'edelkamp':'Edelkamp',
         'karagiorgou':'Karagiorgou', 'proposed':'Proposed', 'full':'Full',
         'wo_activity':'Without activity awareness', 'wo_uturn':'Without hairpin preservation',
         'wo_evidence_topology':'Without topology optimization'}


def catalog():
    """Declare the source-to-product dependency chain for every manuscript item."""
    tasks=[]
    def relative(p):return Path(os.path.relpath(p,ROOT)).as_posix()
    def add(identifier,label,group,deps=(),module=None,args=(),outputs=()):
        tasks.append({'id':identifier,'label':label,'group':group,'deps':list(deps),
                      'module':module,'args':list(args),'outputs':[relative(p) for p in outputs]})
    add('mainling.clean','StudyArea trajectory preprocessing','Data',module='workflow.preprocess',
        outputs=(intermediate('mainling')/'proposed/S2_Clean/all_cleaned_points.csv',))
    for method in CHICAGO:
        name=NAMES[method]
        add('chicago.'+method,'Chicago - '+name,'Chicago',module=f'Chicago.Methods.{name}.Chicago_{name}_run',
            outputs=(network_file('chicago',method),evaluation_file(method)))
    for method in MAINLING:
        name=NAMES[method]
        add('mainling.'+method,'StudyArea - '+name,'StudyArea',('mainling.clean',),
            f'StudyArea.Methods.{name}.StudyArea_{name}_run',outputs=(network_file('mainling',method),))
    entries=('Full','WithoutActivity','WithoutHairpin','WithoutTopology')
    for mode,entry in zip(ABLATIONS,entries):
        add('ablation.'+mode,'Ablation - '+NAMES[mode],'Ablation',
            module='StudyArea.Ablation.StudyArea_Proposed_'+entry,outputs=(network_file('ablation',mode),))
    chi=tuple('chicago.'+m for m in CHICAGO)
    reg=tuple('mainling.'+m for m in MAINLING)
    abl=tuple('ablation.'+m for m in ABLATIONS)
    add('metrics.chicago','Chicago accuracy and graph measures','Metrics',chi,
        'Chicago.Evaluation.Chicago_accuracy',outputs=(table_path('chicago_metrics.csv'),))
    add('metrics.mainling','StudyArea graph measures','Metrics',reg,
        'StudyArea.Evaluation.StudyArea_graph_metrics',outputs=(table_path('mainling_metrics.csv'),))
    add('metrics.ablation','Ablation graph measures','Metrics',abl,
        'workflow.artifacts',('ablation-metrics',),(table_path('ablation_metrics.csv'),))
    add('metrics.all','All reported numerical measures','Metrics',('metrics.chicago','metrics.mainling','metrics.ablation'),
        'workflow.artifacts',('summary',),(table_path('reported_metrics.json'),))
    titles={8:'Chicago reconstructed networks',9:'StudyArea local comparisons',10:'StudyArea map layers',
            11:'Ablation structural responses',12:'Ablation map layers'}
    deps={8:chi,9:reg,10:('mainling.proposed',),11:('metrics.ablation',),12:('mainling.clean',)+abl}
    for number in range(8,13):
        dataset='Chicago' if number==8 else 'StudyArea'
        output=figure_path(number) if number not in (10,12) else map_folder(number)/f'StudyArea_Figure{number}_Layers.json'
        add(f'figure.{number}',f'Figure {number} - {titles[number]}','Figures',deps[number],
            f'{dataset}.Evaluation.{dataset}_Figure{number}',outputs=(output,))
    for number,dependency in [('1',('metrics.chicago',)),('2',()),('3',('metrics.ablation',)),('a1',())]:
        dataset='Chicago' if number=='1' else 'StudyArea';display='A1' if number=='a1' else number
        add('table.'+number,f'Table {display}','Tables',dependency,f'{dataset}.Evaluation.{dataset}_Table{display}',
            outputs=(table_path('table_'+display+'.csv'),))
    return tasks


def all_findings():
    return [t['id'] for t in catalog() if t['group'] in ('Figures','Tables')] + ['metrics.all']


def plan(targets):
    by_id = {t['id']:t for t in catalog()}
    result, visiting, done = [], set(), set()
    def visit(name):
        if name not in by_id:
            raise ValueError('Unknown task: '+name)
        if name in done:
            return
        if name in visiting:
            raise ValueError('Task dependency cycle: '+name)
        visiting.add(name)
        for dependency in by_id[name]['deps']:
            visit(dependency)
        visiting.remove(name)
        done.add(name)
        result.append(by_id[name])
    for target in targets:
        visit(target)
    return result


def control_root():
    return path('outputs')/'.workflow'


def write_json(filename, value):
    filename.parent.mkdir(parents=True, exist_ok=True)
    temporary = filename.with_name(filename.name+'.'+uuid.uuid4().hex+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=True), encoding='utf-8')
    try:
        for attempt in range(25):
            try:
                os.replace(temporary, filename)
                break
            except PermissionError:
                if attempt==24:
                    raise
                time.sleep(.01)
    finally:
        temporary.unlink(missing_ok=True)


def read_status():
    filename = control_root()/'status.json'
    if not filename.exists():
        return {'state':'ready','completed':0,'total':0}
    try:
        state=json.loads(filename.read_text(encoding='utf-8'))
        if state.get('host_id')!=host_id():
            return {'state':'ready','completed':0,'total':0}
        return state
    except (OSError, ValueError):
        return {'state':'reading','completed':0,'total':0}


def process_alive(pid, created=None, tolerance=.02):
    if not isinstance(pid,int) or pid <= 0:
        return False
    if pid == os.getpid() and created is None:
        return True
    try:
        if os.name == 'nt':
            import ctypes
            from ctypes import wintypes
            kernel=ctypes.windll.kernel32
            kernel.OpenProcess.restype=wintypes.HANDLE
            kernel.GetExitCodeProcess.argtypes=(wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD))
            kernel.GetProcessTimes.argtypes=(wintypes.HANDLE,ctypes.POINTER(wintypes.FILETIME),
                                            ctypes.POINTER(wintypes.FILETIME),ctypes.POINTER(wintypes.FILETIME),
                                            ctypes.POINTER(wintypes.FILETIME))
            kernel.CloseHandle.argtypes=(wintypes.HANDLE,)
            handle=kernel.OpenProcess(0x1000, False, pid)
            if not handle:
                return False
            code=ctypes.c_ulong()
            ok=ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
            same=True
            if ok and code.value==259 and created is not None:
                times=[wintypes.FILETIME() for _ in range(4)]
                ok=kernel.GetProcessTimes(handle,*(ctypes.byref(value) for value in times))
                birth=((times[0].dwHighDateTime<<32)|times[0].dwLowDateTime)/10000000-11644473600
                same=bool(ok and abs(birth-float(created))<=tolerance)
            kernel.CloseHandle(handle)
            return bool(ok and code.value == 259 and same)
        os.kill(pid,0)
        if created is not None:
            import psutil
            return abs(psutil.Process(pid).create_time()-float(created))<=tolerance
        return True
    except (OSError, PermissionError):
        return False


def worker_alive(state):
    created=state.get('process_created')
    if created is not None:
        return process_alive(state.get('pid'),created)
    return process_alive(state.get('pid'),state.get('started'),60)


@contextmanager
def status_lock(folder=None):
    """Serialize status transitions before an experiment worker is suspended."""
    folder=control_root() if folder is None else folder
    for attempt in range(250):
        manager=group_lock(folder/'status')
        try:
            manager.__enter__()
            break
        except RuntimeError:
            if attempt==249:
                raise RuntimeError('The run status is busy. Try the control again.')
            time.sleep(.02)
    try:
        yield
    finally:
        manager.__exit__(None,None,None)


def launch(targets, *, action='run'):
    """Launch an independent worker and return its persisted status."""
    if action == 'run':
        plan(targets)
    root=control_root()
    with group_lock(root/'launch'), status_lock(root):
        previous=read_status()
        if previous.get('state') in ('starting','running','pausing','paused','resuming') and worker_alive(previous):
            raise RuntimeError('An experiment is running. Its progress is shown in the window.')
        if previous.get('paused'):
            from .run_control import has_suspended_processes
            if has_suspended_processes(previous['paused']):
                raise RuntimeError('A paused experiment process remains. Use Resume for the recorded run.')
        job=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8]
        log=path('outputs')/'logs/window'/f'{job}.log'
        relative_log=Path(os.path.relpath(log,ROOT)).as_posix()
        log.parent.mkdir(parents=True,exist_ok=True)
        args=[tool('python',required=True),'-B','-u','-m','workflow.engine','worker','--action',action,*targets]
        env=os.environ.copy()
        env.update(PYTHONNOUSERSITE='1',PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
        env['PYTHONPATH']=os.pathsep.join(str(p) for p in RUNTIME_PATHS)
        env=python_environment(args[0],env)
        flags=subprocess.CREATE_NO_WINDOW|subprocess.CREATE_NEW_PROCESS_GROUP if os.name=='nt' else 0
        with log.open('a',encoding='utf-8') as stream:
            child=subprocess.Popen(args,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,
                                   creationflags=flags,start_new_session=os.name!='nt')
        value={'state':'starting','host_id':host_id(),'pid':child.pid,'job':job,'targets':targets,'action':action,
               'log':relative_log,'started':time.time(),'completed':0,'total':0,'current':'Starting'}
        try:
            import psutil
        except ImportError:
            pass  # The dependency installer can run in a new Python environment.
        else:
            try:
                value['process_created']=psutil.Process(child.pid).create_time()
            except psutil.NoSuchProcess:
                pass
        write_json(root/'status.json',value)
        return value


def pause():
    """Pause the active experiment without discarding its in-memory state."""
    from .run_control import suspend_run, has_suspended_processes
    root=control_root()
    with group_lock(root/'launch'), status_lock(root):
        state=read_status()
        if state.get('state')=='paused' and worker_alive(state):
            return state
        if state.get('state') not in ('starting','running') or not worker_alive(state):
            raise RuntimeError('No experiment is running.')
        original=dict(state)
        def progress(record):
            state.update(state='pausing',paused=record,updated=time.time())
            write_json(root/'status.json',state)
        try:
            record=suspend_run(state,ROOT,progress)
        except Exception:
            latest=read_status()
            if has_suspended_processes(latest.get('paused',{})):
                latest.update(state='paused',current='Pause incomplete. Select Resume.')
                write_json(root/'status.json',latest)
            else:
                write_json(root/'status.json',original)
            raise
        state.update(state='paused',paused=record,updated=time.time())
        write_json(root/'status.json',state)
        return state


def resume():
    """Restore a paused worker, or continue the saved selection after an exit."""
    from .run_control import resume_run, has_suspended_processes
    root=control_root()
    with group_lock(root/'launch'), status_lock(root):
        state=read_status()
        if state.get('action')!='run' or not state.get('targets'):
            raise RuntimeError('No experiment is available to resume.')
        alive=worker_alive(state)
        if state.get('state') in ('running','starting') and alive:
            return state
        if state.get('state')=='paused' and alive:
            def progress(record):
                state.update(state='resuming',paused=record,updated=time.time())
                write_json(root/'status.json',state)
            try:
                resume_run(state,ROOT,state['paused'],progress)
            except Exception:
                state['state']='paused'
                write_json(root/'status.json',state)
                raise
            latest=read_status()
            latest.pop('paused',None)
            if latest.get('state') in ('paused','resuming'):
                latest['state']='running'
            latest['updated']=time.time()
            write_json(root/'status.json',latest)
            return latest
        if has_suspended_processes(state.get('paused',{})):
            raise RuntimeError('The experiment coordinator exited while child processes were paused.')
        targets=state['targets']
    return launch(targets,action='run')


def stop():
    """Stop only the verified local experiment tree; keep on-disk checkpoints."""
    from .run_control import stop_run
    root=control_root()
    with group_lock(root/'launch'), status_lock(root):
        state=read_status()
        def progress(record):
            state.update(state='stopping',paused=record,updated=time.time())
            write_json(root/'status.json',state)
        try:
            stop_run(state,ROOT,progress)
        except Exception:
            if state.get('paused'):
                state.update(state='paused',current='Stop incomplete. Resume or try closing again.')
                write_json(root/'status.json',state)
            raise
        state.pop('paused',None)
        state.update(state='stopped',child_pid=None,progress=None,current='Stopped by user; checkpoints retained.',
                     finished=time.time(),updated=time.time())
        write_json(root/'status.json',state)
        return state


def worker(targets, action='run'):
    # Startup state is written by the launcher before the worker takes the global lock.
    for _ in range(50):
        state=read_status()
        if state.get('pid')==os.getpid():
            break
        time.sleep(.1)
    with group_lock(control_root()/'compute'):
        state=read_status()
        if state.get('pid')!=os.getpid():
            raise RuntimeError('The worker does not own the active run record.')
        def update(**values):
            with status_lock():
                latest=read_status()
                if latest.get('job')==state.get('job') and latest.get('pid')==os.getpid():
                    state.update(latest)
                if state.get('state')=='resuming':
                    state['state']='running'
                state.update(values)
                state['updated']=time.time()
                write_json(control_root()/'status.json',state)
        update(state='running',current='Checking environment')
        try:
            from .environment import ensure, check
            if action=='install':
                update(current='Installing Python dependencies')
                ensure(force=True)
                update(state='completed',current='Environment ready',completed=1,total=1,finished=time.time())
                return 0
            if action=='check':
                result=check(native=True)
                write_json(path('outputs')/'environment.json',result)
                update(state='completed',current='Installation checked',environment=result,completed=1,total=1,finished=time.time())
                return 0
            python=ensure()
            steps=plan(targets)
            update(total=len(steps))
            env=os.environ.copy()
            env.update(PYTHONNOUSERSITE='1',PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',GDAL_CACHEMAX='512')
            env['PYTHONPATH']=os.pathsep.join(str(p) for p in RUNTIME_PATHS)
            env=python_environment(python,env)
            for index,task in enumerate(steps):
                update(current=task['label'],task=task['id'],completed=index,
                       stage='',progress=None,runtime_note='')
                print('\nTASK',task['id'],task['label'],flush=True)
                argv=[str(python),'-B','-u','-m',task['module'],*task['args']]
                with subprocess.Popen(argv,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                      text=True,encoding='utf-8',errors='replace') as child:
                    update(child_pid=child.pid)
                    for line in child.stdout:
                        line=english_text(line)
                        print(line,end='',flush=True)
                        progress=parse_progress(line)
                        if progress is not None:
                            update(progress=progress)
                        elif line.startswith('[RUNTIME] '):
                            update(runtime_note=line[len('[RUNTIME] '):].strip())
                        elif line.startswith(('RUN ','SKIP ','[CONSTRUCTION]','[NETWORK EXPORT]','[ASSESSMENT INPUTS]','[ASSESSMENT]')):
                            update(stage=line.strip(),progress=None,runtime_note='')
                    code=child.wait()
                if code:
                    raise RuntimeError(f'{task["label"]} returned exit code {code}. See the run log.')
                for relative in task['outputs']:
                    product=ROOT/relative
                    if not product.is_file() or product.stat().st_size==0:
                        raise FileNotFoundError('Expected output: '+relative)
                update(completed=index+1,child_pid=None,progress=None,runtime_note='')
            update(state='completed',current='Selected tasks completed',finished=time.time())
            write_json(path('outputs')/'runs'/f'{state["job"]}.json',state)
            return 0
        except Exception as error:
            update(state='failed',error=str(error),current=str(error),finished=time.time(),child_pid=None)
            print(type(error).__name__+': '+str(error),flush=True)
            return 1


def main():
    install_english_output()
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ('plan','run','worker'):
        p=sub.add_parser(name)
        p.add_argument('targets',nargs='*')
        p.add_argument('--action',choices=('run','check','install'),default='run')
    clean=sub.add_parser('cleanup-installation');clean.add_argument('pid',type=int)
    sub.add_parser('status')
    sub.add_parser('pause')
    sub.add_parser('resume')
    sub.add_parser('stop')
    args=parser.parse_args()
    if args.command=='cleanup-installation':
        from .environment import cleanup_installation
        cleanup_installation(args.pid);return 0
    if args.command=='status':
        print(json.dumps(read_status(),indent=2));return 0
    if args.command in ('pause','resume','stop'):
        result={'pause':pause,'resume':resume,'stop':stop}[args.command]()
        print(json.dumps(result,indent=2));return 0
    aliases={
        'all':all_findings(),
        'Chicago':['table.1','figure.8','metrics.chicago'],
        'StudyArea':['table.2','table.3','table.a1','figure.9','figure.10','figure.11','figure.12','metrics.mainling'],
    }
    targets=[item for target in args.targets for item in aliases.get(target,[target])]
    if args.command=='plan':
        print(json.dumps(plan(targets),indent=2));return 0
    if args.command=='run':
        print(json.dumps(launch(targets,action=args.action),indent=2));return 0
    return worker(targets,args.action)


if __name__=='__main__':
    raise SystemExit(main())
