"""Shared path resolution and checked, restartable subprocess execution."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from functools import lru_cache
from datetime import datetime
import uuid
from .runtime_language import english_text
from .python_runtime import python_environment

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'settings.json').is_file())
RUNTIME = ROOT / 'Chicago/Evaluation/Runtime'
RUNTIME_PATHS = (ROOT, RUNTIME, ROOT/'StudyArea/Evaluation/Runtime')

SOURCES = {
    'chicago_proposed':'Chicago/Methods/Proposed/Chicago_Proposed_Source',
    'mainling_full':'StudyArea/Methods/Proposed/StudyArea_Proposed_Source',
    'mainling_ablation':'StudyArea/Ablation/StudyArea_Proposed_Ablation_Source',
    'mainling_references':'StudyArea/Evaluation/Runtime/mainling_references',
    'assessment':'Chicago/Evaluation/Chicago_Assessment_Source',
    'assessment_tools':'Chicago/Evaluation/Chicago_Assessment_Tools',
}

def source(name): return ROOT/SOURCES[name]

def dataset_results(group):
    return path('chicago_results' if group=='chicago' else 'studyarea_results')

def intermediate(group):
    if group=='chicago': return dataset_results(group)/'Chicago_Intermediate'
    base=dataset_results(group)/'StudyArea_Intermediate'
    return base/'Ablation' if group=='ablation' else base

def parameter_file(name): return ROOT/PARAMETER_ROLES[name]

PARAMETER_ROLES = {'mainling.json': 'StudyArea/Methods/Proposed/StudyArea_Proposed_parameters.json', 'ablation.json': 'StudyArea/Ablation/StudyArea_Proposed_Ablation_parameters.json', 'references.json': 'Chicago/Evaluation/Chicago_Reference_parameters.json', 'paper_table_A1.json': 'StudyArea/Evaluation/StudyArea_TableA1_parameters.json', 'scenes.json': 'StudyArea/Evaluation/StudyArea_Scenes.json'}



def settings():
    return json.loads((ROOT / 'settings.json').read_text(encoding='utf-8-sig'))


@lru_cache(maxsize=32)
def _shortcut_target(filename, modified, size):
    """Read a local Windows shortcut without a third-party Python dependency."""
    executable=Path(os.environ['SystemRoot'])/'System32/WindowsPowerShell/v1.0/powershell.exe'
    script="[Console]::OutputEncoding=New-Object Text.UTF8Encoding($false);$s=New-Object -ComObject WScript.Shell;[Console]::Write($s.CreateShortcut($env:TRAIL_LOCATION_SHORTCUT).TargetPath)"
    env=os.environ.copy();env['TRAIL_LOCATION_SHORTCUT']=filename
    result=subprocess.run([str(executable),'-NoProfile','-Command',script],env=env,
                          capture_output=True,text=True,encoding='utf-8',errors='replace',
                          creationflags=subprocess.CREATE_NO_WINDOW,check=True)
    target=result.stdout.strip()
    if not target:raise FileNotFoundError('Select this location again in Configuration.')
    return Path(target).resolve()


def configured_location(value):
    candidate=Path(value)
    if candidate.anchor:
        raise ValueError('Choose a relative location in Configuration.')
    candidate=ROOT/candidate
    if candidate.suffix.lower()=='.lnk' and candidate.is_file():
        info=candidate.stat()
        return _shortcut_target(str(candidate),info.st_mtime_ns,info.st_size)
    return candidate.resolve()


def path(name):
    if name == 'mainling_clean':
        return intermediate('mainling')/'proposed/S2_Clean/cleaned_hiking_tracks'
    if name == 'outputs': return dataset_results('chicago')/'Chicago_Runtime'
    selected=configured_location(settings()['paths'][name])
    if name in {'chicago_results','studyarea_results'}:
        package_root=ROOT.resolve()
        if selected!=package_root and package_root not in selected.parents:
            raise ValueError('Results folders must be inside the package folder.')
    return selected


def tool(name, required=False):
    """Resolve only the executable explicitly selected in Configuration."""
    value = settings().get('tools', {}).get(name, '').strip()
    if not value:
        if required:
            raise FileNotFoundError(f'{name}: select an executable in Configuration')
        return f'<configure {name}>'
    candidate=configured_location(value)
    if required and not candidate.is_file():
        raise FileNotFoundError(f'{name}: update its location in Configuration')
    return str(candidate)


def step(name, argv, outputs=(), cwd=ROOT):
    return {'name': name, 'argv': [str(x) for x in argv], 'cwd': str(cwd),
            'outputs': [str(p) for p in outputs]}


def py(name, script, *arguments, outputs=(), cwd=ROOT):
    return step(name, [tool('python'), '-B', '-u', script, *arguments], outputs, cwd)


def module(name, module_name, *arguments, outputs=()):
    return step(name, [tool('python'), '-B', '-u', '-m', module_name, *arguments], outputs, ROOT)


def portable_step(value):
    """Represent package-local command locations independently of the folder name."""
    if isinstance(value,dict):return {key:portable_step(item) for key,item in value.items()}
    if isinstance(value,(list,tuple)):return [portable_step(item) for item in value]
    if isinstance(value,str):
        location=value.replace('\\','/')
        for prefix in {str(ROOT.absolute()).replace('\\','/'),str(ROOT.resolve()).replace('\\','/')}:
            prefix=prefix.rstrip('/')
            if location.casefold()==prefix.casefold():return '.'
            if location.casefold().startswith(prefix.casefold()+'/'):
                return location[len(prefix)+1:]
    return value


def execution_identity():
    """Bind checkpoints to code, parameters, settings and shared input bytes."""
    digest=hashlib.sha256()
    excluded={'Results','Data','Python_Environment','Engines','Locations','__pycache__'}
    candidates=[ROOT/'settings.json']
    for dataset in ('Chicago','StudyArea'):
        for folder, directories, filenames in os.walk(ROOT/dataset, followlinks=False):
            directories[:] = sorted(d for d in directories if d not in excluded)
            candidates += [Path(folder)/name for name in sorted(filenames)
                           if Path(name).suffix.lower() in {'.py','.ps1','.m','.java','.json','.pyx','.yaml','.yml','.xml','.txt'}]
    for candidate in candidates:
        digest.update(candidate.relative_to(ROOT).as_posix().encode())
        digest.update(candidate.read_bytes())
    for name in sorted(settings()['paths']):
        if name.endswith('_results'):continue
        item=path(name)
        selected=sorted(p for p in item.rglob('*') if p.is_file()) if item.is_dir() else [item]
        if item.suffix.lower()=='.shp':selected += [item.with_suffix(s) for s in ('.shx','.dbf','.prj','.cpg') if item.with_suffix(s).is_file()]
        for candidate in selected:
            if not candidate.is_file():raise FileNotFoundError(f'paths.{name}: {candidate}')
            digest.update((name+'/'+candidate.name).encode())
            with candidate.open('rb') as stream:
                for block in iter(lambda:stream.read(1048576),b''):digest.update(block)
    identity=digest.hexdigest()
    # A recorded diagnostic-only repair may retain existing checkpoints. Match
    # the complete code/input digest exactly; any later edit invalidates this alias.
    alias_file=path('outputs')/'runtime_identity_alias.json'
    if alias_file.is_file():
        try:
            alias=json.loads(alias_file.read_text(encoding='utf-8'))
            previous=alias.get('checkpoint_identity','')
            if (alias.get('source_identity')==identity and isinstance(previous,str)
                    and len(previous)==64 and all(c in '0123456789abcdef' for c in previous)):
                return previous
        except (OSError,ValueError,AttributeError):
            pass  # A missing/invalid compatibility record never authorizes reuse.
    return identity


@contextmanager
def group_lock(folder):
    """Hold a nonblocking operating-system lock until the launcher exits."""
    folder.mkdir(parents=True,exist_ok=True)
    with (folder/'launcher.lock').open('a+b') as stream:
        stream.seek(0);stream.write(b'0');stream.flush();stream.seek(0)
        try:
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError('This experiment group is already running') from error
        try: yield
        finally:
            stream.seek(0)
            if os.name=='nt': msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
            else: fcntl.flock(stream.fileno(),fcntl.LOCK_UN)


def execute(document, required_tools=()):
    log_root=path('outputs')/'logs'/document['group']
    with group_lock(log_root):
        _execute_locked(document,required_tools,log_root)


def _execute_locked(document, required_tools, log_root):
    """Log each command; a nonzero child exit never creates a success marker."""
    tool('python', required=True)
    if required_tools:
        from .environment import require_tools
        require_tools(required_tools)
    environment = os.environ.copy()
    if 'matlab' in required_tools:
        from .runtime_language import matlab_environment
        environment = matlab_environment(tool('matlab', required=True), path('outputs')/'matlab_preferences')
    environment['PYTHONNOUSERSITE'] = '1'
    environment['PYTHONUTF8'] = '1'
    environment['PYTHONDONTWRITEBYTECODE'] = '1'
    environment['PYTHONUNBUFFERED'] = '1'
    environment['PYTHONPATH'] = os.pathsep.join(str(p) for p in RUNTIME_PATHS) + os.pathsep + environment.get('PYTHONPATH','')
    environment['GDAL_CACHEMAX'] = '512'
    for name in required_tools:
        environment['PATH'] = str(Path(tool(name, True)).parent) + os.pathsep + environment.get('PATH','')
    environment = python_environment(tool('python', required=True), environment)
    identity = execution_identity()
    for item in document['steps']:
        item=dict(item)
        argv=list(item['argv'])
        for option in item.get('optional_paths',[]):
            if option in argv:
                index=argv.index(option)
                if not Path(argv[index+1]).exists(): del argv[index:index+2]
        item['argv']=argv
        name = item['name']
        marker = log_root / (name+'.success.json')
        ownership = log_root / (name+'.started.json')
        outputs = [Path(p) for p in item['outputs']]
        fingerprint = hashlib.sha256(json.dumps([identity,portable_step(item)],sort_keys=True).encode()).hexdigest()
        if marker.exists():
            saved = json.loads(marker.read_text(encoding='utf-8'))
            if saved.get('fingerprint') == fingerprint and outputs and all(p.exists() and (p.is_dir() or p.stat().st_size>0) for p in outputs):
                print('SKIP', name, flush=True)
                continue
            raise RuntimeError(f'{name}: configuration or output changed; select new dataset Results folders in settings.json')
        # Only archive incomplete preparation products created by this launcher.
        targets=[Path(p).resolve() for p in item.get('restart_targets',[]) if Path(p).exists()]
        if targets:
            saved=json.loads(ownership.read_text(encoding='utf-8')) if ownership.exists() else {}
            if saved.get('fingerprint')!=fingerprint:
                raise RuntimeError(f'{name}: unowned output; select new dataset Results folders')
            output_roots=[dataset_results(group).resolve() for group in ('chicago','mainling')]
            for target in targets:
                if not any(root in target.parents for root in output_roots) or target.is_symlink():
                    raise RuntimeError(f'{name}: restart target must be inside a dataset Results folder')
                if target.is_dir() and ((target/'native/completion.json').exists() or
                                        any((target/'native').glob('*.mat'))):
                    raise RuntimeError(f'{name}: existing native checkpoints retained; inspect the stage log')
            suffix=datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:8]
            for target in targets:
                target.rename(target.with_name(target.name+'.interrupted-'+suffix))
        ownership.write_text(json.dumps({'fingerprint':fingerprint}),encoding='utf-8')
        print('RUN', name, flush=True)
        started = time.perf_counter()
        with (log_root/(name+'.log')).open('a',encoding='utf-8') as log:
            log.write('\n'+json.dumps(item,ensure_ascii=False)+'\n')
            with subprocess.Popen(item['argv'], cwd=item['cwd'], env=environment,
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True,encoding='utf-8',errors='replace') as child:
                for line in child.stdout:
                    line = english_text(line)
                    print(line,end='',flush=True)
                    log.write(line)
                code = child.wait()
        if code:
            raise RuntimeError(f'{name}: exit code {code}; see outputs/logs/{document["group"]}/{name}.log')
        missing = [str(p) for p in outputs if not p.exists() or (p.is_file() and p.stat().st_size==0)]
        if missing:
            raise RuntimeError(f'{name}: expected output missing: {missing}')
        marker.write_text(json.dumps({'fingerprint':fingerprint,'elapsed_seconds':time.perf_counter()-started,
                                      'outputs':portable_step(item['outputs'])},indent=2),encoding='utf-8')


def finish(document, args, required_tools=()):
    if args.plan:
        print(json.dumps(document,ensure_ascii=False,indent=2))
    else:
        execute(document,required_tools)
    return 0
