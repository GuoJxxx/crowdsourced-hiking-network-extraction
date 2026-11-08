"""Resolve compatible Python packages and installed computational engines."""
from __future__ import annotations
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from .common import ROOT, settings, tool, path
from .python_runtime import python_environment

MODULES={'numpy':'numpy','scipy':'scipy','pandas':'pandas','geopandas':'geopandas',
         'shapely':'shapely','pyproj':'pyproj','rasterio':'rasterio','pyogrio':'pyogrio',
         'opencv-python':'cv2','scikit-image':'skimage','Rtree':'rtree','networkx':'networkx',
         'matplotlib':'matplotlib','Pillow':'PIL','h5py':'h5py','psutil':'psutil',
         'packaging':'packaging','fissix':'fissix'}


def python_supported(version):
    return tuple(version[:2]) >= (3,11)


def java_major(text):
    match=re.search(r'(?:openjdk|java|javac)(?: version)?\s+"?(\d+)(?:\.(\d+))?',text,re.I)
    if not match:return None
    major=int(match.group(1))
    return int(match.group(2)) if major==1 and match.group(2) else major


def matlab_supported(release):
    return str(release).strip().lstrip('Rr') in {'2021a','2021b'}


def matlab_platform_supported(release, full_version, windows_build):
    """Windows 11 support starts with R2021b Update 1."""
    release=str(release).strip().lstrip('Rr')
    if not matlab_supported(release):return False
    if windows_build<22000:return True
    if release!='2021b':return False
    update=re.search(r'Update\s+(\d+)',full_version,re.I)
    if update:return int(update.group(1))>=1
    version=re.search(r'9\.11\.0\.(\d+)',full_version)
    return bool(version and int(version.group(1))>=1809720)


def setting_location(section,selected):
    """Convert a same-drive selection to a package-relative location."""
    try:return Path(os.path.relpath(Path(selected),ROOT)).as_posix()
    except ValueError as error:
        raise ValueError('Use Configuration to select a location on another drive.') from error


def python_path():
    return Path(tool('python', required=True))


def packages(python):
    # Import in the selected interpreter so binary-library incompatibilities are
    # reported alongside distribution versions, before model computation starts.
    script="""import contextlib,importlib,importlib.metadata,io,json,sys
sys.path.insert(0,RUNTIME_LOCATION)
from workflow.geospatial import configure_gdal_data,gdal_resources
configure_gdal_data()
modules=MODULE_MAPPING
versions={};errors={}
for name,module in modules.items():
 try:versions[name]=importlib.metadata.version(name)
 except importlib.metadata.PackageNotFoundError:versions[name]=None
 if name=='fissix' and sys.version_info[:2]<(3,13):continue
 try:
  with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
   importlib.import_module(module)
 except Exception as error:errors[name]=type(error).__name__+': '+str(error)
converter={'module':'lib2to3','ready':False,'fissix_required':False}
try:
 with contextlib.redirect_stderr(io.StringIO()):importlib.import_module('lib2to3.refactor')
 converter['ready']=True
except ModuleNotFoundError:
 converter.update(module='fissix',fissix_required=True)
 try:
  with contextlib.redirect_stderr(io.StringIO()):importlib.import_module('fissix.refactor')
  converter['ready']=True
 except Exception as error:errors['source_converter']=type(error).__name__+': '+str(error)
print(json.dumps({'python':list(sys.version_info[:3]),'executable':sys.executable,'packages':versions,'import_errors':errors,'source_converter':converter,'gdal_resources':gdal_resources()}))
""".replace('RUNTIME_LOCATION',repr(str(ROOT/'Chicago/Evaluation/Runtime'))).replace('MODULE_MAPPING',repr(MODULES))
    result=subprocess.run([str(python),'-s','-B','-c',script],env=python_environment(python),capture_output=True,text=True,encoding='utf-8',errors='replace')
    if result.returncode:raise RuntimeError(result.stderr.strip())
    return json.loads(result.stdout)


def satisfied(document):
    if not python_supported(document['python']) or document.get('import_errors'):return False
    try:
        from packaging.requirements import Requirement
    except ImportError:
        try:from pip._vendor.packaging.requirements import Requirement
        except ImportError:return False
    version=document['python']
    marker_env={'python_version':f'{version[0]}.{version[1]}',
                'python_full_version':'.'.join(str(v) for v in version)}
    installed=document['packages']
    for line in (ROOT/'requirements.txt').read_text(encoding='utf-8').splitlines():
        line=line.strip()
        if not line or line.startswith('#'):continue
        requirement=Requirement(line)
        if requirement.marker and not requirement.marker.evaluate(marker_env):continue
        value=installed.get(requirement.name)
        if not value:return False
        if requirement.name=='opencv-python' and value=='4.7.0':value='4.7.0.72'
        if value not in requirement.specifier:return False
    return True


def installation_paths(python):
    script="import json,sys,sysconfig;print(json.dumps({'prefix':sys.prefix,'purelib':sysconfig.get_path('purelib'),'platlib':sysconfig.get_path('platlib')}))"
    result=subprocess.run([str(python),'-I','-B','-c',script],capture_output=True,text=True,encoding='utf-8',check=True)
    return json.loads(result.stdout)


def cleanup_installation(pid):
    """Remove only the temporary download directory owned by this installation."""
    tracker=path('outputs')/'package_installation.json'
    if not tracker.is_file():return
    record=json.loads(tracker.read_text(encoding='utf-8'))
    if int(record['pid'])!=int(pid):raise ValueError('Installation identity changed.')
    folder=Path(record['temporary']);prefix=Path(record['prefix']).resolve()
    if not folder.name.startswith('.trail-package-install-') or folder.resolve().parent!=prefix:
        raise ValueError('Temporary installation location is outside the selected Python environment.')
    if folder.exists():
        attributes=getattr(folder.stat(follow_symlinks=False),'st_file_attributes',0)
        if folder.is_symlink() or attributes & 0x400:raise ValueError('Temporary location cannot be a linked directory.')
        owner=json.loads((folder/'owner.json').read_text(encoding='utf-8'))
        if int(owner['pid'])!=int(pid) or owner['package_root']!=str(ROOT.resolve()):
            raise ValueError('Temporary installation ownership does not match.')
        shutil.rmtree(folder)
    tracker.unlink()


def ensure(force=False):
    """Install into the selected environment, with downloads on the same volume."""
    selected=python_path();found=packages(selected)
    if not python_supported(found['python']):
        raise RuntimeError('Select Python 3.11 or newer in Configuration.')
    if force or not satisfied(found):
        locations=installation_paths(selected);prefix=Path(locations['prefix']).resolve()
        try:temporary=Path(tempfile.mkdtemp(prefix='.trail-package-install-',dir=prefix))
        except OSError as error:raise RuntimeError('The selected Python environment is not writable: '+str(prefix)) from error
        tracker=path('outputs')/'package_installation.json';tracker.parent.mkdir(parents=True,exist_ok=True)
        owner={'pid':os.getpid(),'package_root':str(ROOT.resolve())}
        (temporary/'owner.json').write_text(json.dumps(owner),encoding='utf-8')
        tracker.write_text(json.dumps({**owner,'prefix':str(prefix),'temporary':str(temporary)}),encoding='utf-8')
        env=os.environ.copy()
        for name in ('PIP_TARGET','PIP_PREFIX','PIP_USER','PYTHONUSERBASE','PIP_CACHE_DIR','PYTHONHOME','PYTHONPATH'):env.pop(name,None)
        env.update(TEMP=str(temporary),TMP=str(temporary),TMPDIR=str(temporary),
                   PYTHONNOUSERSITE='1',PIP_CONFIG_FILE=os.devnull,PYTHONDONTWRITEBYTECODE='1')
        print(f'Python executable: {selected}\nPackage destination: {locations["purelib"]}\nTemporary downloads: {temporary}\nPip cache: disabled',flush=True)
        try:
            command=[str(selected),'-I','-B','-m','pip','--isolated','--disable-pip-version-check','install',
                     '--no-user','--no-cache-dir','--prefix',str(prefix),'--only-binary=:all:',
                     '-r',str(ROOT/'requirements.txt')]
            if found.get('source_converter',{}).get('fissix_required'):command.append('fissix>=24.4.24')
            from .runtime_language import stream_command
            stream_command(command,env=env,check=True)
            stream_command([str(selected),'-I','-B','-m','pip','--isolated','check'],env=env,check=True)
        finally:cleanup_installation(os.getpid())
        found=packages(selected)
        if not satisfied(found):raise RuntimeError('Python dependencies are incomplete: '+json.dumps(found,ensure_ascii=False))
    print(f'Python environment ready: {selected}',flush=True)
    return selected


def inspect_tool(name,native=False):
    from .runtime_language import java_arguments, matlab_environment, english_text
    try:
        executable=tool(name)
        resolved=executable if Path(executable).is_file() else None
    except (FileNotFoundError,ValueError):
        resolved=None
    record={'available':bool(resolved),'compatible':False if not resolved else None,
            'requirement':'MATLAB R2021a/R2021b with Statistics and Machine Learning Toolbox' if name=='matlab' else 'JDK 17 or newer'}
    if not resolved or not native:return record
    record['executable']=str(resolved)
    if name=='matlab':
        probe="settings;disp(['MATLAB_VERSION=' version]);disp(['MATLAB_RELEASE=' version('-release')]);disp(['MATLAB_ROOT=' matlabroot]);assert(strncmp(version('-release'),'2021',4),'MATLAB R2021a or R2021b is required');assert(exist('pdist','file')~=0);assert(exist('squareform','file')~=0);squareform(pdist([0 0;1 1]));disp('STATISTICS_READY');"
        args=[resolved,'-batch',probe]
    else:args=java_arguments([resolved,'-version'])
    try:
        native_environment=matlab_environment(resolved,path('outputs')/'matlab_preferences') if name=='matlab' else None
        if name=='matlab':
            args[-1]+="if all(double(getString(message('MATLAB:singularMatrix')))<128),disp('ENGLISH_MESSAGES_READY');end;"
        process=subprocess.run(args,env=native_environment,capture_output=True,text=True,encoding='utf-8',errors='backslashreplace',timeout=120,
                               creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        text=english_text((process.stdout+process.stderr).strip())
        record.update(returncode=process.returncode,version=text)
        if name=='matlab':
            match=re.search(r'MATLAB_RELEASE=(\d{4}[ab])',text)
            release=match.group(1) if match else ''
            english_ready='ENGLISH_MESSAGES_READY' in text
            record.update(release=release,message_language='English' if english_ready else 'not_english',
                          compatible=process.returncode==0 and matlab_supported(release) and 'STATISTICS_READY' in text and english_ready)
            if not english_ready:
                record['error']='English MATLAB diagnostics are required. Check the MATLAB language configuration.'
            full=re.search(r'MATLAB_VERSION=([^\r\n]+)',text)
            if full:record['full_version']=full.group(1)
            build=sys.getwindowsversion().build if os.name=='nt' else 0
            supported=matlab_platform_supported(release,record.get('full_version',''),build)
            record.update(windows_build=build,platform_supported=supported)
            if build>=22000:
                record['recommended_version']='R2021b Update 7'
        else:
            major=java_major(text)
            record.update(major=major,compatible=process.returncode==0 and major is not None and major>=17)
    except (OSError,subprocess.TimeoutExpired,RuntimeError) as error:
        record.update(compatible=False,error=english_text(error))
    return record


def require_tools(names):
    for name in names:
        record=inspect_tool(name,native=True)
        if not record['compatible']:
            raise RuntimeError(name+': '+record['requirement']+'; '+record.get('error',record.get('version','executable not found')))


def check(native=False):
    record=packages(python_path())
    record['installation']=installation_paths(python_path())
    record['python_supported']=python_supported(record['python'])
    record['python_ready']=satisfied(record)
    record['tools']={name:inspect_tool(name,native) for name in ('java','javac','matlab')}
    record['data']={name:{'exists':path(name).exists(),'relative':value,'resolved':str(path(name))} for name,value in settings()['paths'].items() if not name.endswith('_results')}
    record['ready']=record['python_ready'] and all(item.get('compatible') is True for item in record['tools'].values()) and all(item['exists'] for item in record['data'].values())
    print(json.dumps(record,indent=2),flush=True)
    return record
