"""English application diagnostics and native runtime messages."""
import re
import json
import os
import subprocess
import sys
from pathlib import Path


def java_arguments(arguments):
    executable,*options=arguments
    compiler=Path(executable).stem.lower()=='javac'
    prefix='-J-' if compiler else '-'
    return [executable,prefix+'Duser.language=en',prefix+'Duser.country=US',
            prefix+'Dfile.encoding=UTF-8',*options]


def english_text(value):
    """Keep diagnostics readable and represent Unicode paths without changing them."""
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', str(value))
    return text.encode('ascii', errors='backslashreplace').decode('ascii')


class EnglishStream:
    def __init__(self, stream):
        self.stream = stream

    def write(self, value):
        self.stream.write(english_text(value))
        return len(value)

    def writelines(self, values):
        for value in values:
            self.write(value)

    def __getattr__(self, name):
        return getattr(self.stream, name)


def install_english_output():
    if os.name == 'nt':
        import ctypes
        # Select English resources for native Windows process and file errors.
        setter = ctypes.windll.kernel32.SetThreadUILanguage
        setter.argtypes = [ctypes.c_ushort]
        setter.restype = ctypes.c_ushort
        setter(0x0409)
    for name in ('stdout', 'stderr'):
        stream = getattr(sys, name)
        if not isinstance(stream, EnglishStream):
            setattr(sys, name, EnglishStream(stream))


def stream_command(arguments, *, env=None, check=True):
    """Stream installation diagnostics and retain the original process exit code."""
    with subprocess.Popen(arguments, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True,
                          encoding='utf-8', errors='backslashreplace') as child:
        for line in child.stdout:
            print(english_text(line), end='', flush=True)
        code = child.wait()
    if check and code:
        raise subprocess.CalledProcessError(code, arguments)
    return subprocess.CompletedProcess(arguments, code)


def matlab_environment(executable, preferences):
    """Initialize English messages in preferences owned by this runtime folder."""
    preferences = Path(preferences).resolve()
    preferences.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment['MATLAB_PREFDIR'] = str(preferences)
    executable = Path(executable).resolve()
    info = executable.stat()
    identity = {'executable': str(executable), 'size': info.st_size,
                'mtime_ns': info.st_mtime_ns, 'language': 'en_US'}
    marker = preferences/'english_language.json'
    ready = preferences/'english_language.txt'
    if marker.is_file() and ready.is_file():
        try:
            if json.loads(marker.read_text(encoding='utf-8')) == identity and ready.read_text().strip() == 'en_US':
                return environment
        except (OSError, ValueError):
            pass
    print('[SETUP] Preparing English MATLAB diagnostics.', flush=True)
    ready.unlink(missing_ok=True)
    ready_path = str(ready).replace("'", "''")
    # R2021 saves this personal preference in a normal session before batch use.
    command = ("try;settings;languageSettings=settings;"
               "languageSettings.matlab.G11nLanguagePref.PersonalValue='en_US';"
               "fid=fopen('"+ready_path+"','w');assert(fid>=0,'Cannot write MATLAB language configuration');"
               "fprintf(fid,'en_US\\n');fclose(fid);"
               "catch problem;fprintf(2,'ERROR MATLAB language setup [%s]: %s\\n',problem.identifier,problem.message);"
               "exit(2);end;exit(0);")
    log = preferences/'language_setup.log'
    options = {}
    if os.name == 'nt':
        startup = subprocess.STARTUPINFO()
        startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startup.wShowWindow = 0
        options.update(startupinfo=startup, creationflags=subprocess.CREATE_NO_WINDOW)
    result = subprocess.run([str(executable), '-wait', '-nodesktop', '-nosplash',
                             '-logfile', str(log), '-r', command],
                            env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, **options)
    captured = result.stdout.decode('utf-8', errors='backslashreplace')
    if log.is_file():
        captured += log.read_text(encoding='utf-8', errors='backslashreplace')
    log.write_text(english_text(captured), encoding='utf-8')
    if result.returncode or not ready.is_file() or ready.read_text().strip() != 'en_US':
        raise RuntimeError('MATLAB English language setup failed with exit code '
                           +str(result.returncode)+': '+english_text(captured[-2000:]))
    marker.write_text(json.dumps(identity, indent=2, ensure_ascii=True), encoding='utf-8')
    return environment
