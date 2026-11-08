"""Prepare the selected Python's native DLL search path for child processes."""
import os
from pathlib import Path


def python_environment(python, environment=None):
    """Use a Windows Conda installation without requiring shell activation."""
    env = dict(os.environ if environment is None else environment)
    prefix = Path(python).resolve().parent
    if os.name == 'nt' and (prefix / 'conda-meta').is_dir():
        # LAPACK/MKL delay-load DLLs through PATH even after NumPy imports.
        # Prefer the selected environment over libraries from other installations.
        directories = (prefix, prefix / 'Library/mingw-w64/bin',
                       prefix / 'Library/usr/bin', prefix / 'Library/bin',
                       prefix / 'Scripts', prefix / 'bin')
        env['PATH'] = os.pathsep.join(str(p) for p in directories if p.is_dir()) + os.pathsep + env.get('PATH', '')
    return env
