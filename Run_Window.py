"""Open the single desktop window for reconstruction and configuration."""
from pathlib import Path
import os
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
if __name__=='__main__':
    sys.path.insert(0,str(ROOT/'Chicago/Evaluation/Runtime'))
    from workflow.runtime_language import install_english_output
    install_english_output()
    powershell=Path(os.environ['SystemRoot'])/'System32/WindowsPowerShell/v1.0/powershell.exe'
    panel=ROOT/'Chicago/Evaluation/Runtime/workflow/Configure.ps1'
    raise SystemExit(subprocess.call([str(powershell),'-NoProfile','-ExecutionPolicy','Bypass',
                                     '-STA','-WindowStyle','Hidden','-File',str(panel),'-Launch'],cwd=ROOT))
