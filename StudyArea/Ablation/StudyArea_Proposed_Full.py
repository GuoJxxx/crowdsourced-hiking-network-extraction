"""Run the published experiment with its fixed manuscript parameters."""
from pathlib import Path
import sys

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'settings.json').is_file())
for location in (ROOT,ROOT/'Chicago/Evaluation/Runtime',ROOT/'StudyArea/Evaluation/Runtime'):
    sys.path.insert(0,str(location))

if __name__=='__main__':
    from workflow.ablation import main
    raise SystemExit(main(['--mode', 'full']+sys.argv[1:]))
