"""Command-line entry for manuscript reproduction."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
for location in (ROOT,ROOT/'Chicago/Evaluation/Runtime',ROOT/'StudyArea/Evaluation/Runtime'):
    sys.path.insert(0,str(location))

if __name__ == "__main__":
    from workflow.engine import main
    raise SystemExit(main())
