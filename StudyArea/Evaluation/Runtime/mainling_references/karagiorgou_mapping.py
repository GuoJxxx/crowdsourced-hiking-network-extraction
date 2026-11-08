"""Resolve the regional method implementation."""
import sys
from StudyArea.Methods.Karagiorgou import StudyArea_Karagiorgou_mapping as implementation
if __name__ == "__main__":
    raise SystemExit(implementation.main())
sys.modules[__name__] = implementation
