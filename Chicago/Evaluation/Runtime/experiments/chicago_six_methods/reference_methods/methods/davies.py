"""Resolve the method implementation in its named directory."""
import sys
from Chicago.Methods.Davies import Chicago_Davies_backend as implementation
sys.modules[__name__] = implementation
