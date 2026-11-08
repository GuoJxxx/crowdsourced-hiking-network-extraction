"""Resolve the method implementation in its named directory."""
import sys
from Chicago.Methods.Ahmed import Chicago_Ahmed_backend as implementation
sys.modules[__name__] = implementation
