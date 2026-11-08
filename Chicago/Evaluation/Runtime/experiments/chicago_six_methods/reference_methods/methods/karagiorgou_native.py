"""Resolve the method implementation in its named directory."""
import sys
from Chicago.Methods.Karagiorgou import Chicago_Karagiorgou_backend as implementation
sys.modules[__name__] = implementation
