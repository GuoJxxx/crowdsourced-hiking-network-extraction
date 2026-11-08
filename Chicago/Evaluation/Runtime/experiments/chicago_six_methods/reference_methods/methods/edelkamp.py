"""Resolve the method implementation in its named directory."""
import sys
from Chicago.Methods.Edelkamp import Chicago_Edelkamp_backend as implementation
sys.modules[__name__] = implementation
