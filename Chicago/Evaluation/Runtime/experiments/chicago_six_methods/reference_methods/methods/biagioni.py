"""Resolve the method implementation in its named directory."""
import sys
from Chicago.Methods.Biagioni import Chicago_Biagioni_backend as implementation
sys.modules[__name__] = implementation
