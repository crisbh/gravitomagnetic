"""
conftest.py — pytest configuration for the python/tests suite.

vp_utils.py reads a COSMA-specific parameters file at module level.  We intercept
Path.exists and Path.open for that one path before importing so the module loads
cleanly in CI and local environments that don't have the simulation data.
"""

import sys
from io import StringIO
from pathlib import Path

# Make sure `python/` is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

_PARAMS_CONTENT = (
    "Omega0 0.3089\n"
    "OmegaLambda 0.6911\n"
    "OmegaBaryon 0.0486\n"
    "HubbleParam 0.6774\n"
    "BoxSize 500.0\n"
    "UnitLength_in_cm 3.08568e+24\n"
    "UnitMass_in_g 1.989e+43\n"
    "UnitVelocity_in_cm_per_s 100000.0\n"
)

_orig_path_open = Path.open
_orig_path_exists = Path.exists


def _patched_path_open(self, *args, **kwargs):
    if "parameters-usedvalues" in str(self):
        return StringIO(_PARAMS_CONTENT)
    return _orig_path_open(self, *args, **kwargs)


def _patched_path_exists(self):
    if "parameters-usedvalues" in str(self):
        return True
    return _orig_path_exists(self)


Path.open = _patched_path_open
Path.exists = _patched_path_exists
try:
    import vp_utils  # noqa: F401 — triggers module-level build with mock data
finally:
    Path.open = _orig_path_open
    Path.exists = _orig_path_exists
