"""
Unit tests for vp_utils.py — cosmological background functions and C_ell kernels.

These tests are self-contained: they use a simple analytic power spectrum so no
simulation data is required.
"""

import sys
from pathlib import Path
import numpy as np
import pytest

# vp_utils reads a parameters file at import time from a hardcoded path that only
# exists on the COSMA cluster.  We monkeypatch build_cosmo_params_from_file before
# importing so the module can load in CI without the data directory.

MOCK_PARS = {
    "Omega_m": 0.3089,
    "Omega_Lambda": 0.6911,
    "Omega_b": 0.0486,
    "h": 0.6774,
    "H0": 67.74,
    "BoxSize": 500.0,
    "c": 299792.458,
    "m_ele": 9.11e-31,
    "m_H": 1.67e-27,
    "m_He": 6.65e-27,
    "SigmaT": 6.6524e-29,
    "Ombh2": 0.0486 * 0.6774**2,
    "xe": 1.0,
    "Yp": 0.25,
    "rho_c": 1.88e-26 * 0.6774**2,
    "tau_H": 0.07 * 0.75 * 0.0486 * 0.6774**2 / 0.6774,
    "kSZ_bfac": 0.07 * 0.75 * 0.0486 * 0.6774**2 / 0.6774,
    "w_de": -1,
    "khN": 1024 / 500 * np.pi,
    "kN": 1024 / 500 * np.pi * 0.6774,
    "khF": 1 / 500,
    "kF": 1 / 500 * 0.6774,
    "UnitLength_in_cm": 3.08568e24,
    "UnitMass_in_g": 1.989e43,
    "UnitVelocity_in_cm_per_s": 100000.0,
    "simfile_raw": {},
}

# Patch before importing the module so the file-read at module level is bypassed.
import unittest.mock as mock

_patch = mock.patch(
    "vp_utils.build_cosmo_params_from_file",
    return_value=MOCK_PARS,
)

with _patch:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    import importlib
    import vp_utils as _vp_raw  # noqa: E402 — first import triggers module-level code

# Re-expose with the mock so we can use it in tests.
# Since build_cosmo_params_from_file already ran during import, we just use the module.
import vp_utils as vp  # noqa: E402


# ---------------------------------------------------------------------------
# Analytic mock power spectrum: Pk(k) = A * k^n, slope -2 for convergence tests
# ---------------------------------------------------------------------------

def _pk_simple(x):
    """Analytic mock Pk — accepts (k, z) tuple or scalar k."""
    if isinstance(x, tuple):
        k, z = x
    else:
        k = x
    k = np.asarray(k, dtype=float)
    return 1e-3 * k**(-2.0)


# ---------------------------------------------------------------------------
# Background function tests
# ---------------------------------------------------------------------------

def test_chi_of_z_zero():
    assert abs(vp.chi_of_z(0.0)) < 1.0  # should be ~0 Mpc


def test_chi_monotone():
    z = np.linspace(0.1, 3.0, 50)
    chi = vp.chi_of_z(z)
    assert np.all(np.diff(chi) > 0)


def test_hubble_at_zero():
    H0 = vp.parameters_sim["H0"]
    assert abs(vp.Hubble(0.0) - H0) / H0 < 1e-6


def test_n_ele_positive():
    z = np.linspace(0, 3, 20)
    assert np.all(vp.n_ele(z) > 0)


# ---------------------------------------------------------------------------
# ValueError for invalid integration method
# ---------------------------------------------------------------------------

COMMON_KWARGS = dict(
    z_s=1.0,
    ell=500.0,
    kmin=1e-3,
    kmax=10.0,
    Pk=_pk_simple,
    Pk_evol=False,
    N_int=200,
)


@pytest.mark.parametrize("fn", [vp.C_ell_Phi, vp.C_ell_B, vp.C_ell_kSZ, vp.C_ell_B_X_kSZ])
def test_raise_invalid_method(fn):
    with pytest.raises(ValueError, match="Invalid selection"):
        fn(**COMMON_KWARGS, integr_method="bad_method")


# ---------------------------------------------------------------------------
# Integration method consistency: simpson vs trapezoid vs quad
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fn", [vp.C_ell_Phi, vp.C_ell_B])
def test_integration_methods_agree(fn):
    result_simp = fn(**COMMON_KWARGS, integr_method="simpson")
    result_trap = fn(**COMMON_KWARGS, integr_method="trapezoid")
    result_quad = fn(**COMMON_KWARGS, integr_method="quad")

    assert result_simp > 0
    assert abs(result_simp - result_trap) / abs(result_simp) < 0.05
    assert abs(result_simp - result_quad) / abs(result_simp) < 0.05


def test_c_ell_ksz_quad_returns_scalar():
    """Smoke test that C_ell_kSZ with quad returns a finite float (catches chi vs chi_x bug)."""
    result = vp.C_ell_kSZ(**COMMON_KWARGS, integr_method="quad")
    assert np.isfinite(result)
    assert result > 0


def test_c_ell_b_x_ksz_quad_returns_scalar():
    """Smoke test that C_ell_B_X_kSZ with quad returns a finite float (catches chi vs chi_x bug)."""
    result = vp.C_ell_B_X_kSZ(**COMMON_KWARGS, integr_method="quad")
    assert np.isfinite(result)


def test_ksz_integration_methods_agree():
    """C_ell_kSZ simpson vs quad should agree within 10% after n_ele factor fix."""
    result_simp = vp.C_ell_kSZ(**COMMON_KWARGS, integr_method="simpson")
    result_quad = vp.C_ell_kSZ(**COMMON_KWARGS, integr_method="quad")
    assert abs(result_simp - result_quad) / abs(result_simp) < 0.10


def test_b_x_ksz_integration_methods_agree():
    result_simp = vp.C_ell_B_X_kSZ(**COMMON_KWARGS, integr_method="simpson")
    result_quad = vp.C_ell_B_X_kSZ(**COMMON_KWARGS, integr_method="quad")
    assert abs(result_simp - result_quad) / abs(result_simp) < 0.10
