"""
Unit tests for vp_utils.py — cosmological background functions and C_ell kernels.

conftest.py handles the module-level file-read mock so vp_utils imports cleanly
without COSMA data.  Tests here use a simple analytic power spectrum.
"""

import numpy as np
import pytest
import vp_utils as vp


# ---------------------------------------------------------------------------
# Analytic mock power spectrum: Pk(k) = 1e-3 * k^-2
# ---------------------------------------------------------------------------

def _pk_simple(x):
    """Accepts (k, z) tuple (Pk_evol=True path) or scalar k."""
    k = np.asarray(x[0] if isinstance(x, tuple) else x, dtype=float)
    return 1e-3 * k**(-2.0)


COMMON_KWARGS = dict(
    z_s=1.0,
    ell=500.0,
    kmin=1e-3,
    kmax=10.0,
    Pk=_pk_simple,
    Pk_evol=False,
    N_int=200,
)


# ---------------------------------------------------------------------------
# Background functions
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

@pytest.mark.parametrize("fn", [vp.C_ell_Phi, vp.C_ell_B, vp.C_ell_kSZ, vp.C_ell_B_X_kSZ])
def test_raise_invalid_method(fn):
    with pytest.raises(ValueError, match="Invalid selection"):
        fn(**COMMON_KWARGS, integr_method="bad_method")


# ---------------------------------------------------------------------------
# Integration method consistency
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
    """Smoke test — catches chi vs chi_x scope bug in the quad path."""
    result = vp.C_ell_kSZ(**COMMON_KWARGS, integr_method="quad")
    assert np.isfinite(result)
    assert result > 0


def test_c_ell_b_x_ksz_quad_returns_scalar():
    """Smoke test — catches chi vs chi_x scope bug in the quad path."""
    result = vp.C_ell_B_X_kSZ(**COMMON_KWARGS, integr_method="quad")
    assert np.isfinite(result)


def test_ksz_integration_methods_agree():
    """simpson vs quad should agree within 10% after n_ele factor fix."""
    result_simp = vp.C_ell_kSZ(**COMMON_KWARGS, integr_method="simpson")
    result_quad = vp.C_ell_kSZ(**COMMON_KWARGS, integr_method="quad")
    assert abs(result_simp - result_quad) / abs(result_simp) < 0.10


def test_b_x_ksz_integration_methods_agree():
    result_simp = vp.C_ell_B_X_kSZ(**COMMON_KWARGS, integr_method="simpson")
    result_quad = vp.C_ell_B_X_kSZ(**COMMON_KWARGS, integr_method="quad")
    assert abs(result_simp - result_quad) / abs(result_simp) < 0.10
