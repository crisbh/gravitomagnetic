"""
Integration / smoke tests for angular_powerspec_z.py.

These tests create a minimal synthetic dataset in a tmp directory and run
main() to verify the output structure is correct — in particular that the
C_ells/ subdirectory is created (previously missing) and snapshot count is
discovered dynamically rather than hardcoded.
"""

import sys
import json
import numpy as np
import pytest
from pathlib import Path
import unittest.mock as mock


def _write_synthetic_snapshot(pk_matter_dir, pk_curl_dir, idx, z, n_k=10):
    """Write minimal synthetic Pk_matter and Pk_curl .npy files."""
    k = np.logspace(-2, 0, n_k)
    Pk = 1e-2 * k**(-2)
    Pcurl = 1e-3 * k**(-2)

    np.save(pk_matter_dir / f"{idx:03d}.npy", {"k": k, "Pk": Pk, "z": z})
    np.save(pk_curl_dir / f"{idx:03d}.npy", {"k": k, "Pcurl": Pcurl, "z": z})


@pytest.fixture
def synthetic_input(tmp_path):
    """Create a minimal input directory with 3 synthetic snapshots."""
    in_dir = tmp_path / "input"
    pk_m = in_dir / "Pk_matter"
    pk_c = in_dir / "Pk_curl"
    pk_m.mkdir(parents=True)
    pk_c.mkdir(parents=True)

    z_values = [0.5, 1.0, 1.5]
    for i, z in enumerate(z_values):
        _write_synthetic_snapshot(pk_m, pk_c, i, z)

    return in_dir


def test_c_ells_dir_created(synthetic_input, tmp_path):
    """C_ells/ must be created; np.save would fail if it isn't."""
    out_dir = tmp_path / "output"

    # angular_powerspec_z.py calls vp_utils at import which tries to read a
    # parameters file.  Patch build_cosmo_params_from_file to avoid needing the
    # COSMA data path.
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from tests.test_vp_utils import MOCK_PARS  # reuse mock params

    with mock.patch("vp_utils.build_cosmo_params_from_file", return_value=MOCK_PARS):
        import importlib
        import vp_utils
        importlib.reload(vp_utils)

        import angular_powerspec_z as apz
        importlib.reload(apz)

        test_args = mock.MagicMock()
        test_args.in_dir = str(synthetic_input)
        test_args.out_dir = str(out_dir)
        test_args.z_source = 1.5

        with mock.patch("angular_powerspec_z.parse_args", return_value=test_args):
            apz.main()

    assert (out_dir / "C_ells").is_dir(), "C_ells/ subdirectory must be created"
    assert (out_dir / "C_ells" / "ell_grid_z=1.5.npy").exists()
    assert (out_dir / "C_ells" / "C_ells_XY_z=1.5.npy").exists()


def test_snapshot_count_dynamic(synthetic_input, tmp_path):
    """main() should load exactly 3 snapshots — not a hardcoded 25."""
    out_dir = tmp_path / "output2"

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from tests.test_vp_utils import MOCK_PARS

    loaded_indices = []
    original_load = np.load

    def tracking_load(path, **kwargs):
        p = Path(path)
        if "Pk_matter" in str(p) or "Pk_curl" in str(p):
            loaded_indices.append(p.stem)
        return original_load(path, **kwargs)

    with mock.patch("vp_utils.build_cosmo_params_from_file", return_value=MOCK_PARS):
        import importlib
        import vp_utils
        importlib.reload(vp_utils)
        import angular_powerspec_z as apz
        importlib.reload(apz)

        test_args = mock.MagicMock()
        test_args.in_dir = str(synthetic_input)
        test_args.out_dir = str(out_dir)
        test_args.z_source = 1.5

        with mock.patch("angular_powerspec_z.parse_args", return_value=test_args):
            with mock.patch("numpy.load", side_effect=tracking_load):
                apz.main()

    matter_loads = [x for x in loaded_indices if True]  # all tracked loads
    # 3 snapshots × 2 files (Pk_matter + Pk_curl) = 6 loads
    assert len(loaded_indices) == 6, f"Expected 6 file loads for 3 snapshots, got {len(loaded_indices)}"
