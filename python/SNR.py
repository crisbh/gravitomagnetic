import numpy as np
from pathlib import Path
from classy import Class
import vp_utils as utils
import os

base_path = Path('~/nerding/gravitomagnetic/output_cosma').expanduser()

models = ['lcdm', 'frhs', 'ndgp']
surveys = ['LSST', 'Euclid']
experiments = ['Planck', 'SO']

pars_lcdm = utils.build_cosmo_params_from_file(base_path / "lcdm/parameters-usedvalues")
pars_frhs = utils.build_cosmo_params_from_file(base_path / "frhs/parameters-usedvalues")
pars_ndgp = utils.build_cosmo_params_from_file(base_path / "ndgp/parameters-usedvalues")

# Cosmological parameters
params_base = {
    'output': 'tCl,sCl,lCl,mPk',
    'A_s': 2.1e-9,
    'n_s': 0.965,
    'tau_reio': 0.06,
    'lensing': 'yes',
    'l_switch_limber': 50,
    'l_max_scalars': 10000
}

params_lcdm = params_base | {
    'h': pars_lcdm['h'],
    'Omega_cdm': pars_lcdm['Omega_m'] - pars_lcdm['Omega_b'],
    'Omega_b': pars_lcdm['Omega_b'],
}

params_frhs = params_base | {
    'h': pars_frhs['h'],
    'Omega_cdm': pars_frhs['Omega_m'] - pars_frhs['Omega_b'],
    'Omega_b': pars_frhs['Omega_b'],
}

params_ndgp = params_base | {
    'h': pars_ndgp['h'],
    'Omega_cdm': pars_ndgp['Omega_m'] - pars_ndgp['Omega_b'],
    'Omega_b': pars_ndgp['Omega_b'],
}

# Find C_ell_TT from CLASS for the different models
CLASS = {}
C_ell_TT = {}
ells = {}

for m in models:
    CLASS[m] = Class()
    CLASS[m].set(params_base)
    CLASS[m].compute()

    c_ell_unlensed = CLASS[m].raw_cl(10000)   # unlensed spectrum
    # cl_lensed = CLASS[m].lensed_cl(10000)  # lensed spectrum

    ells[m] = c_ell_unlensed['ell']
    C_ell_TT[m] = c_ell_unlensed['tt']  # unlensed tt power spectrum


# First define the survey and experiment parmeters
pars_surv, pars_exp = {}, {}
pars_surv['Euclid'], pars_surv['LSST'] = {}, {}
pars_exp['Planck'], pars_exp['SO'] = {}, {}

pars_surv['Euclid']['n_gal'] = 30
pars_surv['Euclid']['sigma_e'] = 0.22
pars_surv['Euclid']['f_sky'] = 0.36

pars_surv['LSST']['n_gal'] = 40
pars_surv['LSST']['sigma_e'] = 0.22
pars_surv['LSST']['f_sky'] = 0.5

pars_exp['Planck']['theta_fwhm'] = 5
pars_exp['Planck']['Delta_T'] = 3.1
pars_exp['Planck']['f_sky'] = 0.82
pars_exp['Planck']['T_bar'] = 2.7E6

pars_exp['SO']['theta_fwhm'] = 1.4
pars_exp['SO']['Delta_T'] = 6
pars_exp['SO']['f_sky'] = 0.4
pars_exp['SO']['T_bar'] = 2.7E6


# Useful functions for SNR
def noise_convergence(pars_surv):
    sigma_e = pars_surv['sigma_e']
    n_gal = pars_surv['n_gal']
    n_gal*= 60.**2          # galaxies per deg^2
    n_gal*=(180./np.pi)**2  # galaxies per rad^2
    return sigma_e**2/n_gal


def noise_temperature(ell, pars_exp):
    FWHM = pars_exp['theta_fwhm']
    FWHM/= 60.                       # arcmin to deg
    FWHM*= np.pi/180.                # deg to rad
    factor = (pars_exp['Delta_T']/pars_exp['T_bar'])**2
    arg_exp = ell**2 * FWHM**2/(8*np.log(2))
    return factor * np.exp(arg_exp)


def Cov(ell_list, C_ell_kappaB, C_ell_TT, C_ell_kappaGR, pars_surv, pars_exp):
    logell = np.log10(ell_list)
    dlog = logell[1] - logell[0]
    edges = 10**(np.r_[logell[0] - dlog/2, 0.5*(logell[:-1] + logell[1:]), logell[-1] + dlog/2])
    Delta_ell = np.diff(edges)

    factor = Delta_ell * pars_surv['f_sky'] * (2*ell_list + 1)
    contributions = C_ell_kappaB**2 + (C_ell_TT + noise_temperature(ell_list, pars_exp)*(C_ell_kappaGR + noise_convergence(pars_surv)))
    return contributions * factor


def SNR(ell_list, C_ell_kappaB, C_ell_TT, C_ell_kappaGR, pars_surv, pars_exp):
    return np.sqrt(C_ell_kappaB**2 / Cov(ell_list, C_ell_kappaB, C_ell_TT, C_ell_kappaGR, pars_surv, pars_exp))


def main():
    for m in models:
        path_C_ell = base_path / m
        C_ell_XY = np.load(path_C_ell / "C_ells_XY.npy", allow_pickle=True).item()
        ell_grid = np.load(path_C_ell / "ell_grid.npy")
        ell_idx = np.round(ell_grid).astype(int)

        signal_to_noise = SNR(ell_idx, C_ell_XY['B'], C_ell_TT[m][ell_idx], C_ell_XY['Phi'], pars_surv['LSST'], pars_exp['SO'])

        np.save(path_C_ell / "SNR.npy", signal_to_noise)

if __name__ == "__main__":
    main()
