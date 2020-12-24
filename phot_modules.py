"""
    All the functions listed here requires the generation of the particle
    information file.
"""

import os
import sys
os.environ['FLARE'] = '/cosma7/data/dp004/dc-wilk2/flare'
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname("__file__"), '..')))
from functools import partial
import schwimmbad
from SynthObs.SED import models
import FLARE
import FLARE.filters
from FLARE.photom import lum_to_M
import h5py
from swiftsimio import load
import utilities as util


def DTM_fit(Z, Age):
    """
    Fit function from L-GALAXIES dust modeling
    Formula uses Age in Gyr while the supplied Age is in Myr
    """

    D0, D1, alpha, beta, gamma = 0.008, 0.329, 0.017, -1.337, 2.122
    tau = 5e-5 / (D0 * Z)
    DTM = D0 + (D1 - D0) * (1. - np.exp(-alpha * (Z ** beta)
                                        * ((Age / (1e3 * tau)) ** gamma)))
    if np.isnan(DTM) or np.isinf(DTM):
        DTM = 0.

    return DTM


def lum(num, data, kappa, z, BC_fac, cent, campos, IMF='Chabrier_300',
        filters=('FAKE.TH.FUV',), Type='Total', log10t_BC=7.,
        extinction='default', ):

    kinp = np.load('/cosma7/data/dp004/dc-payy1/my_files/'
                   'los/kernel_sph-anarchy.npz',
                   allow_pickle=True)
    lkernel = kinp['kernel']
    header = kinp['header']
    kbins = header.item()['bins']

    S_mass_ini = data.stars.initial_masses.value
    S_Z = data.stars.metal_mass_fractions.value
    S_age = util.calc_ages(z, data.stars.birth_scale_factors.value)
    G_Z = data.gas.metal_mass_fractions.value
    G_sml = data.stars.smoothing_lengths.value
    S_sml = data.gas.smoothing_lengths.value
    G_mass = data.gas.masses.value * 10 ** 10
    S_coords = data.stars.coordinates.value - cent
    G_coords = data.data.gas.coordinates.value - cent
    S_mass = data.stars.masses.value * 10 ** 10

    if S_sml.max() == 0.0:
        print("Ill-defined smoothing lengths")

        last_snap = "%04d" % (num - 1)

        # Define path
        path = '/cosma/home/dp004/dc-rope1/cosma7/SWIFT/hydro_1380_data/ani_hydro_' + last_snap + ".hdf5"

        olddata = load(path)
        old_hsmls = olddata.stars.smoothing_lengths.value
        S_sml[:old_hsmls.size] = old_hsmls
        S_sml[old_hsmls.size:] = np.median(old_hsmls)

    Lums = {f: np.zeros(len(S_mass), dtype=np.float64) for f in filters}

    model = models.define_model(
        F'BPASSv2.2.1.binary/{IMF}')  # DEFINE SED GRID -
    if extinction == 'default':
        model.dust_ISM = (
            'simple', {'slope': -1.})  # Define dust curve for ISM
        model.dust_BC = ('simple', {
            'slope': -1.})  # Define dust curve for birth cloud component
    elif extinction == 'Calzetti':
        model.dust_ISM = ('Starburst_Calzetti2000', {''})
        model.dust_BC = ('Starburst_Calzetti2000', {''})
    elif extinction == 'SMC':
        model.dust_ISM = ('SMC_Pei92', {''})
        model.dust_BC = ('SMC_Pei92', {''})
    elif extinction == 'MW':
        model.dust_ISM = ('MW_Pei92', {''})
        model.dust_BC = ('MW_Pei92', {''})
    elif extinction == 'N18':
        model.dust_ISM = ('MW_N18', {''})
        model.dust_BC = ('MW_N18', {''})
    else:
        ValueError("Extinction type not recognised")

    # Convert coordinates to physical
    S_coords = S_coords / (1 + z)
    G_coords = G_coords / (1 + z)

    # --- create rest-frame luminosities
    F = FLARE.filters.add_filters(filters, new_lam=model.lam)
    model.create_Lnu_grid(
        F)  # --- create new L grid for each filter. In units of erg/s/Hz

    okinds = G_coords[:, 2] < campos

    MetSurfaceDensities = util.get_Z_LOS(S_coords, G_coords[okinds, :],
                                         G_mass[okinds], G_Z[okinds],
                                         G_sml[okinds], (0, 1, 2),
                                         lkernel, kbins)

    Mage = np.nansum(S_mass_ini * S_age) / np.nansum(S_mass_ini)
    Z = np.nanmean(G_Z[okinds])

    MetSurfaceDensities = DTM_fit(Z, Mage) * MetSurfaceDensities

    if Type == 'Total':
        # --- calculate V-band (550nm) optical depth for each star particle
        tauVs_ISM = kappa * MetSurfaceDensities
        tauVs_BC = BC_fac * (S_Z / 0.01)
        fesc = 0.0

    elif Type == 'Pure-stellar':
        tauVs_ISM = np.zeros(len(S_mass_ini))
        tauVs_BC = np.zeros(len(S_mass_ini))
        fesc = 1.0

    elif Type == 'Intrinsic':
        tauVs_ISM = np.zeros(len(S_mass_ini))
        tauVs_BC = np.zeros(len(S_mass_ini))
        fesc = 0.0

    elif Type == 'Only-BC':
        tauVs_ISM = np.zeros(len(S_mass_ini))
        tauVs_BC = BC_fac * (S_Z / 0.01)
        fesc = 0.0

    else:
        tauVs_ISM = None
        tauVs_BC = None
        fesc = None
        ValueError(F"Undefined Type {Type}")

    # --- calculate rest-frame Luminosity. In units of erg/s/Hz
    for f in filters:
        Lnu = models.generate_Lnu_array(model, S_mass_ini, S_age, S_Z,
                                        tauVs_ISM, tauVs_BC, F, f,
                                        fesc=fesc, log10t_BC=log10t_BC)
        Lums[f] = Lnu

    return Lums
