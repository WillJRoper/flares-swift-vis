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


def lum(num, data, kappa, z, BC_fac, cent, IMF='Chabrier_300',
        filters=('FAKE.TH.FUV',), Type='Total', log10t_BC=7.,
        extinction='default'):

    kinp = np.load('/cosma7/data/dp004/dc-payy1/my_files/'
                   'los/kernel_sph-anarchy.npz',
                   allow_pickle=True)
    lkernel = kinp['kernel']
    header = kinp['header']
    kbins = header.item()['bins']

    # print(data.metadata.gas_properties.field_names)
    print(data.metadata.stellar_properties.field_names)

    # S_mass_ini = data.
    S_Z = data.stars.metallicities.value
    # S_age = data.
    G_Z = data.gas.metallicities.value
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


    MetSurfaceDensities = util.get_Z_LOS(S_coords, G_coords,
                                         G_mass, G_Z,
                                         G_sml, (0, 1, 2),
                                         lkernel, kbins)

    # Mage = np.nansum(S_mass_ini * S_age) / np.nansum(Masses)
    # Z = np.nanmean(gasMetallicities)
    #
    # MetSurfaceDensities = DTM_fit(Z, Mage) * MetSurfaceDensities
    #
    # if Type == 'Total':
    #     # --- calculate V-band (550nm) optical depth for each star particle
    #     tauVs_ISM = kappa * MetSurfaceDensities
    #     tauVs_BC = BC_fac * (Metallicities / 0.01)
    #     fesc = 0.0
    #
    # elif Type == 'Pure-stellar':
    #     tauVs_ISM = np.zeros(len(Masses))
    #     tauVs_BC = np.zeros(len(Masses))
    #     fesc = 1.0
    #
    # elif Type == 'Intrinsic':
    #     tauVs_ISM = np.zeros(len(Masses))
    #     tauVs_BC = np.zeros(len(Masses))
    #     fesc = 0.0
    #
    # elif Type == 'Only-BC':
    #     tauVs_ISM = np.zeros(len(Masses))
    #     tauVs_BC = BC_fac * (Metallicities / 0.01)
    #     fesc = 0.0
    #
    # else:
    #     tauVs_ISM = None
    #     tauVs_BC = None
    #     fesc = None
    #     ValueError(F"Undefined Type {Type}")
    #
    # # --- calculate rest-frame Luminosity. In units of erg/s/Hz
    # for f in filters:
    #     Lnu = models.generate_Lnu_array(model, Masses, Ages, Metallicities,
    #                                     tauVs_ISM, tauVs_BC, F, f,
    #                                     fesc=fesc, log10t_BC=log10t_BC)
    #     Lums[f] = Lnu
    #
    # Lums["coords"] = S_coords
    # Lums["smls"] = S_sml
    # Lums["masses"] = S_mass
    # Lums["nstar"] = S_len
    # Lums["begin"] = begin
    # Lums["end"] = end
    #
    # return Lums  # , S_len + G_len


def flux(sim, kappa, tag, BC_fac, inp='FLARES', IMF='Chabrier_300',
         filters=FLARE.filters.NIRCam_W, Type='Total', log10t_BC=7.,
         extinction='default', orientation="sim"):
    kinp = np.load('/cosma/home/dp004/dc-rope1/cosma7/FLARES/'
                   'flares/los_extinction/kernel_sph-anarchy.npz',
                   allow_pickle=True)
    lkernel = kinp['kernel']
    header = kinp['header']
    kbins = header.item()['bins']

    S_mass, S_Z, S_age, S_los, G_Z, S_len, \
    G_len, G_sml, S_sml, G_mass, S_coords, G_coords, \
    S_vels, G_vels, cops, \
    begin, end, gbegin, gend = get_data(sim, tag, inp)

    Fnus = {f: np.zeros(len(S_mass), dtype=np.float64) for f in filters}

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

    z = float(tag[5:].replace('p', '.'))
    F = FLARE.filters.add_filters(filters, new_lam=model.lam * (1. + z))

    cosmo = FLARE.default_cosmo()

    # --- create new Fnu grid for each filter. In units of nJy/M_sol
    model.create_Fnu_grid(F, z, cosmo)

    for jj in range(len(begin)):

        # Extract values for this galaxy
        Masses = S_mass[begin[jj]: end[jj]]
        Ages = S_age[begin[jj]: end[jj]]
        Metallicities = S_Z[begin[jj]: end[jj]]
        gasMetallicities = G_Z[begin[jj]: end[jj]]
        gasSML = G_sml[begin[jj]: end[jj]]
        gasMasses = G_mass[begin[jj]: end[jj]]

        if orientation == "sim":

            starCoords = S_coords[:, begin[jj]: end[jj]].T - cops[:, jj]
            S_coords[:, begin[jj]: end[jj]] = starCoords.T

            MetSurfaceDensities = S_los[begin[jj]:end[jj]]

        elif orientation == "face-on":

            starCoords = S_coords[:, begin[jj]: end[jj]].T - cops[:, jj]
            gasCoords = G_coords[:, begin[jj]: end[jj]].T - cops[:, jj]
            gasVels = G_vels[:, begin[jj]: end[jj]].T

            # Get angular momentum vector
            ang_vec = util.ang_mom_vector(gasMasses, gasCoords, gasVels)

            # Rotate positions
            starCoords = util.get_rotated_coords(ang_vec, starCoords)
            gasCoords = util.get_rotated_coords(ang_vec, gasCoords)
            S_coords[:, begin[jj]: end[jj]] = starCoords.T

            MetSurfaceDensities = util.get_Z_LOS(starCoords, gasCoords,
                                                 gasMasses, gasMetallicities,
                                                 gasSML, (0, 1, 2),
                                                 lkernel, kbins)
        elif orientation == "side-on":

            starCoords = S_coords[:, begin[jj]: end[jj]].T - cops[:, jj]
            gasCoords = G_coords[:, begin[jj]: end[jj]].T - cops[:, jj]
            gasVels = G_vels[:, begin[jj]: end[jj]].T

            # Get angular momentum vector
            ang_vec = util.ang_mom_vector(gasMasses, gasCoords, gasVels)

            # Rotate positions
            starCoords = util.get_rotated_coords(ang_vec, starCoords)
            gasCoords = util.get_rotated_coords(ang_vec, gasCoords)
            S_coords[:, begin[jj]: end[jj]] = starCoords.T

            MetSurfaceDensities = util.get_Z_LOS(starCoords, gasCoords,
                                                 gasMasses, gasMetallicities,
                                                 gasSML, (2, 0, 1),
                                                 lkernel, kbins)
        else:
            MetSurfaceDensities = None
            print(orientation,
                  "is not an recognised orientation. "
                  "Accepted types are 'sim', 'face-on', or 'side-on'")

        Mage = np.nansum(Masses * Ages) / np.nansum(Masses)
        Z = np.nanmean(gasMetallicities)

        MetSurfaceDensities = DTM_fit(Z, Mage) * MetSurfaceDensities

        if Type == 'Total':
            # --- calculate V-band (550nm) optical depth for each star particle
            tauVs_ISM = kappa * MetSurfaceDensities
            tauVs_BC = BC_fac * (Metallicities / 0.01)
            fesc = 0.0

        elif Type == 'Pure-stellar':
            tauVs_ISM = np.zeros(len(Masses))
            tauVs_BC = np.zeros(len(Masses))
            fesc = 1.0

        elif Type == 'Intrinsic':
            tauVs_ISM = np.zeros(len(Masses))
            tauVs_BC = np.zeros(len(Masses))
            fesc = 0.0

        elif Type == 'Only-BC':
            tauVs_ISM = np.zeros(len(Masses))
            tauVs_BC = BC_fac * (Metallicities / 0.01)
            fesc = 0.0

        else:
            tauVs_ISM = None
            tauVs_BC = None
            fesc = None
            ValueError(F"Undefined Type {Type}")

        # --- calculate rest-frame Luminosity. In units of erg/s/Hz
        for f in filters:
            # --- calculate rest-frame flux of each object in nJy
            Fnu = models.generate_Fnu_array(model, Masses, Ages, Metallicities,
                                            tauVs_ISM, tauVs_BC, F, f,
                                            fesc=fesc, log10t_BC=log10t_BC)

            Fnus[f][begin[jj]: end[jj]] = Fnu

    Fnus["coords"] = S_coords
    Fnus["smls"] = S_sml
    Fnus["begin"] = begin
    Fnus["end"] = end

    return Fnus


def get_lines(sim, kappa, tag, BC_fac, inp='FLARES', IMF='Chabrier_300',
              LF=False, lines='HI6563', Type='Total', log10t_BC=7.,
              extinction='default', orientation="sim"):
    kinp = np.load('/cosma/home/dp004/dc-rope1/cosma7/FLARES/'
                   'flares/los_extinction/kernel_sph-anarchy.npz',
                   allow_pickle=True)
    lkernel = kinp['kernel']
    header = kinp['header']
    kbins = header.item()['bins']

    S_mass, S_Z, S_age, S_los, G_Z, S_len, \
    G_len, G_sml, S_sml, G_mass, S_coords, G_coords, \
    S_vels, G_vels, cops, \
    begin, end, gbegin, gend = get_data(sim, tag, inp)

    # --- calculate intrinsic quantities
    if extinction == 'default':
        dust_ISM = ('simple', {'slope': -1.})  # Define dust curve for ISM
        dust_BC = ('simple', {
            'slope': -1.})  # Define dust curve for birth cloud component
    elif extinction == 'Calzetti':
        dust_ISM = ('Starburst_Calzetti2000', {''})
        dust_BC = ('Starburst_Calzetti2000', {''})
    elif extinction == 'SMC':
        dust_ISM = ('SMC_Pei92', {''})
        dust_BC = ('SMC_Pei92', {''})
    elif extinction == 'MW':
        dust_ISM = ('MW_Pei92', {''})
        dust_BC = ('MW_Pei92', {''})
    elif extinction == 'N18':
        dust_ISM = ('MW_N18', {''})
        dust_BC = ('MW_N18', {''})
    else:
        ValueError("Extinction type not recognised")

    lum = np.zeros(len(begin), dtype=np.float64)
    EW = np.zeros(len(begin), dtype=np.float64)

    # --- initialise model with SPS model and IMF.
    # Set verbose=True to see a list of available lines.
    m = models.EmissionLines(F'BPASSv2.2.1.binary/{IMF}', dust_BC=dust_BC,
                             dust_ISM=dust_ISM, verbose=False)
    for jj in range(len(begin)):

        # Extract values for this galaxy
        Masses = S_mass[begin[jj]: end[jj]]
        Ages = S_age[begin[jj]: end[jj]]
        Metallicities = S_Z[begin[jj]: end[jj]]
        gasMetallicities = G_Z[begin[jj]: end[jj]]
        gasSML = G_sml[begin[jj]: end[jj]]
        gasMasses = G_mass[begin[jj]: end[jj]]

        if orientation == "sim":

            MetSurfaceDensities = S_los[begin[jj]:end[jj]]

        elif orientation == "face-on":

            starCoords = S_coords[:, begin[jj]: end[jj]].T - cops[:, jj]
            gasCoords = G_coords[:, begin[jj]: end[jj]].T - cops[:, jj]
            gasVels = G_vels[:, begin[jj]: end[jj]].T

            # Get angular momentum vector
            ang_vec = util.ang_mom_vector(gasMasses, gasCoords, gasVels)

            # Rotate positions
            starCoords = util.get_rotated_coords(ang_vec, starCoords)
            gasCoords = util.get_rotated_coords(ang_vec, gasCoords)

            MetSurfaceDensities = util.get_Z_LOS(starCoords, gasCoords,
                                                 gasMasses,
                                                 gasMetallicities,
                                                 gasSML, (0, 1, 2),
                                                 lkernel, kbins)
        elif orientation == "side-on":

            starCoords = S_coords[:, begin[jj]: end[jj]].T - cops[:, jj]
            gasCoords = G_coords[:, begin[jj]: end[jj]].T - cops[:, jj]
            gasVels = G_vels[:, begin[jj]: end[jj]].T

            # Get angular momentum vector
            ang_vec = util.ang_mom_vector(gasMasses, gasCoords, gasVels)

            # Rotate positions
            starCoords = util.get_rotated_coords(ang_vec, starCoords)
            gasCoords = util.get_rotated_coords(ang_vec, gasCoords)

            MetSurfaceDensities = util.get_Z_LOS(starCoords, gasCoords,
                                                 gasMasses,
                                                 gasMetallicities,
                                                 gasSML, (2, 0, 1),
                                                 lkernel, kbins)
        else:
            MetSurfaceDensities = None
            print(orientation,
                  "is not an recognised orientation. "
                  "Accepted types are 'sim', 'face-on', or 'side-on'")

        if Type == 'Total':
            # --- calculate V-band (550nm) optical depth for each star particle
            tauVs_ISM = kappa * MetSurfaceDensities
            tauVs_BC = BC_fac * (Metallicities / 0.01)
            fesc = 0.0

        elif Type == 'Pure-stellar':
            tauVs_ISM = np.zeros(len(Masses))
            tauVs_BC = np.zeros(len(Masses))
            fesc = 1.0

        elif Type == 'Intrinsic':
            tauVs_ISM = np.zeros(len(Masses))
            tauVs_BC = np.zeros(len(Masses))
            fesc = 0.0

        elif Type == 'Only-BC':
            tauVs_ISM = np.zeros(len(Masses))
            tauVs_BC = BC_fac * (Metallicities / 0.01)
            fesc = 0.0

        else:
            tauVs_ISM = None
            tauVs_BC = None
            fesc = None
            ValueError(F"Undefined Type {Type}")

        o = m.get_line_luminosity(lines, Masses, Ages, Metallicities,
                                  tauVs_BC=tauVs_BC, tauVs_ISM=tauVs_ISM,
                                  verbose=False, log10t_BC=log10t_BC)

        lum[jj] = o['luminosity']
        EW[jj] = o['EW']

    return lum, EW


def get_lum(sim, kappa, tag, BC_fac, IMF='Chabrier_300',
            bins=np.arange(-24, -16, 0.5), inp='FLARES', LF=True,
            filters=('FAKE.TH.FUV'), Type='Total', log10t_BC=7.,
            extinction='default', orientation="sim", masslim=None):

    try:
        Lums = lum(sim, kappa, tag, BC_fac=BC_fac, IMF=IMF, inp=inp, LF=LF,
                   filters=filters, Type=Type, log10t_BC=log10t_BC,
                   extinction=extinction, orientation=orientation,
                   masslim=masslim)

    except Exception as e:
        Lums = {f: np.array([], dtype=np.float64) for f in filters}
        Lums["coords"] = np.array([], dtype=np.float64)
        Lums["smls"] = np.array([], dtype=np.float64)
        Lums["masses"] = np.array([], dtype=np.float64)
        Lums["nstar"] = np.array([], dtype=np.float64)
        Lums["begin"] = np.array([], dtype=np.float64)
        Lums["end"] = np.array([], dtype=np.float64)
        print(e)

    if LF:
        tmp, edges = np.histogram(lum_to_M(Lums), bins=bins)
        return tmp

    else:
        return Lums


def get_lum_all(kappa, tag, BC_fac, IMF='Chabrier_300',
                bins=np.arange(-24, -16, 0.5), inp='FLARES', LF=True,
                filters=('FAKE.TH.FUV'), Type='Total', log10t_BC=7.,
                extinction='default', orientation="sim", numThreads=8,
                masslim=None):
    print(f"Getting luminosities for tag {tag} with kappa={kappa}")

    if inp == 'FLARES':
        df = pd.read_csv('../weight_files/weights_grid.txt')
        weights = np.array(df['weights'])

        sims = np.arange(0, len(weights))

        calc = partial(get_lum, kappa=kappa, tag=tag, BC_fac=BC_fac, IMF=IMF,
                       bins=bins, inp=inp, LF=LF, filters=filters, Type=Type,
                       log10t_BC=log10t_BC, extinction=extinction,
                       orientation=orientation, masslim=masslim)

        pool = schwimmbad.MultiPool(processes=numThreads)
        dat = np.array(list(pool.map(calc, sims)))
        pool.close()

        if LF:
            hist = np.sum(dat, axis=0)
            out = np.zeros(len(bins) - 1)
            err = np.zeros(len(bins) - 1)
            for ii, sim in enumerate(sims):
                err += np.square(np.sqrt(dat[ii]) * weights[ii])
                out += dat[ii] * weights[ii]

            return out, hist, np.sqrt(err)

        else:
            return dat

    else:
        out = get_lum(00, kappa=kappa, tag=tag, BC_fac=BC_fac, IMF=IMF,
                      bins=bins, inp=inp, LF=LF, filters=filters, Type=Type,
                      log10t_BC=log10t_BC, extinction=extinction,
                      orientation=orientation, masslim=masslim)

        return out


def get_flux(sim, kappa, tag, BC_fac, IMF='Chabrier_300', inp='FLARES',
             filters=FLARE.filters.NIRCam, Type='Total', log10t_BC=7.,
             extinction='default', orientation="sim"):
    try:
        Fnus = flux(sim, kappa, tag, BC_fac=BC_fac, IMF=IMF, inp=inp,
                    filters=filters, Type=Type, log10t_BC=log10t_BC,
                    extinction=extinction, orientation=orientation)

    except Exception as e:
        Fnus = np.ones(len(filters)) * np.nan
        print(e)

    return Fnus


def get_flux_all(kappa, tag, BC_fac, IMF='Chabrier_300', inp='FLARES',
                 filters=FLARE.filters.NIRCam, Type='Total', log10t_BC=7.,
                 extinction='default', orientation="sim", numThreads=8):
    print(f"Getting fluxes for tag {tag} with kappa={kappa}")

    if inp == 'FLARES':

        df = pd.read_csv('../weight_files/weights_grid.txt')
        weights = np.array(df['weights'])

        sims = np.arange(0, len(weights))

        calc = partial(get_flux, kappa=kappa, tag=tag, BC_fac=BC_fac, IMF=IMF,
                       inp=inp, filters=filters, Type=Type,
                       log10t_BC=log10t_BC, extinction=extinction,
                       orientation=orientation)

        pool = schwimmbad.MultiPool(processes=numThreads)
        out = np.array(list(pool.map(calc, sims)))
        pool.close()

    else:

        out = get_flux(00, kappa=kappa, tag=tag, BC_fac=BC_fac, IMF=IMF,
                       inp=inp, filters=filters, Type=Type,
                       log10t_BC=log10t_BC, extinction=extinction,
                       orientation=orientation)

    return out
