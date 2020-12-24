import os
import warnings
from scipy.interpolate import interp1d
import astropy.units as u
import matplotlib
import numba as nb
import numpy as np
from astropy.cosmology import Planck13 as cosmo
from photutils import aperture_photometry
from scipy.interpolate import interp1d

os.environ['FLARE'] = '/cosma7/data/dp004/dc-wilk2/flare'

matplotlib.use('Agg')
warnings.filterwarnings('ignore')


def calc_ages(z, a_born):
    # Convert scale factor into redshift
    z_born = 1 / a_born - 1

    # Convert to time in Gyrs
    t = cosmo.age(z)
    t_born = cosmo.age(z_born)

    # Calculate the VR
    ages = (t - t_born).to(u.Myr)

    return ages.value


def calc_srf(z, a_born, mass, t_bin=100):
    # Convert scale factor into redshift
    z_born = 1 / a_born - 1

    # Convert to time in Gyrs
    t = cosmo.age(z)
    t_born = cosmo.age(z_born)

    # Calculate the VR
    age = (t - t_born).to(u.Myr)

    ok = np.where(age.value <= t_bin)[0]
    if len(ok) > 0:

        # Calculate the SFR
        sfr = np.sum(mass[ok]) / (t_bin * 1e6)

    else:
        sfr = 0.0

    return sfr


def calc_srf_from_age(age, mass, t_bin=100):
    ok = np.where(age <= t_bin)[0]
    if len(ok) > 0:

        # Calculate the SFR
        sfr = np.sum(mass[ok]) / (t_bin * 1e6)

    else:
        sfr = np.zeros(len(ok))

    return sfr, ok


def get_Z_LOS(s_cood, g_cood, g_mass, g_Z, g_sml, dimens, lkernel, kbins):
    """

    Compute the los metal surface density (in g/cm^2) for star
    particles inside the galaxy taking the z-axis as the los.

    Args:
        s_cood (3d array): stellar particle coordinates
        g_cood (3d array): gas particle coordinates
        g_mass (1d array): gas particle mass
        g_Z (1d array): gas particle metallicity
        g_sml (1d array): gas particle smoothing length

    """

    conv = (u.solMass/u.Mpc**2).to(u.solMass/u.pc**2)

    n = s_cood.shape[0]
    Z_los_SD = np.zeros(n)
    # Fixing the observer direction as z-axis. Use make_faceon() for changing the
    # particle orientation to face-on
    xdir, ydir, zdir = dimens
    for ii in range(n):
        thisspos = s_cood[ii]
        ok = np.where(g_cood[:, zdir] > thisspos[zdir])[0]
        thisgpos = g_cood[ok]
        thisgsml = g_sml[ok]
        thisgZ = g_Z[ok]
        thisgmass = g_mass[ok]
        x = thisgpos[:, xdir] - thisspos[xdir]
        y = thisgpos[:, ydir] - thisspos[ydir]

        b = np.sqrt(x * x + y * y)
        boverh = b / thisgsml

        ok = np.where(boverh <= 1.)[0]
        kernel_vals = np.array([lkernel[int(kbins * ll)] for ll in boverh[ok]])

        Z_los_SD[ii] = np.sum((thisgmass[ok] * thisgZ[ok] / (
                    thisgsml[ok] * thisgsml[
                ok])) * kernel_vals) * conv  # in units of Msun/pc^2

    return Z_los_SD


def get_rotation_matrix(i_v, unit=None):
    """

    Gives the rotation matrix to orient the z-axis across
    the unit vector 'i_v'

    Args:
        i_v - unit vector

    Returns:
        rotation matrix


    """

    # This solution is from ---
    # https://stackoverflow.com/questions/43507491/imprecision-with-rotation-matrix-to-align-a-vector-to-an-axis

    # This uses the Rodrigues' rotation formula for the re-projection

    # From http://www.j3d.org/matrix_faq/matrfaq_latest.html#Q38
    if unit is None:
        unit = [0.0, 0.0, 1.0]
    # Normalize vector length
    i_v /= np.linalg.norm(i_v)

    # Get axis
    uvw = np.cross(i_v, unit)

    # compute trig values - no need to go through arccos and back
    rcos = np.dot(i_v, unit)
    rsin = np.linalg.norm(uvw)

    # normalize and unpack axis
    if not np.isclose(rsin, 0):
        uvw /= rsin
    u, v, w = uvw

    # Compute rotation matrix - re-expressed to show structure
    return (
            rcos * np.eye(3) +
            rsin * np.array([
        [0, -w, v],
        [w, 0, -u],
        [-v, u, 0]
    ]) +
            (1.0 - rcos) * uvw[:, None] * uvw[None, :]
    )


def get_rotated_coords(vec, coords):
    """

    Given the unit vector (in cartesian), 'vec', generates
    the rotation matrix and rotates the given 'coords' to
    align the z-axis along the unit vector, 'vec'

    Args:
        vec, coords - unit vector to rotate to, coordinates

    Returns:
        rot_coords: rotated coordinates

    """

    rot = get_rotation_matrix(vec)
    rot_coords = (rot @ coords.T).T

    return rot_coords


def ang_mom_vector(this_mass, this_cood, this_vel):
    """

        Get the angular momentum unit vector


    """
    L_tot = np.array([this_mass]).T * np.cross(this_cood, this_vel)
    L_tot_mag = np.sqrt(np.sum(np.nansum(L_tot, axis=0) ** 2))
    L_unit = np.sum(L_tot, axis=0) / L_tot_mag

    return L_unit


def kappa(this_mass, this_coord, this_vel):
    """

    Described in Correa et al.(2017)
    Gives the kinetic energy invested in ordered rotation
    Kco - KE invested in just the co-rotating particles
    Krot - KE invested in ordered rotation


    """

    L_tot = np.array([this_mass]).T * np.cross(this_coord, this_vel)
    L_tot_mag = np.sqrt(np.sum(np.nansum(L_tot, axis=0) ** 2))

    L_unit = np.sum(L_tot, axis=0) / L_tot_mag

    R_z = np.cross(this_scoord, L_unit)
    absR_z = np.sqrt(np.sum(R_z ** 2, axis=1))
    mR = this_mass * absR_z
    K = np.nansum(this_mass * np.sum(this_vel ** 2, axis=1))

    L = np.sum(L_tot * L_unit, axis=1)
    L_co = np.copy(L)
    co = np.where(L_co > 0.)
    L_co = L_co[co]

    L_mR = (L / mR) ** 2
    L_co_mR = (L_co / mR[co]) ** 2
    Krot = np.nansum(this_mass * L_mR) / K

    Kco = np.nansum(this_mass[co] * L_co_mR) / K

    return Kco, Krot


def calc_eigenvec(coods):
    """
    Provided by Chris Lovell, calculates the triaxial quantities
    and the eigen vectors

    Args:
        coods - normed coordinates

    Returns:
        [a, b, c]:
        e_vectors:
    """

    I = np.zeros((3, 3))

    I[0, 0] = np.sum(coods[:, 1] ** 2 + coods[:, 2] ** 2)
    I[1, 1] = np.sum(coods[:, 0] ** 2 + coods[:, 2] ** 2)
    I[2, 2] = np.sum(coods[:, 1] ** 2 + coods[:, 0] ** 2)

    I[0, 1] = I[1, 0] = - np.sum(coods[:, 0] * coods[:, 1])
    I[1, 2] = I[2, 1] = - np.sum(coods[:, 2] * coods[:, 1])
    I[0, 2] = I[2, 0] = - np.sum(coods[:, 2] * coods[:, 0])

    e_values, e_vectors = np.linalg.eig(I)

    sort_idx = np.argsort(e_values)

    e_values = e_values[sort_idx]
    e_vectors = e_vectors[sort_idx, :]

    a = ((5. / (2 * len(coods))) * (
            e_values[1] + e_values[2] - e_values[0])) ** 0.5
    b = ((5. / (2 * len(coods))) * (
            e_values[0] + e_values[2] - e_values[1])) ** 0.5
    c = ((5. / (2 * len(coods))) * (
            e_values[0] + e_values[1] - e_values[2])) ** 0.5

    #     print a, b, c

    return [a, b, c], e_vectors


@nb.jit(nogil=True, parallel=True)
def make_soft_img(pos, Ndim, i, j, imgrange, ls, smooth):
    # Define x and y positions for the gaussians
    Gx, Gy = np.meshgrid(np.linspace(imgrange[0][0], imgrange[0][1], Ndim),
                         np.linspace(imgrange[1][0], imgrange[1][1], Ndim))

    # Initialise the image array
    gsmooth_img = np.zeros((Ndim, Ndim))

    # Loop over each star computing the smoothed gaussian distribution for this particle
    for x, y, l, sml in zip(pos[:, i], pos[:, j], ls, smooth):

        # Compute the image
        g = np.exp(-(((Gx - x) ** 2 + (Gy - y) ** 2) / (2.0 * sml ** 2)))

        # Get the sum of the gaussian
        gsum = np.sum(g)

        # If there are stars within the image in this gaussian add it to the image array
        if gsum > 0:
            gsmooth_img += g * l / gsum

    # gsmooth_img, xedges, yedges = np.histogram2d(pos[:, i], pos[:, j],
    #                                      bins=Ndim,
    #                                      range=imgrange,
    #                                      weights=ls)

    return gsmooth_img


@nb.jit(nogil=True, parallel=True)
def get_img_hlr(img, apertures, app_rs, res, csoft, radii_frac=0.5):
    # Apply the apertures
    phot_table = aperture_photometry(img, apertures, method='subpixel',
                                     subpixels=5)

    # Extract the aperture luminosities
    row = np.lib.recfunctions.structured_to_unstructured(
        np.array(phot_table[0]))
    lumins = row[3:]

    # Get half the total luminosity
    half_l = np.sum(img) * radii_frac

    # Interpolate to increase resolution
    func = interp1d(app_rs, lumins, kind="linear")
    interp_rs = np.linspace(0.001, res / 4, 10000) * csoft
    interp_lumins = func(interp_rs)

    # Get the half mass radius particle
    hmr_ind = np.argmin(np.abs(interp_lumins - half_l))
    hmr = interp_rs[hmr_ind]

    return hmr


def get_pixel_hlr(img, single_pix_area, radii_frac=0.5):

    # Get half the total luminosity
    half_l = np.sum(img) * radii_frac

    # Sort pixels into 1D array ordered by decreasing intensity
    sort_1d_img = np.sort(img.flatten())[::-1]
    sum_1d_img = np.cumsum(sort_1d_img)
    cumal_area = np.full_like(sum_1d_img, single_pix_area)\
                 * np.arange(1, sum_1d_img.size + 1, 1)

    npix = np.argmin(np.abs(sum_1d_img - half_l))
    cumal_area_cutout = cumal_area[np.max((npix - 10, 0)):
                                   np.min((npix + 10, cumal_area.size - 1))]
    sum_1d_img_cutout = sum_1d_img[np.max((npix - 10, 0)):
                                   np.min((npix + 10, cumal_area.size - 1))]

    # Interpolate the arrays for better resolution
    interp_func = interp1d(cumal_area_cutout, sum_1d_img_cutout, kind="linear")
    interp_areas = np.linspace(cumal_area_cutout.min(), cumal_area_cutout.max(),
                               500)
    interp_1d_img = interp_func(interp_areas)

    # Calculate radius from pixel area defined using the interpolated arrays
    pix_area = interp_areas[np.argmin(np.abs(interp_1d_img - half_l))]
    hlr = np.sqrt(pix_area / np.pi)

    return hlr


@nb.njit(nogil=True, parallel=True)
def calc_rad(poss, i, j):
    # Get galaxy particle indices
    rs = np.sqrt(poss[:, i] ** 2 + poss[:, j] ** 2)

    return rs


@nb.njit(nogil=True, parallel=True)
def calc_3drad(poss):
    # Get galaxy particle indices
    rs = np.sqrt(poss[:, 0] ** 2 + poss[:, 1] ** 2 + poss[:, 2] ** 2)

    return rs


def lumin_weighted_centre(poss, ls, i, j):
    cent = np.average(poss[:, (i, j)], axis=0, weights=ls)

    return cent


def calc_light_mass_rad(rs, ls, radii_frac=0.5):

    # Sort the radii and masses
    sinds = np.argsort(rs)
    rs = rs[sinds]
    ls = ls[sinds]

    if ls.size < 100:
        return 0.0

    # Get the cumalative sum of masses
    l_profile = np.cumsum(ls)

    # Get the total mass and half the total mass
    tot_l = np.sum(ls)
    half_l = tot_l * radii_frac

    # Get the half mass radius particle
    hmr_ind = np.argmin(np.abs(l_profile - half_l))
    l_profile_cutout = l_profile[np.max((hmr_ind - 10, 0)):
                                 np.min((hmr_ind + 10, l_profile.size))]
    rs_cutout = rs[np.max((hmr_ind - 10, 0)):
                   np.min((hmr_ind + 10, l_profile.size))]

    if len(rs_cutout) < 3:
        return 0

    # Interpolate the arrays for better resolution
    interp_func = interp1d(rs_cutout, l_profile_cutout, kind="linear")
    interp_rs = np.linspace(rs_cutout.min(),  rs_cutout.max(), 500)
    interp_1d_ls = interp_func(interp_rs)

    new_hmr_ind = np.argmin(np.abs(interp_1d_ls - half_l))
    hmr = interp_rs[new_hmr_ind]

    return hmr


# Weighted quantiles
def weighted_quantile(values, quantiles, sample_weight=None,
                      values_sorted=False, old_style=False):
    """
    Taken from From https://stackoverflow.com/a/29677616/1718096
    Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    :param values: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of
        initial array
    :param old_style: if True, will correct output to be consistent
        with numpy.percentile.
    :return: numpy.array with computed quantiles.
    """

    # do some housekeeping
    values = np.array(values)
    quantiles = np.array(quantiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)
    assert np.all(quantiles >= 0) and np.all(quantiles <= 1), \
        'quantiles should be in [0, 1]'

    # if not sorted, sort values array
    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    if old_style:
        # To be convenient with numpy.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= np.sum(sample_weight)
    return np.interp(quantiles, weighted_quantiles, values)


def binned_weighted_quantile(x, y, weights, bins, quantiles):
    # if ~isinstance(quantiles,list):
    #     quantiles = [quantiles]

    out = np.full((len(bins) - 1, len(quantiles)), np.nan)
    for i, (b1, b2) in enumerate(zip(bins[:-1], bins[1:])):
        mask = (x >= b1) & (x < b2)
        if np.sum(mask) > 0:
            out[i, :] = weighted_quantile(y[mask], quantiles,
                                          sample_weight=weights[mask])

    return np.squeeze(out)
