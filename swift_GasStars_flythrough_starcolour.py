#!/cosma/home/dp004/dc-rope1/.conda/envs/flares-env/bin/python
import matplotlib as ml
ml.use('Agg')
import numpy as np
import sphviewer as sph
from sphviewer.tools import QuickView, cmaps, camera_tools, Blend
import matplotlib.pyplot as plt
from astropy.cosmology import Planck13 as cosmo
import sys
from guppy import hpy; h=hpy()
import os
from swiftsimio import load
import unyt
import gc
import phot_modules as photm
import FLARE


def get_normalised_image(img, vmin=None, vmax=None):

    if vmin == None:
        vmin = np.min(img)
    if vmax == None:
        vmax = np.max(img)

    img = np.clip(img, vmin, vmax)
    img = (img - vmin) / (vmax - vmin)

    return img


def getimage(data, poss, mass, hsml, num, max_pixel, cmap, Type="gas"):

    print('There are', poss.shape[0], 'gas particles in the region')
    
    # Set up particle objects
    P = sph.Particles(poss, mass=mass, hsml=hsml)

    print(np.min(mass))

    # Initialise the scene
    S = sph.Scene(P)

    i = data[num]
    i['xsize'] = 500
    i['ysize'] = 500
    i['roll'] = 0
    S.update_camera(**i)
    R = sph.Render(S)
    R.set_logscale()
    img = R.get_image()

    if Type == "gas":
        vmax =11
        vmin = 6
        print("gas", np.max(img))
        # Convert images to rgb arrays
        rgb = cmap(get_normalised_image(img, vmin=vmin, vmax=vmax))
    else:
        vmax = 21.6
        vmin = 15.0
        print("star", np.min(img[img != 0]), np.max(img))
        # Convert images to rgb arrays
        rgb = get_normalised_image(img, vmin=vmin, vmax=vmax)

    return rgb, R.get_extent()


def single_frame(num, max_pixel, nframes):

    snap = "%04d" % num

    # Define path
    path = '/cosma/home/dp004/dc-rope1/cosma7/SWIFT/' \
           'hydro_1380_data/ani_hydro_' + snap + ".hdf5"

    snap = "%05d" % num

    data = load(path)

    meta = data.metadata
    boxsize = meta.boxsize[0]
    z = meta.redshift

    print("Boxsize:", boxsize)

    filters = ('JWST.NIRCAM.F480M', 'JWST.NIRCAM.F277W', 'JWST.NIRCAM.F150W')

    # Define centre
    cent = np.array([11.76119931, 3.95795609, 1.26561173])
    
    # Define targets
    targets = [[0, 0, 0]]

    ang_v = -360 / (1380 - 60)

    decay = lambda t: (boxsize.value + 5) * np.exp(-0.01637823848547536 * t)
    anti_decay = lambda t: 1.5 * np.exp(0.005139614587492267 * (t - 901))

    id_frames = np.arange(0, 1381, dtype=int)
    rs = np.zeros(len(id_frames), dtype=float)
    rs[0: 151] = decay(id_frames[0:151])
    rs[151:901] = 1.5
    rs[901:] = anti_decay(id_frames[901:])

    simtimes = np.zeros(len(id_frames), dtype=int)
    id_targets = np.zeros(len(id_frames), dtype=int)
    ts = np.full(len(id_frames), 5)
    ps = np.zeros(len(id_frames))
    # ps[0:60] = 0
    # ps[60:] = ang_v * (id_frames[60:] - 60)
    # ps[-2:] = -360
    zoom = np.full(len(id_frames), 1)
    extent = np.full(len(id_frames), 10)

    # Define anchors dict for camera parameters
    anchors = {}
    anchors['sim_times'] = list(simtimes)
    anchors['id_frames'] = list(id_frames)
    anchors['id_targets'] = list(id_targets)
    anchors['r'] = list(rs)
    anchors['t'] = list(ts)
    anchors['p'] = list(ps)
    anchors['zoom'] = list(zoom)
    anchors['extent'] = list(extent)

    print("Processing frame with properties:")
    for key, val in anchors.items():
        print(key, "=", val[num])

    # Define the camera trajectory
    cam_data = camera_tools.get_camera_trajectory(targets, anchors)

    # Get colormap
    # cmap = cmaps.sunlight()
    cmap = ml.cm.magma

    poss = data.gas.coordinates.value
    mass = data.gas.masses.value * 10 ** 10

    # okinds = np.linalg.norm(poss - cent, axis=1) < 1
    # cent = np.average(poss[okinds], weights=rho_gas[okinds], axis=0)
    print("Centered on:", cent)

    poss -= cent
    hsmls = data.gas.smoothing_lengths.value

    poss[np.where(poss > boxsize.value / 2)] -= boxsize.value
    poss[np.where(poss < - boxsize.value / 2)] += boxsize.value

    # Get images
    rgb_gas, extent = getimage(cam_data, poss, mass, hsmls, num, max_pixel,
                               cmap, Type="gas")

    # Get colormap
    cmap = ml.cm.Greys_r

    try:
        poss = data.stars.coordinates.value - cent
        mass = data.stars.masses.value * 10 ** 10
        hsmls = data.stars.smoothing_lengths.value

        if hsmls.max() == 0.0:
            print("Ill-defined smoothing lengths")

            last_snap = "%04d" % (num - 1)

            # Define path
            path = '/cosma/home/dp004/dc-rope1/cosma7/SWIFT/' \
                   'hydro_1380/data/ani_hydro_' + last_snap + ".hdf5"

            data = load(path)
            old_hsmls = data.stars.smoothing_lengths.value
            hsmls[:old_hsmls.size] = old_hsmls
            hsmls[old_hsmls.size:] = np.median(old_hsmls)

        Lum = photm.lum(num, data, kappa=0.007895, z=z, BC_fac=1, cent=cent,
                        campos=rs[num], IMF='Chabrier_300',
                        filters=filters, Type='Total', log10t_BC=7.,
                        extinction='default')

        poss[np.where(poss > boxsize.value / 2)] -= boxsize.value
        poss[np.where(poss < - boxsize.value / 2)] += boxsize.value

        rgb_stars = np.zeros((500, 500, 3))

        for i, f in enumerate(filters):

            print(f)

            # Get images
            rgb_stars[:, :, i], extent = getimage(cam_data, poss, Lum[f],
                                                  hsmls, num, max_pixel,
                                                  cmap, Type="star")

    except AttributeError as e:
        print(e)
        rgb_stars = np.zeros_like(rgb_gas)

    blend = Blend.Blend(rgb_gas, rgb_stars)
    rgb_output = blend.Screen()

    extent = [0, 2 * anchors["r"][num] / anchors["zoom"][num],
              0, 2 * anchors["r"][num] / anchors["zoom"][num]]

    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111)

    ax.imshow(rgb_stars, extent=extent, origin='lower')
    ax.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False,
                   labeltop=False, labelright=False, labelbottom=False)

    ax.text(0.975, 0.05, "$t=$%.1f Gyr" % cosmo.age(z).value,
            transform=ax.transAxes, verticalalignment="top",
            horizontalalignment='right', fontsize=5, color="w")

    ax.plot([0.05, 0.15], [0.025, 0.025], lw=0.75, color='w', clip_on=False,
            transform=ax.transAxes)

    ax.plot([0.05, 0.05], [0.022, 0.027], lw=0.75, color='w', clip_on=False,
            transform=ax.transAxes)
    ax.plot([0.15, 0.15], [0.022, 0.027], lw=0.75, color='w', clip_on=False,
            transform=ax.transAxes)

    axis_to_data = ax.transAxes + ax.transData.inverted()
    left = axis_to_data.transform((0.05, 0.075))
    right = axis_to_data.transform((0.15, 0.075))
    dist = right[0] - left[0]

    if dist > 0.1:
        ax.text(0.1, 0.06, "%.1f cMpc" % dist,
                transform=ax.transAxes, verticalalignment="top",
                horizontalalignment='center', fontsize=5, color="w")
    elif 100 > dist * 10**3 > 1:
        ax.text(0.1, 0.06, "%.1f ckpc" % dist * 10**3,
                transform=ax.transAxes, verticalalignment="top",
                horizontalalignment='center', fontsize=5, color="w")
    else:
        ax.text(0.1, 0.06, "%.1f cpc" % dist * 10**6,
                transform=ax.transAxes, verticalalignment="top",
                horizontalalignment='center', fontsize=5, color="w")

    plt.margins(0, 0)

    fig.savefig('plots/Ani/GasStars_starcolour_flythrough_' + snap + '.png',
                bbox_inches='tight', dpi=1200,
                pad_inches=0)

    plt.close(fig)

if len(sys.argv) > 1:
    single_frame(int(sys.argv[1]), max_pixel=7.5, nframes=1380)
else:

    for num in range(0, 1001):
        single_frame(num, max_pixel=6.5, nframes=1380)
        gc.collect()
