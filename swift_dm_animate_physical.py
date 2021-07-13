#!/cosma/home/dp004/dc-rope1/.conda/envs/flares-env/bin/python
import matplotlib as ml
ml.use('Agg')
import numpy as np
import sphviewer as sph
from sphviewer.tools import QuickView, cmaps, camera_tools
import matplotlib.pyplot as plt
from astropy.cosmology import Planck13 as cosmo
from swiftascmaps import red
import sys
from guppy import hpy; h=hpy()
import os
from swiftsimio import load
import unyt
import gc


def get_normalised_image(img, vmin=None, vmax=None):

    if vmin == None:
        vmin = np.min(img)
    if vmax == None:
        vmax = np.max(img)

    img = np.clip(img, vmin, vmax)
    img = (img - vmin) / (vmax - vmin)

    return img


def getimage(data, poss, hsml, num, z):

    print('There are', poss.shape[0], 'dark matter particles in the region')
    
    # Set up particle objects
    P = sph.Particles(poss, mass=np.ones(poss.shape[0]), hsml=hsml)

    # Initialise the scene
    S = sph.Scene(P)

    i = data[num]
    i['xsize'] = 1920
    i['ysize'] = 1080
    i['roll'] = 0
    S.update_camera(**i)
    R = sph.Render(S)
    R.set_logscale()
    img = R.get_image()

    vmax = 5.0
    vmin = 0.3

    # Get colormaps
    cmap = red

    # Convert images to rgb arrays
    rgb = cmap(get_normalised_image(img, vmin=vmin, vmax=vmax))

    return rgb, R.get_extent()


def single_frame(num, max_pixel, nframes):

    snap = "%04d" % num

    # Define path
    path = '/cosma/home/dp004/dc-rope1/cosma7/SWIFT/DMO_1380_data/ani_hydro_' + snap + ".hdf5"

    snap = "%05d" % num

    data = load(path)

    meta = data.metadata
    boxsize = meta.boxsize[0]
    z = meta.redshift

    print(boxsize, z)

    # Define centre
    cent = np.array([boxsize / 2, boxsize / 2, boxsize / 2])

    # Define targets
    targets = [[0, 0, 0], ]

    # Define anchors dict for camera parameters
    anchors = {}
    anchors['sim_times'] = [0.0, 'same', 'same', 'same', 'same', 'same', 'same', 'same']
    anchors['id_frames'] = np.linspace(0, nframes, 8, dtype=int)
    anchors['id_targets'] = [0, 'same', 'same', 'same', 'same', 'same', 'same', 'same']
    anchors['r'] = [0.0, 'same', 'same', 'same', 'same', 'same', 'same', 'same']
    anchors['t'] = [5, 'same', 'same', 'same', 'same', 'same', 'same', 'same']
    anchors['p'] = [0, 'pass', 'pass', 'pass', 'pass', 'pass', 'pass', -360]
    anchors['zoom'] = [1., 'same', 'same', 'same', 'same', 'same', 'same', 'same']
    anchors['extent'] = [10, 'same', 'same', 'same', 'same', 'same', 'same', 'same']

    # Define the camera trajectory
    cam_data = camera_tools.get_camera_trajectory(targets, anchors)

    poss = data.dark_matter.coordinates.value
    hsmls = data.dark_matter.softenings.value / (1 + z)

    poss -= cent

    poss[np.where(poss > boxsize.value / 2)] -= boxsize.value
    poss[np.where(poss < - boxsize.value / 2)] += boxsize.value

    poss /= (1 + z)

    # Get images
    rgb_DM, extent = getimage(cam_data, poss, hsmls, num, z)

    # extent = [0, 2 * boxsize.value + 4,
    #           0, 2 * boxsize.value + 4]

    dpi = rgb_DM.shape[1]

    fig = plt.figure(figsize=(1, 1.77777777778), dpi=dpi)
    ax = fig.add_subplot(111)

    ax.imshow(rgb_DM, extent=extent, origin='lower')
    ax.tick_params(axis='both', left=False, top=False, right=False,
                   bottom=False, labelleft=False,
                   labeltop=False, labelright=False, labelbottom=False)

    # ax.text(0.975, 0.05, "$t=$%.1f Gyr" % cosmo.age(z).value,
    #         transform=ax.transAxes, verticalalignment="top",
    #         horizontalalignment='right', fontsize=5, color="k")
    #
    # ax.plot([0.05, 0.15], [0.025, 0.025], lw=0.75, color='k', clip_on=False,
    #         transform=ax.transAxes)
    #
    # ax.plot([0.05, 0.05], [0.022, 0.027], lw=0.75, color='k', clip_on=False,
    #         transform=ax.transAxes)
    # ax.plot([0.15, 0.15], [0.022, 0.027], lw=0.75, color='k', clip_on=False,
    #         transform=ax.transAxes)

    # axis_to_data = ax.transAxes + ax.transData.inverted()
    # left = axis_to_data.transform((0.05, 0.075))
    # right = axis_to_data.transform((0.15, 0.075))
    # dist = right[0] - left[0]
    #
    # if dist > 0.1:
    #     ax.text(0.1, 0.06, "%.1f cMpc" % dist,
    #             transform=ax.transAxes, verticalalignment="top",
    #             horizontalalignment='center', fontsize=5, color="k")
    # elif 100 > dist * 10**3 > 1:
    #     ax.text(0.1, 0.06, "%.1f ckpc" % dist * 10**3,
    #             transform=ax.transAxes, verticalalignment="top",
    #             horizontalalignment='center', fontsize=5, color="k")
    # else:
    #     ax.text(0.1, 0.06, "%.1f cpc" % dist * 10**6,
    #             transform=ax.transAxes, verticalalignment="top",
    #             horizontalalignment='center', fontsize=5, color="k")

    plt.margins(0, 0)

    fig.savefig('plots/Ani/Physical/DMphysical_animation_' + snap + '.png',
                bbox_inches='tight', pad_inches=0)
    plt.close(fig)

if len(sys.argv) > 1:
    single_frame(int(sys.argv[1]), max_pixel=7.5, nframes=1380)
else:

    for num in range(0, 1001):
        single_frame(num, max_pixel=6, nframes=1380)
        gc.collect()
