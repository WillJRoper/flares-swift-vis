#!/cosma/home/dp004/dc-rope1/.conda/envs/flares-env/bin/python
import matplotlib as ml
ml.use('Agg')
import numpy as np
import sphviewer as sph
from sphviewer.tools import QuickView, cmaps, camera_tools
import matplotlib.pyplot as plt
from astropy.cosmology import Planck13 as cosmo
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


def getimage(data, poss, hsml, num, max_pixel):

    print('There are', poss.shape[0], 'dark matter particles in the region')
    
    # Set up particle objects
    P = sph.Particles(poss, mass=np.ones(poss.shape[0]), hsml=hsml)

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

    vmax = max_pixel
    vmin = 1

    # Get colormaps
    cmap = cmaps.twilight()

    # Convert images to rgb arrays
    rgb = cmap(get_normalised_image(img, vmin=vmin, vmax=vmax))

    return rgb, R.get_extent()


def single_frame(num, max_pixel, nframes):

    snap = "%04d" % num

    # Define path
    path = '/cosma/home/dp004/dc-rope1/cosma7/SWIFT/DMO_1000/data/mega_dmo_test_' + snap + ".hdf5"

    snap = "%05d" % num

    data = load(path)

    meta = data.metadata
    boxsize = meta.boxsize[0]
    z = meta.redshift

    print(boxsize)
    
    # Define targets
    targets = [[boxsize / 2, boxsize / 2, boxsize / 2]]

    # Define anchors dict for camera parameters
    anchors = {}
    anchors['sim_times'] = [0.0, 'same', 'same', 'same', 'same', 'same', 'same', 'same']
    anchors['id_frames'] = np.linspace(0, nframes, 8, dtype=int)
    anchors['id_targets'] = [0, 'same', 'same', 'same', 'same', 'same', 'same', 'same']
    anchors['r'] = [boxsize.value + 5, 'same', 'same', 'same', 'same', 'same', 'same', 'same']
    anchors['t'] = [5, 'same', 'same', 'same', 'same', 'same', 'same', 'same']
    anchors['p'] = [0, 'pass', 'pass', 'pass', 'pass', 'pass', 'pass', -360]
    anchors['zoom'] = [1., 'same', 'same', 'same', 'same', 'same', 'same', 'same']
    anchors['extent'] = [10, 'same', 'same', 'same', 'same', 'same', 'same', 'same']

    # Define the camera trajectory
    cam_data = camera_tools.get_camera_trajectory(targets, anchors)

    poss = data.dark_matter.coordinates.value
    hsmls = data.dark_matter.softenings.value

    # Get images
    rgb_DM, extent = getimage(cam_data, poss, hsmls, num, max_pixel)

    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111)

    ax.imshow(rgb_DM, extent=extent, origin='lower')
    ax.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False,
                   labeltop=False, labelright=False, labelbottom=False)

    ax.text(0.87, 0.95, "%.3f Gyrs" % cosmo.age(z).value,
             bbox=dict(boxstyle="round,pad=0.3", fc='w', ec="k", lw=1,
                       alpha=0.8),
             transform=ax.transAxes, horizontalalignment='right', fontsize=8)

    fig.savefig('plots/Ani/DM_animation_' + snap + '.png',
                bbox_inches='tight', dpi=300)
    plt.close(fig)

if len(sys.argv) > 1:
    single_frame(sys.argv[1], max_pixel=7.5, nframes=1000)
else:

    for num in range(0, 1001):
        single_frame(num, max_pixel=7.5, nframes=1000)
        gc.collect()
