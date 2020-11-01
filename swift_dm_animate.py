#!/cosma/home/dp004/dc-rope1/.conda/envs/flares-env/bin/python
import matplotlib as ml
ml.use('Agg')
import numpy as np
from sphviewer.tools import QuickView, cmaps
import matplotlib.pyplot as plt
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


def getimage(poss, hsml, max_pixel=None):

    print('There are', poss.shape[0], 'dark matter particles in the region')

    # Set up particle objects
    R = QuickView(poss, hsml=hsml, mass=np.ones(poss.shape[0]), plot=False,
                  r="infinity", logscale=True)

    img = R.get_image()

    if max == None:
        vmax = img.max()
    else:
        vmax = max_pixel
    vmin = vmax * 0.5

    # Get colormaps
    cmap = cmaps.twilight()

    # Convert images to rgb arrays
    rgb = cmap(get_normalised_image(img, vmin=vmin))

    return rgb, R.get_extent()


def single_frame(num, max_pixel):

    snap = "%04d" % num

    # Define path
    path = '/cosma/home/dp004/dc-rope1/cosma7/SWIFT/DMO_test/data/mega_dmo_test_' + snap + ".hdf5"

    data = load(path)

    meta = data.metadata
    boxsize = meta.boxsize[0]

    print(boxsize)

    # # Define anchors dict for camera parameters
    # anchors = {}
    # anchors['sim_times'] = [0.0, 'same', 'same', 'same', 'same', 'same', 'same', 'same']
    # anchors['id_frames'] = [0, 45, 188, 210, 232, 375, 420, 500]
    # anchors['id_targets'] = [0, 'pass', 2, 'pass', 'pass', 'pass', 'pass', 0]
    # anchors['r'] = [boxsize * 3 / 4, 'pass', boxsize / 100, 'same', 'pass', 'pass', 'pass', boxsize * 3 / 4]
    # anchors['t'] = [0, 'pass', 'pass', -180, 'pass', -270, 'pass', -360]
    # anchors['p'] = [0, 'pass', 'pass', 'pass', 'pass', 'pass', 'pass', 360 * 3]
    # anchors['zoom'] = [1., 'same', 'same', 'same', 'same', 'same', 'same', 'same']
    # anchors['extent'] = [10, 'same', 'same', 'same', 'same', 'same', 'same', 'same']
    #
    # # Define the camera trajectory
    # data = camera_tools.get_camera_trajectory(targets, anchors)

    poss = data.dark_matter.coordinates.value
    hsmls = data.dark_matter.softenings.value

    # Get images
    rgb_DM, extent = getimage(poss, hsmls, max_pixel)

    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111)

    ax.imshow(rgb_DM, extent=extent, origin='lower')
    ax.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False,
                   labeltop=False, labelright=False, labelbottom=False)

    fig.savefig('plots/Ani/DM_animation_' + snap + '.png',
                bbox_inches='tight', dpi=300)
    plt.close(fig)

    return rgb_DM.max()


max_pixel = single_frame(100, max_pixel=None)

for num in range(0, 101):
    single_frame(num, max_pixel=max_pixel)
    gc.collect()
