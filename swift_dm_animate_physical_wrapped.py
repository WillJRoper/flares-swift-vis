#!/cosma/home/dp004/dc-rope1/.conda/envs/flares-env/bin/python
import matplotlib as ml
ml.use('Agg')
import numpy as np
import sphviewer as sph
from sphviewer.tools import QuickView, cmaps, camera_tools, Blend
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
import matplotlib.colors as mcolors
from astropy.cosmology import Planck13 as cosmo
# from swiftascmaps import red, evermore, lover, folklore, nineteen_eighty_nine
import cmasher as cmr
import sys
from guppy import hpy; h=hpy()
import os
from swiftsimio import load
import unyt
import gc


def hex_to_rgb(value):
    '''
    Converts hex to rgb colours
    value: string of 6 characters representing a hex colour.
    Returns: list length 3 of RGB values'''
    value = value.strip("#") # removes hash symbol if present
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


def rgb_to_dec(value):
    '''
    Converts rgb to decimal colours (i.e. divides each value by 256)
    value: list (length 3) of RGB values
    Returns: list (length 3) of decimal values'''
    return [v/256 for v in value]


def get_continuous_cmap(hex_list, float_list=None):
    ''' creates and returns a color map that can be used in heat map figures.
        If float_list is not provided, colour map graduates linearly between each color in hex_list.
        If float_list is provided, each color in hex_list is mapped to the respective location in float_list.

        Parameters
        ----------
        hex_list: list of hex code strings
        float_list: list of floats between 0 and 1, same length as hex_list. Must start with 0 and end with 1.

        Returns
        ----------
        colour map'''
    rgb_list = [rgb_to_dec(hex_to_rgb(i)) for i in hex_list]
    if float_list:
        pass
    else:
        float_list = list(np.linspace(0, 1, len(rgb_list)))

    cdict = dict()
    for num, col in enumerate(['red', 'green', 'blue']):
        col_list = [[float_list[i], rgb_list[i][num], rgb_list[i][num]] for i
                    in range(len(float_list))]
        cdict[col] = col_list
    cmp = mcolors.LinearSegmentedColormap('my_cmp', segmentdata=cdict, N=256)
    return cmp


def get_normalised_image(img, vmin=None, vmax=None):

    if vmin == None:
        vmin = np.min(img)
    if vmax == None:
        vmax = np.max(img)

    img = np.clip(img, vmin, vmax)
    img = (img - vmin) / (vmax - vmin)

    return img


def getimage(data, poss, hsml, num, z, cmap):

    print('There are', poss.shape[0], 'dark matter particles in the region')
    
    # Set up particle objects
    P = sph.Particles(poss, mass=np.ones(poss.shape[0]), hsml=hsml)

    # Initialise the scene
    S = sph.Scene(P)

    i = data[num]
    i['xsize'] = 3840
    i['ysize'] = 2160
    i['roll'] = 0
    S.update_camera(**i)
    R = sph.Render(S)
    R.set_logscale()
    img = R.get_image()

    print(img.max(),
          np.percentile(img, 99.99),
          np.percentile(img, 95),
          np.percentile(img, 90),
          np.percentile(img, 67.5),
          np.percentile(img, 50))

    vmax = 6.2
    vmin = 0

    # # Get colormaps
    # cmap2 = cmr.torch_r(np.linspace(0, 1, 128))
    # cmap3 = cmr.swamp(np.linspace(0, 1, 128))
    #
    # # combine them and build a new colormap
    # colors = np.vstack((cmap2, cmap3))
    # cmap = mcolors.LinearSegmentedColormap.from_list('colormap', colors)

    hex_list = ["#000000", "#590925", "#6c1c55", "#7e2e84", "#ba4051",
                "#f6511d", "#ffb400", "#f7ec59", "#fbf6ac", "#ffffff"]

    #cmap = get_continuous_cmap(hex_list, float_list=None)

    img = ndimage.gaussian_filter(img, sigma=(3, 3), order=0)

    # Convert images to rgb arrays
    rgb = cmap(get_normalised_image(img, vmin=vmin, vmax=vmax))

    return rgb, R.get_extent()


def single_frame(num, max_pixel, nframes):

    snap = "%04d" % num

    # Define path
    path = '/cosma/home/dp004/dc-rope1/cosma7/SWIFT/hydro_1380_ani/data/ani_hydro_' + snap + ".hdf5"

    snap = "%05d" % num

    data = load(path)

    meta = data.metadata
    boxsize = meta.boxsize[0]
    z = meta.redshift

    print(boxsize, z)
    print("Physical Box Size:", boxsize / (1 + z))
    print("Boxes in frame:", (1 + z))

    # Define centre
    cent = np.array([boxsize / 2, boxsize / 2, boxsize / 2])

    # Define targets
    targets = [[0, 0, 0], ]

    # Define anchors dict for camera parameters
    anchors = {}
    anchors['sim_times'] = [0.0, 'same', 'same', 'same', 'same', 'same', 'same', 'same']
    anchors['id_frames'] = np.linspace(0, nframes, 8, dtype=int)
    anchors['id_targets'] = [0, 'same', 'same', 'same', 'same', 'same', 'same', 'same']
    anchors['r'] = [5, 'same', 'same', 'same', 'same', 'same', 'same', 'same']
    anchors['t'] = [0, 'same', 'same', 'same', 'same', 'same', 'same', 'same']
    anchors['p'] = [0, 'same', 'same', 'same', 'same', 'same', 'same', 'same']
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

    wrapped_boxes = int(np.ceil(1 + z))
    if wrapped_boxes % 2 == 0:
        wrapped_boxes += 1
    half_wrapped_boxes = int(wrapped_boxes / 2)
    wrapped_poss = np.zeros((poss.shape[0] * wrapped_boxes ** 3, 3))
    wrapped_hsmls = np.zeros(poss.shape[0] * wrapped_boxes ** 3)
    print(wrapped_poss.shape[0]**(1/3))
    n = 0
    for i in range(-half_wrapped_boxes, half_wrapped_boxes, 1):
        for j in range(-half_wrapped_boxes, half_wrapped_boxes, 1):
            for k in range(-half_wrapped_boxes, half_wrapped_boxes, 1):
                print(i, j, k)
                print(n,  np.array([i * boxsize, j * boxsize, k * boxsize]))
                wrapped_poss[poss.shape[0] * n: poss.shape[0] * (n + 1), :] = poss + np.array([i * boxsize / (1 + z), j * boxsize / (1 + z), k * boxsize / (1 + z)])
                wrapped_hsmls[poss.shape[0] * n: poss.shape[0] * (n + 1)] = hsmls
                n += 1

    print(np.min(wrapped_poss, axis=0) * (1 + z), np.max(wrapped_poss, axis=0) * (1 + z))
    print(np.min(wrapped_poss, axis=0) * (1 + z) / boxsize,
          np.max(wrapped_poss, axis=0) * (1 + z) / boxsize)

    # Get images
    cmap = cmr.sepia
    rgb_DM_box, ang_extent = getimage(cam_data, poss, hsmls, num, z, cmap)
    cmap = cmr.neutral
    rgb_DM_wrapped, ang_extent = getimage(cam_data, wrapped_poss,
                                          wrapped_hsmls, num, z, cmap)
    i = cam_data[num]
    extent = [0, 2 * np.tan(ang_extent[1]) * i['r'],
              0, 2 * np.tan(ang_extent[-1]) * i['r']]
    print(ang_extent, extent)

    blend = Blend.Blend(rgb_DM_box, rgb_DM_wrapped)
    rgb_DM = blend.Screen()

    dpi = rgb_DM.shape[0]
    print(dpi, rgb_DM.shape)
    fig = plt.figure(figsize=(1, 1.77777777778), dpi=dpi)
    ax = fig.add_subplot(111)

    ax.imshow(rgb_DM, extent=ang_extent, origin='lower')
    ax.tick_params(axis='both', left=False, top=False, right=False,
                   bottom=False, labelleft=False,
                   labeltop=False, labelright=False, labelbottom=False)

    ax.text(0.975, 0.05, "$t=$%.1f Gyr" % cosmo.age(z).value,
            transform=ax.transAxes, verticalalignment="top",
            horizontalalignment='right', fontsize=1, color="w")

    ax.plot([0.05, 0.15], [0.025, 0.025], lw=0.1, color='w', clip_on=False,
            transform=ax.transAxes)

    ax.plot([0.05, 0.05], [0.022, 0.027], lw=0.15, color='w', clip_on=False,
            transform=ax.transAxes)
    ax.plot([0.15, 0.15], [0.022, 0.027], lw=0.15, color='w', clip_on=False,
            transform=ax.transAxes)

    ax.plot([0.05, 0.15], [0.105, 0.105], lw=0.1, color='w', clip_on=False,
            transform=ax.transAxes)

    ax.plot([0.05, 0.05], [0.102, 0.107], lw=0.15, color='w', clip_on=False,
            transform=ax.transAxes)
    ax.plot([0.15, 0.15], [0.102, 0.107], lw=0.15, color='w', clip_on=False,
            transform=ax.transAxes)

    axis_to_data = ax.transAxes + ax.transData.inverted()
    left = axis_to_data.transform((0.05, 0.075))
    right = axis_to_data.transform((0.15, 0.075))
    dist = extent[1] * (right[0] - left[0]) / (ang_extent[1] - ang_extent[0])

    print(left, right,
          (right[0] - left[0]) / (ang_extent[1] - ang_extent[0]), dist)

    if dist > 0.1:
        ax.text(0.1, 0.145, "%.1f cMpc" % (dist * (1 + z)),
                transform=ax.transAxes, verticalalignment="top",
                horizontalalignment='center', fontsize=1, color="w")
        ax.text(0.1, 0.065, "%.1f pMpc" % dist,
                transform=ax.transAxes, verticalalignment="top",
                horizontalalignment='center', fontsize=1, color="w")
    elif 100 > dist * 10**3 > 1:
        ax.text(0.1, 0.065, "%.1f pkpc" % dist * 10**3,
                transform=ax.transAxes, verticalalignment="top",
                horizontalalignment='center', fontsize=1, color="w")
    else:
        ax.text(0.1, 0.065, "%.1f pkpc" % dist * 10**6,
                transform=ax.transAxes, verticalalignment="top",
                horizontalalignment='center', fontsize=1, color="w")

    plt.margins(0, 0)

    fig.savefig('plots/Ani/Physical/DMphysical_animation_wrapped_' + snap + '.png',
                bbox_inches='tight', pad_inches=0)
    plt.close(fig)

if len(sys.argv) > 1:
    single_frame(int(sys.argv[1]), max_pixel=7.5, nframes=1380)
else:

    for num in range(0, 1001):
        single_frame(num, max_pixel=6, nframes=1380)
        gc.collect()
