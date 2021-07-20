#!/cosma/home/dp004/dc-rope1/.conda/envs/flares-env/bin/python
import matplotlib as ml
ml.use('Agg')
import numpy as np
import sphviewer as sph
from sphviewer.tools import QuickView, cmaps, camera_tools, Blend
import matplotlib.pyplot as plt
from astropy.cosmology import Planck13 as cosmo
import matplotlib.colors as mcolors
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


def getimage(data, poss, mass, hsml, num, max_pixel, cmap, Type="gas"):

    print('There are', poss.shape[0], 'gas particles in the region')
    
    # Set up particle objects
    P = sph.Particles(poss, mass=mass, hsml=hsml)

    print(np.min(mass))

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

    if Type == "gas":
        vmax =11
        vmin = 6
        print("gas", np.max(img))
    else:
        vmax = 13
        vmin = 7.5
        print("star", np.max(img))

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

    print("Boxsize:", boxsize)

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
    ps[0:60] = 0
    ps[60:] = ang_v * (id_frames[60:] - 60)
    ps[-2:] = -360
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

    hex_list = ["#000000", "#590925", "#6c1c55", "#7e2e84", "#ba4051",
                "#f6511d", "#ffb400", "#f7ec59", "#fbf6ac", "#ffffff"]

    cmap = get_continuous_cmap(hex_list, float_list=None)

    poss = data.gas.coordinates.value
    mass = data.gas.masses.value * 10 ** 10
    rho_gas = data.gas.densities.value

    # okinds = np.linalg.norm(poss - cent, axis=1) < 1
    # cent = np.average(poss[okinds], weights=rho_gas[okinds], axis=0)
    print("Centered on:", cent)

    poss -= cent
    hsmls = data.gas.smoothing_lengths.value

    poss[np.where(poss > boxsize.value / 2)] -= boxsize.value
    poss[np.where(poss < - boxsize.value / 2)] += boxsize.value

    # Get images
    rgb_gas, ang_extent = getimage(cam_data, poss, mass, hsmls, num, max_pixel,
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
            path = '/cosma/home/dp004/dc-rope1/cosma7/SWIFT/hydro_1380_ani/data/ani_hydro_' + last_snap + ".hdf5"

            data = load(path)
            old_hsmls = data.stars.smoothing_lengths.value
            hsmls[:old_hsmls.size] = old_hsmls
            hsmls[old_hsmls.size:] = np.median(old_hsmls)

        print(np.min(hsmls), np.max(hsmls))

        poss[np.where(poss > boxsize.value / 2)] -= boxsize.value
        poss[np.where(poss < - boxsize.value / 2)] += boxsize.value

        # Get images
        rgb_stars, ang_extent = getimage(cam_data, poss, mass, hsmls, num,
                                     max_pixel, cmap, Type="star")
    except AttributeError:
        rgb_stars = np.zeros_like(rgb_gas)

    blend = Blend.Blend(rgb_gas, rgb_stars)
    rgb_output = blend.Screen()

    i = cam_data[num]
    extent = [0, 2 * np.tan(ang_extent[1]) * i['r'],
              0, 2 * np.tan(ang_extent[-1]) * i['r']]
    print(ang_extent, extent)

    dpi = rgb_output.shape[0]
    print(dpi, rgb_output.shape)
    fig = plt.figure(figsize=(1, 1.77777777778), dpi=dpi)
    ax = fig.add_subplot(111)

    ax.imshow(rgb_output, extent=extent, origin='lower')
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

    axis_to_data = ax.transAxes + ax.transData.inverted()
    left = axis_to_data.transform((0.05, 0.075))
    right = axis_to_data.transform((0.15, 0.075))
    dist = extent[1] * (right[0] - left[0]) / (ang_extent[1] - ang_extent[0])

    print(left, right,
          (right[0] - left[0]) / (ang_extent[1] - ang_extent[0]), dist)

    if dist > 0.1:
        ax.text(0.1, 0.065, "%.1f cMpc" % dist,
                transform=ax.transAxes, verticalalignment="top",
                horizontalalignment='center', fontsize=1, color="w")
    elif 100 > dist * 10 ** 3 > 1:
        ax.text(0.1, 0.065, "%.1f pkpc" % dist * 10 ** 3,
                transform=ax.transAxes, verticalalignment="top",
                horizontalalignment='center', fontsize=1, color="w")
    else:
        ax.text(0.1, 0.065, "%.1f pkpc" % dist * 10 ** 6,
                transform=ax.transAxes, verticalalignment="top",
                horizontalalignment='center', fontsize=1, color="w")

    plt.margins(0, 0)

    fig.savefig('plots/Ani/GasStars_flythrough_' + snap + '.png',
                bbox_inches='tight', dpi=1200,
                pad_inches=0)

    plt.close(fig)

if len(sys.argv) > 1:
    single_frame(int(sys.argv[1]), max_pixel=7.5, nframes=1380)
else:

    for num in range(0, 1001):
        single_frame(num, max_pixel=6.5, nframes=1380)
        gc.collect()
