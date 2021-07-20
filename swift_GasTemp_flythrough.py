#!/cosma/home/dp004/dc-rope1/.conda/envs/flares-env/bin/python
import matplotlib as ml
ml.use('Agg')
import numpy as np
import sphviewer as sph
from sphviewer.tools import QuickView, cmaps, camera_tools, Blend
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import matplotlib as mpl
from astropy.cosmology import Planck13 as cosmo
import matplotlib.colors as mcolors
import sys
from guppy import hpy; h=hpy()
import os
from swiftsimio import load
import unyt
import gc

mpl.rcParams.update({'font.size': 1})


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


def getimage(data, poss, temp, mass, hsml, num, norm, cmap):

    print('There are', poss.shape[0], 'gas particles in the region')
    
    # Set up particle objects
    P1 = sph.Particles(poss, mass=temp * mass, hsml=hsml)
    P2 = sph.Particles(poss, mass=mass, hsml=hsml)

    # Initialise the scene
    S1 = sph.Scene(P1)
    S2 = sph.Scene(P2)

    i = data[num]
    i['xsize'] = 3840
    i['ysize'] = 2160
    i['roll'] = 0
    S1.update_camera(**i)
    S2.update_camera(**i)
    R1 = sph.Render(S1)
    R2 = sph.Render(S2)
    R1.set_logscale()
    R2.set_logscale()
    img1 = R1.get_image()
    img2 = R2.get_image()
    img = img1 - img2

    vmax = 7.5
    vmin = 3.5
    print("gas temperature", np.min(img), np.max(img))

    # Convert images to rgb arrays
    rgb = cmap(norm(img))

    return rgb, R1.get_extent()


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

    # Get colormap
    # cmap = cmaps.sunlight()

    hex_list = ["#590925", "#6c1c55", "#7e2e84", "#ba4051",
                "#f6511d", "#ffb400", "#f7ec59", "#fbf6ac",
                "#ffffff"]
    float_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1]

    cmap = get_continuous_cmap(hex_list, float_list=float_list)
    norm = plt.Normalize(vmin=3.5, vmax=7.5, clip=True)

    poss = data.gas.coordinates.value
    temp = data.gas.temperatures.value
    mass = data.gas.masses.value * 10 ** 10
    
    print(np.log10(temp.max()),
          np.log10(np.percentile(temp, 99)),
          np.log10(np.percentile(temp, 95)),
          np.log10(np.percentile(temp, 90)),
          np.log10(np.percentile(temp, 67.5)),
          np.log10(np.percentile(temp, 50)))

    # okinds = np.linalg.norm(poss - cent, axis=1) < 1
    # cent = np.average(poss[okinds], weights=rho_gas[okinds], axis=0)
    print("Centered on:", cent)

    poss -= cent
    hsmls = data.gas.smoothing_lengths.value

    poss[np.where(poss > boxsize.value / 2)] -= boxsize.value
    poss[np.where(poss < - boxsize.value / 2)] += boxsize.value

    # Get images
    rgb_output, ang_extent = getimage(cam_data, poss, temp, mass, hsmls, num,
                                      norm, cmap)

    i = cam_data[num]
    extent = [0, 2 * np.tan(ang_extent[1]) * i['r'],
              0, 2 * np.tan(ang_extent[-1]) * i['r']]
    print(ang_extent, extent)

    dpi = rgb_output.shape[0] / 2
    print(dpi, rgb_output.shape)
    fig = plt.figure(figsize=(2, 2 * 1.77777777778), dpi=dpi)
    ax = fig.add_subplot(111)

    ax.imshow(rgb_output, extent=ang_extent, origin='lower')
    ax.tick_params(axis='both', left=False, top=False, right=False, bottom=False, labelleft=False,
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

    ax.text(0.1, 0.055, "%.2f cMpc" % dist,
            transform=ax.transAxes, verticalalignment="top",
            horizontalalignment='center', fontsize=1, color="w")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm._A = []  # # fake up the array of the scalar mappable
    cbaxes = ax.inset_axes([0.05, 0.95, 0.25, 0.015])
    cbar = plt.colorbar(sm, cax=cbaxes, orientation="horizontal")
    cbar.set_ticks([3.5, 5, 6, 7.5])
    labels = ["$\leq3.5$", "5", "6", "$7.5\leq$"]
    cbar.ax.set_xticklabels(labels)
    for tick in cbar.ax.xaxis.get_major_ticks():
        tick.label.set_fontsize("xx-small")
        tick.label.set_color("w")
        tick.label.set_y(3)
    cbar.ax.tick_params(axis='x', color='w', size=0.3, width=0.1)
    cbar.ax.set_xlabel(r"$\log_{10}\left(T / [\mathrm{K}]\right)$", color='w',
                       fontsize=0.2, labelpad=-0.1)
    cbar.outline.set_edgecolor('white')
    cbar.outline.set_linewidth(0.05)

    plt.margins(0, 0)

    fig.savefig('plots/Ani/GasTemp_flythrough_' + snap + '.png',
                bbox_inches='tight',
                pad_inches=0)

    plt.close(fig)

if len(sys.argv) > 1:
    single_frame(int(sys.argv[1]), max_pixel=7.5, nframes=1380)
else:

    for num in range(0, 1001):
        single_frame(num, max_pixel=6.5, nframes=1380)
        gc.collect()
