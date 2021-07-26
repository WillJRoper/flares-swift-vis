#!/cosma/home/dp004/dc-rope1/.conda/envs/flares-env/bin/python
import matplotlib as ml
ml.use('Agg')
import numpy as np
import sphviewer as sph
from sphviewer.tools import QuickView, cmaps, camera_tools, Blend
import matplotlib.pyplot as plt
from astropy.cosmology import Planck13 as cosmo
import matplotlib.colors as mcolors
import scipy.ndimage as ndimage
import sys
from guppy import hpy; h=hpy()
from scipy.spatial import cKDTree
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


def cart_to_spherical(pos):

    s_pos = np.zeros_like(pos)

    xy = pos[:, 0] ** 2 + pos[:, 1] ** 2
    s_pos[:, 0] = np.sqrt(xy + pos[:, 2] ** 2)
    s_pos[:, 1] = np.arctan2(np.sqrt(xy), pos[:, 2]) - (np.pi / 2)
    s_pos[:, 2] = np.arctan2(pos[:, 1], pos[:, 0])

    return s_pos

def spherical_to_equirectangular(pos):

    x = pos[:, 0] * pos[:, 2]
    y = pos[:, 0] * pos[:, 1]

    eq = np.zeros((pos.shape[0], 2))

    eq[:, 0] = x
    eq[:, 1] = y

    return eq, pos[:, 0]


def getimage(data, poss, mass, hsml, num, img_dimens, cmap, Type="gas"):

    print('There are', poss.shape[0], 'gas particles in the region')
    
    # Set up particle objects
    P = sph.Particles(poss, mass=mass, hsml=hsml)

    print(np.min(mass))

    # Initialise the scene
    S = sph.Scene(P)

    i = data[num]
    i['xsize'] = img_dimens
    i['ysize'] = img_dimens
    i['roll'] = 0
    S.update_camera(**i)
    R = sph.Render(S)
    R.set_logscale()
    img = R.get_image()

    img = ndimage.gaussian_filter(img, sigma=(3, 3), order=0)

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


def quartic_spline(q):

    w = np.zeros_like(q)
    okinds1 = q < 1 / 2
    okinds2 = np.logical_and(1 / 2 <= q, q < 3 / 2)
    okinds3 = np.logical_and(3 / 2 <= q, q < 5 / 2)

    w[okinds1] = (5 / 2 - q[okinds1])**4 \
                 - 5 * (3 / 2 - q[okinds1])**4 \
                 + 10 * (1 / 2 - q[okinds1])**4
    w[okinds2] = (5 / 2 - q[okinds2]) ** 4 \
                 - 5 * (3 / 2 - q[okinds2]) ** 4
    w[okinds3] = (5 / 2 - q[okinds3]) ** 4

    return w


def make_spline_img(part_pos, poss, img_dimens, tree, ls, smooth, rs,
                    spline_func=quartic_spline, spline_cut_off=5/2):

    # Initialise the image array
    smooth_img = np.zeros((img_dimens[0], img_dimens[1]))

    # Define x and y positions of pixels
    X, Y = np.meshgrid(np.arange(0, img_dimens[0], 1),
                       np.arange(0, img_dimens[1], 1))

    # Define pixel position array for the KDTree
    pix_pos = np.zeros((X.size, 2), dtype=int)
    pix_pos[:, 0] = X.ravel()
    pix_pos[:, 1] = Y.ravel()

    # Define k constant for 3 dimensions
    k3 = 7 / (478 * np.pi)
    for (i, ipos), l, sml, r in zip(enumerate(part_pos), ls, smooth, rs):

        x_sph1 = r * np.arctan2(poss[i, 1], poss[i, 0])
        x_sph2 = (r + sml) * np.arctan2(poss[i, 1] + sml, poss[i, 0] + sml)

        sml = x_sph2 - x_sph1

        # Query the tree for this particle
        dist, inds = tree.query(ipos, k=part_pos.shape[0],
                                distance_upper_bound=spline_cut_off * sml)

        if type(dist) is float:
            continue

        okinds = dist < spline_cut_off * sml
        dist = dist[okinds]
        inds = inds[okinds]

        # Get the kernel
        w = spline_func(dist / sml)

        # Place the kernel for this particle within the img
        kernel = k3 * w / sml**3
        norm_kernel = kernel / np.sum(kernel)
        smooth_img[pix_pos[inds, 0], pix_pos[inds, 1]] += l * norm_kernel

    return smooth_img


def single_frame(num, max_pixel, nframes):

    snap = "%04d" % num

    # Define path
    path = '/cosma/home/dp004/dc-rope1/cosma7/SWIFT/hydro_1380_ani/data/ani_hydro_' + snap + ".hdf5"

    snap = "%05d" % num

    img_dimens = (4096, 2048)

    data = load(path)

    meta = data.metadata
    boxsize = meta.boxsize[0]
    z = meta.redshift

    print("Boxsize:", boxsize)

    # Define centre
    cent = np.array([11.76119931, 3.95795609, 1.26561173])
    
    # Define targets
    targets = [[0, 0, 0]]

    id_frames = np.arange(0, 1381, dtype=int)
    rs = np.full(len(id_frames), 0., dtype=float)

    simtimes = np.zeros(len(id_frames), dtype=int)
    id_targets = np.zeros(len(id_frames), dtype=int)
    zoom = np.full(len(id_frames), 1)
    extent = np.full(len(id_frames), 10)

    hex_list = ["#000000", "#590925", "#6c1c55", "#7e2e84", "#ba4051",
                "#f6511d", "#ffb400", "#f7ec59", "#fbf6ac", "#ffffff"]
    float_list = [0, 0.2, 0.3, 0.4, 0.45, 0.5, 0.7, 0.8, 0.9, 1]

    cmap = get_continuous_cmap(hex_list, float_list=float_list)

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

    poss = cart_to_spherical(poss)
    print(poss.min(axis=0), poss.max(axis=0))
    poss, rs = spherical_to_equirectangular(poss)
    print(poss.min(axis=0), poss.max(axis=0))

    max_rad = np.sqrt(3 * (boxsize.value / 2)**2)

    # Define range and extent for the images in arc seconds
    imgrange = ((max_rad * -np.pi, max_rad * np.pi),
                (max_rad * -np.pi / 2, max_rad * np.pi / 2))
    imgextent = (max_rad * -np.pi, max_rad * np.pi,
                 max_rad * -np.pi / 2, max_rad * np.pi / 2)

    # Define x and y positions of pixels
    X, Y = np.meshgrid(np.linspace(imgrange[0][0], imgrange[0][1],
                                   img_dimens[0]),
                       np.linspace(imgrange[1][0], imgrange[1][1],
                                   img_dimens[1]))

    # Define pixel position array for the KDTree
    pix_pos = np.zeros((X.size, 2))
    pix_pos[:, 0] = X.ravel()
    pix_pos[:, 1] = Y.ravel()

    # Build KDTree
    tree = cKDTree(pix_pos)

    print("Pixel tree built")

    img = make_spline_img(poss, img_dimens, tree, mass, hsmls, rs)
    img = cmap(img)

    # # Get colormap
    # cmap = ml.cm.Greys_r
    #
    # try:
    #     poss = data.stars.coordinates.value - cent
    #     mass = data.stars.masses.value * 10 ** 10
    #     hsmls = data.stars.smoothing_lengths.value
    #
    #     if hsmls.max() == 0.0:
    #         print("Ill-defined smoothing lengths")
    #
    #         last_snap = "%04d" % (num - 1)
    #
    #         # Define path
    #         path = '/cosma/home/dp004/dc-rope1/cosma7/SWIFT/hydro_1380_ani/data/ani_hydro_' + last_snap + ".hdf5"
    #
    #         data = load(path)
    #         old_hsmls = data.stars.smoothing_lengths.value
    #         hsmls[:old_hsmls.size] = old_hsmls
    #         hsmls[old_hsmls.size:] = np.median(old_hsmls)
    #
    #     print(np.min(hsmls), np.max(hsmls))
    #
    #     poss[np.where(poss > boxsize.value / 2)] -= boxsize.value
    #     poss[np.where(poss < - boxsize.value / 2)] += boxsize.value
    #
    #     for proj_ind in range(6):
    #
    #         ts = np.full(len(id_frames), t_projs[proj_ind])
    #         ps = np.full(len(id_frames), p_projs[proj_ind])
    #
    #         proj = projs[proj_ind]
    #
    #         # Define anchors dict for camera parameters
    #         anchors = {}
    #         anchors['sim_times'] = list(simtimes)
    #         anchors['id_frames'] = list(id_frames)
    #         anchors['id_targets'] = list(id_targets)
    #         anchors['r'] = list(rs)
    #         anchors['t'] = list(ts)
    #         anchors['p'] = list(ps)
    #         anchors['zoom'] = list(zoom)
    #         anchors['extent'] = list(extent)
    #
    #         print(f"Processing projection {proj} with properties:")
    #         for key, val in anchors.items():
    #             print(key, "=", val[num])
    #
    #         # Define the camera trajectory
    #         cam_data = camera_tools.get_camera_trajectory(targets, anchors)
    #
    #         # Get images
    #         star_imgs[proj], ang_extent = getimage(cam_data, poss, mass,
    #                                                hsmls, num,
    #                                                img_dimens, cmap,
    #                                                Type="star")
    # except AttributeError:
    #     for proj_ind in range(6):
    #         proj = projs[proj_ind]
    #         star_imgs[proj] = np.zeros_like(gas_imgs[proj])
    #
    # imgs = {}
    #
    # for proj_ind in range(6):
    #     proj = projs[proj_ind]
    #
    #     blend = Blend.Blend(gas_imgs[proj], star_imgs[proj])
    #     imgs[proj] = blend.Screen()
    #
    # cube = np.zeros((img_dimens * 3,
    #                  img_dimens * 4, 4),
    #                 dtype=np.float32)
    #
    # cube[img_dimens: img_dimens * 2, 0: img_dimens] = imgs[(1, 0, 0)]
    # cube[img_dimens: img_dimens * 2, img_dimens: img_dimens * 2] = imgs[(0, 1, 0)]
    # cube[img_dimens: img_dimens * 2, img_dimens * 2: img_dimens * 3] = imgs[(-1, 0, 0)]
    # cube[img_dimens: img_dimens * 2, img_dimens * 3: img_dimens * 4] = imgs[(0, -1, 0)]
    # cube[img_dimens * 2: img_dimens * 3, img_dimens: img_dimens * 2] = imgs[(0, 0, -1)]
    # cube[0: img_dimens, img_dimens: img_dimens * 2] = imgs[(0, 0, 1)]
    #
    # posx = imgs[(1, 0, 0)]
    # negx = imgs[(-1, 0, 0)]
    # posy = imgs[(0, 1, 0)]
    # negy = imgs[(0, -1, 0)]
    # posz = imgs[(0, 0, 1)]
    # negz = imgs[(0, 0, -1)]
    #
    # squareLength = posx.shape[0]
    # halfSquareLength = squareLength / 2
    #
    # outputWidth = squareLength * 2
    # outputHeight = squareLength * 1
    #
    # output = np.zeros((outputHeight, outputWidth, 4))
    #
    # for loopY in range(0, int(outputHeight)):  # 0..height-1 inclusive
    #
    #     print(loopY)
    #
    #     for loopX in range(0, int(outputWidth)):
    #         # 2. get the normalised u,v coordinates for the current pixel
    #
    #         U = float(loopX) / (outputWidth - 1)  # 0..1
    #         V = float(loopY) / (outputHeight - 1)  # no need for 1-... as the image output needs to start from the top anyway.
    #
    #         # 3. taking the normalised cartesian coordinates calculate the polar coordinate for the current pixel
    #
    #         theta = U * 2 * np.pi
    #         phi = V * np.pi
    #
    #         # 4. calculate the 3D cartesian coordinate which has been projected to a cubes face
    #
    #         cart = convertEquirectUVtoUnit2D(theta, phi, squareLength)
    #
    #         # 5. use this pixel to extract the colour
    #
    #         index = cart["index"]
    #
    #         if (index == "X+"):
    #             output[loopY, loopX] = posx[cart["y"], cart["x"]]
    #         elif (index == "X-"):
    #             output[loopY, loopX] = negx[cart["y"], cart["x"]]
    #         elif (index == "Y+"):
    #             output[loopY, loopX] = posy[cart["y"], cart["x"]]
    #         elif (index == "Y-"):
    #             output[loopY, loopX] = negy[cart["y"], cart["x"]]
    #         elif (index == "Z+"):
    #             output[loopY, loopX] = posz[cart["y"], cart["x"]]
    #         elif (index == "Z-"):
    #             output[loopY, loopX] = negz[cart["y"], cart["x"]]
    #
    dpi = img.shape[0]
    print(dpi, img.shape)
    fig = plt.figure(figsize=(1, 2), dpi=dpi)
    ax = fig.add_subplot(111)

    ax.imshow(img, origin='lower')
    ax.tick_params(axis='both', left=False, top=False, right=False,
                   bottom=False, labelleft=False,
                   labeltop=False, labelright=False, labelbottom=False)

    # ax.text(0.975, 0.05, "$t=$%.1f Gyr" % cosmo.age(z).value,
    #         transform=ax.transAxes, verticalalignment="top",
    #         horizontalalignment='right', fontsize=1, color="w")

    plt.margins(0, 0)

    ax.set_frame_on(False)

    fig.savefig('plots/Ani/360/Equirectangular_flythrough_' + snap + '.png',
                bbox_inches='tight',
                pad_inches=0)

    plt.close(fig)

if len(sys.argv) > 1:
    single_frame(int(sys.argv[1]), max_pixel=7.5, nframes=1380)
else:

    for num in range(0, 1001):
        single_frame(num, max_pixel=6.5, nframes=1380)
        gc.collect()
