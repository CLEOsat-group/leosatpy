#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Name:         transformations.py
# Purpose:      Utilities to determine the transformation, detector orientation,
#               and scale to match the detected sources with a reference catalog
#
#
#
#
# Author:       p4adch (cadam)
#
# Created:      04/29/2022
# Copyright:    (c) p4adch 2010-
#
# History:
#
# 29.04.2022
# - file created and basic methods
#
# -----------------------------------------------------------------------------

""" Modules """
import gc
import os
from copy import copy
import inspect
import logging
import numpy as np
from astropy.wcs import WCS, utils
from fast_histogram import histogram2d

# pipeline-specific modules
import config.base_conf as _base_conf

# -----------------------------------------------------------------------------

""" Meta-info """
__author__ = "Christian Adam"
__copyright__ = 'Copyright 2021, UA, LEOSat observations'
__credits__ = ["Christian Adam, Eduardo Unda-Sanzana, Jeremy Tregloan-Reed"]
__license__ = "Free"
__version__ = "0.1.0"
__maintainer__ = "Christian Adam"
__email__ = "christian.adam84@gmail.com"
__status__ = "Production"
# -----------------------------------------------------------------------------

""" Parameter used in the script """

log = logging.getLogger(__name__)

__taskname__ = 'transformations'

MODULE_PATH = os.path.dirname(inspect.getfile(inspect.currentframe()))


# -----------------------------------------------------------------------------
# changelog
# version 0.1.0 alpha version


def get_scale_and_rotation(observation, catalog, wcsprm, scale_guessed, silent=False):
    """Calculate the scale and rotation compared to the catalog based on the method of Kaiser et al. (1999).

    This should be quite similar to the approach by SCAMP.

    Parameters
    ----------
    observation: Dataframe
        pandas dataframe with sources on the observation
    catalog: Dataframe
        pandas dataframe with nearby sources from online catalogs with accurate astrometric information
    wcsprm: astropy.wcs.wcsprm
        World coordinate system object describing translation between image and skycoord
    silent: bool, optional
        Set to True to suppress most console output

    Returns
    -------
    wcs, signal, report
    """

    # Initialize logging for this user-callable function
    log.setLevel(logging.getLevelName(log.getEffectiveLevel()))

    catalog_on_sensor = wcsprm.s2p(catalog[["RA", "DEC"]], 1)
    catalog_on_sensor = catalog_on_sensor['pixcrd']
    obs_x = [observation["xcentroid"].values]
    cat_x = np.array([catalog_on_sensor[:, 0]])
    obs_y = [observation["ycentroid"].values]
    cat_y = np.array([catalog_on_sensor[:, 1]])

    log_dist_obs = calculate_log_dist(obs_x, obs_y)
    log_dist_cat = calculate_log_dist(cat_x, cat_y)

    angles_obs = calculate_angles(obs_x, obs_y)
    angles_cat = calculate_angles(cat_x, cat_y)

    # get scaling and rotation using cross correlation to match observation and
    # reference catalog.
    # Is also applied to the reflected image.
    corr_ = peak_with_cross_correlation(log_dist_obs,
                                        angles_obs,
                                        log_dist_cat,
                                        angles_cat,
                                        scale_guessed=scale_guessed)

    scaling, rotation, signal, _ = corr_
    corr_reflected = peak_with_cross_correlation(log_dist_obs,
                                                 -1. * angles_obs,
                                                 log_dist_cat,
                                                 angles_cat,
                                                 scale_guessed=scale_guessed)
    scaling_reflected, rotation_reflected, signal_reflected, _ = corr_reflected

    if signal_reflected > signal:
        is_reflected = True
        confidence = signal_reflected / signal
        scaling = scaling_reflected
        rotation = rotation_reflected
    else:
        is_reflected = False
        confidence = signal / signal_reflected

    rot = rotation_matrix(rotation)
    if is_reflected:
        # reflecting, but which direction?? this is a flip of the y-axis
        rot = np.array([[1, 0], [0, -1]]) @ rot

    wcsprm_new = rotate(copy(wcsprm), rot)
    wcsprm_new = scale(wcsprm_new, scaling)

    if is_reflected:
        refl = ""
    else:
        refl = "not "
    if not silent:
        log.info("    Found a rotation of {:.3g} deg and the pixelscale "
                 "was scaled with the factor {:.3g}.".format(rotation / 2 / np.pi * 360., scaling) +
                 " The image was " + refl + "mirrored.")
    log.debug("   The confidence level is {}. values between 1 and 2 are bad. "
              "Much higher values are best.".format(confidence))
    log.debug("   Note that there still might be a 180deg rotation. "
              "If this is the case it should be correct in the next step")

    del observation, catalog_on_sensor, obs_x, obs_y, cat_x, cat_y, \
        log_dist_obs, log_dist_cat, angles_obs, angles_cat, \
        scaling, rotation, signal, _, \
        scaling_reflected, rotation_reflected, signal_reflected, \
        corr_, corr_reflected
    gc.collect()

    return wcsprm_new


def get_offset_with_orientation(observation, catalog, wcsprm, fast=False,
                                report_global="", silent=False):
    """Use offset from cross correlation but with trying 0,90,180,270 rotation.

    Parameters
    ----------
    observation: ~pandas.Dataframe
        pandas dataframe with sources on the observation
    catalog: pd.Dataframe
        pandas dataframe with nearby sources from online catalogs with accurate astrometric information
    wcsprm: astropy.wcs.wcsprm
        World coordinate system object describing translation between image and skycoord
    fast: bool, optional
        If true will run with a subset of the sources to increase speed.
    report_global: str, optional
    silent: bool, optional
        Set to True to suppress most console output

    Returns
    -------
    wcs, signal, report
    """

    # Initialize logging for this user-callable function
    log.setLevel(logging.getLevelName(log.getEffectiveLevel()))

    observation = copy(observation)
    N_SOURCES = observation.shape[0]
    if fast:
        if N_SOURCES > _base_conf.USE_N_SOURCES:
            N_SOURCES = _base_conf.USE_N_SOURCES
        observation = observation.nlargest(N_SOURCES, 'flux')
        N_CATALOG = N_SOURCES * 4
        catalog = catalog.nsmallest(N_CATALOG, 'mag')

    log.debug("> offset_with_orientation, searching for offset while considering 0,90,180,270 rotations")
    if fast:
        log.debug("> running in fast mode")

    # already checking for reflections with the scaling and general rotations,
    # but in case rot_scale is of it is nice to have
    rotations = np.array([[[1, 0], [0, 1]], [[-1, 0], [0, -1]],
                          [[-1, 0], [0, 1]], [[1, 0], [0, -1]],
                          [[0, 1], [1, 0]], [[0, -1], [-1, 0]],
                          [[0, -1], [1, 0]], [[0, 1], [-1, 0]]])

    wcsprm_global = copy(wcsprm)
    results = []
    for rot in rotations:
        log.debug("> Trying rotation {}".format(rot))
        wcsprm = rotate(copy(wcsprm_global), rot)
        report = report_global + "---- Report for rotation {} ---- \n".format(rot)
        wcsprm, signal, report = simple_offset(observation, catalog, wcsprm, report)
        results.append([copy(wcsprm), signal, report])

    signals = [i[1] for i in results]
    median = np.median(signals)
    i = np.argmax(signals)
    wcsprm = results[i][0]
    signal = signals[i]

    report = results[i][2]
    report += "A total of {} sources from the fits file where used. \n".format(N_SOURCES)
    report += "The signal (#stars) is {} times higher than noise outliers for other directions. " \
              "(more than 2 would be nice, typical: 8 for PS)\n".format(signals[i] / median)

    log.debug("We found the following world coordinates: ")
    log.debug(WCS(wcsprm.to_header()))
    log.debug("And here is the report:")
    log.debug(report)
    log.debug("-----------------------------")
    off = wcsprm.crpix - wcsprm_global.crpix
    if not silent:
        log.info("    Found offset {:.3g} in x direction and {:.3g} in y direction".format(off[0], off[1]))

    del observation

    return wcsprm, signal, report


def simple_offset(observation, catalog, wcsprm=None, report=""):
    """Get the best offset in x, y direction.

    Parameters
    ----------
    observation: dataframe
        pandas dataframe with sources on the observation
    catalog: dataframe
        pandas dataframe with nearby sources from online catalogs with accurate astrometric information
    wcsprm: object, optional
        Wold coordinates file.
    report: str
        Previous part of the final report that will be extended by the method.

    Returns
    -------
    wcsprm, signal, report
    """
    report += "simple_offset approach via a histogram \n"

    cat_x = np.array([catalog["xcentroid"].values])
    cat_y = np.array([catalog["ycentroid"].values])
    if wcsprm is not None:
        catalog_on_sensor = wcsprm.s2p(catalog[["RA", "DEC"]], 1)
        catalog_on_sensor = catalog_on_sensor['pixcrd']
        cat_x = np.array([catalog_on_sensor[:, 0]])
        cat_y = np.array([catalog_on_sensor[:, 1]])

    obs_x = np.array([observation["xcentroid"].values])
    obs_y = np.array([observation["ycentroid"].values])

    dist_x = (obs_x - cat_x.T).flatten()
    dist_y = (obs_y - cat_y.T).flatten()

    binwidth = _base_conf.OFFSET_BINWIDTH  # would there be a reason to make it bigger than 1?
    bins = [np.arange(min(dist_x), max(dist_x) + binwidth, binwidth),
            np.arange(min(dist_y), max(dist_y) + binwidth, binwidth)]

    H, x_edges, y_edges = np.histogram2d(dist_x, dist_y, bins=bins)

    del cat_x, cat_y, obs_x, obs_y, bins, binwidth
    gc.collect()

    # finding the peak for the x and y distance where the two sets overlap and take the first peak
    peak = np.argwhere(H == np.max(H))[0]

    # sum up signal in fixed aperture 1 pixel in each direction around the peak,
    # so a 3x3 array, total 9 pixel
    signal = np.sum(H[peak[0] - 1:peak[0] + 2, peak[1] - 1:peak[1] + 2])
    signal_wide = np.sum(H[peak[0] - 4:peak[0] + 5, peak[1] - 4:peak[1] + 5])
    report += "signal wide (64pixel) - signal (9pixel)  = {}. " \
              "If this value is large then " \
              "there might be rotation or scaling issues. \n".format(signal_wide - signal)
    del observation, H
    gc.collect()

    x_shift = (x_edges[peak[0]] + x_edges[peak[0] + 1]) / 2
    y_shift = (y_edges[peak[1]] + y_edges[peak[1] + 1]) / 2

    report += "We find an offset of {} in the x direction and {} " \
              "in the y direction \n".format(x_shift, y_shift)
    report += "{} sources are fitting well with this offset. \n".format(signal)

    if wcsprm is not None:
        current_central_pixel = wcsprm.crpix
        new_central_pixel = [current_central_pixel[0] + x_shift, current_central_pixel[1] + y_shift]
        wcsprm.crpix = new_central_pixel

        return wcsprm, signal, report
    else:
        return [x_shift, y_shift]


def fine_transformation(observation, catalog, wcsprm, threshold=10,
                        compare_threshold=10, skip_rot_scale=False, silent=False):
    """Final improvement of registration. This requires that the wcs is already accurate to a few pixels.

    Parameters
    ----------
    observation: dataframe
        pandas dataframe with sources on the observation
    catalog: dataframe
        pandas dataframe with nearby sources from online catalogs with accurate astrometric information
    wcsprm: astropy.wcs.wcsprm
            World coordinate system object describing translation between image and skycoord
    threshold: float, optional
        maximum separation in pixel to consider two sources matches
    compare_threshold: float, optional
        maximum separation in pixel to consider two sources matches
    skip_rot_scale: bool, optional
        If True, rotate and scaling function are not applied to transformation
    silent: bool, optional
        Set to True to suppress most console output

    Returns
    -------
    wcsprm, score
    """

    # Initialize logging for this user-callable function
    log.setLevel(logging.getLevelName(log.getEffectiveLevel()))

    wcsprm_original = wcsprm
    wcsprm = copy(wcsprm)

    if threshold == 20 or threshold == 100:
        observation = observation.nlargest(5, "flux")
    _, _, obs_xy, cat_xy, _, _, _ = \
        find_matches(observation, catalog, wcsprm, threshold=threshold)

    if len(obs_xy[:, 0]) < 4:
        return wcsprm_original, 0  # not enough matches

    # angle
    angle_offset = -calculate_angles([obs_xy[:, 0]],
                                     [obs_xy[:, 1]]) + calculate_angles([cat_xy[:, 0]],
                                                                        [cat_xy[:, 1]])
    log_dist_obs = calculate_log_dist([obs_xy[:, 0]], [obs_xy[:, 1]])
    log_dist_cat = calculate_log_dist([cat_xy[:, 0]], [cat_xy[:, 1]])
    threshold_min = np.log(10)  # minimum distance to make useful scaling or angle estimation
    if threshold == 10:
        threshold_min = np.log(100)

    mask = (log_dist_obs > threshold_min) & (log_dist_cat > threshold_min)
    scale_offset = -log_dist_obs + log_dist_cat

    # only take useful data points
    angle_offset = angle_offset[mask]
    scale_offset = scale_offset[mask]

    rotation = np.mean(angle_offset)
    scaling = np.e ** (np.mean(scale_offset))

    del angle_offset, scale_offset, mask, log_dist_obs, log_dist_cat

    rot = rotation_matrix(rotation)
    if not skip_rot_scale:
        wcsprm = rotate(wcsprm, np.array(rot))
        if 0.9 < scaling < 1.1:
            wcsprm = scale(wcsprm, scaling)
        else:
            if not silent:
                log.warning("    Fine transformation failed. Scaling calculation failed")
            return wcsprm_original, 0

    # need to recalculate positions
    _, _, obs_xy, cat_xy, _, _, _ = \
        find_matches(observation, catalog, wcsprm, threshold=threshold)

    if len(obs_xy[:, 0]) < 4:
        return wcsprm_original, 0

    # offset:
    x_shift = np.mean(obs_xy[:, 0] - cat_xy[:, 0])
    y_shift = np.mean(obs_xy[:, 1] - cat_xy[:, 1])

    current_central_pixel = wcsprm.crpix
    new_central_pixel = [current_central_pixel[0] + x_shift, current_central_pixel[1] + y_shift]
    wcsprm.crpix = new_central_pixel

    _, _, _, _, _, score, _ = \
        find_matches(observation, catalog, wcsprm, threshold=compare_threshold)

    del _, observation, catalog

    return wcsprm, score


def find_matches(obs, cat, wcsprm=None, threshold=10):
    """Match observation with reference catalog using minimum distance."""

    # check if the input has data
    cat_has_data = cat[["xcentroid", "ycentroid"]].any(axis=0).any()
    obs_has_data = obs[["xcentroid", "ycentroid"]].any(axis=0).any()

    if not cat_has_data or not obs_has_data:
        return None, None, None, None, None, None, False

    # convert obs to numpy
    obs_xy = obs[["xcentroid", "ycentroid"]].to_numpy()

    # set up the catalog data; use RA, Dec if used with wcsprm
    cat_xy = cat[["xcentroid", "ycentroid"]].to_numpy()
    if wcsprm is not None:
        cat_xy = wcsprm.s2p(cat[["RA", "DEC"]], 1)
        cat_xy = cat_xy['pixcrd']

    # calculate the distances
    dist_xy = np.sqrt((obs_xy[:, 0] - cat_xy[:, 0, np.newaxis])**2
                      + (obs_xy[:, 1] - cat_xy[:, 1, np.newaxis])**2)

    idx_arr = np.where(dist_xy == np.min(dist_xy, axis=0))
    min_dist_xy = dist_xy[idx_arr]
    del dist_xy

    mask = min_dist_xy < threshold
    cat_idx = idx_arr[0][mask]
    obs_idx = idx_arr[1][mask]
    min_dist_xy = min_dist_xy[mask]

    # print(obs_xy[idx_arr[1][mask], :])
    obs_matched = obs.iloc[obs_idx]
    cat_matched = cat.iloc[cat_idx]

    del obs, cat

    if len(min_dist_xy) == 0:  # meaning the list is empty
        best_score = 0
    else:
        rms = np.sqrt(np.mean(np.square(min_dist_xy)))
        best_score = len(obs_xy) / (rms + 10)  # start with the current best score

    obs_xy = obs_xy[obs_idx, :]
    cat_xy = cat_xy[cat_idx, :]

    return obs_matched, cat_matched, obs_xy, cat_xy, min_dist_xy, best_score, True


def cross_corr_to_fourier_space(a):
    """Transform 2D array into fourier space. Use padding and normalization."""

    aa = (a - np.nanmean(a)) / np.nanstd(a)

    # wraps around so half the size should be fine, pads 2D array with zeros
    aaa = np.pad(aa, (2, 2), 'constant')
    ff_a = np.fft.fft2(aaa)

    del a, aa, aaa
    gc.collect()

    return ff_a


def peak_with_cross_correlation(log_distance_obs: np.ndarray, angle_obs: np.ndarray,
                                log_distance_cat: np.ndarray, angle_cat: np.ndarray,
                                scale_guessed: bool = False):
    """Find the best relation between the two sets using cross correlation.
    Either the positional offset or the scale+angle between them.

    Parameters
    ----------
    log_distance_obs: array
        first axis to consider of observations (log distance)
    angle_obs: array
        second axis to consider of observations (angle)
    log_distance_cat: array
        first axis to consider of catalog data (log distance)
    angle_cat: array
        first axis to consider of catalog data (angle)

    Returns
    -------
    scaling, rotation, signal
    """

    if not scale_guessed:
        minimum_distance = np.log(2)  # minimum pixel distance
        maximum_distance = max(log_distance_obs)
    else:
        # broader distance range if the scale is just a guess so there is a higher chance to find the correct one
        minimum_distance = min([min(log_distance_cat),
                                min(log_distance_obs) if min(log_distance_obs) > np.log(1) else np.log(1)])
        maximum_distance = max([max(log_distance_cat),
                                max(log_distance_obs)])

    # use a broader distance range for the scale,
    # so there is a higher chance to find the correct one
    # minimum_distance = min([min(log_distance_cat), min(log_distance_obs)])
    # maximum_distance = max([max(log_distance_cat), max(log_distance_obs)])
    bins_dist, binwidth_dist = np.linspace(minimum_distance, maximum_distance,
                                           _base_conf.NUM_BINS_DIST, retstep=True)

    minimum_ang = min([min(angle_cat), min(angle_obs)])
    maximum_ang = max([max(angle_cat), max(angle_obs)])

    bins_ang, binwidth_ang = np.linspace(minimum_ang, maximum_ang,
                                         360 * _base_conf.BINS_ANG_FAC, retstep=True)

    # min max of both
    bins = np.array([len(bins_dist), len(bins_ang)])  # .astype('uint64')

    vals = np.array([log_distance_obs, angle_obs])  # .astype('float32')
    ranges = np.array([[minimum_distance, maximum_distance],
                       [minimum_ang, maximum_ang]])  # .astype('float32')
    H_obs = histogram2d(*vals, bins=bins,
                        range=ranges)
    vals = np.array([log_distance_cat, angle_cat])  # .astype('float32')
    H_cat = histogram2d(*vals, bins=bins,
                        range=ranges)

    ff_obs = cross_corr_to_fourier_space(H_obs)
    ff_cat = cross_corr_to_fourier_space(H_cat)

    cross_corr = ff_obs * np.conj(ff_cat)

    # frequency cut off
    step = 1  # maybe in arcsec??, this is usually the time-step to get a frequency
    frequ = np.fft.fftfreq(ff_obs.size, d=step).reshape(ff_obs.shape)
    max_frequ = np.max(frequ)  # frequencies are symmetric - to +
    threshold = _base_conf.FREQU_THRESHOLD * max_frequ

    cross_corr[(frequ < threshold) & (frequ > -threshold)] = 0  # how to choose the frequency cut off?

    cross_corr = np.real(np.fft.ifft2(cross_corr))
    cross_corr = np.fft.fftshift(cross_corr)  # the zero shift is at (0,0), this moves it to the middle

    # take first peak
    peak = np.argwhere(cross_corr == np.max(cross_corr))[0]
    around_peak = cross_corr[peak[0] - 1:peak[0] + 2, peak[1] - 1:peak[1] + 2]

    # finding the sub pixel shift of the true peak
    peak_x_subpixel = np.sum(np.sum(around_peak, axis=1)
                             * (np.arange(around_peak.shape[0]) + 1)) / np.sum(around_peak) - 2
    peak_y_subpixel = np.sum(np.sum(around_peak, axis=0)
                             * (np.arange(around_peak.shape[1]) + 1)) / np.sum(around_peak) - 2

    # sum up signal in fixed aperture 1 pixel in each direction around the peak, so a 3x3 array, total 9 pixel
    signal = np.sum(cross_corr[peak[0] - 1:peak[0] + 2, peak[1] - 1:peak[1] + 2])

    # is that correct? yes, I think so, shape is an uneven number and index counting starts at 0
    middle_x = cross_corr.shape[0] / 2.
    middle_y = cross_corr.shape[1] / 2.

    x_shift = (peak[0] + peak_x_subpixel - middle_x) * binwidth_dist
    y_shift = (peak[1] + peak_y_subpixel - middle_y) * binwidth_ang

    del cross_corr, middle_x, middle_y, frequ, \
        log_distance_obs, log_distance_cat, angle_obs, angle_cat, \
        bins_dist, bins_ang, binwidth_dist, binwidth_ang, \
        vals, H_obs, H_cat, ff_obs, ff_cat
    gc.collect()

    # get the scale and rotation
    scaling = np.e ** (-x_shift)
    rotation = y_shift  # / 2 / np.pi * 360 maybe easier in rad

    return scaling, rotation, signal, [x_shift, y_shift]


def frange(x, y, jump=1.0):
    """https://gist.github.com/axelpale/3e780ebdde4d99cbb69ffe8b1eada92c"""
    i = 0.0
    x = float(x)  # Prevent yielding integers.
    y = float(y)  # Comparison converts y to float every time otherwise.
    x0 = x
    epsilon = jump / 2.0
    yield float("%g" % x)  # yield always first value
    while x + epsilon < y:
        i += 1.0
        x = x0 + i * jump
        yield float("%g" % x)


def calculate_dist(data_x, data_y):
    """Calculate the distance between positions."""
    data_x = np.array(data_x)
    data_y = np.array(data_y)

    dist_x = (data_x - data_x.T)
    dist_y = (data_y - data_y.T)

    # only use off diagonal elements
    dist_x = dist_x[np.where(~np.eye(dist_x.shape[0], dtype=bool))]
    dist_y = dist_y[np.where(~np.eye(dist_y.shape[0], dtype=bool))]

    dist = np.sqrt(dist_x ** 2 + dist_y ** 2)

    del data_x, data_y, dist_x, dist_y
    gc.collect()

    return dist


def calculate_log_dist(data_x, data_y):
    """Calculate logarithmic distance between points."""
    log_dist = np.log(calculate_dist(data_x, data_y) + np.finfo(float).eps)

    del data_x, data_y
    gc.collect()
    return log_dist


def calculate_angles(data_x, data_y):
    """Calculate the angle with the x-axis."""
    data_x = np.array(data_x)
    data_y = np.array(data_y)

    # get all pairs: vector differences, then
    vec_x = data_x - data_x.T
    vec_y = data_y - data_y.T
    vec_x = vec_x[np.where(~np.eye(vec_x.shape[0], dtype=bool))]
    vec_y = vec_y[np.where(~np.eye(vec_y.shape[0], dtype=bool))]

    # get the angle with x-axis.
    angles = np.arctan2(vec_x, vec_y)

    # make sure angles are between 0 and 2 Pi
    angles = angles % (2 * np.pi)

    # shift to -pi to pi
    angles[np.where(angles > np.pi)] = -1 * (2 * np.pi - angles[np.where(angles > np.pi)])

    del data_x, data_y, vec_x, vec_y
    gc.collect()

    return angles


def rotation_matrix(angle):
    rot = [[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]]
    del angle
    return rot


def rotate(wcsprm, rot):
    """Help method for offset_with_orientation. Set the different rotations in the header."""
    # hdr["PC1_1"] = rot[0][0]
    # hdr["PC1_2"] = rot[1][0]
    # hdr["PC2_1"] = rot[0][1]
    # hdr["PC2_2"] = rot[1][1]

    pc = wcsprm.get_pc()
    pc_rotated = rot @ pc
    wcsprm.pc = pc_rotated

    return wcsprm


def scale(wcsprm, scale_factor):
    """Apply the scale to wcs."""

    pc = wcsprm.get_pc()
    pc_scaled = scale_factor * pc
    wcsprm.pc = pc_scaled

    return wcsprm


def translate_wcsprm(wcsprm):
    """ Moving scaling to cdelt, out of the pc matrix.

    Parameters
    ----------
    wcsprm: astropy.wcs.wcsprm
        World coordinate system object describing translation between image and skycoord

    Returns
    -------
    wcsprm
    """

    # Initialize logging for this user-callable function
    log.setLevel(logging.getLevelName(log.getEffectiveLevel()))

    wcs = WCS(wcsprm.to_header())
    log.debug(wcs)
    scales = utils.proj_plane_pixel_scales(wcs)

    cdelt = wcsprm.get_cdelt()
    scale_ratio = scales / cdelt
    pc = np.array(wcsprm.get_pc())

    pc[0, 0] = pc[0, 0] / scale_ratio[0]
    pc[1, 0] = pc[1, 0] / scale_ratio[1]
    pc[0, 1] = pc[0, 1] / scale_ratio[0]
    pc[1, 1] = pc[1, 1] / scale_ratio[1]

    wcsprm.pc = pc
    wcsprm.cdelt = scales

    return wcsprm, scales


def shift_translation(src_image, target_image):
    """
    Translation registration by cross-correlation (like FFT up-sampled cross-correlation)
    Parameters :
     - src_image : ndarray of the reference image.
     - target_image : ndarray of the image to register.
      Must be same dimensionality as src_image.
     Return :
      - shift : ndarray of the shift pixels vector required to register the target_image the with src_image.
    """
    # The two images need to have the same size
    if src_image.shape != target_image.shape:
        raise ValueError("Registration Error : Images need to have the same size")

    # In order to calcul the FFT(Fast Fourier Transform), we need convert the data type into real data
    src_image = np.array(src_image, dtype=np.complex128, copy=False)
    target_image = np.array(target_image, dtype=np.complex128, copy=False)
    src_freq = np.fft.fftn(src_image)
    target_freq = np.fft.fftn(target_image)

    # In order to calcul the shift in pixel,
    # we need to compute the cross-correlation by an IFFT(Inverse Fast Fourier Transform)
    shape = src_freq.shape
    image_product = src_freq * target_freq.conj()
    cross_correlation = np.fft.ifftn(image_product)

    # Locate maximum in order to calcul the shift between two numpy array
    maxima = np.unravel_index(np.argmax(np.abs(cross_correlation)), cross_correlation.shape)
    midpoints = np.array([np.fix(axis_size / 2.) for axis_size in shape])

    shift = np.array(maxima, dtype=np.float64)
    shift[shift > midpoints] -= np.array(shape)[shift > midpoints]

    # If its only one row or column the shift along that dimension has no effect : we set to zero.
    for dim in range(src_freq.ndim):
        if shape[dim] == 1:
            shift[dim] = 0

    del src_image, target_image, src_freq, target_freq, image_product, cross_correlation
    gc.collect()

    return shift
