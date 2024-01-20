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
import math
import os
from copy import copy
import inspect
import logging
import fast_histogram as fhist

import matplotlib.pyplot as plt
import numpy as np
from astropy.wcs import WCS, utils

from . import base_conf

# -----------------------------------------------------------------------------

""" Meta-info """
__author__ = "Christian Adam"
__copyright__ = 'Copyright 2021-2023, CLEOSat group'
__credits__ = ["Eduardo Unda-Sanzana, Jeremy Tregloan-Reed, Christian Adam"]
__license__ = "GPL-3.0 license"
__maintainer__ = "Christian Adam"
__email__ = "christian.adam84@gmail.com"
__status__ = "Production"

__taskname__ = 'transformations'
# -----------------------------------------------------------------------------

""" Parameter used in the script """

log = logging.getLogger(__name__)

MODULE_PATH = os.path.dirname(inspect.getfile(inspect.currentframe()))


# -----------------------------------------------------------------------------


def get_scale_and_rotation(observation, catalog, wcsprm, scale_guessed,
                           dist_bin_size, ang_bin_size):
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
    scale_guessed:
    dist_bin_size:
    ang_bin_size:

    Returns
    -------
    wcs, signal, report
    """

    # Initialize logging for this user-callable function
    log.setLevel(logging.getLevelName(log.getEffectiveLevel()))

    # manage observations
    obs_x = [observation["xcentroid"].values]
    obs_y = [observation["ycentroid"].values]

    log_dist_obs = calculate_log_dist(obs_x, obs_y)
    angles_obs = calculate_angles(obs_x, obs_y)

    # manage reference catalog
    catalog_on_sensor = wcsprm.s2p(catalog[["RA", "DEC"]], 0)
    catalog_on_sensor = catalog_on_sensor['pixcrd']

    cat_x = np.array([catalog_on_sensor[:, 0]])
    cat_y = np.array([catalog_on_sensor[:, 1]])

    log_dist_cat = calculate_log_dist(cat_x, cat_y)
    angles_cat = calculate_angles(cat_x, cat_y)

    # Get scaling and rotation using cross-correlation to match observation and
    # reference catalog. Is also applied to the reflected image.
    corr_ = peak_with_cross_correlation(log_dist_obs,
                                        angles_obs,
                                        log_dist_cat,
                                        angles_cat,
                                        scale_guessed=scale_guessed,
                                        dist_bin_size=dist_bin_size,
                                        ang_bin_size=ang_bin_size)
    scaling, rotation, signal, _ = corr_
    corr_reflected = peak_with_cross_correlation(log_dist_obs,
                                                 -1. * angles_obs,
                                                 log_dist_cat,
                                                 angles_cat,
                                                 scale_guessed=scale_guessed,
                                                 dist_bin_size=dist_bin_size,
                                                 ang_bin_size=ang_bin_size)
    scaling_reflected, rotation_reflected, signal_reflected, _ = corr_reflected

    if signal_reflected > signal:
        is_reflected = True
        confidence = np.abs(signal_reflected / signal)
        scaling = scaling_reflected
        rotation = rotation_reflected
    else:
        is_reflected = False
        confidence = np.abs(signal / signal_reflected)

    rot = rotation_matrix(rotation)
    if is_reflected:
        # reflecting, but which direction? this is a flip of the y-axis
        rot = np.array([[1, 0], [0, -1]]) @ rot

    # check if the result makes sense
    wcsprm_new = copy(wcsprm)
    if 0.45 < scaling < 4.5:
        wcsprm_new = rotate(copy(wcsprm), rot)
        wcsprm_new = scale(wcsprm_new, scaling)

    if is_reflected:
        refl = ""
    else:
        refl = "not "

    log.debug("    Found a rotation of {:.3g} deg and the pixelscale "
              "was scaled with the factor {:.3g}.".format(rotation / 2 / np.pi * 360., scaling) +
              " The image was " + refl + "mirrored.")
    log.debug("    The confidence level is {}. Values between 1 and 2 are bad. "
              "Much higher values are best.".format(confidence))
    log.debug("    Note that there still might be a 180deg rotation. "
              "If this is the case it should be correct in the next step")

    base_conf.clean_up(observation, catalog_on_sensor, obs_x, obs_y, cat_x, cat_y,
                       log_dist_obs, log_dist_cat, angles_obs, angles_cat, _,
                       scaling_reflected, rotation_reflected, signal_reflected,
                       corr_, corr_reflected)

    return wcsprm_new, (scaling, rotation, signal, confidence)


def get_offset_with_orientation(observation, catalog, wcsprm,
                                report_global=""):
    """Use offset from cross-correlation but with trying a 0,90,180,270-degree rotation.

    Parameters
    ----------
    observation: ~pandas.Dataframe
        pandas dataframe with sources on the observation
    catalog: pd.Dataframe
        pandas dataframe with nearby sources from online catalogs with accurate astrometric information
    wcsprm: astropy.wcs.wcsprm
        World coordinate system object describing translation between image and skycoord
    report_global: str, optional

    Returns
    -------
    wcs, signal, report
    """

    # Initialize logging for this user-callable function
    log.setLevel(logging.getLevelName(log.getEffectiveLevel()))

    observation = copy(observation)
    N_SOURCES = observation.shape[0]

    log.debug("> offset_with_orientation, searching for offset while considering 0,90,180,270 rotations")

    # Already checking for reflections with the scaling and general rotations,
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
        wcsprm, signal, report, xy_shift = simple_offset(observation, catalog, wcsprm, report=report)
        results.append([copy(wcsprm), signal, report, xy_shift])

    signals = [i[1] for i in results]
    median = np.nanmedian(signals)
    i = np.nanargmax(signals)
    wcsprm = results[i][0]
    signal = signals[i]
    xy_shift = results[i][3]

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
    log.debug("    Found offset {:.3g} in x direction and {:.3g} in y direction".format(off[0], off[1]))

    del observation

    return wcsprm, (xy_shift, off), signal, report


def simple_offset(observation, catalog, wcsprm=None, offset_binwidth=1, report=""):
    """Get the best offset in x, y direction.

    Parameters
    ----------
    observation: dataframe
        pandas dataframe with sources on the observation
    catalog: dataframe
        pandas dataframe with nearby sources from online catalogs with accurate astrometric information
    wcsprm: object, optional
        Wold coordinates file.
    offset_binwidth:
    report: str
        Previous part of the final report that will be extended by the method.

    Returns
    -------
    wcsprm, signal, report
    """
    report += "simple_offset approach via a histogram \n"

    if wcsprm is not None:
        catalog_on_sensor = wcsprm.s2p(catalog[["RA", "DEC"]], 0)
        catalog_on_sensor = catalog_on_sensor['pixcrd']
        cat_x = np.array([catalog_on_sensor[:, 0]])
        cat_y = np.array([catalog_on_sensor[:, 1]])
    else:
        cat_x = np.array([catalog["xcentroid"].values])
        cat_y = np.array([catalog["ycentroid"].values])

    obs_x = np.array([observation["xcentroid"].values])
    obs_y = np.array([observation["ycentroid"].values])

    dist_x = (obs_x - cat_x.T).flatten()
    dist_y = (obs_y - cat_y.T).flatten()

    # Compute the minimum and maximum values for both axes
    min_x, max_x = dist_x.min(), dist_x.max()
    min_y, max_y = dist_y.min(), dist_y.max()

    # Compute the number of bins for both axes
    num_bins_x = int(np.ceil((max_x - min_x) / offset_binwidth)) + 1
    num_bins_y = int(np.ceil((max_y - min_y) / offset_binwidth)) + 1

    # Compute the edges for both axes
    x_edges = np.linspace(min_x, max_x, num_bins_x)
    y_edges = np.linspace(min_y, max_y, num_bins_y)

    vals = [dist_x, dist_y]
    bins = [num_bins_x, num_bins_y]
    ranges = [[min_x, max_x], [min_y, max_y]]
    # Compute the histogram
    H = fhist.histogram2d(*vals, bins=bins, range=ranges)

    base_conf.clean_up(cat_x, cat_y, obs_x, obs_y, bins, vals, ranges)

    # find the peak for the x and y distance where the two sets overlap and take the first peak
    peak = np.argwhere(H == np.max(H))[0]

    # sum up the signal in the fixed aperture 1 pixel in each direction around the peak,
    # so a 3x3 array, total 9 pixel
    signal = np.sum(H[peak[0] - 1:peak[0] + 2, peak[1] - 1:peak[1] + 2])
    signal_wide = np.sum(H[peak[0] - 4:peak[0] + 5, peak[1] - 4:peak[1] + 5])
    report += "signal wide (64pixel) - signal (9pixel)  = {}. " \
              "If this value is large then " \
              "there might be rotation or scaling issues. \n".format(signal_wide - signal)

    del observation, H

    x_shift = (x_edges[peak[0]] + x_edges[peak[0] + 1]) / 2
    y_shift = (y_edges[peak[1]] + y_edges[peak[1] + 1]) / 2

    report += "We find an offset of {} in the x direction and {} " \
              "in the y direction \n".format(x_shift, y_shift)
    report += "{} sources are fitting well with this offset. \n".format(signal)

    del x_edges, y_edges

    if wcsprm is not None:
        current_central_pixel = wcsprm.crpix
        new_central_pixel = [current_central_pixel[0] + x_shift,
                             current_central_pixel[1] + y_shift]
        wcsprm.crpix = new_central_pixel

        return wcsprm, signal, report, [x_shift, y_shift]
    else:
        return [x_shift, y_shift]


def fine_transformation(observation, catalog, wcsprm, threshold=10,
                        compare_threshold=10., skip_rot_scale=False, silent=False):
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

    # find matches
    _, _, obs_xy, cat_xy, _, _, _ = \
        find_matches(observation, catalog, wcsprm, threshold=threshold)

    if len(obs_xy[:, 0]) < 4:
        return wcsprm_original, 0, (None, None, [0, 0])  # not enough matches

    # angle
    angle_offset = -calculate_angles([obs_xy[:, 0]],
                                     [obs_xy[:, 1]]) + calculate_angles([cat_xy[:, 0]],
                                                                        [cat_xy[:, 1]])

    log_dist_obs = calculate_log_dist([obs_xy[:, 0]], [obs_xy[:, 1]])
    log_dist_cat = calculate_log_dist([cat_xy[:, 0]], [cat_xy[:, 1]])

    threshold_min = np.log(20)  # minimum distance to make useful scaling or angle estimation
    if threshold == 10:
        threshold_min = np.log(200)

    mask = (log_dist_obs > threshold_min) & (log_dist_cat > threshold_min)
    scale_offset = -log_dist_obs + log_dist_cat

    # only take useful data points
    angle_offset = angle_offset[mask]
    scale_offset = scale_offset[mask]

    rotation = np.nanmedian(angle_offset)
    scaling = np.e ** (np.nanmedian(scale_offset))

    base_conf.clean_up(angle_offset, scale_offset, mask, log_dist_obs, log_dist_cat)

    rot = rotation_matrix(rotation)
    if not skip_rot_scale:
        wcsprm = rotate(wcsprm, np.array(rot))
        if 0.9 < scaling < 1.1:
            wcsprm = scale(wcsprm, scaling)
        else:
            if not silent:
                log.warning("    Fine transformation failed. Scaling calculation failed")
            return wcsprm_original, 0, (None, None, [0, 0])

    # Recalculate positions
    _, _, obs_xy, cat_xy, _, _, _ = \
        find_matches(observation, catalog, wcsprm, threshold=threshold)

    if len(obs_xy[:, 0]) < 4:
        return wcsprm_original, 0, (None, None, [0, 0])

    # Get offset and update the central reference pixel
    x_shift = np.nanmedian(obs_xy[:, 0] - cat_xy[:, 0])
    y_shift = np.nanmedian(obs_xy[:, 1] - cat_xy[:, 1])

    current_central_pixel = wcsprm.crpix
    new_central_pixel = [current_central_pixel[0] + x_shift,
                         current_central_pixel[1] + y_shift]
    wcsprm.crpix = new_central_pixel

    # Recalculate score
    _, _, _, _, _, score, _ = \
        find_matches(observation, catalog, wcsprm, threshold=compare_threshold)

    del _, observation, catalog

    return wcsprm, score, (scaling, rotation, [x_shift, y_shift])


def find_matches(obs, cat, wcsprm=None, threshold=10.):
    """Match observation with reference catalog using minimum distance."""

    # check if the input has data
    cat_has_data = cat[["xcentroid", "ycentroid"]].any(axis=0).any()
    obs_has_data = obs[["xcentroid", "ycentroid"]].any(axis=0).any()

    if not cat_has_data or not obs_has_data:
        return None, None, None, None, None, None, False

    # convert obs to numpy
    obs_xy = obs[["xcentroid", "ycentroid"]].to_numpy()

    # set up the catalog data; use RA, Dec if used with wcsprm
    if wcsprm is not None:
        cat_xy = wcsprm.s2p(cat[["RA", "DEC"]], 1)
        cat_xy = cat_xy['pixcrd']
    else:
        cat_xy = cat[["xcentroid", "ycentroid"]].to_numpy()

    # calculate the distances
    dist_xy = np.sqrt((obs_xy[:, 0] - cat_xy[:, 0, np.newaxis]) ** 2
                      + (obs_xy[:, 1] - cat_xy[:, 1, np.newaxis]) ** 2)

    idx_arr = np.where(dist_xy == np.min(dist_xy, axis=0))
    min_dist_xy = dist_xy[idx_arr]
    del dist_xy

    mask = min_dist_xy <= threshold
    cat_idx = idx_arr[0][mask]
    obs_idx = idx_arr[1][mask]
    min_dist_xy = min_dist_xy[mask]

    obs_matched = obs.iloc[obs_idx]
    cat_matched = cat.iloc[cat_idx]

    obs_matched = obs_matched.sort_values(by='mag', ascending=True)
    obs_matched.reset_index(drop=True, inplace=True)

    cat_matched = cat_matched.sort_values(by='mag', ascending=True)
    cat_matched.reset_index(drop=True, inplace=True)

    columns_to_add = ['xcentroid', 'ycentroid', 'fwhm', 'fwhm_err', 'include_fwhm']
    cat_matched = cat_matched.assign(**obs_matched[columns_to_add].to_dict(orient='series'))
    del obs, cat

    obs_xy = obs_xy[obs_idx, :]
    cat_xy = cat_xy[cat_idx, :]

    if len(min_dist_xy) == 0:  # meaning the list is empty
        best_score = 0
    else:
        rms = np.sqrt(np.nanmean(np.square(min_dist_xy))) / len(obs_matched)
        best_score = len(obs_xy) / rms  # start with the current best score

    return obs_matched, cat_matched, obs_xy, cat_xy, min_dist_xy, best_score, True


def cross_corr_to_fourier_space(a):
    """Transform 2D array into fourier space. Use padding and normalization."""

    aa = (a - np.nanmean(a)) / np.nanstd(a)

    # wraps around so half the size should be fine, pads 2D array with zeros
    aaa = np.pad(aa, (2, 2), 'constant')
    ff_a = np.fft.fft2(aaa)

    return ff_a


def create_bins(min_value, max_value, bin_size, is_distance=True):
    """ Create bins for histogram """

    bin_step = np.deg2rad(bin_size)
    diff = max_value - min_value

    if is_distance:
        bin_step = bin_size
        diff = (np.e ** max_value - np.e ** min_value)

    N = math.ceil(diff / bin_step)
    N += 1

    bins, binwidth = np.linspace(min_value, max_value, N, retstep=True, dtype='float32')

    return bins, binwidth


def peak_with_cross_correlation(log_distance_obs: np.ndarray, angle_obs: np.ndarray,
                                log_distance_cat: np.ndarray, angle_cat: np.ndarray,
                                scale_guessed: bool = False,
                                dist_bin_size: float = 1,
                                ang_bin_size: float = 0.1,
                                frequ_threshold: float = 0.02):
    """Find the best relation between the two sets using cross-correlation.
    Either the positional offset or the scale+angle between them.

    Parameters
    ----------
    log_distance_obs:
        first axis to consider of observations (log distance)
    angle_obs:
        second axis to consider of observations (angle)
    log_distance_cat:
        first axis to consider of catalog data (log distance)
    angle_cat:
        first axis to consider of catalog data (angle)
    scale_guessed:

    dist_bin_size:
    ang_bin_size:
    frequ_threshold

    Returns
    -------
    scaling, rotation, signal
    """

    # set limits for distances and angles
    minimum_distance = min(log_distance_obs)
    if not scale_guessed:
        maximum_distance = max(log_distance_obs)
    else:
        # broader distance range if the scale is just a guess, so there is a higher chance to find the correct one
        maximum_distance = max([max(log_distance_cat),
                                max(log_distance_obs)])

    minimum_ang = min([min(angle_cat), min(angle_obs)])
    maximum_ang = max([max(angle_cat), max(angle_obs)])

    # create bins and ranges for histogram
    bins_dist, binwidth_dist = create_bins(minimum_distance, maximum_distance,
                                           dist_bin_size, is_distance=True)
    bins_ang, binwidth_ang = create_bins(minimum_ang, maximum_ang,
                                         ang_bin_size, is_distance=False)

    bins = [len(bins_dist), len(bins_ang)]
    ranges = [[minimum_distance, maximum_distance],
              [minimum_ang, maximum_ang]]

    # create a 2d histogram for the observations
    vals = [log_distance_obs, angle_obs]
    H_obs = fhist.histogram2d(*vals, bins=bins, range=ranges).astype(dtype=np.complex128)

    # create a 2d histogram for the reference sources
    vals = [log_distance_cat, angle_cat]
    H_cat = fhist.histogram2d(*vals, bins=bins, range=ranges).astype(dtype=np.complex128)

    # apply FFT
    ff_obs = cross_corr_to_fourier_space(H_obs)
    ff_cat = cross_corr_to_fourier_space(H_cat)

    # normalize the FFT
    ff_obs = (ff_obs - np.nanmean(ff_obs)) / np.nanstd(ff_obs)
    ff_cat = (ff_cat - np.nanmean(ff_cat)) / np.nanstd(ff_cat)

    # calculate cross-correlation
    cross_corr = ff_obs * np.conj(ff_cat)

    # frequency cut off
    step = 1  # maybe in arcsec?, this is usually the time-step to get a frequency
    frequ = np.fft.fftfreq(ff_obs.size, d=step).reshape(ff_obs.shape)
    max_frequ = np.max(frequ)  # frequencies are symmetric - to +
    threshold = frequ_threshold * max_frequ

    # Apply the mask to the cross_corr array
    cross_corr[(frequ < threshold) & (frequ > -threshold)] = 0  # how to choose the frequency cut off?

    # Inverse transform from frequency to spatial domain
    cross_corr = np.real(np.fft.ifft2(cross_corr))
    cross_corr = np.fft.fftshift(cross_corr)  # the zero shift is at (0,0), this moves it to the middle

    # take the first peak
    # alternative: peak = np.unravel_index(np.argmax(np.abs(cross_corr)), cross_corr.shape)
    peak = np.argwhere(cross_corr == np.max(cross_corr))[0]  # original
    around_peak = cross_corr[peak[0] - 1:peak[0] + 2, peak[1] - 1:peak[1] + 2]

    # find the sub pixel shift of the true peak
    peak_x_subpixel = np.sum(np.sum(around_peak, axis=1)
                             * (np.arange(around_peak.shape[0]) + 1)) / np.sum(around_peak) - 2
    peak_y_subpixel = np.sum(np.sum(around_peak, axis=0)
                             * (np.arange(around_peak.shape[1]) + 1)) / np.sum(around_peak) - 2

    # sum up the signal in a fixed aperture 1 pixel in each direction around the peak, so a 3x3 array, total 9 pixel
    signal = np.sum(cross_corr[peak[0] - 1:peak[0] + 2, peak[1] - 1:peak[1] + 2])

    # get mid-point
    middle_x = cross_corr.shape[0] / 2
    middle_y = cross_corr.shape[1] / 2

    # calculate final shift
    x_shift = (peak[0] + peak_x_subpixel - middle_x) * binwidth_dist
    y_shift = (peak[1] + peak_y_subpixel - middle_y) * binwidth_ang

    # extract scale and rotation from shift
    scaling = np.e ** (-x_shift)
    rotation = y_shift

    base_conf.clean_up(cross_corr, middle_x, middle_y,
                       log_distance_obs, log_distance_cat,
                       angle_obs, angle_cat,
                       bins_dist, bins_ang,
                       binwidth_dist, binwidth_ang,
                       vals, H_obs, H_cat, ff_obs, ff_cat)

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

    base_conf.clean_up(data_x, data_y, dist_x, dist_y)

    return dist


def calculate_log_dist(data_x, data_y):
    """Calculate logarithmic distance between points."""
    log_dist = np.log(calculate_dist(data_x, data_y) + np.finfo(float).eps)

    del data_x, data_y

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
    angles = angles % (2. * np.pi)

    # shift to -pi to pi
    angles[np.where(angles > np.pi)] = -1 * (2. * np.pi - angles[np.where(angles > np.pi)])

    base_conf.clean_up(data_x, data_y, vec_x, vec_y)

    return angles


def rotation_matrix(angle):
    """Return the corresponding rotation matrix"""
    rot = [[np.cos(angle), np.sin(angle)],
           [-np.sin(angle), np.cos(angle)]]
    del angle
    return rot


def rotate(wcsprm, rot):
    """Help method for offset_with_orientation.
    Set the different rotations in the header.
    """
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

    # compute the scales corresponding to celestial axes
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
    Parameters
    ----------
     src_image: ndarray of the reference image.
     target_image: ndarray of the image to register. Must be the same dimensionality as src_image.

    Returns
    -------
      - shift : ndarray of the shift pixels vector required to register the target_image the with src_image.
    """
    # The two images need to have the same size
    if src_image.shape != target_image.shape:
        raise ValueError("Registration Error : Images need to have the same size")

    # convert to real number for FFT(Fast Fourier Transform)
    src_image = np.array(src_image, dtype=np.complex128, copy=False)
    target_image = np.array(target_image, dtype=np.complex128, copy=False)
    src_freq = np.fft.fftn(src_image)
    target_freq = np.fft.fftn(target_image)

    # compute the cross-correlation by an IFFT(Inverse Fast Fourier Transform)
    shape = src_freq.shape
    image_product = src_freq * target_freq.conj()
    cross_correlation = np.fft.ifftn(image_product)

    # Locate the maximum to calculate the shift between the two numpy arrays
    maxima = np.unravel_index(np.argmax(np.abs(cross_correlation)), cross_correlation.shape)
    midpoints = np.array([np.fix(axis_size / 2) for axis_size in shape])

    # calculate the shift in pixel
    shift = np.array(maxima, dtype=np.float64)
    shift[shift > midpoints] -= np.array(shape)[shift > midpoints]

    # If its only one row or column, the shift along that dimension has no effect: set to zero.
    for dim in range(src_freq.ndim):
        if shape[dim] == 1:
            shift[dim] = 0

    base_conf.clean_up(src_image, target_image,
                       src_freq, target_freq,
                       image_product, cross_correlation)
    return shift

# Can be used for debugging
# def plot_3d_cross_corr(cross_corr, binwidth_dist, binwidth_ang):
#     import matplotlib.pyplot as plt
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#
#     x = np.arange(-cross_corr.shape[0] / 2, cross_corr.shape[0] / 2) * binwidth_dist
#     y = np.arange(-cross_corr.shape[1] / 2, cross_corr.shape[1] / 2) * binwidth_ang
#     x, y = np.meshgrid(y, x)
#     z = cross_corr
#
#     surf = ax.plot_surface(y, x, z, cmap='viridis', linewidth=0, antialiased=False)
#     fig.colorbar(surf, shrink=0.5, aspect=5)
#
#     ax.set_xlabel('X-axis')
#     ax.set_ylabel('Y-axis')
#     ax.set_zlabel('Cross-correlation')
#
#
# def plot_3d_surface(array: np.ndarray,
#                     ranges: np.ndarray,
#                     xlabel: str = 'X-axis',
#                     ylabel: str = 'Y-axis',
#                     zlabel: str = 'Z-axis'):
#     """Create a 3D surface plot for a 2D array."""
#     import matplotlib.pyplot as plt
#     # print(array.shape)
#     # Generate a meshgrid for the X and Y coordinates
#     log_distance_bins = np.linspace(ranges[0][0], ranges[0][1], array.shape[0])
#     angle_bins = np.linspace(np.rad2deg(ranges[1][0]),
#                              np.rad2deg(ranges[1][1]), array.shape[1])
#     X, Y = np.meshgrid(angle_bins, log_distance_bins)
#     # # Generate a meshgrid for the X and Y coordinates
#     # X, Y = np.meshgrid(np.arange(array.shape[1]), np.arange(array.shape[0]))
#
#     # Create a 3D plot
#     fig = plt.figure()
#     # ax = fig.gca(projection='3d')
#     # surf = ax.plot_surface(X, Y, array, cmap='viridis', linewidth=0, antialiased=False)
#     ax = fig.add_subplot(111, projection='3d')
#     surf = ax.plot_surface(Y, X, array, cmap='viridis', linewidth=0, antialiased=False)
#
#     # Add a color bar
#     fig.colorbar(surf, shrink=0.5, aspect=5)
#
#     # Set plot labels
#     # ax.set_xlabel(xlabel)
#     # ax.set_ylabel(ylabel)
#     ax.set_zlabel(zlabel)
#     ax.set_ylabel('Angle (deg)')
#     ax.set_xlabel('Log Distance')
#     # Display the plot
#     # plt.show()
