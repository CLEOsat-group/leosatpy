#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Name:         satellites.py
# Purpose:      Utilities for the detection of satellite trails in images.
#
#
#
#
# Author:       p4adch (cadam)
#
# Created:      05/06/2022
# Copyright:    (c) p4adch 2010-
#
# History:
#
# 06.05.2022
# - file created and basic methods
#
# -----------------------------------------------------------------------------
from __future__ import annotations

import sys

""" Modules """

import os
import inspect
import logging
import gc
import numpy as np
import pandas as pd
from pyorbital import astronomy
from geopy import distance
from datetime import datetime
from dateutil import parser

import requests
import math
import fast_histogram as fhist
from lmfit import Model
import ephem

from scipy import ndimage as nd
from skimage import morphology as morph
from skimage.draw import line
from skimage.measure import regionprops_table
from skimage.transform import hough_line_peaks, hough_line

from astropy.stats import (sigma_clipped_stats, sigma_clip)
from astropy import constants as const
from photutils.segmentation import (SegmentationImage, detect_sources, detect_threshold)

from sklearn.cluster import AgglomerativeClustering

try:
    import matplotlib
except ImportError:
    plt = None
else:
    import matplotlib as mpl
    from matplotlib import pyplot as plt
    import matplotlib.lines as mlines
    import matplotlib.gridspec as gridspec  # GRIDSPEC !
    from matplotlib.ticker import AutoMinorLocator, LogLocator
    from astropy.visualization import LinearStretch, LogStretch, SqrtStretch
    from astropy.visualization.mpl_normalize import ImageNormalize
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # matplotlib parameter
    matplotlib.use('Qt5Agg')
    matplotlib.rc("lines", linewidth=1.2)
    matplotlib.rc('figure', dpi=150, facecolor='w', edgecolor='k')
    matplotlib.rc('text.latex', preamble=r'\usepackage{sfmath}')
    matplotlib.rc('pdf', fonttype=42)
    matplotlib.rc('ps', fonttype=42)

# pipeline-specific modules
import config.base_conf as _base_conf

# -----------------------------------------------------------------------------

""" Meta-info """
__author__ = "Christian Adam"
__copyright__ = 'Copyright 2021, UA, LEOSat observations'
__credits__ = ["Christian Adam, Eduardo Unda-Sanzana, Jeremy Tregloan-Reed"]
__license__ = "Free"
__version__ = "0.2.0"
__maintainer__ = "Christian Adam"
__email__ = "christian.adam84@gmail.com"
__status__ = "Production"

__taskname__ = 'satellites'
# -----------------------------------------------------------------------------

""" Parameter used in the script """
log = logging.getLogger(__name__)

MODULE_PATH = os.path.dirname(inspect.getfile(inspect.currentframe()))


# -----------------------------------------------------------------------------
# changelog
# version 0.1.0 alpha version;
# version 0.2.0 added calculations for satellite data
# version 0.3.0 added calculations for solar phase angle
# version 0.4.0 added calculations for angular velocity

def get_angular_distance(observer: ephem.Observer,
                         sat_az_alt: list,
                         prev_sat_az_alt: list) -> float:
    """ Calculate angular distance between two coordinates """

    ra, dec = observer.radec_of(np.radians(sat_az_alt[0]),
                                np.radians(sat_az_alt[1]))

    prev_ra, prev_dec = observer.radec_of(np.radians(prev_sat_az_alt[0]),
                                          np.radians(prev_sat_az_alt[1]))

    dtheta = 2. * np.arcsin(np.sqrt(np.sin(0.5 * (dec - prev_dec)) ** 2.
                                    + np.cos(dec) * np.cos(prev_dec)
                                    * np.sin(0.5 * (ra - prev_ra)) ** 2.))

    dtheta *= 206264.806

    return dtheta


def get_average_magnitude(flux, flux_err, std_fluxes, std_mags, mag_corr, mag_scale=None):
    """Calculate mean magnitude"""

    # get difference magnitude and magnitude difference
    _error = ((flux + flux_err) / (std_fluxes[:, 0] - std_fluxes[:, 1])) / \
             (flux / std_fluxes[:, 0])

    diff_mag = -2.5 * np.log10(flux / std_fluxes[:, 0])
    diff_mag_err = np.sqrt((-2.5 * np.log10(flux * _error / std_fluxes[:, 0]) - diff_mag) ** 2)

    # observed satellite magnitude relative to std stars
    mag = std_mags[:, 0] + diff_mag + mag_corr[0]
    mag = mag + mag_scale if mag_scale is not None else mag
    mag_err = np.sqrt(diff_mag_err ** 2 + std_mags[:, 1] ** 2 + mag_corr[1] ** 2)

    mask_mag = np.array(sigma_clip(mag,
                                   sigma=3.,
                                   maxiters=None,
                                   cenfunc=np.nanmedian).mask)
    mask_mag_err = np.array(sigma_clip(mag_err,
                                       sigma=3.,
                                       maxiters=None,
                                       cenfunc=np.nanmedian).mask)

    mask = np.ma.mask_or(mask_mag, mask_mag_err)
    mag_cleaned = mag[~mask]
    mag_err_cleaned = mag_err[~mask]

    mag_avg_w = np.average(mag_cleaned, weights=1. / mag_err_cleaned ** 2)
    mag_avg_err = np.nanmean(mag_err_cleaned)

    del flux, flux_err, std_fluxes, std_mags, mag_corr
    gc.collect()

    return mag_avg_w, mag_avg_err


def sun_inc_elv_angle(obsDate: str | datetime, geo_loc: tuple):
    """Calculate solar azimuth, elevation, and incidence angle for a given location on Earth"""

    # get satellite latitude and longitude
    (geo_lat, geo_lon) = geo_loc

    dtobj = obsDate
    if isinstance(obsDate, str):
        # convert the time of observation
        d = parser.parse(obsDate)
        dtobj = datetime(d.year, d.month, d.day,
                         d.hour, d.minute, d.second)

    # get sun elevation and azimuth angle from
    sun_el, sun_az = astronomy.get_alt_az(dtobj, geo_lon, geo_lat)
    sunEL = sun_el * 180. / math.pi  # from radians to degrees

    # solar incidence angle theta
    theta = 90. + sunEL

    # return results
    return theta


def get_elevation(lat, long):
    """ Return elevation from latitude, longitude based on open SRTM elevation data """

    # query address
    query = ('https://api.open-elevation.com/api/v1/lookup'
             f'?locations={lat},{long}')

    # json object, various ways you can extract value
    r = requests.get(query).json()

    # one approach is to use a pandas json functionality:
    elevation = pd.json_normalize(r, 'results')['elevation'].values[0]

    return elevation


def get_radius_earth(B):
    """Get radius of Earth for a given observation site latitude"""
    B = math.radians(B)  # converting into radians

    a = _base_conf.REARTH_EQU  # Radius at sea level at the equator
    b = _base_conf.REARTH_POL  # Radius at poles

    c = (a ** 2 * math.cos(B)) ** 2
    d = (b ** 2 * math.sin(B)) ** 2
    e = (a * math.cos(B)) ** 2
    f = (b * math.sin(B)) ** 2

    R = math.sqrt((c + d) / (e + f))
    return R


def ang_distance(lat1, lat2, lon1, lon2):

    # Haversine formula
    if lon2 > lon1:
        delta_lambda = math.radians(lon2 - lon1)
    else:
        delta_lambda = math.radians(lon1 - lon2)

    phi_1 = math.radians(lat1)
    phi_2 = math.radians(lat2)

    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = 2. * math.pi - delta_lambda if delta_lambda > math.pi else delta_lambda

    a = math.sin(delta_phi / 2.0) ** 2 + math.cos(phi_1) * math.cos(phi_2) * math.sin(delta_lambda / 2.0) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return c


def get_solar_phase_angle(sat_az, sat_alt, geo_loc, obs_range, obsDate):
    """Get the solar phase angle via Sun-Sat angle"""

    au = _base_conf.AU_TO_KM
    sun = ephem.Sun()
    sun.compute(obsDate)

    a = sun.earth_distance * au
    b = obs_range

    # get satellite latitude and longitude
    (geo_lat, geo_lon) = geo_loc

    dtobj = obsDate
    if isinstance(obsDate, str):
        # convert the time of observation
        d = parser.parse(obsDate)
        dtobj = datetime(d.year, d.month, d.day,
                         d.hour, d.minute, d.second)

    # get sun elevation and azimuth angle from
    sun_el, sun_az = astronomy.get_alt_az(dtobj, geo_lon, geo_lat)

    sun_az = np.rad2deg(sun_az)
    sun_alt = np.rad2deg(sun_el)

    sun_az = sun_az + 360. if sun_az < 0. else sun_az

    angle_c = ang_distance(sun_alt, sat_alt, sun_az, sat_az)
    c = math.sqrt(math.pow(a, 2) + math.pow(b, 2) - 2 * a * b * math.cos(angle_c))
    angle_a = math.acos((math.pow(b, 2) + math.pow(c, 2) - math.pow(a, 2)) / (2 * b * c))
    angle_b = math.pi - angle_a - angle_c

    phase_angle = math.pi - angle_c - angle_b
    phase_angle = np.rad2deg(phase_angle)
    sun_sat_ang = np.rad2deg(angle_c)

    return sun_sat_ang, phase_angle, sun_az, sun_alt


def get_observer_angle(sat_lat, geo_lat, sat_h_orb_km, h_obs_km, sat_range_km):
    """"""
    # Initialize logging for this user-callable function
    log.setLevel(logging.getLevelName(log.getEffectiveLevel()))

    # get radius of Earth for a given observation site latitude
    Rearth_obs = get_radius_earth(geo_lat)
    Rearth_h_obs = Rearth_obs + h_obs_km

    # get radius of Earth for a given satellite latitude
    Rearth_sat = get_radius_earth(sat_lat)
    Rearth_h_sat = Rearth_sat + sat_h_orb_km

    # use cosine theorem to calc obs phase angle
    theta_2 = math.acos((sat_range_km**2 + Rearth_h_sat**2 - Rearth_h_obs**2) /
                        (2. * sat_range_km * Rearth_h_sat))
    theta_2 = np.rad2deg(theta_2)

    return theta_2


def get_obs_range(sat_elev, h_orb, h_obs_km, lat):
    """Get satellite-observer range.

    Adopted from get_sat_observer_range.m by Angel Otarola, aotarola@tmt.org

    """
    # Initialize logging for this user-callable function
    log.setLevel(logging.getLevelName(log.getEffectiveLevel()))

    # get radius of Earth for a given observation site latitude
    Rearth_obs = get_radius_earth(lat)
    Rearth = Rearth_obs + h_obs_km

    # derived quantities
    # S = satellite position along its orbit
    # O = observer position on the surface
    # C = Geo-center

    gamma = 90. + sat_elev  # degrees, Angle with center in the observer - <SOC>

    # Length of the CO side = Rearth
    # Length of the CS side = Rearth + Horb

    CO = Rearth  # km
    CS = Rearth + h_orb  # km

    # use the law of sin for derivation of the beta angle <CSO>
    beta = math.asin(CO / CS * math.sin(gamma * math.pi / 180.)) * 180. / math.pi  # degrees

    # derivation of the alpha angle, <OCS>
    alpha = 180. - gamma - beta  # degrees,

    # Use of the cos law for derivation of the distance from Observer to
    # Satellite (OS), accounting for its elevation (or zenith) angle
    OS = math.sqrt(CO ** 2 + CS ** 2 - 2. * CO * CS * math.cos(alpha * math.pi / 180.))

    return OS


def poly_func(x, a0, a1, a2):
    """ Second order polynom """
    return a0 + a1 * x + a2 * x ** 2


def compute_indices(c, ws, length):
    # default setting: % operations to accommodate odd/even window sizes
    low, high = c - (ws // 2), c + (ws // 2) + ws % 2

    # correction for overlap with the borders of the array
    if low < 0:
        low, high = 0, ws
    elif high > length:
        low, high = -ws, None

    return low, high


def extract_peak_window(arr, coords, window_size=(3, 3)):
    # extract array shapes and window sizes into single variables
    len_r, len_c = arr.shape
    wsr, wsc = window_size

    # extract coords and correct for 0-indexing
    r, c = coords
    r0, c0 = r, c

    row_low, row_high = compute_indices(r0, wsr, len_r)
    col_low, col_high = compute_indices(c0, wsc, len_c)

    return arr[row_low:row_high, col_low:col_high], (row_low, row_high), (col_low, col_high)


def calculate_trail_parameter(fit_result):
    """ Calculate length, width and orientation.

    Reference: Closed form line-segment extraction using the Hough transform
               Xu, Zezhong ; Shin, Bok-Suk ; Klette, Reinhard
    Bibcode: 2015PatRe..48.4012X
    """

    # get parameter
    a0 = fit_result.params['a0'].value
    a1 = fit_result.params['a1'].value
    a2 = fit_result.params['a2'].value

    err_a0 = fit_result.params['a0'].stderr
    err_a1 = fit_result.params['a1'].stderr
    err_a2 = fit_result.params['a2'].stderr

    a = a1 ** 2. / (4. * a2)

    # length
    L = np.sqrt(12.) * np.sqrt(a2 + a0 - a)
    c = np.sqrt((4. * a0 * a2 - a1 ** 2. + 4. * a2 ** 2.) / a2)
    err_L = np.sqrt((2. * np.sqrt(3) / c * err_a0) ** 2
                    + (np.sqrt(3) / c * a1 / a2 * err_a1) ** 2
                    + (((np.sqrt(3) * a1 ** 2) / (2. * a2 ** 2)
                        + 2 * np.sqrt(3)) * err_a2 / c) ** 2)

    # width
    T = np.sqrt(12.) * np.sqrt(a0 - a)
    d = np.sqrt((4. * a0 * a2 - a1 ** 2.) / a2)
    err_T = np.sqrt((2. * np.sqrt(3) / d * err_a0) ** 2
                    + (np.sqrt(3) / d / a2 * err_a1) ** 2
                    + ((np.sqrt(3) * a1 ** 2) / (2. * a2 ** 2) * err_a2 / d) ** 2)
    # orientation
    theta_p = - a1 / (2. * a2)
    theta_p_err = np.sqrt((err_a1 / (2. * a2)) ** 2
                          + (a1 * err_a2 / (2. * a2 ** 2)) ** 2)

    e = np.sqrt(1. - ((T / 2.) ** 2. / (L / 2.) ** 2.))

    del fit_result
    gc.collect()

    return (L, err_L,
            T, err_T, theta_p, theta_p_err,
            np.rad2deg(theta_p), np.rad2deg(theta_p_err), e)


def fit_trail_params(Hij, rhos, thetas, theta_0, image_size, cent_ind, win_size):
    """Perform fit on a given sub-window of the Hough space."""

    fit_results = {}

    # extract sub-window from input
    h_peak_win, row_ind, col_ind = extract_peak_window(Hij,
                                                       (cent_ind[0], cent_ind[1]),
                                                       (win_size[0], win_size[1]))
    fit_results['Hij_sub'] = h_peak_win

    # set-up axis values, x=rho, y=theta
    rho_j = rhos[row_ind[0]:row_ind[1]]
    theta_i = thetas[col_ind[0]:col_ind[1]]
    theta_i_rad = np.deg2rad(theta_i)
    theta_i_rad_new = np.linspace(start=theta_i_rad[0], stop=theta_i_rad[-1],
                                  num=int(len(theta_i_rad) * 100), endpoint=True)
    fit_results['Hij_rho'] = rho_j
    fit_results['Hij_theta'] = theta_i

    # calculate mean and variance for each column in the peak region
    mu = np.dot(h_peak_win.T, rho_j) / np.sum(h_peak_win, axis=0)
    var = np.empty(h_peak_win.shape[1])
    for col in range(h_peak_win.shape[1]):
        diff = np.subtract(rho_j, mu[col]) ** 2
        var[col] = (np.dot(h_peak_win.T, diff[:, None]) /
                    np.sum(h_peak_win[:, col])).flatten()[col]

    # fit quadratic model
    pmodel = Model(poly_func)
    params = pmodel.make_params(a0=1., a1=1., a2=1.)
    quad_fit_result = pmodel.fit(var, params, x=theta_i_rad,
                                 weights=np.sqrt(1. / var),
                                 scale_covar=True)
    fit_results['quad_fit'] = quad_fit_result
    fit_results['quad_xnew'] = theta_i_rad_new
    fit_results['quad_x'] = theta_i_rad
    fit_results['quad_y'] = var

    # get result parameter
    (L, err_L, T, err_T,
     theta_p_rad, e_theta_p_rad,
     theta_p_deg, e_theta_p_deg, e) = calculate_trail_parameter(quad_fit_result)

    # set-up axis for fit; apply alternate transformation if 45deg < theta < 135deg
    x = np.tan(theta_i_rad)
    y = mu / np.cos(theta_i_rad)
    y_label = r'$\mu / \cos(\theta)$'
    ang_switch = False
    if np.pi / 4. < theta_0 < 3. * np.pi / 4.:
        x = 1. / np.tan(theta_i_rad)
        y = mu / np.sin(theta_i_rad)
        y_label = r'$\mu / \sin(\theta)$'
        ang_switch = True

    xnew = np.linspace(start=x[0], stop=x[-1],
                       num=int(len(x) * 100), endpoint=True)

    # fit linear model
    pmodel = Model(poly_func)
    params = pmodel.make_params(a0=1., a1=1., a2=0.)

    # fix quadratic term
    params['a2'].vary = False
    lin_fit_result = pmodel.fit(y, params, x=x, scale_covar=True)
    fit_results['lin_fit'] = lin_fit_result
    fit_results['lin_xnew'] = xnew
    fit_results['lin_x'] = x
    fit_results['lin_y'] = y
    fit_results['lin_y_label'] = y_label

    # transform to image coordinates
    xc = lin_fit_result.params['a0'].value + image_size[1] / 2.
    yc = lin_fit_result.params['a1'].value + image_size[0] / 2.
    xc_err = lin_fit_result.params['a0'].stderr
    yc_err = lin_fit_result.params['a1'].stderr
    if ang_switch:
        xc = lin_fit_result.params['a1'].value + image_size[1] / 2.
        yc = lin_fit_result.params['a0'].value + image_size[0] / 2.
        xc_err = lin_fit_result.params['a1'].stderr
        yc_err = lin_fit_result.params['a0'].stderr

    del Hij, h_peak_win
    gc.collect()

    result = ((xc, yc), (xc_err, yc_err),
              L, err_L, T, err_T, theta_p_rad + np.pi / 2., e_theta_p_rad,
              theta_p_deg + 90., e_theta_p_deg, e, fit_results)

    return result


# noinspection PyAugmentAssignment
def create_hough_space_vectorized(image: np.ndarray,
                                  dtheta: float = 0.05,
                                  drho: float = 1,
                                  silent: bool = False):
    """ Create Hough transform parameter space H(theta, rho).

    Vectorized implementation.

    Parameters
    ----------
        image:
            Input image
        drho:
            Number of rho values.
            Defaults to None.
        dtheta:
            Number of angle values.
            Defaults to 720=0.5deg
        silent:
            If True, minimal output.
            Defaults to False.
    """

    # Initialize logging for this user-callable function
    log.setLevel(logging.getLevelName(log.getEffectiveLevel()))

    if not silent:
        log.info("  > Perform Hough transform to determine "
                 "the length, width and orientation of the detected satellite trail.")

    # image properties
    edge_height, edge_width = image.shape[:2]
    edge_height_half, edge_width_half = edge_height // 2, edge_width // 2

    # get step-size for rho and theta
    d = np.sqrt(np.square(edge_height) + np.square(edge_width))

    N_theta = math.ceil(180 / dtheta)
    N_theta += 1
    thetas, binwidth_theta = np.linspace(-90, 90,
                                         N_theta, retstep=True, dtype='float32')
    N_rho = math.ceil(2 * d / drho)
    N_rho += 1
    rhos, binwidth_rho = np.linspace(-d, d,
                                     N_rho, retstep=True, dtype='float32')

    # get cosine and sinus
    cos_thetas = np.cos(np.deg2rad(thetas)).astype('float32')
    sin_thetas = np.sin(np.deg2rad(thetas)).astype('float32')

    # extract edge points which are at least larger than sigma * background rms and convert them
    edge_points = np.argwhere(image != 0.)
    edge_points = edge_points - np.array([[edge_height_half, edge_width_half]])

    # calculate matrix product
    rho_values = np.matmul(edge_points, np.array([sin_thetas, cos_thetas],
                                                 dtype='float32')).astype('float32')
    del image, cos_thetas, sin_thetas, edge_points

    # get the hough space
    bins = [len(thetas), len(rhos)]
    vals = np.array([np.tile(thetas, rho_values.shape[0]),
                     rho_values.ravel().astype('float32')], dtype='float32')

    ranges = np.array([[np.min(thetas), np.max(thetas)],
                       [np.min(rhos), np.max(rhos)]]).astype('float32')
    accumulator = fhist.histogram2d(*vals, bins=bins, range=ranges).astype('float32')

    # rhos in rows and thetas in columns (rho, theta)
    accumulator = np.transpose(accumulator)

    del rho_values, vals, ranges, bins
    gc.collect()

    return accumulator, rhos, thetas, drho, dtheta


def quantize(df, colname='orientation', tolerance=0.005):
    model = AgglomerativeClustering(distance_threshold=tolerance,
                                    linkage='complete',
                                    n_clusters=None).fit(df[[colname]].values)
    df = (df
          .assign(groups=model.labels_,
                  center=df[[colname]]
                  .groupby(model.labels_)
                  .transform(lambda v: (v.max() + v.min()) / 2.))
          )

    del model
    gc.collect()

    return df


def get_min_trail_length(config: dict):
    """"""
    sat_id = config['sat_id']
    pixscale = config['pixscale']

    # get nominal orbit altitude
    sat_h_orb_ref = _base_conf.SAT_HORB_REF['ONEWEB']
    if 'STARLINK' in sat_id:
        sat_h_orb_ref = _base_conf.SAT_HORB_REF['STARLINK']
    if 'BLUEWALKER' in sat_id:
        sat_h_orb_ref = _base_conf.SAT_HORB_REF['BLUEWALKER']

    H_sat = sat_h_orb_ref * 1000.
    R = const.R_earth.value + H_sat
    y = np.sqrt(const.G.value * const.M_earth.value / R ** 3)
    y *= 6.2831853
    return np.rad2deg(y) * 3600. / pixscale * 0.1


def detect_sat_trails(image: np.ndarray,
                      config: dict,
                      alpha: float = 10.,
                      sigma_blurr: float = 3.,
                      borderLen: int = 1,
                      mask: np.ndarray = None,
                      silent: bool = False):
    """ Find satellite trails in image and extract region properties.

    Use a modified version of the Hough line detection to detect the trails and to estimate length, width,
    orientation of multiple satellite trails.

    Based on the method for the detection of extended line-segments by
    Xu, Z., Shin, B.-S., Klette, R. 2015.
    Closed form line-segment extraction using the Hough transform.
    Pattern Recognition 48, 4012–4023, doi:10.1016/j.patcog.2015.06.008

    """

    # Initialize logging for this user-callable function
    log.setLevel(logging.getLevelName(log.getEffectiveLevel()))

    if not isinstance(image, np.ndarray):
        raise ValueError("Input image not ndarray object.")

    n_trail = 0
    properties = ['label', 'centroid', 'area', 'orientation',
                  'axis_major_length',
                  'axis_minor_length',
                  'eccentricity',
                  'feret_diameter_max', 'coords']

    trail_data = {'reg_info_lst': []}

    # get minimum expected trail length for satellite
    min_trail_length = get_min_trail_length(config)

    # zero out the borders with width given by borderLen
    if borderLen > 0:
        len_x, len_y = image.shape
        image[0:borderLen, 0:len_y] = 0
        image[len_x - borderLen:len_x, 0:len_y] = 0
        image[0:len_x, 0:borderLen] = 0
        image[0:len_x, len_y - borderLen:len_y] = 0

    if mask is not None:
        m = np.where(mask, 0, 1)
        image *= m

    blurred_f = nd.gaussian_filter(image, 2.)
    filter_blurred_f = nd.gaussian_filter(blurred_f, sigma_blurr)
    sharpened = blurred_f + alpha * (blurred_f - filter_blurred_f)

    mean, _, std = sigma_clipped_stats(sharpened, grow=False)
    sharpened -= mean

    threshold = detect_threshold(sharpened, 1., mean, std)
    sources = detect_sources(sharpened, threshold=threshold,
                             npixels=9, connectivity=8, mask=mask)
    segm = SegmentationImage(sources.data)
    segm_init = segm.copy()

    # create regions
    regs = regionprops_table(segm.data, properties=properties)
    df_init = pd.DataFrame(regs)
    df_init['orientation_deg'] = df_init['orientation'] * 180. / np.pi

    # first filter roundish objects
    labels = df_init.query("eccentricity >= 0.985 and area > @min_trail_length")['label']
    if labels.size == 0:
        if not silent:
            log.info("  ==> NO satellite trails detected.")
        del image, blurred_f, filter_blurred_f, sharpened, \
            threshold, sources, segm, segm_init, \
            regs, df_init
        gc.collect()
        return None, False

    segm.keep_labels(labels=labels, relabel=False)

    regs = regionprops_table(segm.data, properties=properties)
    df = pd.DataFrame(regs)
    df['orientation_deg'] = df['orientation'] * 180. / np.pi

    if df.shape[0] == 1:
        if df['eccentricity'].values[0] >= 0.999 and \
                df['axis_major_length'].values[0] > min_trail_length:
            if not silent:
                log.info("  ==> 1 satellite trails detected.")
            reg_dict = get_trail_properties(segm, df, config, silent=True)
            reg_dict['detection_imgs'].update({'img_sharp': sharpened,
                                               'segm_map': segm_init})

            trail_data['reg_info_lst'].append(reg_dict)
            n_trail = 1
        else:
            if not silent:
                log.info("  ==> NO satellite trails detected.")
            del image, blurred_f, filter_blurred_f, sharpened, \
                threshold, sources, segm, segm_init, \
                regs, df
            gc.collect()
            return None, False
    else:
        df1 = quantize(df, colname='orientation_deg', tolerance=config['MAX_DISTANCE'])
        grouped = df1.groupby(by='groups')
        for name, group in grouped:
            if (group['eccentricity'] <= 0.995).all() or group['axis_major_length'].sum() < min_trail_length:
                df1 = df1.drop(grouped.get_group(name).index)
        if df1.empty:
            if not silent:
                log.info("  ==> NO satellite trails detected.")

            del image, blurred_f, filter_blurred_f, sharpened, \
                threshold, sources, segm, segm_init, \
                regs, df
            gc.collect()

            return None, False

        s = df1.groupby(by='groups')['axis_major_length'].sum().reset_index()
        res = s[s['axis_major_length'] == s.groupby(['groups'])['axis_major_length'].transform('max')]
        ind = res.sort_values(by='axis_major_length', ascending=False)['groups'].values

        # !!! select only the longest for now !!!
        ind = ind[0]
        df2 = grouped.get_group(ind)

        segm.keep_labels(labels=df['label'], relabel=False)
        reg_dict = get_trail_properties(segm, df2, config, silent=True)

        if config['roi_offset'] is not None:
            coords = np.array(reg_dict['coords'])
            coords[0] = coords[0] + config['roi_offset'][0]
            coords[1] = coords[1] + config['roi_offset'][1]
            reg_dict['coords'] = tuple(coords)

        reg_dict['detection_imgs'].update({'img_sharp': sharpened,
                                           'segm_map': segm_init})

        trail_data['reg_info_lst'].append(reg_dict)

        del df1, df2, reg_dict
        gc.collect()

    trail_data['N_trails'] = n_trail

    del image, blurred_f, filter_blurred_f, sharpened, \
        threshold, sources, segm, segm_init, \
        regs, df

    gc.collect()

    return trail_data, True


def get_distance(loc1, loc2):
    """Distance between observer and satellite on the surface of earth"""
    return distance.distance(loc1, loc2, ellipsoid='WGS-84').km


def get_trail_properties(segm_map, df, config,
                         silent: bool = False):
    """
    Perform Hough transform and determine trail parameter, i.e. length, width, and
    orientation

    """

    theta_bin_size = config['THETA_BIN_SIZE']
    rho_bin_size = config['RHO_BIN_SIZE']

    binary_img = np.zeros(segm_map.shape)
    for row in df.itertuples(name='label'):
        binary_img[segm_map.data == row.label] = 1

    # apply dilation
    footprint = np.ones((1, 1))
    dilated = nd.binary_dilation(binary_img, structure=footprint, iterations=3)

    # create hough space and get peaks
    hspace_smooth, peaks, dist, theta = get_hough_transform(dilated,
                                                            theta_bin_size=theta_bin_size,
                                                            rho_bin_size=rho_bin_size,
                                                            silent=silent)

    if len(peaks) > 1:
        edge_height, edge_width = dilated.shape[:2]
        edge_height_half, edge_width_half = edge_height // 2, edge_width // 2
        del hspace_smooth

        sel_dict = {i: [] for i in range(len(peaks))}
        for i in range(len(peaks)):
            angle = peaks[i][1]
            dist = peaks[i][2]

            a = np.cos(np.deg2rad(angle))
            b = np.sin(np.deg2rad(angle))
            x0 = (a * dist) + edge_width_half
            y0 = (b * dist) + edge_height_half
            x1 = int(x0 + edge_width * (-b))
            y1 = int(y0 + edge_height * a)
            x2 = int(x0 - edge_width * (-b))
            y2 = int(y0 - edge_height * a)

            rr, cc = line(y1, x1, y2, x2)

            # check that columns are within image boundaries
            idx = (cc > 0) & (cc < dilated.shape[1])
            rr = rr[idx]
            cc = cc[idx]

            # check that rows are within image boundaries
            idx = (rr > 0) & (rr < dilated.shape[0])
            rr = rr[idx]
            cc = cc[idx]

            # find the unique labels that make up the particular trail
            label_list = np.unique(segm_map.data[rr, cc])
            [sel_dict[i].append(row.Index) for row in df.itertuples(name='label') if row.label in label_list]

        # select only the label that belong to the first peak
        # this can be extended later to let the user choose which to pick
        # set 0 for first entry
        sel_idx = config['PARALLEL_TRAIL_SELECT']
        _df = df.loc[sel_dict[sel_idx]]
        binary_img = np.zeros(segm_map.shape)
        for row in _df.itertuples(name='label'):
            binary_img[segm_map.data == row.label] = 1

        # apply dilation
        footprint = np.ones((1, 1))
        dilated = nd.binary_dilation(binary_img, structure=footprint, iterations=5)

        # create hough space and get peaks
        hspace_smooth, peaks, dist, theta = get_hough_transform(dilated,
                                                                theta_bin_size=theta_bin_size,
                                                                rho_bin_size=rho_bin_size,
                                                                silent=silent)

    indices = np.where(hspace_smooth == peaks[0][0])
    row_ind, col_ind = indices[0][0], indices[1][0]
    center_idx = (row_ind, col_ind)

    # set sub-window size
    theta_sub_win_size = math.ceil(config['THETA_SUB_WIN_SIZE'] / theta_bin_size)
    rho_sub_win_size = math.ceil(4. * np.sqrt(2.)
                                 * config['RHO_SUB_WIN_RES_FWHM']
                                 * config['fwhm'] / rho_bin_size)
    sub_win_size = (rho_sub_win_size, theta_sub_win_size)

    theta_init = np.deg2rad(theta[col_ind])
    fit_res = fit_trail_params(hspace_smooth, dist, theta, theta_init,
                               dilated.shape, center_idx, sub_win_size)

    reg_dict = {'coords': fit_res[0],  # 'coords':  reg.centroid[::-1],
                'coords_err': fit_res[1],
                'width': fit_res[2], 'e_width': fit_res[3],
                'height': fit_res[4], 'e_height': fit_res[5],
                'orient_rad': fit_res[6],
                'e_orient_rad': fit_res[7],
                'orient_deg': fit_res[8],
                'e_orient_deg': fit_res[9],
                'param_fit_dict': fit_res[11],
                'detection_imgs': {
                    'img_sharp': None,
                    'segm_map': None,
                    'img_labeled': segm_map,
                    'trail_mask': dilated
                }}

    del segm_map, binary_img, fit_res, dilated, df, hspace_smooth, \
        indices, row_ind, col_ind, dist, theta
    gc.collect()

    return reg_dict


def get_hough_transform(image: np.ndarray, sigma: float = 2.,
                        theta_bin_size=0.02,
                        rho_bin_size=1,
                        silent: bool = False):
    """ Perform a Hough transform on the selected region and fit parameter """

    # generate Hough space H(theta, rho)
    res = create_hough_space_vectorized(image=image,
                                        dtheta=theta_bin_size,
                                        drho=rho_bin_size,
                                        silent=silent)
    hspace, dist, theta, _, _ = res
    # apply gaussian filter to smooth the image
    hspace_smooth = nd.gaussian_filter(hspace, sigma)

    del image, res, hspace, _

    # find peaks in H-space and determine length, width, orientation, and
    # centroid coordinates of each satellite trail
    peaks = list(zip(*hough_line_peaks(hspace_smooth, theta, dist)))

    return hspace_smooth, peaks, dist, theta
