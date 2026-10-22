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

""" Modules """
from __future__ import annotations

import cv2
import ephem
import fast_histogram as fhist
import inspect
import logging
import math
import numpy as np
import os
import pandas as pd
import requests

from datetime import datetime
from functools import lru_cache
from dateutil import parser
from lmfit import Model
from lmfit.models import (ConstantModel, GaussianModel,
                          ExponentialGaussianModel, LognormalModel)

from pyorbital import astronomy

from scipy import ndimage as nd
# from scipy.sparse import csr_matrix, csr_array
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.stats import norm

from skimage.draw import line
from skimage.measure import regionprops_table
from skimage.transform import hough_line_peaks

from astropy import constants as const
from astropy import units as u
from astropy.coordinates import Angle
from photutils.aperture import RectangularAperture
from astropy.stats import (sigma_clipped_stats, sigma_clip, mad_std, gaussian_fwhm_to_sigma)

from photutils.segmentation import (SegmentationImage, detect_sources,
                                    detect_threshold, SourceCatalog)

from sklearn.cluster import AgglomerativeClustering

try:
    import matplotlib
except ImportError:
    plt = None
else:
    import matplotlib
    import matplotlib as mpl
    from matplotlib import cm
    from matplotlib import pyplot as plt
    import matplotlib.lines as mlines
    import matplotlib.gridspec as gridspec  # GRIDSPEC !
    from matplotlib.ticker import AutoMinorLocator, LogLocator
    from astropy.visualization import LinearStretch, LogStretch, SqrtStretch
    from astropy.visualization.mpl_normalize import ImageNormalize
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib.collections import PolyCollection

    # matplotlib parameter
    try:
        matplotlib.use('Qt5Agg')
    except ImportError:
        pass
    matplotlib.rc("lines", linewidth=1.2)
    matplotlib.rc('figure', dpi=150, facecolor='w', edgecolor='k')
    matplotlib.rc('text.latex', preamble=r'\usepackage{sfmath}')
    matplotlib.rc('pdf', fonttype=42)
    matplotlib.rc('ps', fonttype=42)

# pipeline-specific modules
from . import base_conf as bc

# -----------------------------------------------------------------------------

""" Meta-info """
__author__ = "Christian Adam"
__copyright__ = 'Copyright 2021-2023, CLEOSat group'
__credits__ = ["Eduardo Unda-Sanzana, Jeremy Tregloan-Reed, Christian Adam"]
__license__ = "GPL-3.0 license"
__maintainer__ = "Christian Adam"
__email__ = "christian.adam84@gmail.com"
__status__ = "Production"

__taskname__ = 'satellites'
# -----------------------------------------------------------------------------

""" Parameter used in the script """
log = logging.getLogger(__name__)

MODULE_PATH = os.path.dirname(inspect.getfile(inspect.currentframe()))

norm_type = cv2.NORM_MINMAX


# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Marked for deletion
# -----------------------------------------------------------------------------
#
# def get_angular_distance(observer: ephem.Observer,
#                          sat_az_alt: list,
#                          prev_sat_az_alt: list) -> float:
#     """ Calculate angular distance between two coordinates.
#
#     Parameters
#     ----------
#     observer: ephem.Observer
#         Observer object
#     sat_az_alt: list
#         Satellite azimuth and altitude [degrees]
#     prev_sat_az_alt: list
#         Previous satellite azimuth and altitude [degrees]
#
#     Returns
#     -------
#     float
#         Angular distance [arcsec]
#     """
#     # Convert current position to RA/Dec
#     ra, dec = observer.radec_of(np.radians(sat_az_alt[0]),
#                                 np.radians(sat_az_alt[1]))
#     # Convert previous position to RA/Dec
#     prev_ra, prev_dec = observer.radec_of(np.radians(prev_sat_az_alt[0]),
#                                           np.radians(prev_sat_az_alt[1]))
#     # Calculate angular distance using haversine formula
#     dtheta = 2. * np.arcsin(np.sqrt(np.sin(0.5 * (dec - prev_dec)) ** 2.
#                                     + np.cos(dec) * np.cos(prev_dec)
#                                     * np.sin(0.5 * (ra - prev_ra)) ** 2.))
#     # Convert to arcseconds
#     dtheta *= 206264.806
#
#     return dtheta
# -----------------------------------------------------------------------------


def get_average_magnitude(flux, flux_err, std_fluxes, std_mags, mag_corr, mag_corr_sat=0.,
                          mag_scale=None, area=None):
    """Calculate mean magnitude.

    Parameters
    ----------
    flux: float
        Measured flux (in electrons)
    flux_err: float
        Uncertainty in the measured flux (in electrons)
    std_fluxes: np.ndarray
        Standard star fluxes and their uncertainties (in electrons)
    std_mags: np.ndarray
        Standard star magnitudes and their uncertainties
    mag_corr: tuple
        Magnitude correction and its uncertainty
    mag_corr_sat: float
        Magnitude correction for the satellite.
    mag_scale: float | None
        Magnitude scale correction (e.g., for exposure time).
    area: float | None
        Area correction (e.g., for aperture size).

    Returns
    -------
    tuple
        Magnitude and its uncertainty
    """

    # Calculate the error in the magnitude difference using error propagation
    flux_ratio = flux / std_fluxes[:, 0]
    flux_ratio_err = np.sqrt((flux_err / flux) ** 2 + (std_fluxes[:, 1] / std_fluxes[:, 0]) ** 2) * flux_ratio

    diff_mag = -2.5 * np.log10(flux / std_fluxes[:, 0])
    diff_mag_err = 2.5 / np.log(10) * flux_ratio_err / flux_ratio

    # Observed satellite magnitude relative to std stars
    mag = diff_mag + std_mags[:, 0] + mag_corr[0] + mag_corr_sat
    mag = mag + mag_scale if mag_scale is not None else mag
    mag = mag + 2.5 * np.log10(area) if area is not None else mag

    mag_err = np.sqrt(diff_mag_err ** 2 + std_mags[:, 1] ** 2 + mag_corr[1] ** 2)

    mask = np.array(sigma_clip(mag,
                               sigma=3.,
                               maxiters=10,
                               cenfunc=np.nanmedian,
                               stdfunc=mad_std).mask)

    mag_cleaned = mag[~mask]
    mag_err_cleaned = mag_err[~mask]

    mag_avg_w = np.nanmean(mag_cleaned)
    mag_avg_err = np.nanmean(mag_err_cleaned)

    bc.clean_up(flux, flux_err, std_fluxes, std_mags, mag_corr)

    return mag_avg_w, mag_avg_err


def get_magnitude_zpmag(flux, flux_err, zp_mag, mag_corr_sat=0., mag_scale=None, area=None):
    """Calculate satellite magnitude using the zero-point magnitude from the header.

    Reference:
    https://www.gnu.org/software/gnuastro/manual/html_node/Brightness-flux-magnitude.html

    Parameters
    ----------
    flux: float
        Measured flux (in electrons)
    flux_err: float
        Uncertainty in the measured flux (in electrons)
    zp_mag: float
        Zero-point magnitude
    mag_corr_sat: float
        Magnitude correction for the satellite.
    mag_scale: float | None
        Magnitude scale correction (e.g., for exposure time).
    area: float | None
        Area correction (e.g., for aperture size).

    Returns
    -------
    tuple
        Magnitude and its uncertainty
    """
    # The flux = counts * t_exp -> zp_mag in mag
    mag = -2.5 * np.log10(flux) + zp_mag + mag_corr_sat
    mag = mag + mag_scale if mag_scale is not None else mag
    mag = mag + 2.5 * np.log10(area) if area is not None else mag

    # Calculate uncertainty
    mag_err = 2.5 / np.log(10) * (flux_err / flux)

    return mag, mag_err


def sun_inc_elv_angle(obsDate: str | datetime, geo_loc: tuple):
    """Calculate solar azimuth, elevation, and incidence angle for a given location on Earth.

    Parameters
    ----------
    obsDate: str | datetime
        Time of observation (UTC)
    geo_loc: tuple
        Geographic location (latitude, longitude) of the observation site [degrees]

    Returns
    -------
    float
        Solar incidence angle theta [degrees]
    """
    # Get satellite latitude and longitude
    (geo_lat, geo_lon) = geo_loc

    dtobj = obsDate
    if isinstance(obsDate, str):
        # Convert the time of observation
        d = parser.parse(obsDate)
        dtobj = datetime(d.year, d.month, d.day,
                         d.hour, d.minute, d.second)

    # Get sun elevation and azimuth angle from
    sun_el, sun_az = astronomy.get_alt_az(dtobj, geo_lon, geo_lat)
    sunEL = sun_el * 180. / math.pi  # Convert from radians to degrees

    # Solar incidence angle theta
    theta = 90. + sunEL

    # Return results
    return theta


def get_radius_earth(B):
    """Get the radius of Earth for a given observation site latitude.

    Parameters
    ----------
    B: float
        Latitude of observation site [degrees]

    Returns
    -------
    float
        Radius of Earth at given latitude [km]
    """
    B = math.radians(B)  # converting into radians

    a = bc.REARTH_EQU  # Radius at sea level at the equator
    b = bc.REARTH_POL  # Radius at poles

    c = (a ** 2 * math.cos(B)) ** 2
    d = (b ** 2 * math.sin(B)) ** 2
    e = (a * math.cos(B)) ** 2
    f = (b * math.sin(B)) ** 2

    R = math.sqrt((c + d) / (e + f))
    return R


def ang_distance(lat1, lat2, lon1, lon2):
    """ Calculate angular distance between two coordinates in radians.

    Parameters
    ----------
    lat1: float
        Latitude of point 1 [degrees]
    lat2: float
        Latitude of point 2 [degrees]
    lon1: float
        Longitude of point 1 [degrees]
    lon2: float
        Longitude of point 2 [degrees]

    Returns
    -------
    float
        Angular distance [radians]
    """
    # Get delta lambda
    if lon2 > lon1:
        delta_lambda = math.radians(lon2 - lon1)
    else:
        delta_lambda = math.radians(lon1 - lon2)
    # Get phi_1 and phi_2
    phi_1 = math.radians(lat1)
    phi_2 = math.radians(lat2)
    # Calculate delta phi and calculate adjusted delta lambda
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = 2. * math.pi - delta_lambda if delta_lambda > math.pi else delta_lambda
    # Haversine formula
    a = math.sin(delta_phi / 2.0) ** 2 + math.cos(phi_1) * math.cos(phi_2) * math.sin(delta_lambda / 2.0) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return c


def get_solar_phase_angle(sat_az, sat_alt, geo_loc, obs_range, obsDate):
    """Get the solar phase angle via the Sun-Sat angle.

    Parameters
    ----------
    sat_az: float
        Satellite azimuth angle [degrees]
    sat_alt: float
        Satellite altitude angle [degrees]
    geo_loc: tuple
        Geographic location (latitude, longitude) of the observation site [degrees]
    obs_range: float
        Satellite-observer range [km]
    obsDate: str | datetime
        Time of observation (UTC)

    Returns
    -------
    tuple
        Sun-Sat angle [degrees], phase angle [degrees], solar azimuth [degrees], solar elevation [degrees]
    """
    # Setup constants and compute sun position
    au = bc.AU_TO_KM
    sun = ephem.Sun()
    sun.compute(obsDate)
    # Get sun-earth distance in km
    a = sun.earth_distance * au
    b = obs_range  # satellite-observer distance in km

    # Get satellite latitude and longitude
    (geo_lat, geo_lon) = geo_loc

    dtobj = obsDate
    if isinstance(obsDate, str):
        # Convert the time of observation
        d = parser.parse(obsDate)
        dtobj = datetime(d.year, d.month, d.day,
                         d.hour, d.minute, d.second)

    # Get sun elevation and azimuth angle from
    sun_el, sun_az = astronomy.get_alt_az(dtobj, geo_lon, geo_lat)

    # Get solar azimuth and altitude in degrees
    sun_alt = np.rad2deg(sun_el)
    sun_az = np.rad2deg(sun_az)
    sun_az = sun_az + 360. if sun_az < 0. else sun_az

    # Calculate angle c using the cosine theorem
    angle_c = ang_distance(sun_alt, sat_alt, sun_az, sat_az)
    c = math.sqrt(math.pow(a, 2) + math.pow(b, 2) - 2 * a * b * math.cos(angle_c))
    angle_a = math.acos((math.pow(b, 2) + math.pow(c, 2) - math.pow(a, 2)) / (2 * b * c))
    angle_b = math.pi - angle_a - angle_c
    # Get phase angle and sun-sat angle
    phase_angle = math.pi - angle_c - angle_b
    phase_angle = np.rad2deg(phase_angle)
    sun_sat_ang = np.rad2deg(angle_c)

    return sun_sat_ang, phase_angle, sun_az, sun_alt


def get_observer_angle(sat_lat, geo_lat, sat_h_orb_km, h_obs_km, sat_range_km):
    """ Get the observer angle (satellite elevation angle) from the ground station.

    Parameters
    ----------
    sat_lat: float
        Satellite sub-point latitude [degrees]
    geo_lat: float
        Ground station latitude [degrees]
    sat_h_orb_km: float
        Satellite orbit altitude [km]
    h_obs_km: float
        Observer altitude [km]
    sat_range_km: float
        Satellite-observer range [km]

    Returns
    -------
    float
        Observer phase angle in degrees.
    """
    # Initialize logging for this user-callable function
    log.setLevel(logging.getLevelName(log.getEffectiveLevel()))

    # Get the radius of Earth for a given observation site latitude
    Rearth_obs = get_radius_earth(geo_lat)
    Rearth_h_obs = Rearth_obs + h_obs_km

    # Get the radius of Earth for a given satellite latitude
    Rearth_sat = get_radius_earth(sat_lat)
    Rearth_h_sat = Rearth_sat + sat_h_orb_km

    # Use cosine theorem to calculate the obs phase angle
    theta_2 = math.acos((sat_range_km ** 2 + Rearth_h_sat ** 2 - Rearth_h_obs ** 2) /
                        (2. * sat_range_km * Rearth_h_sat))
    theta_2 = np.rad2deg(theta_2)

    return theta_2


def get_obs_range(sat_elev, h_orb, h_obs_km, lat):
    """Get satellite-observer range.

    Adopted from get_sat_observer_range.m by Angel Otarola, aotarola@tmt.org

    Parameters
    ----------
    sat_elev: float
        Satellite elevation angle in degrees.
    h_orb: float
        Satellite orbit altitude in km.
    h_obs_km: float
        Observer altitude in km.
    lat: float
        Observer latitude in degrees.

    Returns
    -------
    float
        Distance from observer to satellite in km.
    """
    # Initialize logging for this user-callable function
    log.setLevel(logging.getLevelName(log.getEffectiveLevel()))

    # Get the radius of Earth for a given observation site latitude
    Rearth_obs = get_radius_earth(lat)
    Rearth = Rearth_obs + h_obs_km

    # Derived quantities
    # S = satellite position along its orbit
    # O = observer position on the surface
    # C = Geo-center

    gamma = 90. + sat_elev  # degrees, Angle with the center in the observer - <SOC>

    # Length of the CO side = Rearth
    # Length of the CS side = Rearth + Horb

    CO = Rearth  # km
    CS = Rearth + h_orb  # km

    # Use the law of sin for derivation of the beta angle <CSO>
    beta = math.asin(CO / CS * math.sin(gamma * math.pi / 180.)) * 180. / math.pi  # degrees

    # Derivation of the alpha angle, <OCS>
    alpha = 180. - gamma - beta  # degrees,

    # Use of the cos law for derivation of the distance from Observer to
    # Satellite (OS), accounting for its elevation (or zenith) angle
    OS = math.sqrt(CO ** 2 + CS ** 2 - 2. * CO * CS * math.cos(alpha * math.pi / 180.))

    return OS


def poly_func(x, a0, a1, a2):
    """ Second order polynom """
    return a0 + a1 * x + a2 * x ** 2


def compute_indices(c, ws, length):
    """ Compute low and high indices for a given center, window size, and array length.

    Parameters
    ----------
    c: int
        center index
    ws: int
        window size
    length: int
        length of the array

    Returns
    -------
    tuple
        low index, high index
    """
    # Default setting: % operations to accommodate odd/even window sizes
    low, high = c - (ws // 2), c + (ws // 2) + ws % 2

    # Correction for overlap with the borders of the array
    if low < 0:
        low, high = 0, ws
    elif high > length:
        low, high = -ws, None

    return low, high


def extract_peak_window(arr, coords, window_size=(3, 3)):
    """ Extract sub-array from given coordinates with specified window size.

    Parameters
    ----------
    arr: np.ndarray
        Array from which to extract the sub-array.
    coords: tuple
        Coordinates (row, column) of the center of the sub-array.
    window_size: tuple
        Size of the sub-array (rows, columns).

    Returns
    -------
    tuple
        Sub-array, row indices, column indices
    """
    # Extract array shapes and window sizes into single variables
    len_r, len_c = arr.shape
    wsr, wsc = window_size

    # Extract coords and correct for 0-indexing
    r, c = coords
    r0, c0 = r, c

    row_low, row_high = compute_indices(r0, wsr, len_r)
    col_low, col_high = compute_indices(c0, wsc, len_c)

    return arr[row_low:row_high, col_low:col_high], (row_low, row_high), (col_low, col_high)


def calculate_trail_parameter(fit_result):
    """ Calculate length, width and orientation.

    Reference: Closed form line-segment extraction using the Hough transform
               Xu, Zezhong; Shin, Bok-Suk; Klette, Reinhard
    Bibcode: 2015PatRe..48.4012X
    """
    # Get parameter
    a0 = fit_result.params['a0'].value
    a1 = fit_result.params['a1'].value
    a2 = fit_result.params['a2'].value

    err_a0 = fit_result.params['a0'].stderr
    err_a1 = fit_result.params['a1'].stderr
    err_a2 = fit_result.params['a2'].stderr

    a = a1 ** 2. / (4. * a2)

    # Length
    L = np.sqrt(12.) * np.sqrt(a2 + a0 - a)
    c = np.sqrt((4. * a0 * a2 - a1 ** 2. + 4. * a2 ** 2.) / a2)
    err_L = np.sqrt((2. * np.sqrt(3) / c * err_a0) ** 2
                    + (np.sqrt(3) / c * a1 / a2 * err_a1) ** 2
                    + (((np.sqrt(3) * a1 ** 2) / (2. * a2 ** 2)
                        + 2 * np.sqrt(3)) * err_a2 / c) ** 2)

    # Width
    T = np.sqrt(12.) * np.sqrt(a0 - a)
    d = np.sqrt((4. * a0 * a2 - a1 ** 2.) / a2)
    err_T = np.sqrt((2. * np.sqrt(3) / d * err_a0) ** 2
                    + (np.sqrt(3) / d / a2 * err_a1) ** 2
                    + ((np.sqrt(3) * a1 ** 2) / (2. * a2 ** 2) * err_a2 / d) ** 2)
    # Orientation
    theta_p = - a1 / (2. * a2)
    theta_p_err = np.sqrt((err_a1 / (2. * a2)) ** 2
                          + (a1 * err_a2 / (2. * a2 ** 2)) ** 2)

    e = np.sqrt(1. - ((T / 2.) ** 2. / (L / 2.) ** 2.))

    del fit_result

    return (L, err_L,
            T, err_T, theta_p, theta_p_err,
            np.rad2deg(theta_p), np.rad2deg(theta_p_err), e)


def fit_trail_params(Hij, rhos, thetas, theta_0, image_size, cent_ind, win_size,
                     fit_method='leastsq'):
    """Perform fit on a given sub-window of the Hough space.

    Parameters
    ----------
    Hij: np.ndarray
        Hough space array.
    rhos: np.ndarray
        Rho values.
    thetas: np.ndarray
        Theta values.
    theta_0: float
        Initial theta value [radians].
    image_size: tuple
        Size of the original image (height, width).
    cent_ind: tuple
        Center indices (row, column) of the peak in the Hough space.
    win_size: tuple
        Window size (rows, columns) for the sub-array extraction.
    fit_method: str
        Fitting method to use (default: 'leastsq').

    Returns
    -------
    tuple
        Fitted parameters and fit results.
    """
    fit_results = {}

    # Extract sub-window from input
    h_peak_win, row_ind, col_ind = extract_peak_window(Hij,
                                                       (cent_ind[0], cent_ind[1]),
                                                       (win_size[0], win_size[1]))
    fit_results['Hij_sub'] = h_peak_win

    # Set-up axis values, x=rho, y=theta
    rho_j = rhos[row_ind[0]:row_ind[1]]
    theta_i = thetas[col_ind[0]:col_ind[1]]
    theta_i_rad = np.deg2rad(theta_i)
    theta_i_rad_new = np.linspace(start=theta_i_rad[0], stop=theta_i_rad[-1],
                                  num=int(len(theta_i_rad) * 100), endpoint=True)
    fit_results['Hij_rho'] = rho_j
    fit_results['Hij_theta'] = theta_i

    # Calculate mean and variance for each column in the peak region
    mu = np.dot(h_peak_win.T, rho_j) / np.sum(h_peak_win, axis=0)
    var = np.empty(h_peak_win.shape[1])
    for col in range(h_peak_win.shape[1]):
        diff = np.subtract(rho_j, mu[col]) ** 2
        var[col] = (np.dot(h_peak_win.T, diff[:, None]) /
                    np.sum(h_peak_win[:, col])).flatten()[col]

    # Fit quadratic model
    pmodel = Model(poly_func, independent_vars='x')
    params = pmodel.make_params(a0=1., a1=1., a2=1.)

    # WARNING: This might cause problems in the future.
    # lmfit v1.1.0 works fine, but since 1.2.0 and above the fit fails; reason unknown
    quad_fit_result = pmodel.fit(data=var,
                                 params=params,
                                 x=theta_i_rad,
                                 method=fit_method,
                                 weights=np.sqrt(1. / var),
                                 nan_policy='omit',
                                 scale_covar=True)

    fit_results['quad_fit'] = quad_fit_result
    fit_results['quad_xnew'] = theta_i_rad_new
    fit_results['quad_x'] = theta_i_rad
    fit_results['quad_y'] = var

    # Get result parameter
    (L, err_L, T, err_T,
     theta_p_rad, e_theta_p_rad,
     theta_p_deg, e_theta_p_deg, e) = calculate_trail_parameter(quad_fit_result)
    # print(L, err_L, T, err_T,
    #       theta_p_rad, e_theta_p_rad,
    #       theta_p_deg, e_theta_p_deg, e)

    # Set-up axis for fit; apply alternate transformation if 45deg < theta < 135deg
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

    # Fit linear model
    pmodel = Model(poly_func, independent_vars='x')
    params = pmodel.make_params(a0=1., a1=1., a2=0.)

    # Fit quadratic model with a2 fixed
    params['a2'].vary = False
    lin_fit_result = pmodel.fit(data=y,
                                params=params,
                                x=x,
                                method=fit_method,
                                nan_policy='omit',
                                scale_covar=True)
    fit_results['lin_fit'] = lin_fit_result
    fit_results['lin_xnew'] = xnew
    fit_results['lin_x'] = x
    fit_results['lin_y'] = y
    fit_results['lin_y_label'] = y_label

    # Transform to image coordinates
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

    result = ((xc, yc), (xc_err, yc_err),
              L, err_L, T, err_T, theta_p_rad + np.pi / 2., e_theta_p_rad,
              theta_p_deg + 90., e_theta_p_deg, e, fit_results)

    return result


@lru_cache(maxsize=None)
def _precompute_hough_grid(img_height: int,
                           img_width: int,
                           dtheta: float,
                           drho: float):
    """
    Precompute Hough grid values for a given image shape and bin sizes.

    Parameters
    ----------
    img_height: int
        Height of the image.
    img_width: int
        Width of the image.
    dtheta: float
        Step size for theta in degrees.
    drho: float
        Step size for rho.

    Returns
    -------
    tuple
        rhos, thetas, sin_thetas, cos_thetas
    """
    d = float(np.hypot(img_height, img_width))

    # Rho grid
    N_rho = math.ceil(2.0 * d / drho) + 1
    rhos, _ = np.linspace(-d, d, N_rho,
                          retstep=True)

    # Theta grid \[-90, 90\]
    N_theta = math.ceil(180.0 / dtheta) + 1
    thetas, _ = np.linspace(-90.0, 90.0, N_theta,
                            retstep=True)

    thetas_rad = np.deg2rad(thetas, dtype=np.float32)
    cos_thetas = np.cos(thetas_rad)
    sin_thetas = np.sin(thetas_rad)

    return rhos, thetas, sin_thetas, cos_thetas


# noinspection PyAugmentAssignment
def create_hough_space_vectorized(image: np.ndarray,
                                  dtheta: float = 0.05,
                                  drho: float = 1,
                                  theta_window: tuple = None):
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
    theta_window:
        Optional ``(theta_center, theta_half_width)`` in degrees. If given (and the
        half-width is positive), the theta search is restricted to a window around
        ``theta_center`` by INDEX-SLICING the cached global theta grid. The slice keeps
        the exact same grid points as the full range, so the interior of the window is
        byte-identical to the full computation. Defaults to None (full ``[-90, 90]``).
    """
    # Initialize logging for this user-callable function
    log.setLevel(logging.getLevelName(log.getEffectiveLevel()))

    # Image properties
    img_height, img_width = image.shape[:2]
    img_height_half, img_width_half = img_height // 2, img_width // 2

    # Get (rhos, thetas, sin, cos) from cache
    rhos, thetas, sin_thetas, cos_thetas = _precompute_hough_grid(
        img_height=img_height,
        img_width=img_width,
        dtheta=float(dtheta),
        drho=float(drho)
    )

    # Optionally restrict the theta grid to a window around a known orientation.
    # IMPORTANT: this must be an index slice of the cached global grid, NEVER a fresh
    # linspace - a fresh grid would shift the grid points and break byte-identity with
    # the full-range result. The caller (get_hough_transform) guards that the peak stays
    # comfortably interior and otherwise falls back to the full range.
    if theta_window is not None:
        theta_center, theta_half_width = theta_window
        if theta_half_width is not None and theta_half_width > 0:
            ilo = int(np.searchsorted(thetas, theta_center - theta_half_width, side='left'))
            ihi = int(np.searchsorted(thetas, theta_center + theta_half_width, side='right'))
            ilo = max(0, ilo)
            ihi = min(len(thetas), ihi)
            if ihi - ilo >= 2:
                thetas = thetas[ilo:ihi]
                sin_thetas = sin_thetas[ilo:ihi]
                cos_thetas = cos_thetas[ilo:ihi]

    # Extract edge points
    edge_points = np.argwhere(image.astype(np.int8))
    edge_points -= np.array([[img_height_half, img_width_half]])
    # # Extract edge points
    # edge_sparse = csr_array(image.astype(np.int8))
    # edge_points = np.vstack(edge_sparse.nonzero()).T
    # edge_points -= np.array([[img_height_half, img_width_half]])

    # Calculate matrix product
    rho_values = np.matmul(edge_points.astype(np.int16),
                           np.array([sin_thetas, cos_thetas],
                                    dtype='float32')).astype('float32')

    del image, cos_thetas, sin_thetas, edge_points

    # Get the hough space using fast histogram
    bins = [len(thetas), len(rhos)]
    vals = [np.tile(thetas, rho_values.shape[0]), rho_values.ravel()]
    ranges = [[np.min(thetas), np.max(thetas)], [np.min(rhos), np.max(rhos)]]

    accumulator = fhist.histogram2d(*vals, bins=bins, range=ranges)

    # Rhos in rows and thetas in columns (rho, theta)
    accumulator = np.transpose(accumulator)

    bc.clean_up(rho_values, ranges, bins)

    return accumulator, rhos, thetas, drho, dtheta


def quantize(df, colname='orientation', tolerance=0.005):
    """ Use agglomerative clustering to quantize and group values by their orientation.

    Parameters
    ----------
    df: pd.DataFrame
        A DataFrame containing the values to be clustered.
    colname: str
        The name of the column in df to be clustered.
    tolerance: float
        The maximum distance between clusters.

    Returns
    -------
        pd.DataFrame
            The input DataFrame with additional columns 'groups' and 'center'.
    """
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

    return df


def get_min_trail_length(config: dict):
    """

    Parameters
    ----------
    config

    Returns
    -------

    """
    sat_id = config['sat_id']
    pixscale = config['pixscale']

    if isinstance(pixscale, str):
        pixscale = float(pixscale)

    # Get nominal orbit altitude
    sat_h_orb_ref = bc.get_nominal_orbit_altitude(sat_id)

    H_sat = sat_h_orb_ref * 1000.
    R = const.R_earth.value + H_sat
    y = np.sqrt(const.G.value * const.M_earth.value / R ** 3)
    y *= 6.2831853
    return np.rad2deg(y) * 3600. / pixscale * 0.05  # 5% of the angular distance


def detect_sat_trails(image: np.ndarray,
                      config: dict,
                      image_orig: np.ndarray,
                      amount: float = 10.,
                      sigma_blurr: float = 4.,
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

    properties = ['label', 'centroid', 'area', 'orientation',
                  'axis_major_length',
                  'axis_minor_length',
                  'eccentricity',
                  # 'feret_diameter_max', 'coords'
                  ]

    trail_data = {'reg_info_lst': []}
    orig_data = {'image': image_orig}

    # Get minimum expected trail length for satellite
    # todo: improve this function and justify the 5% factor
    #  better to use the actual exposure time and calculate the expected length, see get_obs_range().
    min_trail_length = get_min_trail_length(config)

    # Mask saturated pixel
    saturation_limit = config['sat_lim']
    saturation_mask = (image >= saturation_limit)
    image[saturation_mask] = 0.

    # Combine masks
    if mask is not None:
        mask |= saturation_mask
    else:
        mask = saturation_mask
    log.debug("    Create segmentation map from image")

    if mask is not None:

        sigcl_mean, sigcl_median, sigcl_std = sigma_clipped_stats(image,
                                                                  sigma=3.0,
                                                                  mask=mask,
                                                                  maxiters=10)
        image = np.where(mask, sigcl_mean, image)

    # Apply un-sharpen mask
    blurred_f = nd.gaussian_filter(image, 3.)
    filter_blurred_f = nd.gaussian_filter(blurred_f, sigma_blurr)
    sharpened = blurred_f + amount * (blurred_f - filter_blurred_f)

    # Apply gaussian to get rid of unwanted edges
    sharpened = nd.gaussian_filter(sharpened, 3.)

    # Zero out the borders with width given by borderLen
    len_y, len_x = sharpened.shape
    borderWidth_x = int(np.ceil(len_x * 0.015625))
    borderWidth_y = int(np.ceil(len_y * 0.015625))
    sharpened[0:borderWidth_y, 0:len_x] = 0.
    sharpened[len_y - borderWidth_y:len_y, 0:len_x] = 0.
    sharpened[0:len_y, 0:borderWidth_x] = 0.
    sharpened[0:len_y, len_x - borderWidth_x:len_x] = 0.

    if mask is not None:
        sigcl_mean, sigcl_median, sigcl_std = sigma_clipped_stats(sharpened,
                                                                  sigma=3.0,
                                                                  mask=mask,
                                                                  maxiters=10)
        sharpened = np.where(mask, sigcl_mean, sharpened)

    del blurred_f, filter_blurred_f

    # Determine the detection threshold
    threshold = detect_threshold(sharpened, background=None, nsigma=1.5, mask=mask)

    # Get sources and create a segmentation map
    sources = detect_sources(sharpened, threshold=threshold,
                             npixels=9, connectivity=8, mask=mask)
    segm = SegmentationImage(sources.data)

    # Only for CTIO BW3 2022-11-10
    if config['use_sat_mask']:
        raper = RectangularAperture([[959.63, 1027.3], [903.93, 1073.75]],
                                    w=2755, h=10,
                                    theta=Angle(47.1, u.degree))
        m = raper.to_mask('center')
        m0 = m[0].to_image(segm.shape)
        m1 = m[1].to_image(segm.shape)
        m1[m1 > 0] = 2
        m0 += m1

        segm = SegmentationImage(m0.astype(int))

    # Make a copy
    segm_init = segm.copy()

    # First filter roundish objects
    cat = SourceCatalog(sharpened, segm)
    orig_data['segm'] = segm_init
    orig_data['cat'] = SourceCatalog(image, segm)

    # Compute image properties and return them as a pandas-compatible table.
    regs = regionprops_table(segm.data, properties=properties)
    df_orig = pd.DataFrame(regs)
    df_orig['orientation_deg'] = df_orig['orientation'] * 180. / np.pi
    orig_data['df'] = df_orig
    del regs, df_orig

    # Mask almost horizontal segments close to x-axis
    lim_ang = 2.
    border_mask_ang_x = np.abs(cat.orientation.data) > lim_ang
    border_mask_x1 = cat.centroid[:, 0] > sharpened.shape[1] * 0.05
    border_mask_x2 = cat.centroid[:, 0] < sharpened.shape[1] - sharpened.shape[1] * 0.05
    border_mask_x = border_mask_ang_x & border_mask_x1 & border_mask_x2

    # Mask almost vertical segments close to y-axis
    border_mask_ang_y = 90. - np.abs(cat.orientation.data) > lim_ang
    border_mask_y1 = cat.centroid[:, 1] > sharpened.shape[1] * 0.05
    border_mask_y2 = cat.centroid[:, 1] < sharpened.shape[0] - sharpened.shape[0] * 0.05
    border_mask_y = border_mask_ang_y & border_mask_y1 & border_mask_y2

    # Combine the border masks
    border_mask_combined = border_mask_x | border_mask_y

    # Eccentricity mask. Choose 99.5 percentile as limit
    e_lim = np.percentile(cat.eccentricity, 99.5)
    orig_data['e_lim'] = e_lim
    log.debug(f"    Initial filter. Keep segments with ecc >= {e_lim:.4f}")
    ecc_mask = cat.eccentricity.data >= e_lim

    # Combine eccentricity and border masks
    combined_mask = ecc_mask & border_mask_combined

    # Create labels
    labels = cat.labels[combined_mask]

    # Clean up
    bc.clean_up(cat, ecc_mask, combined_mask)

    # First check if there are any labels
    if labels.size == 0:
        if not silent:
            log.info(f"{' ':<4}No trail detected")
        bc.clean_up(sharpened, labels, segm, segm_init)
        return None, False

    # Keep only the good label, i.e. those segments that passed the filters, but do not relabel
    segm.keep_labels(labels=labels, relabel=False)

    # Create an initial regions dataframe
    regs = regionprops_table(segm.data, properties=properties)
    df_init = pd.DataFrame(regs)
    df_init['orientation_deg'] = df_init['orientation'] * 180. / np.pi  # Convert to degrees
    del regs

    # Now check how many segments are left
    # If only one segment is left, check its eccentricity.
    # 0.999 is equal to a semi-major to semi-minor axis ratio of 44.7, i.e. the object is 44.7 times longer than wide
    if df_init.shape[0] == 1:
        if df_init['eccentricity'].values[0] < 0.999:
            if not silent:
                log.info(f"{' ':<4}No trail detected")
            bc.clean_up(sharpened, segm, segm_init, labels, df_init)
            return None, False
        else:
            df_search = df_init.copy()
    else:

        # Group segments according to orientation +- tolerance
        df_quantize = quantize(df_init, colname='orientation_deg',
                               tolerance=config['MAX_DISTANCE'])

        # Filter groups based on eccentricity and total length
        grouped = df_quantize.groupby(by='groups')

        # Collect groups to drop, then drop all at once
        groups_to_drop = []
        for name, group in grouped:
            log.debug(f"Group {name}: labels={group['label'].tolist()}, " 
                      f"orientations={group['orientation_deg'].tolist()}")

            all_low_eccentricity = (group['eccentricity'] < 0.995).all()
            total_length_too_short = group['axis_major_length'].sum() < min_trail_length

            if all_low_eccentricity or total_length_too_short:
                groups_to_drop.append(name)

        df_quantize = df_quantize[~df_quantize['groups'].isin(groups_to_drop)]

        if df_quantize.empty:
            if not silent:
                log.info(f"{' ':<4}No trail detected")
            bc.clean_up(sharpened, segm, segm_init,
                                labels, df_init, df_quantize, grouped)
            return None, False
        else:
            # Sort groups according to the total length in the group
            s = df_quantize.groupby(by='groups')['axis_major_length'].sum().reset_index()  # sum of lengths
            res = s[s['axis_major_length'] == s.groupby(['groups'])['axis_major_length'].transform('max')]
            idx_sorted = res.sort_values(by='axis_major_length', ascending=False)['groups'].values

            # !!! We select only the longest for now !!!
            ind = idx_sorted[0]
            df_search = grouped.get_group(ind)

            bc.clean_up(df_quantize, s, res, ind, grouped)

    # Compute trail properties
    reg_dict, n_trail_total = get_trail_properties(segm, df_search, config, orig_data, silent=silent)

    if reg_dict is None:
        if not silent:
            log.info(f"{' ':<4}No trail detected")
        bc.clean_up(sharpened, segm, segm_init, labels, df_init, df_search, orig_data)
        return None, False

    # Check results of fit for non-finite results
    check_props = np.array([reg_dict['width'], reg_dict['e_width'],
                            reg_dict['height'], reg_dict['e_height']])

    if not np.isfinite(check_props).all():
        if not silent:
            log.warning(f"{' ':<4}Trail parameter fit {bc.fail_str}")
        has_trail = False
        n_trail_total -= 1
    else:
        log.debug(f"    Trail parameter fit {bc.pass_str}")
        has_trail = True

    if config['roi_offset'] is not None:
        coords = np.array(reg_dict['coords'])
        coords[0] = coords[0] + config['roi_offset'][0]
        coords[1] = coords[1] + config['roi_offset'][1]
        reg_dict['coords'] = tuple(coords)

    reg_dict['detection_imgs'].update({'img_sharp': sharpened,
                                       'segm_map': [segm_init.data,
                                                    segm_init.make_cmap(seed=12345)]})

    trail_data['reg_info_lst'].append(reg_dict)

    del df_search, reg_dict

    trail_data['N_trails'] = n_trail_total
    if not silent:
        log.info(f"{' ':<4}Trails detected: {n_trail_total}")

    # Clean up
    bc.clean_up(sharpened, segm, segm_init, df_init, labels)

    return trail_data, has_trail


def get_labels_along_trail(segm_data, angle, dist, img_shape):
    """
    Find all segment labels that lie along a given trail defined by angle and distance.

    Parameters
    ----------
    segm_data : np.ndarray
        Segmentation map data.
    angle : float
        Angle of the trail in degrees.
    dist : float
        Distance from the origin to the trail.
    img_shape : tuple
        Shape of the image (height, width).

    Returns
    -------
    label_list : np.ndarray
        Array of unique segment labels along the trail.
    """

    height, width = img_shape
    height_half, width_half = height // 2, width // 2

    # Calculate the image diagonal
    image_diagonal = np.sqrt(height ** 2 + width ** 2)

    a = np.cos(np.deg2rad(angle))
    b = np.sin(np.deg2rad(angle))
    x0 = (a * dist) + width_half
    y0 = (b * dist) + height_half

    x1 = int(x0 + image_diagonal * b)
    y1 = int(y0 - image_diagonal * a)
    x2 = int(x0 - image_diagonal * b)
    y2 = int(y0 + image_diagonal * a)

    rr, cc = line(y1, x1, y2, x2)

    # Clip the line to the image dimensions
    rr, cc = rr.clip(0, height - 1), cc.clip(0, width - 1)

    # Find the unique labels that make up the particular trail
    label_list = np.unique(segm_data[rr, cc])
    label_list = np.setdiff1d(label_list, np.array([0]), assume_unique=True)

    del segm_data, rr, cc

    return label_list

def get_trail_properties(segm_map, df, config, orig_data,
                         silent: bool = False):
    """
    Perform Hough transform and determine trail parameter, i.e. length, width, and
    orientation

    Parameters
    ----------
    segm_map : SegmentationImage
        Segmentation map containing the segments to analyze.
    df : pd.DataFrame
        A DataFrame containing the properties of the segments.
    config : dict
        Configuration parameters for the analysis.
    orig_data : dict
        Original data for reference.
    silent : bool, optional
        If True, suppresses logging output.

    Returns
    -------
    tuple
        A dictionary with trail properties and the number of trails detected.
    """
    theta_bin_size = config['THETA_BIN_SIZE']
    rho_bin_size = config['RHO_BIN_SIZE']

    # Estimate the trail orientation from the (already orientation-grouped) segments to
    # restrict the Hough theta search to a window around it. Verified mapping:
    # Hough peak theta ~ -(regionprops orientation_deg). The window is opt-in via the
    # config key THETA_WINDOW_HALF_DEG (default 5 deg); a value <= 0 disables it and
    # reproduces the original full-range behaviour exactly.
    theta_half_width = float(config.get('THETA_WINDOW_HALF_DEG', 5.0)) \
        if hasattr(config, 'get') else 5.0
    theta_center = None
    if theta_half_width and theta_half_width > 0 and 'orientation_deg' in df:
        weights = df['axis_major_length'].to_numpy(dtype=float)
        angles = df['orientation_deg'].to_numpy(dtype=float)
        if np.isfinite(weights).any() and np.nansum(weights) > 0:
            theta_center = -float(np.average(angles, weights=weights))
        else:
            theta_center = -float(np.mean(angles))
    # Interior guard: the peak must lie at least (half the theta fit sub-window +
    # gaussian smoothing radius + a small margin) cells from the window edges, otherwise
    # the windowed Hough falls back to the exact full-range computation.
    guard_cells = math.ceil(config['THETA_SUB_WIN_SIZE'] / theta_bin_size) // 2 + 12

    # Create a binary image
    binary_img = np.zeros(segm_map.shape)
    for row in df.itertuples(name='label'):
        binary_img[segm_map.data == row.label] = 1

    # Apply dilation
    struct1 = nd.generate_binary_structure(2, 1)
    dilated = nd.binary_dilation(binary_img, structure=struct1, iterations=1)

    del binary_img

    log.debug(f"    Create Hough space accumulator array (This may take a second!)")
    # Create hough space and get peaks
    hspace_smooth, peaks, dist, theta = get_hough_transform(dilated,
                                                            theta_bin_size=theta_bin_size,
                                                            rho_bin_size=rho_bin_size,
                                                            theta_center=theta_center,
                                                            theta_half_width=theta_half_width,
                                                            guard_cells=guard_cells)

    n_trail = 0
    if len(peaks) > 1:

        # Delete old result
        del hspace_smooth

        # Make a copy of the original
        df_tmp = df.copy()

        sel_dict = {}
        for i in range(len(peaks)):
            angle = peaks[i][1]
            dist = peaks[i][2]

            # Find segments along the trail
            label_list = get_labels_along_trail(segm_map.data, angle, dist, dilated.shape[:2])

            index_list = []
            if not df_tmp.empty:
                # Select only the segment labels that belong to this peak
                [index_list.append(row.Index) for row in df_tmp.itertuples(name='label') if row.label in label_list]
                df_tmp.drop(index_list, inplace=True)
            else:
                break
            sel_dict[i] = index_list
            n_trail += 1

        # Select only the label that belong to the first peak
        # this can be extended later to let the user choose which to pick
        # Set 0 for first entry
        sel_idx = config['PARALLEL_TRAIL_SELECT']
        _df = df.loc[sel_dict[sel_idx]]
        binary_img = np.zeros(segm_map.shape)
        for row in _df.itertuples(name='label'):
            binary_img[segm_map.data == row.label] = 1
        del _df

        # Apply dilation
        struct1 = nd.generate_binary_structure(2, 1)
        dilated = nd.binary_dilation(binary_img, structure=struct1, iterations=1)

        # Create hough space and get peaks
        hspace_smooth, peaks, dist, theta = get_hough_transform(dilated,
                                                                theta_bin_size=theta_bin_size,
                                                                rho_bin_size=rho_bin_size,
                                                                theta_center=theta_center,
                                                                theta_half_width=theta_half_width,
                                                                guard_cells=guard_cells)
        del binary_img
    else:
        if check_for_bad_column(peaks[0], orig_data, config):
            bc.clean_up(segm_map, dilated, df, hspace_smooth,
                        dist, theta)
            return None, 0
        n_trail = 1

    indices = np.where(hspace_smooth == peaks[0][0])
    row_ind, col_ind = indices[0][0], indices[1][0]
    center_idx = (row_ind, col_ind)

    # Set sub-window size
    theta_sub_win_size = math.ceil(config['THETA_SUB_WIN_SIZE'] / theta_bin_size)
    rho_sub_win_size = math.ceil(4. * np.sqrt(2.)
                                 * config['RHO_SUB_WIN_RES_FWHM']
                                 * float(config['fwhm']) / rho_bin_size)
    sub_win_size = (rho_sub_win_size, theta_sub_win_size)

    theta_init = np.deg2rad(theta[col_ind])
    log.debug("    Determine trail parameter: length, width, and orientation")
    fit_res = fit_trail_params(hspace_smooth, dist, theta, theta_init,
                               dilated.shape, center_idx, sub_win_size,
                               config['TRAIL_PARAMS_FITTING_METHOD'])
    # Relabel segmentation map
    segm_map.relabel_consecutive(start_label=1)
    reg_dict = {'coords': fit_res[0],  # 'coords': reg.centroid[::-1],
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
                    'img_labeled': [segm_map.data,
                                    segm_map.make_cmap(seed=12345)],
                    'trail_mask': dilated
                }}
    # Clean up
    bc.clean_up(segm_map, fit_res, dilated, df, hspace_smooth,
                        indices, row_ind, col_ind, dist, theta)
    return reg_dict, n_trail

def check_for_bad_column(peak, data, config):
    """ Check if the detected trail is due to a saturated column.

    Parameters
    ----------
    peak : tuple
        A tuple containing the peak information (value, angle, distance).
    data : dict
        A dictionary containing the original data including segmentation map and dataframe.
    config : dict
        Configuration parameters for the analysis.

    Returns
    -------
    is_bad_column : bool
        True if the trail is due to a saturated column, False otherwise.
    """
    is_bad_column = False

    # Check if the trail is the result of a saturated star
    angle = peak[1]
    dist = peak[2]

    angel_lim = 0.5
    angle_test = abs(angle) < angel_lim

    # Get segment labels along the trail from the original data
    label_list = get_labels_along_trail(data['segm'].data, angle, dist, data['segm'].shape[:2])

    # For each entry in the label_list get the corresponding row in the original data and check for saturation,
    # orientation, and eccentricity
    for label in label_list:
        label_idx = data['segm'].get_index(label)
        segm_df = data['df'][data['df']['label'] == label]
        segment_cutout = data['segm'].segments[label_idx].make_cutout(data['image'])

        saturation_limit = config['sat_lim']
        saturation_mask = (segment_cutout >= saturation_limit)

        saturation_test = saturation_mask.sum() > 3
        ecc_test = segm_df['eccentricity'].values[0] < data['e_lim']

        if angle_test and saturation_test and ecc_test:
            is_bad_column = True
            break

    return is_bad_column

def get_hough_transform(image: np.ndarray, sigma: float = 2.,
                        theta_bin_size=0.02,
                        rho_bin_size=1,
                        theta_center=None,
                        theta_half_width=None,
                        guard_cells: int = 0):
    """ Perform a Hough transform on the selected region and fit parameter.

    Parameters
    ----------
    image : np.ndarray
        Input binary image.
    sigma : float
        Standard deviation for Gaussian kernel used to smooth Hough space.
    theta_bin_size : float
        Bin size for theta in degrees.
    rho_bin_size : float
        Bin size for rho.
    theta_center : float, optional
        Centre (deg) of a theta window to restrict the search to, i.e. the expected
        Hough angle of the trail (~ -orientation_deg). If None (default), the full
        ``[-90, 90]`` range is used, i.e. the original behaviour.
    theta_half_width : float, optional
        Half-width (deg) of the theta window. If None or <= 0, no windowing is applied.
    guard_cells : int
        Minimum number of theta cells the detected peak must sit inside the window
        edges for the windowed result to be accepted. If the peak is closer to an edge
        (or no peak is found in the window), the function falls back to the full-range
        Hough, which reproduces the original result exactly.

    Returns
    -------
    tuple
        Hough space, peaks, distances, and angles.
    """

    def _hough(theta_window):
        # Generate Hough space H(theta, rho)
        res = create_hough_space_vectorized(image=image,
                                            dtheta=theta_bin_size,
                                            drho=rho_bin_size,
                                            theta_window=theta_window)
        hspace_, dist_, theta_, _, _ = res
        # Apply gaussian filter to smooth the accumulator (float32; see Tier 2)
        hspace_smooth_ = nd.gaussian_filter(hspace_.astype(np.float32), sigma)
        del res, hspace_, _
        # Find peaks in H-space
        peaks_ = list(zip(*hough_line_peaks(hspace_smooth_, theta_, dist_)))
        return hspace_smooth_, peaks_, dist_, theta_

    # Windowed Hough around the known orientation, with an interior-peak guard.
    # When the peak is comfortably interior, the smoothed accumulator around it - and
    # hence the downstream centroid/fit - is byte-identical to the full-range result.
    if theta_center is not None and theta_half_width is not None and theta_half_width > 0:
        hspace_smooth, peaks, dist, theta = _hough((theta_center, theta_half_width))
        if peaks:
            # Locate the top peak's column by angle (robust to float value collisions)
            top_col = int(np.argmin(np.abs(theta - peaks[0][1])))
            ncols = hspace_smooth.shape[1]
            if guard_cells <= top_col <= ncols - 1 - guard_cells:
                del image
                return hspace_smooth, peaks, dist, theta
        # Guard failed (estimate too far off, or no peak in window) -> exact fallback.
        del hspace_smooth, peaks, dist, theta

    # Full-range Hough: the default behaviour and the fallback path.
    hspace_smooth, peaks, dist, theta = _hough(None)
    del image
    return hspace_smooth, peaks, dist, theta


def perpendicular_line_profile(center, L, theta, perp_length, num_samples=10, use_pixel_sampling=False):
    """
    Calculate the start and end points of line profiles perpendicular to a main line,
    centered at the main line's midpoint, evenly spaced along it.

    Parameters
    ----------
    center : tuple
        (x, y) coordinates of the center point of the main line.
    L : float
        Length of the main line.
    theta : float
        Angle of the main line in degrees.
    perp_length : float
        Length of the perpendicular lines.
    num_samples : int
        Number of perpendicular lines to sample along the main line.
    use_pixel_sampling : bool
        If True, the number of samples is determined by the pixel distance along the main line.

    Returns
    -------
    list of tuples
        Each tuple contains two points (start, end) defining a perpendicular line profile.

    Notes
    -----
    The main line is defined by its center point, length, and angle.
    If the use_pixel_sampling is True, the function calculates the number of samples
    based on the pixel distance between the start and end points of the main line.
    This ensures that a sample is taken for every pixel along the main line.

    The perpendicular lines are calculated to be symmetrically spaced about the center
    point of the main line, with one line passing through the center if the number
    of samples is odd, or evenly spaced on either side if the number is even.
    """
    # Convert theta to radians
    theta_rad = np.radians(theta)

    # Calculate start, end, and center points of the main line
    half_length = L / 2
    start_point = [center[0] - half_length * np.cos(theta_rad), center[1] - half_length * np.sin(theta_rad)]
    end_point = [center[0] + half_length * np.cos(theta_rad), center[1] + half_length * np.sin(theta_rad)]
    center_point = [center[0], center[1]]

    # Calculate the number of samples if using pixel sampling
    if use_pixel_sampling:
        pixel_distance = int(np.hypot(end_point[0] - start_point[0], end_point[1] - start_point[1]))
        num_samples = max(pixel_distance, 1)  # Ensure at least one sample

    # Slope of the main line
    m = np.tan(theta_rad)

    # Slope of the perpendicular line
    perp_slope = -1 / m if m != 0 else np.inf

    # Calculate the spacing and positions of the perpendicular lines
    if num_samples > 1:
        spacing = L / (num_samples - 1)
        line_points = [start_point]
        for i in range(1, num_samples - 1):
            offset = i * spacing - half_length
            point = [center[0] + offset * np.cos(theta_rad), center[1] + offset * np.sin(theta_rad)]
            line_points.append(point)
        line_points.append(end_point)
    else:
        line_points = [center_point]

    # Calculate points on the perpendicular lines
    perp_profiles = []
    half_perp_length = perp_length / 2
    for point in line_points:
        if perp_slope != np.inf:  # Non-vertical lines
            dx = half_perp_length / np.sqrt(1 + perp_slope ** 2)
            dy = perp_slope * dx
        else:  # Vertical lines
            dx = 0
            dy = half_perp_length

        perp_start = [point[0] - dx, point[1] - dy]
        perp_end = [point[0] + dx, point[1] + dy]

        perp_profiles.append((perp_start, perp_end))

    return perp_profiles


def get_pixel_values(img_array, line_profiles, length):
    """
    Extract pixel values from an image array along specified line profiles.

    This function iterates over line profiles, each defined by a start and end point,
    extracts the pixel values along these lines from the image array, and pads or trims
    the results to ensure a consistent profile length.

    Parameters
    ----------
    img_array : np.ndarray
        2D array representing the image from which to extract pixel values.
    line_profiles : list of tuples
        Each tuple contains two points (start, end) defining a line profile.
    length : int
        The desired length of each extracted line profile.

    Returns
    -------
    np.ndarray
        A 2D array where each column corresponds to the pixel values along a line profile,
        padded with NaNs to ensure consistent length.
    """
    profiles_pixel_values = []
    for start, end in line_profiles:
        # Calculate the coordinates of the line using skimage's line function
        rr, cc = line(int(round(start[1])), int(round(start[0])),
                      int(round(end[1])), int(round(end[0])))

        # Create an array of NaN values
        line_values = np.full(length, np.nan)

        # Replace values within the image bounds
        valid_indices = (rr >= 0) & (rr < img_array.shape[0]) & \
                        (cc >= 0) & (cc < img_array.shape[1])
        valid_rr, valid_cc = rr[valid_indices], cc[valid_indices]
        line_values[:len(valid_rr)] = img_array[valid_rr, valid_cc]

        profiles_pixel_values.append(line_values)
        del rr, cc

    profiles_pixel_values = np.array(profiles_pixel_values).T
    del img_array,
    return profiles_pixel_values


def fit_single_model(x, y, fwhm, yerr=None, model_type='gaussian'):
    """
    Fits a line profile using a single specified model type.

    Parameters
    ----------
    x : array-like
        The x data.
    y : array-like
        The y data.
    fwhm : float
        Full width at half-maximum.
    yerr : array-like, optional
        The y error data.
    model_type : str
        Type of model to use for fitting. Options are 'gaussian', 'lognormal', or 'expgaussian'.

    Returns
    -------
    result : lmfit.model.ModelResult
        The result of the fit.
    """
    sky_mod = ConstantModel(prefix='const_', independent_vars=['x'], nan_policy='omit')
    sky_mod.set_param_hint(name='c', value=0.)
    pars = sky_mod.make_params()

    center_init = x[np.nanargmax(y)]
    sigma_init = fwhm * gaussian_fwhm_to_sigma
    amp = np.nanmax(y)
    xmin = 0
    xmax = len(x)

    if model_type == 'gaussian':
        model = GaussianModel(prefix='m_', nan_policy='omit')
        model_params = dict(center=dict(value=center_init,
                                        min=xmin, max=xmax
                                        ),
                            sigma=dict(value=sigma_init, min=0.),
                            amplitude=dict(value=amp, min=0.))
    elif model_type == 'lognormal':
        model = LognormalModel(prefix='m_', nan_policy='omit')
        model_params = dict(center=dict(value=center_init,
                                        min=xmin, max=xmax
                                        ),
                            sigma=dict(value=sigma_init, min=0.),
                            amplitude=dict(value=amp, min=0.))
    elif model_type == 'expgaussian':
        model = ExponentialGaussianModel(prefix='m_', nan_policy='omit')
        model_params = dict(center=dict(value=center_init,
                                        min=xmin, max=xmax
                                        ),
                            sigma=dict(value=sigma_init, min=0.),
                            amplitude=dict(value=amp),
                            gamma=dict(value=1, min=0.))
    else:
        raise ValueError("Invalid model type. "
                         "Choose 'gaussian', 'lognormal', or 'expgaussian'.")

    pars.update(model.make_params(**model_params))

    mod = model + sky_mod

    weights = 1. / yerr if yerr is not None else None

    fitting_method = 'nelder'

    result = mod.fit(data=y, x=x,
                     params=pars,
                     method=fitting_method,
                     weights=weights,
                     calc_covar=True,
                     scale_covar=True,
                     nan_policy='omit')

    return result


def adjust_center_init(x, center_init, shift_percentage=10.):
    """ Adjust the initial center position by a random shift within a specified percentage of the data range.

    Parameters
    ----------
    x : array-like
        The x data.
    center_init : float
        The initial center position.
    shift_percentage : float
        Percentage of the data range to use for shifting.

    Returns
    -------
    new_center : float
        The adjusted center position.
    """
    x_min, x_max = np.min(x), np.max(x)
    x_range = x_max - x_min
    max_shift = x_range * shift_percentage
    shift_direction = np.random.choice([-1, -1])
    shift_amount = np.random.uniform(0, max_shift)
    new_center = center_init + shift_direction * shift_amount
    new_center = np.clip(new_center, x_min, x_max)
    return new_center


def fit_line_profile_two_comps(x, y, fwhm, model_type='gaussian',
                               yerr=None,
                               shift_percentage=10.):
    """ Fits a line profile using a two-component specified model type.

    Parameters
    ----------
    x : array-like
        The x data.
    y : array-like
        The y data.
    fwhm : float
        Full width at half-maximum.
    model_type : str
        Type of model to use for fitting. Options are 'gaussian', 'lognormal', or 'expgaussian'.
    yerr : array-like, optional
        The y error data.
    shift_percentage : float
        Percentage to shift the center of the second component.

    Returns
    -------
    result : lmfit.model.ModelResult
        The result of the fit.
    """
    sky_mod = ConstantModel(prefix='const_', independent_vars=['x'], nan_policy='omit')
    sky_mod.set_param_hint(name='c', value=0.)
    pars = sky_mod.make_params()

    center_init = x[np.nanargmax(y)]
    sigma_init = fwhm * gaussian_fwhm_to_sigma
    amp = np.nanmax(y)
    xmin = 0.
    xmax = len(x)

    if model_type == 'gaussian':
        mod1 = GaussianModel(prefix='m1_', nan_policy='omit')
        mod2 = GaussianModel(prefix='m2_', nan_policy='omit')
    elif model_type == 'lognormal':
        mod1 = LognormalModel(prefix='m1_', nan_policy='omit')
        mod2 = LognormalModel(prefix='m2_', nan_policy='omit')
    elif model_type == 'expgaussian':
        mod1 = ExponentialGaussianModel(prefix='m1_', nan_policy='omit')
        mod2 = ExponentialGaussianModel(prefix='m2_', nan_policy='omit')
    else:
        raise ValueError("Invalid model type. "
                         "Choose 'gaussian', 'lognormal', or 'expgaussian'.")

    # Parameters for the first model component
    common_params_1 = dict(center=dict(value=center_init,
                                       min=xmin, max=xmax
                                       ),
                           sigma=dict(value=sigma_init, min=1e-5),
                           amplitude=dict(value=amp, min=0.))
    if model_type == 'expgaussian':
        common_params_1['gamma'] = dict(value=1., min=1e-15)

    pars.update(mod1.make_params(**common_params_1))

    # Adjusted center for the second model component
    adjusted_center = adjust_center_init(x, center_init, shift_percentage)
    common_params_2 = dict(center=dict(value=adjusted_center,
                                       min=xmin, max=xmax
                                       ),
                           sigma=dict(value=1.5 * sigma_init, min=1e-5),
                           amplitude=dict(value=amp, min=0.))
    if model_type == 'expgaussian':
        common_params_2['gamma'] = dict(value=1., min=1e-15)

    pars.update(mod2.make_params(**common_params_2))

    mod = mod1 + mod2 + sky_mod

    weights = 1. / yerr if yerr is not None else None

    fitting_method = 'nelder'

    result = mod.fit(data=y, x=x,
                     params=pars,
                     method=fitting_method,
                     weights=weights,
                     calc_covar=True,
                     scale_covar=True,
                     nan_policy='omit')

    return result


def composite_gaussian(x, params):
    """ Compute the sum of multiple Gaussian functions.

    Parameters
    ----------
    x : float or np.ndarray
        The input value(s) where the composite Gaussian is evaluated.
    params : dict
        Dictionary containing parameters of the Gaussian components. Each entry should have keys 'A', 'mu', and 'sigma'.

    Returns
    -------
    float or np.ndarray
        The value(s) of the composite Gaussian at the input x.
    """
    return sum(p['A'] * np.exp(-0.5 * ((x - p['mu']) / p['sigma'])**2) for p in params.values())


def find_gaussian_bounds(params, sigma, xtol=1e-5, rtol=1e-5, maxiter=500):
    """ Find bounds that contain a specified percentage of the area under a composite Gaussian curve.

    Parameters
    ----------
    params : dict
        Dictionary containing parameters of the Gaussian components. Each entry should have keys 'A', 'mu', and 'sigma'.
    sigma : float
        The number of standard deviations to define the bounds.
    xtol : float
        Absolute tolerance for root finding.
    rtol : float
        Relative tolerance for root finding.
    maxiter : int
        Maximum number of iterations for root finding.

    Returns
    -------
    tuple
        Lower and upper bounds that contain the specified percentage of the area.
    """
    total_area = quad(composite_gaussian, -np.inf, np.inf, args=(params,))[0]

    lower_percentile = norm.cdf(-sigma)
    upper_percentile = norm.cdf(sigma)

    def find_bound(target_area):
        def integral(x):
            return quad(composite_gaussian, -np.inf, x, args=(params,))[0] - target_area
        return integral

    # Adjusting the bounds based on the parameters
    mu_values = [p['mu'] for p in params.values()]
    sigma_values = [p['sigma'] for p in params.values()]
    min_bound = min(mu_values) - 5 * max(sigma_values)
    max_bound = max(mu_values) + 5 * max(sigma_values)

    x_low = brentq(find_bound(lower_percentile * total_area), min_bound, max_bound,
                   xtol=xtol, rtol=rtol, maxiter=maxiter)
    x_up = brentq(find_bound(upper_percentile * total_area), min_bound, max_bound,
                  xtol=xtol, rtol=rtol, maxiter=maxiter)

    return x_low, x_up
