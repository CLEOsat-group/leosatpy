#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Name:         sources.py
# Purpose:      Utilities to support creation of an accurate source catalog
#               using object detection with photutils
#
#               from progress.spinner import Spinner
#               spinner = Spinner('Loading ')
#
#
# Author:       p4adch (cadam)
#
# Created:      04/27/2022
# Copyright:    (c) p4adch 2010-
#
# History:
#
# 27.04.2022
# - file created and basic methods
#
# -----------------------------------------------------------------------------

""" Modules """
from __future__ import annotations

import gc
import os
import sys
import re
from typing import Optional

import numpy as np
import pandas as pd

import inspect
import logging
import requests

import astropy.wcs
from astropy import units as u
from astropy.table import (
    Table, Column)
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.time import Time
from astropy.io import (
    ascii, fits)
from astropy.stats import (sigma_clipped_stats,
                           mad_std, sigma_clip, SigmaClip)

from astropy.convolution import (
    convolve, Tophat2DKernel)
from astropy.nddata import NDData
from astropy.nddata import Cutout2D

from astroquery.gaia import Gaia

from lmfit import Model

from photutils.aperture import CircularAperture
from photutils.segmentation import (
    detect_sources, detect_threshold, deblend_sources, SourceCatalog)
from photutils.detection import (
    DAOStarFinder, IRAFStarFinder, find_peaks)
from photutils.psf import extract_stars
from photutils.background import (
    Background2D,  # For estimating the background
    SExtractorBackground, StdBackgroundRMS, MMMBackground, MADStdBackgroundRMS,
    BkgZoomInterpolator)
from photutils.centroids import centroid_2dg

import scipy.spatial as spsp
from scipy.spatial import KDTree
from scipy.stats import chi2
from sklearn.metrics.pairwise import euclidean_distances

try:
    import matplotlib

except ImportError:
    plt = None
else:
    import matplotlib
    import matplotlib as mpl
    import matplotlib.lines as mlines
    import matplotlib.gridspec as gridspec  # GRIDSPEC !
    from matplotlib import pyplot as plt
    from matplotlib.ticker import (
        AutoMinorLocator, LogLocator)
    from astropy.visualization import (
        LinearStretch, LogStretch, SqrtStretch)
    from astropy.visualization.mpl_normalize import ImageNormalize
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # matplotlib parameter
    matplotlib.use('Qt5Agg')
    matplotlib.rc("lines", linewidth=1.2)
    matplotlib.rc('figure', dpi=150, facecolor='w', edgecolor='k')
    matplotlib.rc('text.latex', preamble=r'\usepackage{sfmath}')
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    # mpl.rcParams['font.family'] = 'Arial'

# pipeline-specific modules
from . import photometry as phot
from . import base_conf as bc
from . import transformations as imtrans

# -----------------------------------------------------------------------------

""" Meta-info """
__author__ = "Christian Adam"
__copyright__ = 'Copyright 2021-2023, CLEOSat group'
__credits__ = ["Eduardo Unda-Sanzana, Jeremy Tregloan-Reed, Christian Adam"]
__license__ = "GPL-3.0 license"
__maintainer__ = "Christian Adam"
__email__ = "christian.adam84@gmail.com"
__status__ = "Production"

__taskname__ = 'sources'

# -----------------------------------------------------------------------------

""" Parameter used in the script """
log = logging.getLogger(__name__)

MODULE_PATH = os.path.dirname(inspect.getfile(inspect.currentframe()))


# -----------------------------------------------------------------------------


def gauss_sigma2fwhm(sigma):
    """ Convert sigma value to full width at half maximum (FWHM) for gaussian function.

    :param sigma: Sigma value (standard deviation) of the gaussian profile
    :type sigma: float
    :return: Full width at half maximum value
    :rtype: float

    """

    fwhm = 2 * np.sqrt(2 * np.log(2)) * sigma

    return fwhm


def gauss_fwhm2sigma(fwhm):
    """ Convert full width at half maximum (FWHM) for gaussian
    function to sigma value.

    :param fwhm: Full width at half maximum of the gaussian profile
    :type fwhm: float
    :return: Sigma value (standard deviation) of the gaussian profile
    :rtype: float

    """

    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))

    return sigma


def gauss2d(x, y, xc, yc, amp, sigma, sky):
    """2D Gaussian function"""

    G = amp * np.exp(-1. * ((x - xc) ** 2 + (y - yc) ** 2) / (2. * sigma ** 2)) + sky

    return G


def moffat_fwhm(alpha, beta):
    """ Calculate FWHM from Moffat function.

    :param alpha: Alpha corresponds to the fitted width of the moffat function
    :type alpha: float
    :param beta: Beta value describes the wings of the moffat function
    :type beta: float
    :return: Full width at half maximum of moffat function
    :rtype: float

    """

    fwhm = 2. * alpha * np.sqrt((2. ** (1. / beta)) - 1.)

    return fwhm


def moffat2d(x, y, xc, yc, amp, alpha, beta, sky):
    """2D Moffat function"""

    M = amp * (1. + ((x - xc) ** 2 + (y - yc) ** 2) / alpha ** 2) ** -beta + sky

    return M


def resid_func(x, y, xc, yc, amp, alpha, beta, sky):
    """ Residual function for Moffat"""

    M = moffat2d(x, y, xc, yc, amp, alpha, beta, sky)

    return M


def remove_close_elements(df, distance_threshold, flux_col=5):
    """Remove elements within a given radius around each point, keeping the brightest."""

    # Sort the dataframe so that the brightest objects come first

    # Retrieve the column name using the index
    if isinstance(flux_col, int):
        column_name = df.columns[flux_col]
        ascending = True
    else:
        # Use the column name directly
        column_name = flux_col
        ascending = False

    df_sorted = df.sort_values(by=column_name, ascending=ascending)

    # Convert the x and y centroid columns to a numpy array for KDTree
    points = df_sorted[['xcentroid', 'ycentroid']].to_numpy()

    # Create a KDTree for efficient neighborhood queries
    tree = KDTree(points)

    # Initialize a mask to keep track of which points to retain
    mask = np.ones(len(df_sorted), dtype=bool)

    # Iterate through each point from brightest to faintest
    for i in range(len(points)):
        # If the current point is already marked as False, skip it
        if not mask[i]:
            continue

        # Find all points within the distance_threshold of the current point
        close_points = tree.query_ball_point(points[i], distance_threshold)

        # For all close points (excluding the current one), mark them as False
        for idx in close_points:
            if idx != i:
                mask[idx] = False

    # Apply the mask to the sorted dataframe to get the filtered result
    df_masked = df_sorted[mask]

    # Return the result sorted in ascending order of flux (optional, depending on use case)
    # df_masked = df_masked.sort_values(by=flux_col, ascending=True)

    return df_masked, np.sum(~mask)


def build_source_catalog(data, mask, input_catalog=None, known_fwhm=None,
                         **config):
    """
    Detect sources in the image and determine the Full Width Half Maximum (FWHM) of the image.

    Returns
    -------

    """

    # Initialize logging for this user-callable function
    log.setLevel(logging.getLevelName(log.getEffectiveLevel()))

    default_box_size = config['SOURCE_BOX_SIZE']
    init_fwhm = config['FWHM_INIT_GUESS'] if known_fwhm is None else known_fwhm
    nsigma = config['THRESHOLD_VALUE']
    fwhm_multiplier = config['ISOLATE_SOURCES_FWHM_SEP']

    box_size = int(np.ceil(fwhm_multiplier * init_fwhm)) + 0.5
    # print(nsigma, init_fwhm, default_box_size, box_size, box_size * 2)
    box_size = default_box_size if known_fwhm is None else 2. * box_size

    log.info(f'{" ":<4}> Find sources and estimate Full Width Half Maximum (FWHM)')
    if mask is None:
        mask = np.zeros(data.shape, dtype=bool)

    mask = make_border_mask(mask, borderLen=box_size // 2)

    # Create threshold map
    threshold_value = detect_threshold(data, nsigma=nsigma, mask=mask)

    npixels = 9
    connectivity = 8
    if config['bin_str'] == '4x4':
        npixels = 3
        connectivity = 8
    try:
        # Detect and deblend sources
        segm = detect_sources(data, threshold_value, npixels=npixels, connectivity=connectivity, mask=mask)
        segm_deblend = deblend_sources(data, segm,
                                       npixels=npixels, nlevels=32, contrast=0.001, connectivity=connectivity,
                                       progress_bar=False)
    except ValueError:
        log.critical("{" ":<4}>> Source detection failed")
        return None, None, False, ['', '', '']

    # Create a catalog of sources
    catalog = SourceCatalog(data, segm_deblend, mask=mask)

    # Select columns
    sources = catalog.to_table(
        columns=['label', 'xcentroid', 'ycentroid',
                 'max_value', 'segment_flux',
                 'eccentricity', 'fwhm', 'gini']).to_pandas()

    # Rename the flux column
    sources.rename(columns={"segment_flux": "flux"}, inplace=True)
    sources['mag'] = -2.5 * np.log10(sources['flux'].values)

    # Sort by maximum value
    sources.sort_values(by='flux', ascending=False, inplace=True)

    log.info(f'{" ":<6}Initial number of sources: {len(sources)}')

    log.info(f'{" ":<6}Run profile fit (This may take a second.)')
    sources = fit_source_cat(data, sources, box_size, init_fwhm,
                             update_position=True, **config)
    df = sources.copy()

    log.info(f"{' ':<6}>> Sources found: {np.sum(sources['fit_mask'])}")
    log.info(f'{" ":<4}> Filter outlier')

    # List of columns to apply sigma clipping on, in sequence
    columns_to_clip = ['eccentricity', 'fwhm']
    if np.sum(sources['fit_mask']) > 11:
        columns_to_clip = ['eccentricity', 'fwhm']

    # Create the initial mask
    cumulative_mask = df['fit_mask'].values

    # Sequentially apply sigma clipping on each column
    for column in columns_to_clip:
        # Apply sigma clipping only on rows that have not been masked previously
        masked_data = sigma_clip(df[column][cumulative_mask], masked=True,
                                 sigma=5,
                                 cenfunc=np.nanmedian,
                                 stdfunc=mad_std)
        current_mask = ~masked_data.mask  # Get a boolean mask of valid (unmasked) values
        log.info(f'{" ":<6}{f"Masked {column} outlier":<28}: {np.sum(~current_mask)}')

        # Update the cumulative mask based on the current mask
        temp_mask = cumulative_mask.copy()
        temp_mask[cumulative_mask] = current_mask  # Update only where data was previously unmasked
        cumulative_mask = temp_mask

    # Apply the cumulative mask to the final column
    df['masked_fwhm'] = df['fwhm'].where(cumulative_mask)

    df['include_fwhm'] = cumulative_mask

    sources = df.copy()

    sources['id'] = np.arange(1, len(sources) + 1)

    image_fwhm = np.nanmedian(sources['masked_fwhm'].values)
    image_fwhm_err = np.nanstd(sources['masked_fwhm'].values)

    if input_catalog is None:
        log.info(f'{" ":<2}{f"==> Detected sources":<20}: {len(sources[cumulative_mask])}')
        log.info(f'{" ":<6}{f"FWHM":<16}: {image_fwhm:.3f} +/- {image_fwhm_err:.3f} px')
        return (image_fwhm, image_fwhm_err), sources, True, None
    else:
        obs_df = sources[cumulative_mask]
        log.info(f'{" ":<4}>> Detected sources: {len(obs_df)}')

        ref_df = input_catalog.copy()
        ref_df.reset_index(drop=True, inplace=True)
        log.info(f'{" ":<4}> X-match with reference catalog')
        log.info(f'{" ":<6}Initial number of reference sources: {len(ref_df)}')
        hsize = (box_size - 1) // 2
        boundary_mask = ((ref_df['xcentroid'] > hsize)
                         & (ref_df['xcentroid'] < (data.shape[1] - 1 - hsize))
                         & (ref_df['ycentroid'] > hsize)
                         & (ref_df['ycentroid'] < (data.shape[0] - 1 - hsize)))

        ref_df = ref_df[boundary_mask]

        log.info(f'{" ":<6}Sources near boundary removed: {np.sum(~boundary_mask):d}')
        ref_df, n_crowded = remove_close_elements(df=ref_df,
                                                  distance_threshold=hsize)

        log.info(f'{" ":<6}Crowded sources removed: {n_crowded:d}')

        ref_df.reset_index(drop=True, inplace=True)

        matches = imtrans.find_matches(obs_df, ref_df, threshold=image_fwhm, sort=False)
        sources = matches[1][matches[1]['include_fwhm']].copy()

        sources['id'] = np.arange(1, len(sources) + 1)

        image_fwhm = np.nanmedian(sources['fwhm'].values)
        image_fwhm_err = np.nanstd(sources['fwhm'].values)
        log.info(f'{" ":<2}==> Usable sources: {len(sources)}, '
                 f'FWHM: {image_fwhm:.3f} +/- {image_fwhm_err:.3f} [ pixels ]')

        return (image_fwhm, image_fwhm_err), sources, True, None


def fit_source_cat(data, df, cutout_size, known_fwhm=None, update_position=False, **config):
    """
    Fit multiple sources in the input data using either a Gaussian or Moffat model.

    This function takes a dataframe of source positions and iteratively fits each source within the provided
    data using either a Gaussian or Moffat model. The fitted parameters (such as FWHM and SNR) are appended
    to the input dataframe for each source.

    Parameters
    ----------
    data : ndarray
        2D numpy array containing the pixel values of the image in which sources are to be fitted.
    df : DataFrame
        Pandas DataFrame containing the source positions (with columns like 'xcentroid' and 'ycentroid').
    cutout_size : int
        The size of the cutout around each source to be fitted.
    known_fwhm : float, optional
        The known full width at half maximum (FWHM) value to use for initial guesses. If None, the value from
        the configuration is used. Default is None.
    update_position : bool, optional
        If True, update the centroid positions in the input dataframe based on the fit results. Default is False.
    config : dict
        Configuration parameters, including:

        - sat_lim : float
            Saturation limit for the source amplitude.
        - FWHM_INIT_GUESS : float
            Initial guess for FWHM if known_fwhm is not provided.
        - FWHM_LIM_MIN : float
            Minimum acceptable value for the FWHM.
        - FWHM_LIM_MAX : float
            Maximum acceptable value for the FWHM.
        - USE_GAUSS : bool
            If True, use a Gaussian model; otherwise, use a Moffat model.
        - DEFAULT_MOFF_BETA : float
            Default beta value for the Moffat model.
        - FITTING_METHOD : str
            Method to be used for fitting (e.g., 'least_square').

    Returns
    -------
    DataFrame
        A Pandas DataFrame containing the original source data along with the fitted FWHM, FWHM error,
        signal-to-noise ratio, and a fit mask indicating successful fits.

    Notes
    -----
    - The function uses either a Gaussian or Moffat model to fit each source.
    - Sources that do not meet the specified criteria (e.g., saturation, poor fit quality) are excluded from the results.
    - The results are appended to the input dataframe and returned as a new dataframe with additional columns.
    """

    sat_lim = config['sat_lim']
    fwhm_guess = config['FWHM_INIT_GUESS'] if known_fwhm is None else known_fwhm
    min_good_fwhm = config['FWHM_LIM_MIN']
    max_good_fwhm = config['FWHM_LIM_MAX']
    use_gauss = config["USE_GAUSS"]
    default_moff_beta = config['DEFAULT_MOFF_BETA']
    fitting_method = config['FITTING_METHOD']
    total_sources = len(df.index)

    # sigmaclip_fwhm_sigma = config['SIGMACLIP_FWHM_SIGMA']

    col_names = ['fit_fwhm', 'fit_fwhm_err', 's2n']
    result_arr = np.empty((total_sources, len(col_names)))
    nan_arr = np.array([[np.nan] * len(col_names)])
    bool_arr = np.zeros(total_sources, dtype=bool)

    # fit sources
    for i in range(total_sources):
        idx = df.index.values[i]
        x0 = df['xcentroid'].loc[[idx]]
        y0 = df['ycentroid'].loc[[idx]]

        # Create the Cutout2D object
        cutout_obj = Cutout2D(data=data, position=(x0, y0), size=cutout_size,
                              mode='partial', fill_value=0)

        cutout = cutout_obj.data

        _, sky, std = sigma_clipped_stats(data=cutout, cenfunc=np.nanmedian,
                                          stdfunc=np.nanstd, sigma=5.)
        cutout -= sky

        # estimate the signal-to-noise ratio
        snr_stars = np.nansum(cutout) / np.sqrt(np.nanstd(cutout) * cutout_size ** 2
                                                + ((1. / 2.) ** 2) * cutout_size ** 2)

        fit_result = fit_single_source(cutout, cutout_size, fwhm_guess, use_gauss=use_gauss,
                                       default_moff_beta=default_moff_beta,
                                       fitting_method=fitting_method)

        to_add = nan_arr
        if fit_result is not None:
            A = fit_result.params['amp'].value
            A_err = fit_result.params['amp'].stderr
            x_fitted = fit_result.params['xc'].value
            y_fitted = fit_result.params['yc'].value
            fwhm_fit = fit_result.params['fwhm'].value
            fwhm_fit_err = fit_result.params['fwhm'].stderr
            if fwhm_fit_err is not None:

                x_corrected, y_corrected = cutout_obj.to_original_position((x_fitted, y_fitted))
                # print(snr_stars, np.nansum(cutout) / std, A / std, fwhm_fit, fwhm_fit_err)

                if not ((A > sat_lim)
                        or (A <= A_err)
                        or (A_err is None)
                        or (snr_stars < 0)
                        or (fwhm_fit <= fwhm_fit_err)
                        or (fwhm_fit <= min_good_fwhm)
                        or (max_good_fwhm - 1 <= fwhm_fit)):
                    to_add = [fwhm_fit, fwhm_fit_err, snr_stars]
                    bool_arr[i] = True
                    if update_position:
                        df['xcentroid'] = df['xcentroid'].replace([x0], x_corrected)
                        df['ycentroid'] = df['ycentroid'].replace([y0], y_corrected)
                else:
                    to_add = nan_arr

        result_arr[i] = to_add
        bc.print_progress_bar(i+1, total_sources, length=80,
                              color=bc.BCOLORS.OKGREEN, use_lock=False)
        del cutout_obj, cutout

    sys.stdout.write('\n')

    # combine the fwhm result with the source table
    out_df = pd.concat(
        [
            df,
            pd.DataFrame(
                result_arr,
                index=df.index,
                columns=col_names
            )
        ], axis=1
    )

    out_df['fit_mask'] = bool_arr

    # reset index
    out_df.reset_index(inplace=True, drop=True)

    return out_df


def fit_single_source(data, box_size, fwhm_guess=None,
                      use_gauss=True,
                      default_moff_beta=4.765,
                      fitting_method='least_square'):
    """
    Fit a single source in the input data using either a Gaussian or Moffat model.

    This function fits a single point source in a 2D array (e.g., an image) using a specified model
    (Gaussian or Moffat).

    Parameters
    ----------
    data : ndarray
        2D numpy array containing the pixel values of the source to be fitted.
    box_size : int
        The size of the box around the source to be fitted.
    fwhm_guess : float, optional
        An initial guess for the full width at half maximum (FWHM).
    use_gauss : bool, optional
        If True, use a Gaussian model; otherwise, use a Moffat model. Default is True.
    default_moff_beta : float, optional
        Default beta value for the Moffat model. Default is 4.765.
    fitting_method : str, optional
        Method to be used for fitting (e.g., 'least_square'). Default is 'least_square'.

    Returns
    -------
    ModelResult
        A ModelResult object containing the fit results and related information.
        If the fitting fails, returns None.

    Notes
    -----
    - The function uses either a Gaussian or Moffat model based on the configuration.
    - The Moffat model has an additional parameter, 'beta', which controls the shape of the distribution.
    - The fitting is performed using the lmfit package's Model function, with specified bounds
      for the fit parameters to ensure physically reasonable results.
    """

    # create a mesh grid
    x_pix = np.arange(0, box_size)
    y_pix = np.arange(0, box_size)
    xx, yy = np.meshgrid(x_pix, y_pix)

    # choose the detection model Gauss or Moffat
    if use_gauss:
        fwhm_expr = '2 * sqrt(2 * log(2)) * sigma'

        def model_resid_func(x, y, xc, yc, amp, sigma, sky):
            """ Residual function for Gauss"""
            G = gauss2d(x, y, xc, yc, amp, sigma, sky)
            return G
    else:
        fwhm_expr = '2 * alpha * sqrt(2**(1 / beta) - 1)'

        def model_resid_func(x, y, xc, yc, amp, alpha, beta, sky):
            """ Residual function for Moffat"""
            M = moffat2d(x, y, xc, yc, amp, alpha, beta, sky)
            return M

    # basic model function setup
    model_func = Model(model_resid_func, independent_vars=('x', 'y'))
    model_func.set_param_hint(name='xc', value=box_size / 2.,
                              min=1, max=box_size - 1)
    model_func.set_param_hint(name='yc', value=box_size / 2.,
                              min=1, max=box_size - 1)

    # use Gaussian model
    if use_gauss:
        model_func.set_param_hint(name='sigma', value=fwhm_guess / 2.35482)
    # use Moffat model
    else:
        model_func.set_param_hint(name='alpha',
                                  value=3.,
                                  min=0.,
                                  max=30.)
        model_func.set_param_hint(name='beta',
                                  value=default_moff_beta,
                                  vary=False)

    # set FWHM
    model_func.set_param_hint(name='fwhm',
                              expr=fwhm_expr)

    # set initial amplitude estimate
    model_func.set_param_hint('amp',
                              value=np.nanmax(data),
                              min=0.85 * np.nanmin(data),
                              max=1.5 * np.nanmax(data))

    # set initial background estimate
    model_func.set_param_hint('sky', value=np.nanmedian(data), vary=True)

    # make parameters
    fit_params = model_func.make_params()

    # fit model
    try:

        result = model_func.fit(data=data, x=xx, y=yy,
                                method=fitting_method,
                                params=fit_params,
                                calc_covar=True, scale_covar=True,
                                nan_policy='omit', max_nfev=100)
        return result
    except (Exception,):
        return None


# def auto_build_source_catalog(data,
#                               img_std,
#                               mask,
#                               use_catalog=None,
#                               fwhm=None,
#                               source_box_size=25,
#                               fwhm_init_guess=4., threshold_value=25,
#                               min_source_no=3,
#                               max_source_no=1000,
#                               fudge_factor=5,
#                               fine_fudge_factor=0.1,
#                               max_iter=50, fitting_method='least_square',
#                               use_gauss=True,
#                               lim_threshold_value=3.,
#                               default_moff_beta=4.765,
#                               min_good_fwhm=1,
#                               max_good_fwhm=30,
#                               sigmaclip_fwhm_sigma=3.,
#                               isolate_sources_fwhm_sep=5., init_iso_dist=25.,
#                               sat_lim=65536.,
#                               silent=False):
#     """ Automatically detect and extract sources.
#
#     Credit: https://github.com/Astro-Sean/autophot/blob/master/autophot/packages/find.py
#
#     """
#
#     # Initialize logging for this user-callable function
#     log.setLevel(logging.getLevelName(log.getEffectiveLevel()))
#
#     # Try to use PSF derived from the image as detection kernel
#     # The kernel must be derived from well-isolated sources not near the edge of the image
#     # kernel_psf = False
#     using_catalog_sources = False
#     if use_catalog is not None:
#         using_catalog_sources = True
#
#     init_fwhm = fwhm_init_guess
#     if fwhm is not None:
#         init_fwhm = fwhm
#
#     # decrease the detection threshold by
#     m = 0
#
#     # increase the detection threshold by
#     n = 0
#
#     # decrease the increment size
#     decrease_increment = False
#
#     # backstop
#     failsafe = 0
#
#     # check if a few more sources can be found
#     check = False
#     check_len = -np.inf
#
#     update_fwhm_guess = False
#     brightest = 25  # number of brightest sources to be selected for the initial run
#
#     # create a mesh grid
#     x_pix = np.arange(0, source_box_size)
#     y_pix = np.arange(0, source_box_size)
#     xx, yy = np.meshgrid(x_pix, y_pix)
#
#     # choose the detection model Gauss or Moffat
#     if use_gauss:
#         log.info('    Using Gaussian Profile for FWHM fitting')
#
#         model_fwhm = gauss_sigma2fwhm
#         fwhm_expr = '2 * sqrt(2 * log(2)) * sigma'
#
#         def model_resid_func(x, y, xc, yc, amp, sigma, sky):
#             """ Residual function for Gauss"""
#             G = gauss2d(x, y, xc, yc, amp, sigma, sky)
#             return G
#     else:
#
#         log.info('    Using Moffat Profile for FWHM fitting')
#
#         model_fwhm = moffat_fwhm
#         fwhm_expr = '2 * alpha * sqrt(2**(1 / beta) - 1)'
#
#         def model_resid_func(x, y, xc, yc, amp, alpha, beta, sky):
#             """ Residual function for Moffat"""
#             M = moffat2d(x, y, xc, yc, amp, alpha, beta, sky)
#             return M
#
#     # basic model function setup
#     model_func = Model(model_resid_func, independent_vars=('x', 'y'))
#     model_func.set_param_hint(name='xc', value=source_box_size / 2.,
#                               min=0, max=source_box_size)
#     model_func.set_param_hint(name='yc', value=source_box_size / 2.,
#                               min=0, max=source_box_size)
#
#     # use Gaussian model
#     if use_gauss:
#         model_func.set_param_hint(name='sigma', value=1.)
#     # use Moffat model
#     else:
#         model_func.set_param_hint(name='alpha',
#                                   value=3.,
#                                   min=0.,
#                                   max=30.)
#         model_func.set_param_hint(name='beta',
#                                   value=default_moff_beta,
#                                   vary=False)
#
#     # set FWHM
#     model_func.set_param_hint(name='fwhm',
#                               expr=fwhm_expr)
#
#     nddata = NDData(data=data)
#     isolated_sources = []
#
#     # run a loop until enough sources are found
#     while True:
#
#         # for photometry use the photometric reference catalog
#         if using_catalog_sources:
#             sources = use_catalog
#             sources.reset_index(drop=True, inplace=True)
#         # else search for sources
#         else:
#
#             # this is a break condition to avoid an infinite loop
#             if failsafe > max_iter:
#                 break
#             else:
#                 failsafe += 1
#
#             # check if the threshold value is still good
#             threshold_value_check = threshold_value + n - m
#
#             # if the threshold <= 0 reverse previous drop and fine_fudge factor
#             if threshold_value_check < lim_threshold_value and not decrease_increment:
#                 log.warning(
#                     '    Threshold value has gone below background limit [%d sigma] - '
#                     'increasing by smaller increment ' % lim_threshold_value)
#
#                 # revert previous decrease
#                 decrease_increment = True
#                 m = fudge_factor
#
#             elif threshold_value_check < lim_threshold_value and decrease_increment:
#                 log.critical('    FWHM detection failed - cannot find suitable threshold value ')
#                 return None, None, False, ['', '', '']
#             else:
#                 threshold_value = round(threshold_value + n - m, 3)
#
#             # if the threshold goes negative, use a smaller fudge factor
#             if decrease_increment:
#                 fudge_factor = fine_fudge_factor
#
#             # prepare algorithm
#             # print(init_fwhm)
#             daofind = DAOStarFinder(fwhm=init_fwhm * 2.,
#                                     threshold=threshold_value * img_std,
#                                     exclude_border=True,
#                                     peakmax=sat_lim,
#                                     # minsep_fwhm=1,
#                                     brightest=brightest,
#                                     # sigma_radius=1.5
#                                     # min_separation=3*init_fwhm
#                                     )
#
#             # extract sources
#             # print(~mask)
#             # plt.title('0')
#             # plt.imshow(data)
#             # plt.show()
#             sources = daofind.find_stars(data, mask)
#
#             # decrease threshold if no sources were found
#             if sources is None:
#                 log.warning('    Sources == None at %.1f sigma - decreasing threshold' % threshold_value)
#                 m = fudge_factor
#                 continue
#
#             # Sort based on flux to identify the brightest sources for use as a kernel
#             sources.sort('flux', reverse=True)
#             sources = sources.to_pandas()
#         # print(sources)
#
#         src_positions = np.array(list(zip(sources['xcentroid'], sources['ycentroid'])))
#         plt.figure()
#         plt.title('1')
#         plt.imshow(data, vmin=0, vmax=100)
#         plt.scatter(src_positions[:, 0], src_positions[:, 1], c='r', alpha=0.5)
#         plt.figure()
#         plt.title('1b')
#         plt.imshow(mask * 1)
#
#         from photutils.detection import find_peaks
#         from photutils.segmentation import detect_threshold
#
#         threshold_value = detect_threshold(data, nsigma=5, background=0.)
#         all_sources = find_peaks(data, threshold_value, box_size=15,
#                                  border_width=int(15),
#                                  # npeaks = 100,
#                                  centroid_func=centroid_2dg,
#                                  mask=mask).to_pandas()
#         # # print(len(all_sources))
#         # src_positions = np.array(list(zip(all_sources['x_peak'], all_sources['y_peak'])))
#         # plt.figure()
#         # plt.title('find peaks')
#         # plt.imshow(data, vmin=0, vmax=100)
#         # plt.scatter(src_positions[:, 0], src_positions[:, 1], c='r', alpha=0.5)
#         # plt.show()
#         # input()
#         if not silent:
#             log.info('    Number of sources before cleaning [ %.1f sigma ]: %d ' % (threshold_value, len(sources)))
#
#         # perform some checks
#         if len(sources) == 0 and not using_catalog_sources:
#             log.warning('No sources')
#             m = fudge_factor
#             continue
#
#         if check and len(sources) >= 10 and not using_catalog_sources:
#             pass
#
#         elif check and len(sources) > check_len and not using_catalog_sources:
#             if not silent:
#                 log.info('    More sources found - trying again')
#
#             m = fudge_factor
#             check_len = len(sources)
#             check = False
#
#             continue
#
#         # Last ditch attempt to recover a few more sources
#         elif len(sources) < 5 and not check and not using_catalog_sources:
#             if not silent:
#                 log.info('    Sources found but attempting to go lower')
#             m = fudge_factor
#             check_len = len(sources)
#             check = True
#             continue
#
#         # search for fwhm
#         if not update_fwhm_guess:
#             if not silent:
#                 log.info('    Updating search FWHM value')
#
#             new_fwhm_guess = []
#             new_r_squared_guess = []
#             src_tmp = sources.copy()
#
#             # take the 50 brightest sources
#             no_sources = 50 if len(src_tmp.index.values) > 50 else len(src_tmp.index.values)
#             # src_tmp = src_tmp.head(no_sources)
#
#             for i in list(src_tmp.index.values):
#                 try:
#                     idx = src_tmp.index.values[i]
#                     # print(src_tmp.loc[[idx]])
#                     stars_tbl = Table()
#                     stars_tbl['x'] = src_tmp['xcentroid'].loc[[idx]]
#                     stars_tbl['y'] = src_tmp['ycentroid'].loc[[idx]]
#                     stars = extract_stars(nddata, stars_tbl, size=source_box_size)
#                     stars = stars.data
#                     # stars[stars < 0] = 0.
#                     _, sky, _ = sigma_clipped_stats(data=stars, cenfunc=np.nanmedian,
#                                                     stdfunc=np.nanstd, sigma=5.)
#                     stars -= sky
#
#                     # estimate the signal-to-noise ratio
#                     snr_stars = np.nansum(stars) / np.sqrt(np.nanstd(stars) * source_box_size ** 2
#                                                            + ((1. / 2.) ** 2) * source_box_size ** 2)
#                     print(snr_stars)
#
#                     # jump to the next object
#                     if np.nanmax(stars) >= sat_lim or np.isnan(np.max(stars)):
#                         continue
#
#                     # set initial amplitude estimate
#                     model_func.set_param_hint('amp',
#                                               value=np.nanmax(stars),
#                                               min=0.85 * np.nanmax(stars),
#                                               max=1.5 * np.nanmax(stars))
#
#                     # set initial background estimate
#                     model_func.set_param_hint('sky', value=0, vary=False)
#
#                     # make parameters
#                     fit_params = model_func.make_params()
#
#                     # fit model
#                     result = model_func.fit(data=stars, x=xx, y=yy,
#                                             method=fitting_method,
#                                             params=fit_params,
#                                             calc_covar=True, scale_covar=True,
#                                             nan_policy='omit', max_nfev=100)
#                     # print(result.fit_report())
#
#                     # extract FWHM
#                     fwhm_fit = result.params['fwhm'].value
#                     fwhm_fit_err = result.params['fwhm'].stderr
#                     # print(fwhm_fit, fwhm_fit_err)
#                     # plt.figure()
#                     # plt.title('Stars')
#                     # plt.imshow(stars)
#                     # plt.show()
#
#                     # check result
#                     if (max_good_fwhm <= fwhm_fit <= min_good_fwhm) \
#                             or (result.params['amp'] >= sat_lim) \
#                             or (result.params['fwhm'].stderr is None) \
#                             or (snr_stars <= 0.):
#                         new_fwhm_guess.append(np.nan)
#                         continue
#
#                     new_r_squared_guess.append(result.rsquared)
#                     new_fwhm_guess.append(fwhm_fit)
#                     m = 0
#
#                     if len(new_fwhm_guess) == no_sources:
#                         break
#
#                     # print(new_fwhm_guess, new_r_squared_guess)
#                 except (Exception,):
#                     pass
#
#             new_r_squared_guess = np.array(new_r_squared_guess)
#             new_fwhm_guess = np.array(new_fwhm_guess)
#             # print(new_fwhm_guess, new_r_squared_guess)
#
#             all_nan_check = (np.where(new_fwhm_guess == np.nan, True, False)).all()
#             if all_nan_check and not decrease_increment and not using_catalog_sources:
#                 log.warning('    No FWHM values - decreasing threshold')
#                 m = fudge_factor
#                 continue
#             else:
#
#                 init_fwhm = np.nanpercentile(new_fwhm_guess, 50.,
#                                              method='median_unbiased')
#
#                 if ~np.isnan(init_fwhm):  # and len(new_fwhm_guess[~np.isnan(new_fwhm_guess)]) >= 3:
#                     if not silent:
#                         log.info('    Updated guess for FWHM: %.1f pixels ' % init_fwhm)
#
#                     init_r2 = np.nanpercentile(new_r_squared_guess, 84.135,
#                                                method='median_unbiased')  # 84.135
#
#                     if not silent:
#                         log.info('    Updated limit for R-squared: %.3f' % init_r2)
#
#                     brightest = None
#                     update_fwhm_guess = True
#                     continue
#                 if not using_catalog_sources:
#                     log.warning('    Not enough FWHM values - decreasing threshold')
#                     m = fudge_factor
#                     continue
#
#         if len(sources) > max_source_no and not using_catalog_sources:
#             log.warning('    Too many sources - increasing threshold')
#             if n == 0:
#                 threshold_value *= 2
#             elif m != 0:
#                 decrease_increment = True
#                 n = fine_fudge_factor
#                 fudge_factor = fine_fudge_factor
#             else:
#                 n = fudge_factor
#             continue
#
#         elif len(sources) > 5000 and m != 0 and not using_catalog_sources:
#             log.warning('    Picking up noise - increasing threshold')
#             fudge_factor = fine_fudge_factor
#             n = fine_fudge_factor
#             m = 0
#             decrease_increment = True
#             continue
#
#         elif len(sources) < min_source_no and not decrease_increment and not using_catalog_sources:
#             log.warning('    Too few sources - decreasing threshold')
#             m = fudge_factor
#             continue
#
#         elif len(sources) == 0 and not using_catalog_sources:
#             log.warning('    No sources - decreasing threshold')
#             m = fudge_factor
#             continue
#
#         # src_positions = np.array(list(zip(sources['xcentroid'], sources['ycentroid'])))
#         # plt.figure()
#         # plt.title('2')
#         # plt.imshow(data)
#         # plt.scatter(src_positions[:, 0], src_positions[:, 1], c='r', alpha=0.5)
#
#         # exclude sources close to image boundary
#         len_with_boundary = len(sources)
#         hsize = (source_box_size - 1) // 2
#         boundary_mask = ((sources['xcentroid'] > hsize) & (sources['xcentroid'] < (data.shape[1] - 1 - hsize)) &
#                          (sources['ycentroid'] > hsize) & (sources['ycentroid'] < (data.shape[0] - 1 - hsize)))
#
#         sources = sources[boundary_mask]
#
#         # report
#         n_near_boundary = len_with_boundary - len(sources)
#         if n_near_boundary > 0:
#             if not silent:
#                 log.info(f'    Sources removed near boundary: {n_near_boundary:d}')
#
#         #
#         isolated_sources, _ = remove_close_elements(sources, init_iso_dist / 2, 'flux')
#         if not using_catalog_sources:
#             isolated_sources.sort_values(by='flux', ascending=False, inplace=True)
#
#         # src_positions = np.array(list(zip(sources['xcentroid'], sources['ycentroid'])))
#         # iso_src_positions = np.array(list(zip(isolated_sources['xcentroid'], isolated_sources['ycentroid'])))
#         # print(src_positions)
#         # plt.figure()
#         # plt.title('3')
#         # plt.imshow(data)
#         # plt.scatter(src_positions[:, 0], src_positions[:, 1], c='r', alpha=0.75)
#         # plt.scatter(iso_src_positions[:, 0], iso_src_positions[:, 1], c='b', alpha=0.5)
#         # plt.show()
#
#         n_crowded = len(sources) - len(isolated_sources)
#         if n_crowded > 0:
#             log.info(f'    Crowded sources removed: {n_crowded:d}')
#
#         del sources
#
#         # get minimum distance to neighbour
#         v = isolated_sources[['xcentroid', 'ycentroid']]
#         dist = euclidean_distances(v, v)
#         dist = np.floor(dist)
#         minval = np.min(np.where(dist == 0., dist.max(), dist), axis=0)
#         isolated_sources['min_sep'] = minval
#
#         # reset index
#         isolated_sources.reset_index(drop=True, inplace=True)
#
#         if len(isolated_sources) < min_source_no:
#             log.warning('    Less than min source available after isolating sources')
#             m = fudge_factor
#             continue
#
#         if not silent:
#             log.info(f'    Source for FWHM estimation: {len(isolated_sources.index)}')
#             log.info('    Run fit (This may take a second.)')
#
#         col_names = ['fwhm', 'fwhm_err', 'median']
#         result_arr = np.empty((len(isolated_sources.index), len(col_names)))
#         nan_arr = np.array([[np.nan] * 3])
#         saturated_source = 0
#         high_fwhm = 0
#         for i in range(len(isolated_sources.index)):
#
#             idx = isolated_sources.index.values[i]
#             x0 = isolated_sources['xcentroid'].loc[[idx]]
#             y0 = isolated_sources['ycentroid'].loc[[idx]]
#
#             stars_tbl = Table()
#             stars_tbl['x'] = x0
#             stars_tbl['y'] = y0
#
#             try:
#                 stars = extract_stars(nddata, stars_tbl, size=source_box_size)
#             except (Exception,):
#                 result_arr[i] = nan_arr
#                 continue
#
#             stars = stars.data
#
#             if np.nanmax(stars) >= sat_lim or np.isnan(np.max(stars)):
#                 result_arr[i] = nan_arr
#                 continue
#
#             try:
#                 _, sky, _ = sigma_clipped_stats(data=stars, cenfunc=np.nanmedian,
#                                                 stdfunc=np.nanstd, sigma=3.)
#                 # print(sky)
#                 stars -= sky
#                 model_func.set_param_hint('amp', value=np.nanmax(stars),
#                                           min=0.85 * np.nanmax(stars),
#                                           max=1.5 * np.nanmax(stars))
#
#                 model_func.set_param_hint('sky', value=0, vary=False)
#
#                 if not use_gauss:
#                     alpha = init_fwhm / (2 * np.sqrt(2 ** (1 / default_moff_beta) - 1))
#                     model_func.set_param_hint('alpha', value=alpha)
#
#                 fit_params = model_func.make_params()
#
#                 result = model_func.fit(data=stars, x=xx, y=yy,
#                                         method=fitting_method,
#                                         params=fit_params, calc_covar=True,
#                                         scale_covar=True,
#                                         nan_policy='omit', max_nfev=100)
#                 # print(result.fit_report())
#
#                 # fwhm_fit = 2. * result.params['alpha'] * np.sqrt((2. ** (1. / result.params['beta'])) - 1.)
#                 fwhm_fit = result.params['fwhm'].value
#                 fwhm_fit_err = result.params['fwhm'].stderr
#
#                 A = result.params['amp'].value
#                 A_err = result.params['amp'].stderr
#                 x_fitted = result.params['xc'].value
#                 y_fitted = result.params['yc'].value
#                 bkg_approx = result.params['sky'].value
#                 # print(isolated_sources['flux'].loc[[idx]],
#                 #       A, bkg_approx, fwhm_fit, fwhm_fit_err, result.rsquared, init_r2)
#
#                 to_add = nan_arr
#                 if fwhm_fit_err is not None:
#
#                     corrected_x = x_fitted - source_box_size / 2 + x0
#                     corrected_y = y_fitted - source_box_size / 2 + y0
#
#                     if A > sat_lim:
#                         to_add = nan_arr
#                         saturated_source += 1
#                     elif (max_good_fwhm - 1 <= fwhm_fit) or (fwhm_fit <= min_good_fwhm) or (fwhm_fit <= fwhm_fit_err):
#                         to_add = nan_arr
#                         high_fwhm += 1
#                     elif A <= A_err or A_err is None:
#                         to_add = nan_arr
#                     else:
#                         to_add = np.array([fwhm_fit, fwhm_fit_err, bkg_approx])
#                         isolated_sources['xcentroid'] = isolated_sources['xcentroid'].replace([x0], corrected_x)
#                         isolated_sources['ycentroid'] = isolated_sources['ycentroid'].replace([y0], corrected_y)
#                     #
#                     # plt.figure()
#                     # plt.imshow(stars)
#                     # plt.show()
#
#             except (Exception,):
#                 to_add = nan_arr
#
#             result_arr[i] = to_add
#
#         if saturated_source != 0:
#             log.info(f'    Saturated sources removed: {saturated_source:d}')
#         # print(high_fwhm, type(high_fwhm))
#         if high_fwhm != 0:
#             log.info(f'    Sources with bad fwhm '
#                      f'[limit: {min_good_fwhm:.1f}, {max_good_fwhm:.1f} pixels]: {high_fwhm:d}')
#
#         # combine the fwhm result with the source table
#         isolated_sources = pd.concat(
#             [
#                 isolated_sources,
#                 pd.DataFrame(
#                     result_arr,
#                     index=isolated_sources.index,
#                     columns=col_names
#                 )
#             ], axis=1
#         )
#
#         # reset index
#         isolated_sources.reset_index(inplace=True, drop=True)
#         print(isolated_sources)
#         print('here')
#         if not using_catalog_sources:
#
#             if len(isolated_sources['fwhm'].values) == 0:
#                 log.warning('    No sigma values taken')
#                 continue
#
#             if len(isolated_sources) < min_source_no:
#                 log.warning('    Less than min source after sigma clipping: %d' % len(isolated_sources))
#                 threshold_value += m
#                 if n == 0:
#                     decrease_increment = True
#                     n = fine_fudge_factor
#                     fudge_factor = fine_fudge_factor
#                 else:
#                     n = fudge_factor
#                 m = 0
#                 continue
#
#             FWHM_mask = sigma_clip(isolated_sources['fwhm'].values,
#                                    sigma=sigmaclip_fwhm_sigma,
#                                    masked=True,
#                                    maxiters=10,
#                                    cenfunc=np.nanmedian,
#                                    stdfunc=np.nanstd)
#             plt.figure()
#             plt.hist(isolated_sources['fwhm'].values, bins='auto')
#             plt.show()
#             if np.sum(FWHM_mask.mask) == 0 or len(isolated_sources) < 5:
#                 isolated_sources['include_fwhm'] = [True] * len(isolated_sources)
#                 fwhm_array = isolated_sources['fwhm'].values
#             else:
#                 fwhm_array = isolated_sources[~FWHM_mask.mask]['fwhm'].values
#                 isolated_sources['include_fwhm'] = ~FWHM_mask.mask
#                 # log.info('    Removed %d FWHM outliers' % (np.sum(FWHM_mask.mask)))
#
#             median_mask = sigma_clip(isolated_sources['median'].values,
#                                      sigma=3.,
#                                      masked=True,
#                                      maxiters=10,
#                                      cenfunc=np.nanmedian,
#                                      stdfunc=mad_std)
#
#             if np.sum(median_mask) == 0 or np.sum(~median_mask.mask) < 5:
#                 isolated_sources['include_median'] = [True] * len(isolated_sources)
#                 pass
#
#             else:
#                 isolated_sources['include_median'] = ~median_mask.mask
#                 # log.info('    Removed %d median outliers' % (np.sum(median_mask.mask)))
#
#             if not silent:
#                 log.info('    Usable sources found [ %d sigma ]: %d' % (threshold_value,
#                                                                         len(isolated_sources)))
#
#             image_fwhm = np.nanmedian(fwhm_array)
#             if len(fwhm_array) == 0:
#                 log.warning('    Less than min source after sigma clipping: %d' % len(isolated_sources))
#                 threshold_value -= n
#
#                 if m == 0:
#                     decrease_increment = True
#                     m = fine_fudge_factor
#                     fudge_factor = fine_fudge_factor
#                 else:
#                     m = fudge_factor
#
#                 n = 0
#                 continue
#
#             if len(isolated_sources) > 3:
#                 too_close = (isolated_sources['min_sep'] <= isolate_sources_fwhm_sep * image_fwhm)
#                 isolated_sources = isolated_sources[~too_close]
#                 log.info(f'    Sources within minimum separation '
#                          f'[ {(isolate_sources_fwhm_sep * image_fwhm):.0f} pixel ]: {too_close.sum():d}')
#
#         break
#
#     # isolated_sources = isolated_sources.dropna(subset=['fwhm', 'fwhm_err'], inplace=False)
#     FWHM_mask = sigma_clip(isolated_sources['fwhm'].values,
#                            sigma=sigmaclip_fwhm_sigma,
#                            masked=True,
#                            maxiters=10,
#                            cenfunc=np.nanmedian,
#                            stdfunc=np.nanstd)
#
#     if np.sum(FWHM_mask.mask) == 0 or len(isolated_sources) < 5:
#         isolated_sources['include_fwhm'] = [True] * len(isolated_sources)
#         fwhm_array = isolated_sources['fwhm'].values
#     else:
#         fwhm_array = isolated_sources[~FWHM_mask.mask]['fwhm'].values
#         isolated_sources['include_fwhm'] = ~FWHM_mask.mask
#         log.info(f'    FWHM outliers removed: {np.sum(FWHM_mask.mask):d}')
#
#     image_fwhm = np.nanmedian(fwhm_array)
#     image_fwhm_err = np.nanstd(fwhm_array)
#
#     if not silent:
#         log.info(f'    FWHM: {image_fwhm:.3f} +/- {image_fwhm_err:.3f} [ pixels ]')
#     isolated_sources['cat_id'] = np.arange(1, len(isolated_sources) + 1)
#
#     return (image_fwhm, image_fwhm_err), isolated_sources, True, None
#

def clean_catalog_trail(imgarr, mask, catalog, fwhm, r=5.):
    """Remove objects from the source catalog.

    Remove objects from the source catalog where any
    pixel within a radius = r * sigma_kernel falls within the area of any trail in the image.

    Parameters
    ----------
    imgarr: np.ndarray
        Input image
    mask: np.ndarray, bool
        Mask containing all found satellite trails
    catalog: pandas.Dataframe
        Source table with positions on the detector
    fwhm: float
        FWHM used to define the radius around a source. None of the pixels within this area
        must overlap with the trail region.
    r: float
        Multiplier to allow adjustment of radius

    Returns
    -------
    catalog_cleaned: ~pd.Dataframe
        Cleaned source catalog
    """

    radius = r * fwhm

    trl_ind = np.where(mask * imgarr > 0)

    if not np.any(trl_ind, axis=1).any():
        return None, None
    del trl_ind

    src_pos = np.array(list(zip(catalog['xcentroid'], catalog['ycentroid'])))

    unwanted_indices = []
    for i in range(len(src_pos)):
        aperture = CircularAperture(src_pos[i], radius).to_mask(method='center')
        mask2 = aperture.multiply(mask)
        if mask2 is not None:
            if np.any(mask2[mask2 > 0]):
                unwanted_indices.append([i, True])
        else:
            unwanted_indices.append([i, False])
        del mask2, aperture

    unwanted_indices = np.array(unwanted_indices)

    if list(unwanted_indices):
        desired_df = catalog.drop(catalog.index[unwanted_indices[:, 0]], axis=0)
        idx = np.where(unwanted_indices[:, 1])
        removed_df = catalog.iloc[unwanted_indices[:, 0][idx]]
        del idx, imgarr, catalog, unwanted_indices, mask
        gc.collect()
        return desired_df, removed_df
    else:
        del imgarr, unwanted_indices, mask
        gc.collect()
        return catalog, pd.DataFrame()


def load_background_file(bkg_fname, silent=False):
    """Load 2D background data from file"""
    if not silent:
        log.info(f"  > Load 2D background from file: ../auxiliary/{bkg_fname[1]}.fits")

    # load fits file
    with fits.open(f'{bkg_fname[0]}.fits') as hdul:
        hdul.verify('fix')
        hdr = hdul[0].header
        bkg_background = hdul[0].data.astype('float32')
        bkg_rms = hdul[1].data.astype('float32')
        bkg_median = hdr['BKGMED']
        bkg_rms_median = hdr['RMSMED']

    return bkg_background, bkg_median, bkg_rms, bkg_rms_median


def compute_2d_background(imgarr, mask, box_size, win_size,
                          bkg_estimator=MMMBackground,
                          rms_estimator=MADStdBackgroundRMS,
                          estimate_bkg=True, bkg_fname=Optional[list], silent=False):
    """Compute a 2D background for the input array.
    This function uses `~photutils.background.Background2D` to determine
    an adaptive background that takes into account variations in flux
    across the image.

    Parameters
    ----------
    imgarr: ndarray
        An NDarray of science data for which the background needs to be computed
    mask:
    box_size: integer
        The box_size along each axis for Background2D to use.
    win_size: integer
        The window size of the 2D median filter to apply to the low-resolution map as the
        `filter_size` parameter in Background2D.
    bkg_estimator: function
        The name of the function to use as the estimator of the background.
    rms_estimator: function
        The name of the function to use for estimating the RMS in the background.
    estimate_bkg: bool, optional
        If True, the 2D background is estimated from a given image. Else
    bkg_fname: filename, optional
        Name of the file containing the background, rms image and data from 2D background estimation.
        If not None, the background estimation is skipped
    silent: bool, optional
        Set to True to suppress most console output

    Returns
    -------
    bkg_background: np.ndarray
        The NDarray has the same shape as the input image array which contains the determined
        background across the array. If Background2D fails for any reason, a simpler
        sigma-clipped single-valued array will be computed instead.
    bkg_median: float
        The median value (or single sigma-clipped value) of the computed background.
    bkg_rms: float
        NDarray the same shape as the input image array which contains the RMS of the
        background across the array.  If Background2D fails for any reason, a simpler
        sigma-clipped single-valued array will be computed instead.
    bkg_rms_median: float
        The median value (or single sigma-clipped value) of the RMS of the computed
        background.

    """
    # Initialize logging for this user-callable function
    log.setLevel(logging.getLevelName(log.getEffectiveLevel()))

    bkg_background, bkg_median, bkg_rms, bkg_rms_median = None, None, np.array([]), None

    # Load background image data
    if not estimate_bkg and bkg_fname is not None:
        del imgarr, mask
        return load_background_file(bkg_fname, silent=silent)

    if not silent:
        log.info("  > Estimate 2D background from image. (This may take a second.)")

    imgarr[np.isnan(imgarr)] = 0.

    bkg = None
    exclude_percentiles = [10, 25, 50, 75]
    for percentile in exclude_percentiles:

        try:

            # create a simple source mask
            threshold = detect_threshold(imgarr, nsigma=3., mask=mask,
                                         sigma_clip=SigmaClip(sigma=3.0, maxiters=None))

            segment_img = detect_sources(imgarr, threshold, mask=mask,
                                         npixels=9, connectivity=8)

            src_mask = segment_img.make_source_mask(footprint=None)

            if mask is not None:
                src_mask |= mask

            # estimate the background
            bkg = my_background(imgarr,
                                mask=mask,
                                box_size=(box_size, box_size),
                                filter_size=(win_size, win_size),
                                exclude_percentile=percentile,
                                bkg_estimator=bkg_estimator,
                                bkgrms_estimator=rms_estimator,
                                edge_method="pad")

            # print(bkg.background_median, bkg.background_rms_median)
            # plt.figure()
            # norm = ImageNormalize(stretch=LinearStretch())
            # plt.imshow(bkg.background, origin='lower', cmap='Greys_r', norm=norm,
            #            interpolation='nearest')
            # bkg.plot_meshes(outlines=True, marker='.', color='cyan', alpha=0.3)
            # plt.show()

        except (Exception,):
            bkg = None
            continue

        if bkg is not None:
            bkg_background = bkg.background.astype('float32')
            bkg_median = bkg.background_median
            bkg_rms = bkg.background_rms.astype('float32')
            bkg_rms_median = bkg.background_rms_median
            break

    # If Background2D does not work at all, define default scalar values for
    # the background to be used in source identification
    if bkg is None:
        if not silent:
            log.warning("    Background2D failure detected. "
                        "Using alternative background calculation instead....")

        # detect the sources
        threshold = detect_threshold(imgarr, nsigma=2.0,
                                     sigma_clip=SigmaClip(sigma=3.0, maxiters=10))
        segment_img = detect_sources(imgarr, threshold, npixels=5)
        src_mask = segment_img.make_source_mask(footprint=None)
        if mask is not None:
            src_mask |= mask

        sigcl_mean, sigcl_median, sigcl_std = sigma_clipped_stats(imgarr,
                                                                  sigma=3.0,
                                                                  mask=src_mask,
                                                                  maxiters=10)
        bkg_median = max(0.0, sigcl_median)
        bkg_rms_median = sigcl_std

        # create background frame shaped like imgarr populated with sigma-clipped median value
        bkg_background = np.full_like(imgarr, bkg_median).astype('float32')

        # create background frame shaped like imgarr populated with sigma-clipped standard deviation value
        bkg_rms = np.full_like(imgarr, sigcl_std).astype('float32')

    if not silent:
        log.info(f"    Save 2D background to file: ../auxiliary/{bkg_fname[1]}.fits")

    # write results to fits
    hdu1 = fits.PrimaryHDU(data=bkg_background)
    hdu1.header.set('BKGMED', bkg_median)
    hdu1.header.set('RMSMED', bkg_rms_median)
    hdu2 = fits.ImageHDU(data=bkg_rms)
    new_hdul = fits.HDUList([hdu1, hdu2])
    new_hdul.writeto(f'{bkg_fname[0]}.fits', output_verify='ignore', overwrite=True)

    del imgarr, hdu1, hdu2, new_hdul
    gc.collect()

    return bkg_background, bkg_median, bkg_rms, bkg_rms_median


def compute_radius(wcs: WCS, naxis1: int, naxis2: int,
                   ra: float = None, dec: float = None) -> float:
    """Compute the radius from the center to the furthest edge of the WCS.

    Parameters
    -----------
    wcs:
        World coordinate system object describing translation between image and skycoord.
    naxis1:
        Axis length used to calculate the image footprint.
    naxis2:
        Axis length used to calculate the image footprint.
    ra:

    dec:


    Returns
    -------
    radius:
        Radius of field-of-view in arcmin.
    """

    if ra is None and dec is None:
        ra, dec = wcs.wcs.crval

    img_center = SkyCoord(ra=ra * u.degree, dec=dec * u.degree)

    # calculate footprint
    wcs_foot = wcs.calc_footprint(axes=(naxis1, naxis2))

    # get corners
    img_corners = SkyCoord(ra=wcs_foot[:, 0] * u.degree,
                           dec=wcs_foot[:, 1] * u.degree)

    # make sure the radius is less than 1 deg because of GAIAdr3 search limit
    separations = img_center.separation(img_corners).value
    radius = separations.max()

    return radius


def convert_astrometric_table(table: Table, catalog_name: str) -> Table:
    """Convert a table with varying column names into a more standardized table"""
    # Initialize logging for this user-callable function
    log.setLevel(logging.getLevelName(log.getEffectiveLevel()))

    cat_name = catalog_name.upper()
    if cat_name in bc.SUPPORTED_CATALOGS:
        cat_dict = bc.SUPPORTED_CATALOGS[cat_name]
        table.meta['converted'] = True
        log.debug(f'Converting supported catalog: {catalog_name}')
    else:
        table.meta['converted'] = False
        log.debug(f'NOT converting catalog: {catalog_name}')
        return table

    # translate the old column name into the new standardized name
    # for each column specified in this dict
    for new, old in cat_dict.items():
        if new != 'epoch':
            # print(old)
            rename_colname(table, old, new)
        elif old != '':
            # Add decimal year epoch column as new column for existing 'epoch'-like column
            # determine format of data in old time/date column
            val = table[old][0]
            if val > 2400000:
                dfmt = 'jd'
            elif val > 10000:
                dfmt = 'mjd'
            else:
                dfmt = None

            if dfmt:
                # Insure 'epoch' is decimal year and add it as a new column
                new_times = Time(table[old], format=dfmt).decimalyear
                time_col = Column(data=new_times, name=new)
                table.add_column(time_col)
            else:

                table.rename_column(old, new)
        else:
            # Insure at least an empty column is provided for 'epoch'
            rename_colname(table, old, new)

    return table

def make_border_mask(mask, borderLen=10):
    """"""

    if not isinstance(borderLen, int):
        borderLen = int(borderLen)

    # Create a mask of the same size as the image, initially False (no mask)
    # mask = np.zeros(image.shape, dtype=bool)

    # Set the top and bottom borders to True
    mask[:borderLen, :] = True
    mask[-borderLen:, :] = True

    # Set the left and right borders to True
    mask[:, :borderLen] = True
    mask[:, -borderLen:] = True

    return mask

def extract_source_catalog(imgarr,
                           source_catalog=None,
                           known_fwhm=None,
                           silent=False, **config):
    """ Extract and source catalog using photutils.

    The catalog returned by this function includes sources found in the
    input image with the positions translated to the coordinate frame
    defined by the reference WCS `refwcs`.

    Parameters
    ----------
    imgarr: np.ndarray
        Input image as an astropy.io.fits HDUList.
    source_catalog:
    known_fwhm:
    config: dict
        Dictionary containing the configuration
    silent: bool, optional
        Set to True to suppress most console output

    Returns
    -------
    source_cat: `~astropy.table.Table`
        Astropy Tables containing sources from image.
    """
    if not isinstance(imgarr, np.ndarray):
        raise ValueError(f"Input {imgarr} not a np.ndarray object.")

    # Initialize logging for this user-callable function
    log.setLevel(logging.getLevelName(log.getEffectiveLevel()))

    log.info("> Extract sources from image")

    img_mask = config['image_mask']

    if config['telescope_keyword'] == 'CTIO 4.0-m telescope':
        imgarr_bkg_subtracted = imgarr
        # bkg_rms_median = config['sky_noise']
    else:
        # compute 2D background
        bkg_arr, _, _, _ = compute_2d_background(imgarr,
                                                 mask=img_mask,
                                                 box_size=config['BKG_BOX_SIZE'],
                                                 win_size=config['BKG_MED_WIN_SIZE'],
                                                 estimate_bkg=config['estimate_bkg'],
                                                 bkg_fname=config['bkg_fname'],
                                                 silent=silent)

        # subtract background from the image
        imgarr_bkg_subtracted = imgarr - bkg_arr

    # apply bad pixel mask if present
    if img_mask is not None:
        imgarr_bkg_subtracted[img_mask] = 0  # np.nan  #bkg_rms_median

    if not silent:
        if source_catalog is not None:
            log.info("  > Build source catalog from known sources")
        else:
            log.info("  > Build source catalog from detected sources")

    #
    fwhm, source_cat, state, fail_msg = build_source_catalog(data=imgarr_bkg_subtracted,
                                                             input_catalog=source_catalog,
                                                             known_fwhm=known_fwhm,
                                                             mask=img_mask, **config)

    if not state or len(source_cat) == 0:
        del imgarr, imgarr_bkg_subtracted
        gc.collect()
        return None, None, False, fail_msg

    del imgarr, imgarr_bkg_subtracted
    gc.collect()

    return source_cat, fwhm, True, None


def get_reference_catalog_astro(ra, dec, sr: float = 0.5,
                                epoch: Time = None, catalog: str = 'GAIADR3',
                                mag_lim= -1, silent: bool = False):
    """ Extract reference catalog from VO web service.

    Queries the catalog available at the ``SERVICELOCATION`` specified
    for this module to get any available astrometric source catalog entries
    around the specified position in the sky based on a cone-search.

    todo: update this!!!

    Parameters
    ----------
    ra : float
        Right Ascension (RA) of center of field-of-view (in decimal degrees)
    dec : float
        Declination (Dec) of center of field-of-view (in decimal degrees)
    sr : float, optional
        Search radius (in decimal degrees) from field-of-view center to use
        for sources from catalog. Default: 0.5 degrees
    epoch : float, optional
        Catalog positions returned for this field-of-view will have their
        proper motions applied to represent their positions at this date, if
        a value is specified at all, for catalogs with proper motions.
    catalog : str, optional
        Name of catalog to query, as defined by web-service.  Default: 'GSC242'
    mag_lim : float or int, optional
    silent : bool, optional
        Set to True to suppress most console output

    Returns
    -------
    csv : CSV object
        CSV object of returned sources with all columns as provided by catalog
    """

    # Initialize logging for this user-callable function
    log.setLevel(logging.getLevelName(log.getEffectiveLevel()))

    coord = SkyCoord(ra=ra, dec=dec, unit=(u.degree, u.degree),
                     obstime=epoch
                     )
    radius = u.Quantity(sr, u.deg)

    ref_table = None
    if 'GAIA' in catalog:

        Gaia.ROW_LIMIT = -1  # Ensure the default row limit.

        if not silent:
            log.info("> Get astrometric reference catalog via astroquery")
            log.info(f"  Downloading data... This may take a while!!! Don't panic")

        mag_lim_str = f"""AND phot_g_mean_mag <= {mag_lim}"""
        if mag_lim == -1:
            mag_lim_str = """"""

        query = f"""SELECT * \
            FROM {bc.DEF_ASTROQUERY_CATALOGS[catalog.upper()]} AS g
            WHERE 1 = CONTAINS(POINT('ICRS', g.ra, g.dec), CIRCLE('ICRS',
            {coord.ra.value}, {coord.dec.value}, {radius.value})) \
            AND parallax IS NOT NULL AND parallax > 0  AND parallax > parallax_error \
            AND pmra IS NOT NULL \
            AND pmdec IS NOT NULL \
            {mag_lim_str} \
            ORDER BY phot_g_mean_mag ASC    
        """

        # run the query
        j = Gaia.launch_job_async(query)
        ref_table = j.get_results()

    # Add catalog name as metadata
    ref_table.meta['catalog'] = catalog
    ref_table.meta['epoch'] = epoch

    # Convert a common set of columns into standardized column names
    ref_table = convert_astrometric_table(ref_table, catalog)

    # sort table by magnitude, fainter to brightest
    ref_table.sort('mag', reverse=True)

    return ref_table[['RA', 'DEC', 'mag', 'objID']].to_pandas(), catalog


def download_phot_ref_cat(ra, dec, sr=0.1, epoch=None,
                          num_sources=None,
                          catalog='GSC243',
                          full_catalog=False, silent=False):
    """
    Extract reference catalog from VO web service.

    Queries the catalog available at the ``SERVICELOCATION`` specified
    for this module to get any available astrometric source catalog entries
    around the specified position in the sky based on a cone-search.

    Parameters
    ----------
    ra: float
        Right Ascension (RA) of center of field-of-view (in decimal degrees)
    dec: float
        Declination (Dec) of center of field-of-view (in decimal degrees)
    sr: float, optional
        Search radius (in decimal degrees) from field-of-view center to use
        for sources from catalog. Default: 0.1 degrees
    epoch: float, optional
        Catalog positions returned for this field-of-view will have their
        proper motions applied to represent their positions at this date, if
        a value is specified at all, for catalogs with proper motions.
    num_sources: int, None, optional
        Maximum number of the brightest/faintest sources to return in catalog.
        If `num_sources` is negative, return that number of the faintest
        sources.  By default, all sources are returned.
    catalog: str, optional
        Name of catalog to query, as defined by web-service.  Default: 'GSC242'
    full_catalog: bool, optional
        Return the full set of columns provided by the web service.
    silent: bool, optional
        Set to True to suppress most console output

    Returns
    -------
    csv: CSV object
        A CSV object of returned sources with all columns as provided by catalog
    """

    # Initialize logging for this user-callable function
    log.setLevel(logging.getLevelName(log.getEffectiveLevel()))

    serviceType = 'vo/CatalogSearch.aspx'
    spec_str = 'RA={}&DEC={}&SR={}&FORMAT={}&CAT={}&MINDET=5&MAXOBJ=5000'
    headers = {'Content-Type': 'text/csv'}
    fmt = 'CSV'
    epoch_str = '&EPOCH={:.3f}'

    base_spec = spec_str.format(ra, dec, sr, fmt, catalog)
    spec = base_spec + epoch_str.format(epoch) if epoch is not None else base_spec
    if not silent:
        log.info("> Get photometric reference catalog via VO web service")

    url_chk = url_checker(f'{bc.SERVICELOCATION}/{serviceType}')
    if not url_chk[0]:
        log.error(f"  {url_chk[1]}. ")
        sys.exit(1)
    if not silent:
        log.info(f"  {url_chk[1]}. Downloading data... This may take a while!!! Don't panic")

    serviceUrl = f'{bc.SERVICELOCATION}/{serviceType}?{spec}'
    log.debug("Getting catalog using: \n    {}".format(serviceUrl))
    rawcat = requests.get(serviceUrl, headers=headers)

    # convert from bytes to a String
    r_contents = rawcat.content.decode()
    rstr = r_contents.split('\r\n')

    # remove initial line describing the number of sources returned
    # CRITICAL to proper interpretation of CSV data
    if rstr[0].startswith('Error'):
        # Try again without EPOCH
        serviceUrl = f'{bc.SERVICELOCATION}/{serviceType}?{base_spec}'
        log.debug(f"Getting catalog using: \n    {serviceUrl}")
        rawcat = requests.get(serviceUrl, headers=headers)
        r_contents = rawcat.content.decode()  # convert from bytes to a String
        rstr = r_contents.split('\r\n')

    # If there is still an error returned by the web-service, report the exact error
    if rstr[0].startswith('Error'):
        log.warning(f"Catalog generation FAILED with: \n{rstr}")

    del rstr[0], rawcat, r_contents
    gc.collect()

    ref_table = Table.read(rstr, format='ascii.csv')

    if not ref_table:
        return ref_table

    # Add catalog name as meta data
    ref_table.meta['catalog'] = catalog
    ref_table.meta['epoch'] = epoch

    # Convert a common set of columns into standardized column names
    ref_table = convert_astrometric_table(ref_table, catalog)

    if not full_catalog:
        ref_table = ref_table['RA', 'DEC', 'mag', 'objID']

    # sort table by magnitude, fainter to brightest
    ref_table.sort('mag', reverse=True)

    if num_sources is not None:
        idx = -1 * num_sources
        ref_table = ref_table[:idx] if num_sources < 0 else ref_table[idx:]

    return ref_table.to_pandas(), catalog


def get_photometric_catalog(fname, loc, imgarr, hdr, wcsprm,
                            catalog, silent=False, **config):
    """"""

    # Initialize logging for this user-callable function
    log.setLevel(logging.getLevelName(log.getEffectiveLevel()))

    fwhm = config['img_fwhm']

    photo_ref_cat_fname = f'{loc}/{fname}_trail_img_photo_ref_cat'

    if catalog.upper() not in bc.SUPPORTED_CATALOGS:
        log.warning(f"Given photometry catalog '{catalog}' NOT SUPPORTED. "
                    "Defaulting to GSC243")
        catalog = "GSC243"

    if config["photo_ref_cat_fname"] is not None:
        photo_ref_cat_fname = config["photo_ref_cat_fname"]

    # get the observation date
    # if ('time-obs'.upper() in hdr and 'telescop'.upper() in hdr and
    #         hdr['telescop'.upper()] != 'CTIO 4.0-m telescope'):
    #     time_string = f"{hdr['date-obs'.upper()]}T{hdr['time-obs'.upper()]}"
    # else:
    #     time_string = hdr['date-obs'.upper()]
    #
    # frmt = bc.has_fractional_seconds(time_string)
    #
    # t = pd.to_datetime(time_string,
    #                    format=frmt, utc=False)
    #
    # epoch = Time(t)

    # set RA and DEC using the image center
    wcs = WCS(hdr)

    ra, dec = wcs.all_pix2world(hdr['NAXIS1'] / 2.,
                                hdr['NAXIS2'] / 2., 0)

    # estimate the radius of the FoV for ref. catalog creation
    fov_radius = compute_radius(wcs,
                                naxis1=hdr['NAXIS1'],
                                naxis2=hdr['NAXIS2'],
                                ra=ra, dec=dec)

    # check for source catalog file. If present and not force extraction, use these catalogs
    chk_ref_cat_photo = os.path.isfile(photo_ref_cat_fname + '.cat')

    read_ref_cat_photo = True
    if config["force_download"] or not chk_ref_cat_photo:
        read_ref_cat_photo = False

    # get photometric reference catalog
    if read_ref_cat_photo:
        log.info("> Load photometric references catalog from file")
        ref_tbl_photo, _, ref_catalog_photo = read_catalog(photo_ref_cat_fname)
    else:
        # get reference catalog
        ref_tbl_photo, ref_catalog_photo = \
            download_phot_ref_cat(ra=ra, dec=dec, sr=fov_radius,
                                  catalog=catalog,
                                  full_catalog=True, silent=silent)

        # add positions to table
        pos_on_det = wcs.wcs_world2pix(ref_tbl_photo[["RA", "DEC"]].values, 0)  # ['pixcrd']
        ref_tbl_photo["xcentroid"] = pos_on_det[:, 0]
        ref_tbl_photo["ycentroid"] = pos_on_det[:, 1]

        # remove non-stars from catalog
        if 'classification' in ref_tbl_photo:
            ref_tbl_photo = ref_tbl_photo[ref_tbl_photo.classification == 0]

        # mask positions outside the image boundaries
        mask = (ref_tbl_photo["xcentroid"] > 0) & \
               (ref_tbl_photo["xcentroid"] < hdr['NAXIS1']) & \
               (ref_tbl_photo["ycentroid"] > 0) & \
               (ref_tbl_photo["ycentroid"] < hdr['NAXIS2'])
        ref_tbl_photo = ref_tbl_photo[mask]

        if not silent:
            log.info("> Save photometric references catalog.")
        save_catalog(cat=ref_tbl_photo, wcsprm=wcsprm, out_name=photo_ref_cat_fname,
                     mode='ref_photo', catalog=ref_catalog_photo)

    if not silent:
        log.info("> Check photometric references catalog.")
    df, std_fkeys, mag_conv, exec_state, fail_msg = select_std_stars(ref_tbl_photo,
                                                                     ref_catalog_photo,
                                                                     config['_filter_val'],
                                                                     num_std_max=config['NUM_STD_MAX'],
                                                                     num_std_min=config['NUM_STD_MIN'])

    if not exec_state:
        return None, None, None, None, False, fail_msg

    # confirm sources
    src_tbl, kernel_fwhm, exec_state, fail_msg = extract_source_catalog(imgarr=imgarr,
                                                                        source_catalog=df,
                                                                        known_fwhm=fwhm,
                                                                        silent=silent,
                                                                        **config)
    if not exec_state:
        return None, None, None, None, False, fail_msg

    # from photutils.aperture import CircularAperture
    #
    # src_pos = np.array(list(zip(ref_tbl_photo['xcentroid'], ref_tbl_photo['ycentroid'])))
    # aperture = CircularAperture(src_pos, r=12)
    #
    # fig = plt.figure(figsize=(10, 6))
    #
    # gs = gridspec.GridSpec(1, 1)
    # ax = fig.add_subplot(gs[0, 0])
    # ax.imshow(imgarr, origin='lower', interpolation='nearest')
    # aperture.plot(axes=ax, **{'color': 'red', 'lw': 1.25, 'alpha': 0.75})
    # plt.show()

    return src_tbl, std_fkeys, mag_conv, kernel_fwhm, exec_state, None


def get_src_and_cat_info(fname, loc, imgarr, hdr, wcsprm,
                         silent=False, **config):
    """
    Extract astrometric positions and photometric data for sources in the
    input images' field-of-view.

    """

    # Initialize logging for this user-callable function
    log.setLevel(logging.getLevelName(log.getEffectiveLevel()))

    # get the observation date
    if ('time-obs'.upper() in hdr and 'telescop'.upper() in hdr and
            hdr['telescop'.upper()] != 'CTIO 4.0-m telescope'):
        time_string = f"{hdr['date-obs'.upper()]}T{hdr['time-obs'.upper()]}"
    else:
        time_string = hdr['date-obs'.upper()]

    # frmt = bc.has_fractional_seconds(time_string)
    t = pd.to_datetime(time_string,
                       format='ISO8601', utc=False)

    epoch = Time(t)

    # get RA and DEC value
    ra, dec = wcsprm.crval

    # set FoV
    fov_radius = config["fov_radius"]

    # Convert pointing to string for catalog name
    coo = SkyCoord(ra=ra * u.degree, dec=dec * u.degree)
    ra_str = coo.ra.to_string(unit=u.hourangle, sep='', precision=2, pad=True)
    dec_str = coo.dec.to_string(sep='', precision=2, alwayssign=True, pad=True)

    # set default catalog name
    src_cat_fname = f'{loc}/{fname}_src_cat'
    astro_ref_cat_fname = f'{loc}/ref_cat_{ra_str}{dec_str}'

    # check input and overwrite the defaults if necessary
    if config["src_cat_fname"] is not None:
        src_cat_fname = config["src_cat_fname"]
    if config["ref_cat_fname"] is not None:
        astro_ref_cat_fname = config["ref_cat_fname"]

    # check for source catalog file. If present and not force extraction, use these catalogs
    chk_src_cat = os.path.isfile(src_cat_fname + '.cat')
    chk_ref_cat_astro = os.path.isfile(astro_ref_cat_fname + '.cat')

    read_src_cat = not (config["force_extract"] or not chk_src_cat)
    read_ref_cat_astro = not (config["force_download"] or not chk_ref_cat_astro)

    if read_src_cat:
        if not silent:
            log.info("> Load image source catalog from file")
        src_tbl, kernel_fwhm, _ = read_catalog(src_cat_fname)
    else:
        # detect sources in image and get positions
        src_tbl, kernel_fwhm, state, fail_msg = \
            extract_source_catalog(imgarr=imgarr,
                                   silent=silent, **config)
        if not state:
            del imgarr, src_tbl, kernel_fwhm
            gc.collect()
            return (None for _ in range(6)), False, fail_msg

        if not silent:
            log.info("> Save image source catalog.")
        save_catalog(cat=src_tbl, wcsprm=wcsprm, out_name=src_cat_fname,
                     kernel_fwhm=kernel_fwhm)

    # get astrometric reference catalog
    if read_ref_cat_astro:
        # get reference catalog for precise positions from file
        log.info("> Load astrometric references catalog from file")
        ref_tbl_astro, _, ref_catalog_astro = read_catalog(astro_ref_cat_fname)

    else:
        # get reference catalog for precise positions from the web
        ref_tbl_astro, ref_catalog_astro = \
            get_reference_catalog_astro(ra=ra, dec=dec, sr=fov_radius,
                                        epoch=epoch,
                                        catalog='GAIAdr3',
                                        mag_lim=config['ref_cat_mag_lim'],
                                        silent=silent)

        if not silent:
            log.info("> Save astrometric references catalog.")
            save_catalog(cat=ref_tbl_astro, wcsprm=wcsprm,
                         out_name=astro_ref_cat_fname,
                         mode='ref_astro', catalog=ref_catalog_astro)

        # add positions to table
        pos_on_det = wcsprm.s2p(ref_tbl_astro[["RA", "DEC"]].values, 0)['pixcrd']
        ref_tbl_astro["xcentroid"] = pos_on_det[:, 0]
        ref_tbl_astro["ycentroid"] = pos_on_det[:, 1]

        del pos_on_det

    del imgarr
    if len(src_tbl) == 0:
        log.critical(f"  No detected sources to proceed!!! Skipping further steps.")
        return (None for _ in range(6)), False, None

    return (src_tbl, ref_tbl_astro, ref_catalog_astro,
            src_cat_fname, astro_ref_cat_fname,
            kernel_fwhm), True, None


def my_background(img, box_size, mask=None, interp=None, filter_size=(1, 1),
                  exclude_percentile=90, bkg_estimator=None,
                  bkgrms_estimator=None, edge_method='pad'):
    """ Run photutils background estimation with SigmaClip and MedianBackground"""

    if bkg_estimator is None:
        bkg_estimator = SExtractorBackground
    if bkgrms_estimator is None:
        bkgrms_estimator = StdBackgroundRMS
    if interp is None:
        interp = BkgZoomInterpolator()

    return Background2D(img, box_size,
                        sigma_clip=SigmaClip(sigma=3., maxiters=10),
                        exclude_percentile=exclude_percentile,
                        mask=mask,
                        bkg_estimator=bkg_estimator(),
                        bkgrms_estimator=bkgrms_estimator(),
                        edge_method=edge_method,
                        interpolator=interp, filter_size=filter_size)


def rename_colname(table: Table, colname: str, newcol: str):
    """Convert column name in table to user-specified name"""
    # If the table is missing a column, add a column with values of None
    if colname != '':
        table.rename_column(colname, newcol)
    else:
        empty_column = Column(data=[None] * len(table), name=newcol, dtype=np.float64)
        table.add_column(empty_column)


def read_catalog(cat_fname: str, tbl_format: str = "ecsv") -> tuple:
    """Load catalog from file"""

    cat = ascii.read(cat_fname + '.cat', format=tbl_format)

    kernel_fwhm = 4.
    kernel_fwhm_err = np.nan
    catalog = 'GAIAdr3'

    if 'kernel_fwhm' in cat.meta.keys():
        kernel_fwhm = cat.meta['kernel_fwhm']
    if 'kernel_fwhm_err' in cat.meta.keys():
        kernel_fwhm_err = cat.meta['kernel_fwhm_err']
    fwhm = (kernel_fwhm, kernel_fwhm_err)

    if 'catalog' in cat.meta.keys():
        catalog = cat.meta['catalog']

    cat_df = cat.to_pandas()
    del cat

    # exclude non-star objects
    if 'classification' in cat_df:
        cat_df = cat_df[cat_df.classification == 0]
    cat_df.reset_index()

    return cat_df, fwhm, catalog


def save_catalog(cat: pd.DataFrame,
                 wcsprm: astropy.wcs.Wcsprm,
                 out_name: str, kernel_fwhm: tuple = None, mag_corr: tuple = None,
                 std_fkeys: tuple = None,
                 catalog: str = None, mode: str = 'src', tbl_format: str = "ecsv"):
    """Save given catalog to .cat file"""

    # convert wcs parameters to wcs
    wcs = WCS(wcsprm.to_header())

    cat_out = cat.copy()

    if mode == 'src':
        # get position on sky
        pos_on_sky = wcs.wcs_pix2world(cat[["xcentroid", "ycentroid"]], 0)
        cat_out["RA"] = pos_on_sky[:, 0]
        cat_out["DEC"] = pos_on_sky[:, 1]
        cols = ['RA', 'DEC', 'xcentroid', 'ycentroid', 'mag', 'cat_id',
                'sharpness', 'roundness1', 'roundness2',
                'npix', 'sky', 'peak', 'flux']

    else:
        # get position on the detector
        pos_on_det = wcs.wcs_world2pix(cat[["RA", "DEC"]].values, 0)  # ['pixcrd']
        cat_out["xcentroid"] = pos_on_det[:, 0]
        cat_out["ycentroid"] = pos_on_det[:, 1]
        cols = ['RA', 'DEC', 'xcentroid', 'ycentroid', 'mag', 'objID']

    if mode == 'ref_astro':
        cat_out = cat_out[cols]

    # convert to astropy.Table and add meta-information
    cat_out = Table.from_pandas(cat_out, index=False)
    kernel_fwhm = (None, None) if kernel_fwhm is None else kernel_fwhm
    mag_corr = (None, None) if mag_corr is None else mag_corr
    std_fkeys = (None, None) if std_fkeys is None else std_fkeys
    cat_out.meta = {'kernel_fwhm': kernel_fwhm[0],
                    'kernel_fwhm_err': kernel_fwhm[1],
                    'catalog': catalog,
                    'std_fkeys': std_fkeys,
                    'mag_corr': mag_corr[0],
                    'mag_corr_err': mag_corr[1]}

    # write file to disk in the given format
    ascii.write(cat_out, out_name + '.cat', overwrite=True, format=tbl_format)

    del cat_out, cat
    gc.collect()


def select_reference_catalog(band: str, source: str = "auto") -> str:
    """ Select catalog based on the given band and the selected mode.

    Parameters
    ----------
    band:
        A str representing the filter band of the observation
    source:
        A str which defines which catalog to query.
        If 'auto' the catalog is selected based on the given filter band.
        The Standard is 'GSC242' for GSC 2.4.2.

    Returns
    -------
    catalog_name
        The selected catalog for photometry as str.

    # todo: add preferable catalog and allow multiple catalogs for a specific band
    """

    # Initialize logging for this user-callable function
    log.setLevel(logging.getLevelName(log.getEffectiveLevel()))

    catalog_name = source

    if band not in bc.SUPPORTED_BANDS:
        log.error("Given or observed filter band are not supported. "
                  "Possible filter are: {}".format(", ".join(bc.SUPPORTED_BANDS.keys())))
        sys.exit(1)

    if source != "auto":
        if source not in bc.SUPPORTED_CATALOGS:
            log.error("Given catalog not supported. "
                      "Possible catalogs are: "
                      "{}".format(", ".join(bc.SUPPORTED_CATALOGS.keys())))
            sys.exit(1)
        else:
            if source not in bc.SUPPORTED_BANDS[band]:
                log.error("Given band is not supported for this catalog. "
                          "Possible catalog for "
                          "{} band: {}".format(band,
                                               ", ".join(bc.SUPPORTED_BANDS[band])))
                sys.exit(1)
    if source == 'auto':
        catalog_name = bc.SUPPORTED_BANDS[band][0]

    return catalog_name


def select_std_stars(ref_cat: pd.DataFrame,
                     catalog: str, band: str,
                     num_std_max: int = None,
                     num_std_min: int = 5,
                     silent: bool = False) -> tuple:
    """
    Select standard stars for aperture photometry from photometric reference catalog.

    If no photometric reference stars are available in the observed band, color conversion
    is applied using transformations between SDSS Magnitudes and UBVRcIc.

    References:
    -----------
    - Lupton, R. 2005,
      http://classic.sdss.org/dr4/algorithms/sdssUBVRITransform.html#Lupton2005
    - Jester, S. 2005,
      https://classic.sdss.org/dr4/algorithms/sdssUBVRITransform.php#Jester2005

    Parameters
    ----------
    ref_cat : DataFrame
        Standard star catalog data.
    catalog : str
        Standard star catalog name.
    band : str
        Filter used in the observation.
    num_std_max : int, optional, default None
        Maximum number of standard stars.
    num_std_min : int, optional, default 5
        Minimum number of standard stars.
    silent : bool, optional, default False
        Suppress outputs.

    Returns
    -------
    ref_cat_filtered : DataFrame or None
        Catalog of selected standard stars or None if not enough standard stars are found.
    """

    # check for error column
    has_mag_conv = False

    # make a copy to work with
    cat_srt = ref_cat.copy()

    # exclude non-star sources from GSC2.4
    if 'classification' in ref_cat:
        cat_srt = ref_cat.drop(ref_cat[ref_cat.classification > 0].index)

    # reset indices
    cat_srt = cat_srt.reset_index()

    # convert FITS filter to catalog filter and error (if available)
    prim_filter_keys = bc.CATALOG_FILTER_EXT[catalog][band]['Prim']
    alt_filter_keys = bc.CATALOG_FILTER_EXT[catalog][band]['Alt']

    # flatten the list of filter keys
    flat_filter_keys = sum(prim_filter_keys, [])

    # Remove rows with NaN values
    df = cat_srt.dropna(subset=flat_filter_keys)

    # filter zero error values
    if band == 'w':
        non_zero_error_condition = (df[flat_filter_keys[0][1]] != 0.) & (df[flat_filter_keys[1][1]] != 0.)
    else:
        non_zero_error_condition = df[flat_filter_keys[1]] != 0.

    # apply the non-zero error condition
    df = df[non_zero_error_condition]

    #
    if len(df) < num_std_min and alt_filter_keys is not None and band != 'w':
        if not silent:
            log.warning(f"  ==> No or less than {num_std_min} standard stars "
                        f"in {flat_filter_keys[0]} band.")
            log.warning(f"      Using alternative band with known magnitude conversion.")

        alt_filter_keys = np.asarray(alt_filter_keys, str)
        alt_cat = cat_srt.dropna(subset=alt_filter_keys.flatten())

        x1 = alt_cat[alt_filter_keys[0, :]].to_numpy()
        x2 = alt_cat[alt_filter_keys[1, :]].to_numpy()
        x3 = alt_cat[alt_filter_keys[2, :]].to_numpy()

        has_data = np.any(np.array([x1, x2, x3]), axis=1).all()
        if not has_data:
            log.critical("  Insufficient data for magnitude conversion.")
            fail_msg = ['StdStarMagError', 'Insufficient data for magnitude conversion', '']
            del ref_cat, cat_srt, df, alt_cat
            return None, flat_filter_keys, has_mag_conv, False, fail_msg

        if band == 'U':
            alt_mags, alt_mags_err = phot.convert_sdss_to_U_jester(u=x1, g=x2, r=x3)
        else:
            alt_mags, alt_mags_err = phot.convert_sdss_to_BVRI_lupton(f=prim_filter_keys[0][0],
                                                                      x1=x1, x2=x2, x3=x3)

        new_df = pd.DataFrame({flat_filter_keys[0]: alt_mags,
                               flat_filter_keys[1]: alt_mags_err}, index=alt_cat.index)

        cat_srt = cat_srt.fillna(99999999).astype(np.int64, errors='ignore')
        cat_srt = cat_srt.replace(99999999, np.nan)

        cat_srt.update(new_df)
        has_mag_conv = True

    elif band == 'w':
        alt_filter_keys = np.asarray(flat_filter_keys, str)
        alt_cat = cat_srt.dropna(subset=alt_filter_keys.flatten())

        x1 = alt_cat[alt_filter_keys[0, :]].to_numpy()
        x2 = alt_cat[alt_filter_keys[1, :]].to_numpy()

        has_data = np.any(np.array([x1, x2]), axis=1).all()
        if not has_data:
            log.critical("Insufficient data for magnitude conversion.")
            del ref_cat, cat_srt, df, alt_cat

            fail_msg = ['StdStarMagError', 'Insufficient data for magnitude conversion', '']
            return None, flat_filter_keys, has_mag_conv, False, fail_msg

        alt_mags = x2[:, 0] + 0.23 * (x1[:, 0] - x2[:, 0])
        alt_mags_err = np.sqrt((0.23 * x1[:, 1]) ** 2 + (0.77 * x2[:, 1]) ** 2)

        filter_keys = ['WMag', 'WMagErr']
        new_df = pd.DataFrame({filter_keys[0]: alt_mags,
                               filter_keys[1]: alt_mags_err}, index=alt_cat.index)

        cat_srt = cat_srt.fillna(99999999).astype(np.int64, errors='ignore')
        cat_srt = cat_srt.replace(99999999, np.nan)

        cat_srt = pd.concat([cat_srt, new_df], axis=1)
        has_mag_conv = True

    # sort table by magnitude, brightest to fainter
    df = cat_srt.sort_values(by=[flat_filter_keys[0]], axis=0, ascending=True, inplace=False)
    df = df.dropna(subset=flat_filter_keys, inplace=False)
    if df.empty:
        log.critical("  No sources with uncertainties remaining!!!")
        del ref_cat, cat_srt, df

        fail_msg = ['StdStarMagError', 'No sources with uncertainties remaining', '']
        return None, flat_filter_keys, has_mag_conv, False, fail_msg

    # select the std stars by number
    if num_std_max is not None:
        idx = num_std_max
        df = df[:idx]

    # result column names
    cols = ['objID', 'RA', 'DEC', 'xcentroid', 'ycentroid']
    cols += flat_filter_keys

    df = df[cols]

    del ref_cat, cat_srt

    return df, flat_filter_keys, has_mag_conv, True, None


def url_checker(url: str) -> tuple[bool, str]:
    """Simple check if the URL is reachable"""
    try:
        # Get Url
        get = requests.get(url)
    # Exception
    except requests.exceptions.RequestException as e:
        # print URL with Errs
        return False, f"{url}: is Not reachable \nErr: {e}"
    else:
        # if the request succeeds
        if get.status_code == 200:
            return True, "URL is reachable"
        else:
            return False, f"URL: is Not reachable, status_code: {get.status_code}"


# def find_worst_residual_near_center(resid: np.ndarray):
#     """Find the pixel location of the worst residual, avoiding the edges"""
#
#     yc, xc = resid.shape[0] / 2., resid.shape[1] / 2.
#     radius = resid.shape[0] / 3.
#
#     y, x = np.mgrid[0:resid.shape[0], 0:resid.shape[1]]
#
#     mask = np.sqrt((y - yc) ** 2 + (x - xc) ** 2) < radius
#     rmasked = resid * mask
#
#     return np.unravel_index(np.argmax(rmasked), resid.shape)


# def plot_mask(scene, bkgd, mask, zmin, zmax, worst=None, smooth=0):
#     """Make a three-panel plot of:
#          * the mask for the whole image,
#          * the scene times the mask
#          * a zoomed-in region, with the mask shown as contours
#     """
#     from astropy.convolution import Gaussian2DKernel
#     if worst is None:
#         y, x = find_worst_residual_near_center(bkgd)
#     else:
#         x, y = worst
#     plt.figure(figsize=(20, 10))
#     plt.subplot(131)
#     plt.imshow(mask, vmin=0, vmax=1, cmap=plt.cm.get_cmap('gray'), origin='lower')
#     plt.subplot(132)
#     if smooth == 0:
#         plt.imshow((scene - bkgd) * (1 - mask), vmin=zmin, vmax=zmax, origin='lower')
#     else:
#         smoothed = convolve((scene - bkgd) * (1 - mask), Gaussian2DKernel(smooth))
#         plt.imshow(smoothed * (1 - mask), vmin=zmin / smooth, vmax=zmax / smooth,
#                    origin='lower')
#     plt.subplot(133)
#     plt.imshow(scene - bkgd, vmin=zmin, vmax=zmax)
#     plt.contour(mask, colors='red', alpha=0.2)
#     plt.ylim(y - 100, y + 100)
#     plt.xlim(x - 100, x + 100)
#     return x, y

# def mask_image(image,
#                vignette=-1.,
#                vignette_rectangular=-1.,
#                cutouts=None,
#                only_rectangle=None, silent=False):
#     """Mask image"""
#
#     # Initialize logging for this user-callable function
#     log.setLevel(logging.getLevelName(log.getEffectiveLevel()))
#
#     imgarr = image.copy()
#
#     # only search sources in a circle with radius <vignette>
#     if (0. < vignette < 2.) & (vignette != -1.):
#         sidelength = np.max(imgarr.shape)
#         x = np.arange(0, imgarr.shape[1])
#         y = np.arange(0, imgarr.shape[0])
#         if not silent:
#             log.info("    Only search sources in a circle "
#                      "with radius {}px".format(vignette * sidelength / 2.))
#         vignette = vignette * sidelength / 2.
#         mask = (x[np.newaxis, :] - sidelength / 2) ** 2 + \
#                (y[:, np.newaxis] - sidelength / 2) ** 2 < vignette ** 2
#         imgarr[~mask] = np.nan
#
#     # ignore a fraction of the image at the corner
#     if (0. < vignette_rectangular < 1.) & (vignette_rectangular != -1.):
#         if not silent:
#             log.info("    Ignore {0:0.1f}% of the image at the corner. ".format((1. - vignette_rectangular) * 100.))
#         sidelength_x = imgarr.shape[1]
#         sidelength_y = imgarr.shape[0]
#         cutoff_left = (1. - vignette_rectangular) * sidelength_x
#         cutoff_right = vignette_rectangular * sidelength_x
#         cutoff_bottom = (1. - vignette_rectangular) * sidelength_y
#         cutoff_top = vignette_rectangular * sidelength_y
#         x = np.arange(0, imgarr.shape[1])
#         y = np.arange(0, imgarr.shape[0])
#         left = x[np.newaxis, :] > cutoff_left
#         right = x[np.newaxis, :] < cutoff_right
#         bottom = y[:, np.newaxis] > cutoff_bottom
#         top = y[:, np.newaxis] < cutoff_top
#         mask = (left * bottom) * (right * top)
#         imgarr[~mask] = np.nan
#
#     # cut out rectangular regions of the image, [(xstart, xend, ystart, yend)]
#     if cutouts is not None and all(isinstance(el, list) for el in cutouts):
#         x = np.arange(0, imgarr.shape[1])
#         y = np.arange(0, imgarr.shape[0])
#         for cutout in cutouts:
#             if not silent:
#                 log.info("    Cutting out rectangular region {} of image. "
#                          "(xstart, xend, ystart, yend)".format(cutout))
#             left = x[np.newaxis, :] > cutout[0]
#             right = x[np.newaxis, :] < cutout[1]
#             bottom = y[:, np.newaxis] > cutout[2]
#             top = y[:, np.newaxis] < cutout[3]
#             mask = (left * bottom) * (right * top)
#             imgarr[mask] = np.nan
#
#     # use only_rectangle within image format: (xstart, xend, ystart, yend)
#     if only_rectangle is not None and isinstance(only_rectangle, tuple):
#         x = np.arange(0, imgarr.shape[1])
#         y = np.arange(0, imgarr.shape[0])
#         if not silent:
#             log.info("    Use only rectangle {} within image. "
#                      "(xstart, xend, ystart, yend)".format(only_rectangle))
#         left = x[np.newaxis, :] > only_rectangle[0]
#         right = x[np.newaxis, :] < only_rectangle[1]
#         bottom = y[:, np.newaxis] > only_rectangle[2]
#         top = y[:, np.newaxis] < only_rectangle[3]
#         mask = (left * bottom) * (right * top)
#         imgarr[~mask] = np.nan
#
#     return imgarr
#
