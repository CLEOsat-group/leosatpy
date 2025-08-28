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

from astropy.nddata import Cutout2D

from astroquery.gaia import Gaia

from lmfit import Model

from photutils.aperture import CircularAperture
from photutils.segmentation import (
    detect_sources, detect_threshold, deblend_sources, SourceCatalog)
from photutils.background import (
    Background2D,  # For estimating the background
    SExtractorBackground, StdBackgroundRMS, MMMBackground, MADStdBackgroundRMS,
    BkgZoomInterpolator)

from scipy.spatial import KDTree

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
    Parameters
    ----------
    sigma : float
        Sigma value (standard deviation) of the gaussian profile.

    Returns
    -------
    float
        Full width at half maximum value.
    """

    fwhm = 2 * np.sqrt(2 * np.log(2)) * sigma

    return fwhm


def gauss_fwhm2sigma(fwhm):
    """
    Convert full width at half maximum (FWHM) for gaussian function to sigma value.

    Parameters
    ----------
    fwhm : float
        Full width at half maximum of the gaussian profile.

    Returns
    -------
    float
        Sigma value (standard deviation) of the gaussian profile.
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
    box_size = default_box_size if known_fwhm is None else 2. * box_size

    log.info(f'{" ":<4}> Find sources and estimate Full Width Half Maximum (FWHM)')
    if mask is None:
        mask = np.zeros(data.shape, dtype=bool)

    mask = make_border_mask(mask, borderLen=box_size // 2)

    # Create threshold map
    threshold_value = detect_threshold(data, background=0., nsigma=nsigma, mask=mask)

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
        log.critical(f"{' ':<4}>> Source detection failed")
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

    col_names = ['fit_fwhm', 'fit_fwhm_err', 's2n']
    result_arr = np.empty((total_sources, len(col_names)))
    nan_arr = np.array([[np.nan] * len(col_names)])
    bool_arr = np.zeros(total_sources, dtype=bool)

    # Fit sources
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

        # Estimate the signal-to-noise ratio
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
        bc.print_progress_bar(i + 1, total_sources, length=80,
                              color=bc.BCOLORS.OKGREEN, use_lock=False)
        del cutout_obj, cutout

    sys.stdout.write('\n')

    # Combine the fwhm result with the source table
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

    # Reset index
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
    # Create a mesh grid
    x_pix = np.arange(0, box_size)
    y_pix = np.arange(0, box_size)
    xx, yy = np.meshgrid(x_pix, y_pix)

    # Choose the detection model Gauss or Moffat
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

    # Basic model function setup
    model_func = Model(model_resid_func, independent_vars=('x', 'y'))
    model_func.set_param_hint(name='xc', value=box_size / 2.,
                              min=1, max=box_size - 1)
    model_func.set_param_hint(name='yc', value=box_size / 2.,
                              min=1, max=box_size - 1)

    # Use Gaussian model
    if use_gauss:
        model_func.set_param_hint(name='sigma', value=fwhm_guess / 2.35482)
    # Use Moffat model
    else:
        model_func.set_param_hint(name='alpha',
                                  value=3.,
                                  min=0.,
                                  max=30.)
        model_func.set_param_hint(name='beta',
                                  value=default_moff_beta,
                                  vary=False)

    # Set the FWHM
    model_func.set_param_hint(name='fwhm',
                              expr=fwhm_expr)

    # Set the initial amplitude estimate
    model_func.set_param_hint('amp',
                              value=np.nanmax(data),
                              min=0.85 * np.nanmin(data),
                              max=1.5 * np.nanmax(data))

    # Set the initial background estimate
    model_func.set_param_hint('sky', value=np.nanmedian(data), vary=True)

    # Make fit parameters
    fit_params = model_func.make_params()

    # Run the model fit
    try:

        result = model_func.fit(data=data, x=xx, y=yy,
                                method=fitting_method,
                                params=fit_params,
                                calc_covar=True, scale_covar=True,
                                nan_policy='omit', max_nfev=100)
        return result
    except (Exception,):
        return None


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
    r: float, optional
        Multiplier to allow adjustment of radius. Default is 5.

    Returns
    -------
    catalog_cleaned : pd.Dataframe
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

    # Load fits file
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
    mask: ndarray
        Image mask. True values indicate masked pixels that will be excluded from any calculations.
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
        background across the array. If Background2D fails for any reason, a simpler
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

            # Create a simple source mask
            threshold = detect_threshold(imgarr, nsigma=3., mask=mask,
                                         sigma_clip=SigmaClip(sigma=3.0, maxiters=None))

            segment_img = detect_sources(imgarr, threshold, mask=mask,
                                         npixels=9, connectivity=8)

            src_mask = segment_img.make_source_mask(footprint=None)

            if mask is not None:
                src_mask |= mask

            # Estimate the background
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

        # Detect the sources
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

        # Create background frame shaped like imgarr populated with sigma-clipped median value
        bkg_background = np.full_like(imgarr, bkg_median).astype('float32')

        # Create background frame shaped like imgarr populated with sigma-clipped standard deviation value
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


def compute_radius(wcs_obj, naxis1, naxis2,
                   ra=None, dec=None):
    """Compute the radius from the center to the furthest edge of the WCS.

    Parameters
    -----------
    wcs_obj : WCS
        World coordinate system object describing translation between image and skycoord.
    naxis1 : int
        Axis length used to calculate the image footprint.
    naxis2 : int
        Axis length used to calculate the image footprint.
    ra : float, optional
        Right Ascension (RA) of center of field-of-view (in decimal degrees)
    dec: float, optional
        Declination (Dec) of center of field-of-view (in decimal degrees)

    Returns
    -------
    radius : float
        Radius of field-of-view in arcmin.
    """
    if ra is None and dec is None:
        ra, dec = wcs_obj.wcs.crval

    # Set image center coordinates
    img_center = SkyCoord(ra=ra * u.degree, dec=dec * u.degree)

    # Calculate footprint
    wcs_foot = wcs_obj.calc_footprint(axes=(naxis1, naxis2))

    # Get corners
    img_corners = SkyCoord(ra=wcs_foot[:, 0] * u.degree,
                           dec=wcs_foot[:, 1] * u.degree)

    # Make sure the radius is less than 1 deg because of GAIAdr3 search limit
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

    # Translate the old column name into the new standardized name
    # for each column specified in this dict
    for new, old in cat_dict.items():
        if new != 'epoch':
            # Check if the object id is lower or upper case
            if new == 'objID' and ("source_id" in table.colnames or "SOURCE_ID" in table.colnames):
                match_upper = [col for col in table.colnames if old.upper() == col]
                if not match_upper:
                    old = old.lower()
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
                # Make sure 'epoch' is decimal year and add it as a new column
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
    """
    Make a border mask for an image.
    """
    if not isinstance(borderLen, int):
        borderLen = int(borderLen)

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
        # Compute the 2D background
        bkg_arr, _, _, _ = compute_2d_background(imgarr,
                                                 mask=img_mask,
                                                 box_size=config['BKG_BOX_SIZE'],
                                                 win_size=config['BKG_MED_WIN_SIZE'],
                                                 estimate_bkg=config['estimate_bkg'],
                                                 bkg_fname=config['bkg_fname'],
                                                 silent=silent)

        # Subtract the extracted background from the image
        imgarr_bkg_subtracted = imgarr - bkg_arr

    # Apply bad pixel mask if present
    if img_mask is not None:
        imgarr_bkg_subtracted[img_mask] = 0  # np.nan  #bkg_rms_median

    if not silent:
        if source_catalog is not None:
            log.info("  > Build source catalog from known sources")
        else:
            log.info("  > Build source catalog from detected sources")

    # Build the source catalog
    fwhm, source_cat, state, fail_msg = build_source_catalog(data=imgarr_bkg_subtracted,
                                                             input_catalog=source_catalog,
                                                             known_fwhm=known_fwhm,
                                                             mask=img_mask, **config)

    del imgarr, imgarr_bkg_subtracted
    gc.collect()

    if not state or len(source_cat) == 0:
        return None, None, False, fail_msg

    return source_cat, fwhm, True, None


def download_astro_ref_cat(ra, dec, sr=0.5,
                           epoch=None, catalog='GAIADR3',
                           mag_lim=-1, silent=False):
    """ Download reference catalog via VO web service.

    Fetch astrometric source catalog entries from the specified ``SERVICELOCATION``
    using a cone-search around the given sky position.

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
        Name of catalog to query, as defined by web-service. Default: 'GAIADR3'
    mag_lim : float or int, optional
        Magnitude limit for the sources to be retrieved. Default is -1 (no limit).
    silent : bool, optional
        Set to True to suppress most console output

    Returns
    -------
    csv : CSV object
        A CSV object of returned sources with all columns as provided by catalog
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

        # Create the query
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

        # Run the query
        j = Gaia.launch_job_async(query)
        ref_table = j.get_results()

    # Add the catalog name as metadata
    ref_table.meta['catalog'] = catalog
    ref_table.meta['epoch'] = epoch

    # Convert a common set of columns into standardized column names
    ref_table = convert_astrometric_table(ref_table, catalog)

    # Sort the table by magnitude, fainter to brightest
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
    ra : float
        Right Ascension (RA) of center of field-of-view (in decimal degrees)
    dec : float
        Declination (Dec) of center of field-of-view (in decimal degrees)
    sr : float, optional
        Search radius (in decimal degrees) from field-of-view center to use
        for sources from catalog. Default: 0.1 degrees
    epoch : float, optional
        Catalog positions returned for this field-of-view will have their
        proper motions applied to represent their positions at this date, if
        a value is specified at all, for catalogs with proper motions.
    num_sources : int, None, optional
        Maximum number of the brightest/faintest sources to return in catalog.
        If `num_sources` is negative, return that number of the faintest
        sources. By default, all sources are returned.
    catalog : str, optional
        Name of catalog to query, as defined by web-service. Default: 'GSC243'
    full_catalog : bool, optional
        Return the full set of columns provided by the web service.
    silent : bool, optional
        Set to True to suppress most console output

    Returns
    -------
    csv : CSV object
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
    log.info("Getting catalog using: \n    {}".format(serviceUrl))
    rawcat = requests.get(serviceUrl, headers=headers)
    print(rawcat)
    # Convert from bytes to a String
    r_contents = rawcat.content.decode()
    rstr = r_contents.split('\r\n')
    print(rstr)
    # Remove initial line describing the number of sources returned
    # CRITICAL to proper interpretation of CSV data
    if rstr[0].startswith('Error'):
        # Try again without EPOCH
        serviceUrl = f'{bc.SERVICELOCATION}/{serviceType}?{base_spec}'
        log.info(f"Getting catalog using: \n    {serviceUrl}")
        rawcat = requests.get(serviceUrl, headers=headers)
        r_contents = rawcat.content.decode()  # Convert from bytes to a String
        rstr = r_contents.split('\r\n')

    # If there is still an error returned by the web-service, report the exact error
    if rstr[0].startswith('Error'):
        log.warning(f"Catalog generation FAILED with: \n{rstr}")

    del rstr[0], rawcat, r_contents
    gc.collect()
    print(rstr)
    ref_table = Table.read(rstr, format='ascii.csv')
    print(ref_table)
    if not ref_table:
        return ref_table

    # Add catalog name as metadata
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
    print(ref_table, catalog)
    return ref_table.to_pandas(), catalog


def get_photometric_catalog(fname, loc, imgarr, hdr, wcsprm,
                            catalog, silent=False, **config):
    """

    Parameters
    ----------
    fname
    loc
    imgarr
    hdr
    wcsprm
    catalog
    silent
    config

    Returns
    -------

    """
    # Initialize logging for this user-callable function
    log.setLevel(logging.getLevelName(log.getEffectiveLevel()))

    fwhm = float(config['img_fwhm'])

    photo_ref_cat_fname = f'{loc}/{fname}_trail_img_photo_ref_cat'

    if catalog.upper() not in bc.SUPPORTED_CATALOGS:
        log.warning(f"Given photometry catalog '{catalog}' NOT SUPPORTED. "
                    "Defaulting to GSC243")
        catalog = "GSC243"

    if config["photo_ref_cat_fname"] is not None:
        photo_ref_cat_fname = config["photo_ref_cat_fname"]

    # Get the observation date
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

    # Set RA and DEC using the image center
    wcs = WCS(hdr)

    ra, dec = wcs.all_pix2world(hdr['NAXIS1'] / 2.,
                                hdr['NAXIS2'] / 2., 0)

    # Estimate the radius of the FoV for ref. catalog creation
    fov_radius = compute_radius(wcs,
                                naxis1=hdr['NAXIS1'],
                                naxis2=hdr['NAXIS2'],
                                ra=ra, dec=dec)

    # Check for source catalog file. If present and not force extraction, use these catalogs
    chk_ref_cat_photo = os.path.isfile(photo_ref_cat_fname + '.cat')

    read_ref_cat_photo = True
    if config["force_download"] or not chk_ref_cat_photo:
        read_ref_cat_photo = False

    # Get photometric reference catalog
    if read_ref_cat_photo:
        log.info("> Load photometric references catalog from file")
        ref_tbl_photo, _, ref_catalog_photo = read_catalog(photo_ref_cat_fname)
    else:
        # Get reference catalog
        ref_tbl_photo, ref_catalog_photo = \
            download_phot_ref_cat(ra=ra, dec=dec, sr=fov_radius,
                                  catalog=catalog,
                                  full_catalog=True, silent=silent)

        # Add positions to table
        pos_on_det = wcs.wcs_world2pix(ref_tbl_photo[["RA", "DEC"]].values, 0)  # ['pixcrd']
        ref_tbl_photo["xcentroid"] = pos_on_det[:, 0]
        ref_tbl_photo["ycentroid"] = pos_on_det[:, 1]

        # Remove non-stars from catalog
        if 'classification' in ref_tbl_photo:
            ref_tbl_photo = ref_tbl_photo[ref_tbl_photo.classification == 0]

        # Mask positions outside the image boundaries
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

    # Get the observation date
    if ('time-obs'.upper() in hdr and 'telescop'.upper() in hdr and
            hdr['telescop'.upper()] != 'CTIO 4.0-m telescope'):
        time_string = f"{hdr['date-obs'.upper()]}T{hdr['time-obs'.upper()]}"
    else:
        if hdr['TELESCOP'] == 'FUT':
            time_string = hdr['date-beg'.upper()]
        else:
            time_string = hdr['date-obs'.upper()]

    # frmt = bc.has_fractional_seconds(time_string)
    t = pd.to_datetime(time_string,
                       format='ISO8601', utc=False)

    epoch = Time(t)

    # Get RA and DEC value
    ra, dec = wcsprm.crval

    # Set FoV
    fov_radius = config["fov_radius"]

    # Convert pointing to string for catalog name
    coo = SkyCoord(ra=ra * u.degree, dec=dec * u.degree)
    ra_str = coo.ra.to_string(unit=u.hourangle, sep='', precision=2, pad=True)
    dec_str = coo.dec.to_string(sep='', precision=2, alwayssign=True, pad=True)

    # Set default catalog name
    src_cat_fname = f'{loc}/{fname}_src_cat'
    astro_ref_cat_fname = f'{loc}/ref_cat_{ra_str}{dec_str}'

    # Check input and overwrite the defaults if necessary
    if config["src_cat_fname"] is not None:
        src_cat_fname = config["src_cat_fname"]
    if config["ref_cat_fname"] is not None:
        astro_ref_cat_fname = config["ref_cat_fname"]

    # Check for source catalog file. If present and not force extraction, use these catalogs
    chk_src_cat = os.path.isfile(src_cat_fname + '.cat')
    chk_ref_cat_astro = os.path.isfile(astro_ref_cat_fname + '.cat')

    read_src_cat = not (config["force_extract"] or not chk_src_cat)
    read_ref_cat_astro = not (config["force_download"] or not chk_ref_cat_astro)

    if read_src_cat:
        if not silent:
            log.info("> Load image source catalog from file")
        src_tbl, kernel_fwhm, _ = read_catalog(src_cat_fname)
    else:
        # Detect sources in image and get positions
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

    # Get astrometric reference catalog
    if read_ref_cat_astro:
        # Get reference catalog for precise positions from file
        log.info("> Load astrometric references catalog from file")
        ref_tbl_astro, _, ref_catalog_astro = read_catalog(astro_ref_cat_fname)

    else:
        # Get reference catalog for precise positions from the web
        ref_tbl_astro, ref_catalog_astro = \
            download_astro_ref_cat(ra=ra, dec=dec, sr=fov_radius,
                                   epoch=epoch,
                                   catalog='GAIAdr3',
                                   mag_lim=config['ref_cat_mag_lim'],
                                   silent=silent)

        if not silent:
            log.info("> Save astrometric references catalog.")
            save_catalog(cat=ref_tbl_astro, wcsprm=wcsprm,
                         out_name=astro_ref_cat_fname,
                         mode='ref_astro', catalog=ref_catalog_astro)

        # Add positions to table
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

    # Exclude non-star objects
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
    # Convert wcs parameters to wcs
    wcs = WCS(wcsprm.to_header())

    cat_out = cat.copy()

    if mode == 'src':
        # Get position on sky
        pos_on_sky = wcs.wcs_pix2world(cat[["xcentroid", "ycentroid"]], 0)
        cat_out["RA"] = pos_on_sky[:, 0]
        cat_out["DEC"] = pos_on_sky[:, 1]
        cols = ['RA', 'DEC', 'xcentroid', 'ycentroid', 'mag', 'cat_id',
                'sharpness', 'roundness1', 'roundness2',
                'npix', 'sky', 'peak', 'flux']

    else:
        # Get position on the detector
        pos_on_det = wcs.wcs_world2pix(cat[["RA", "DEC"]].values, 0)  # ['pixcrd']
        cat_out["xcentroid"] = pos_on_det[:, 0]
        cat_out["ycentroid"] = pos_on_det[:, 1]
        cols = ['RA', 'DEC', 'xcentroid', 'ycentroid', 'mag', 'objID']

    if mode == 'ref_astro':
        cat_out = cat_out[cols]

    # Convert to astropy.Table and add meta-information
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
    # Check for error column
    has_mag_conv = False

    # Make a copy to work with
    cat_srt = ref_cat.copy()

    # Exclude non-star sources from GSC2.4
    if 'classification' in ref_cat:
        cat_srt = ref_cat.drop(ref_cat[ref_cat.classification > 0].index)

    # Reset indices
    cat_srt = cat_srt.reset_index()

    # Convert FITS filter to catalog filter and error (if available)
    prim_filter_keys = bc.CATALOG_FILTER_EXT[catalog][band]['Prim']
    alt_filter_keys = bc.CATALOG_FILTER_EXT[catalog][band]['Alt']

    # flatten the list of filter keys
    flat_filter_keys = sum(prim_filter_keys, [])

    # Remove rows with NaN values
    df = cat_srt.dropna(subset=flat_filter_keys)

    # Filter zero error values
    if band == 'w':
        non_zero_error_condition = (df[flat_filter_keys[0][1]] != 0.) & (df[flat_filter_keys[1][1]] != 0.)
    else:
        non_zero_error_condition = df[flat_filter_keys[1]] != 0.

    # Apply the non-zero error condition
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

    # Select the std stars by number
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
#     # Only search sources in a circle with radius <vignette>
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
