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

from astropy.convolution import (
    convolve, Tophat2DKernel)
from astropy.nddata import NDData

from astroquery.gaia import Gaia

from lmfit import Model

from photutils.segmentation import (
    detect_sources, detect_threshold)
from photutils.detection import (
    DAOStarFinder)
from photutils.psf import extract_stars
from photutils.background import (
    Background2D,  # For estimating the background
    SExtractorBackground, StdBackgroundRMS, MMMBackground, MADStdBackgroundRMS,
    BkgZoomInterpolator)
from photutils.aperture import CircularAperture

import scipy.spatial as spsp
from scipy.spatial import KDTree
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
from . import base_conf as _base_conf

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


def moffat2d(x, y, amp, xc, yc, alpha, beta, sky):
    """2D Moffat function"""

    M = amp * (1. + ((x - xc) ** 2 + (y - yc) ** 2) / alpha ** 2) ** -beta + sky

    return M


def resid_func(x, y, amp, xc, yc, alpha, beta, sky):
    """ Residual function for Moffat"""
    M = moffat2d(x, y, amp, xc, yc, alpha, beta, sky)
    return M


def remove_close_elements(df, distance_threshold):
    """Remove elements within a given radius around each point"""

    # make sure the faintest objects are first
    df_sorted = df.sort_values(by=df.columns[5], ascending=False)

    points = df_sorted[['xcentroid', 'ycentroid']].to_numpy()

    tree = KDTree(points)
    close_points = tree.query_ball_point(points, distance_threshold)
    mask = np.ones(len(df_sorted), dtype=bool)
    for i, point in enumerate(close_points):
        if len(point) > 1:
            mask[i] = False

    df_masked = df_sorted[mask]
    df_masked = df_masked.sort_values(by=df.columns[5], ascending=True)

    return df_masked


def auto_build_source_catalog(data,
                              img_std,
                              use_catalog=None,
                              fwhm=None,
                              source_box_size=25,
                              fwhm_init_guess=4., threshold_value=25,
                              min_source_no=3,
                              max_source_no=1000,
                              fudge_factor=5,
                              fine_fudge_factor=0.1,
                              max_iter=50, fitting_method='least_square',
                              lim_threshold_value=3.,
                              default_moff_beta=4.765,
                              min_good_fwhm=1,
                              max_good_fwhm=30,
                              sigmaclip_fwhm_sigma=3.,
                              isolate_sources_fwhm_sep=5., init_iso_dist=25.,
                              sat_lim=65536.,
                              silent=False):
    """ Automatically detect and extract sources.

    Credit: https://github.com/Astro-Sean/autophot/blob/master/autophot/packages/find.py

    """

    # Initialize logging for this user-callable function
    log.setLevel(logging.getLevelName(log.getEffectiveLevel()))

    # Try to use PSF derived from the image as detection kernel
    # The kernel must be derived from well-isolated sources not near the edge of the image
    # kernel_psf = False
    using_catalog_sources = False
    if use_catalog is not None:
        using_catalog_sources = True
    init_fwhm = fwhm_init_guess
    if fwhm is not None:
        init_fwhm = fwhm

    # initial r_squared
    init_r2 = 0.7

    # decrease
    m = 0

    # increase
    n = 0

    decrease_increment = False

    # backstop
    failsafe = 0

    # check if a few more sources can be found
    check = False
    check_len = -np.inf

    update_guess = False
    brightest = 25

    x_pix = np.arange(0, source_box_size)
    y_pix = np.arange(0, source_box_size)
    xx, yy = np.meshgrid(x_pix, y_pix)

    model_func = Model(resid_func, independent_vars=('x', 'y'))

    model_func.set_param_hint(name='xc', value=source_box_size / 2.,
                              min=0, max=source_box_size)
    model_func.set_param_hint(name='yc', value=source_box_size / 2.,
                              min=0, max=source_box_size)

    model_func.set_param_hint(name='alpha', value=3.,
                              min=0., max=30.)
    model_func.set_param_hint(name='beta', value=default_moff_beta,
                              vary=False)
    model_func.set_param_hint(name='fwhm',
                              min=min_good_fwhm, max=max_good_fwhm,
                              expr='2 * alpha * sqrt(2**(1 / beta) - 1)')

    from scipy import ndimage as nd
    nddata = NDData(data=data)
    isolated_sources = []
    while True:
        if using_catalog_sources:
            sources = use_catalog
            sources.reset_index(drop=True, inplace=True)
        else:

            if failsafe > max_iter:
                break
            else:
                failsafe += 1

            # check if the threshold value is still good
            threshold_value_check = threshold_value + n - m

            # if <=0 reverse previous drop and fine_fudge factor
            if threshold_value_check < lim_threshold_value and not decrease_increment:
                log.warning(
                    '    Threshold value has gone below background limit [%d sigma] - '
                    'increasing by smaller increment ' % lim_threshold_value)

                # revert previous decrease
                decrease_increment = True
                m = fudge_factor

            elif threshold_value_check < lim_threshold_value and decrease_increment:
                log.critical('    FWHM detection failed - cannot find suitable threshold value ')
                return None, None, False, ['', '', '']
            else:
                threshold_value = round(threshold_value + n - m, 3)

            # if the threshold goes negative, use a smaller fudge factor
            if decrease_increment:
                fudge_factor = fine_fudge_factor
            # print(sat_lim)
            daofind = DAOStarFinder(fwhm=init_fwhm,
                                    threshold=threshold_value * img_std,
                                    exclude_border=True,
                                    peakmax=sat_lim,
                                    brightest=brightest)

            sources = daofind(data)
            if sources is None:
                log.warning('    Sources == None at %.1f sigma - decreasing threshold' % threshold_value)
                m = fudge_factor
                continue

            # Sort based on flux to identify the brightest sources for use as a kernel
            sources.sort('flux', reverse=True)
            sources = sources.to_pandas()

        if not silent:
            log.info('    Number of sources before cleaning [ %.1f sigma ]: %d ' % (threshold_value, len(sources)))

        if len(sources) == 0 and not using_catalog_sources:
            log.warning('No sources')
            m = fudge_factor
            continue

        if check and len(sources) >= 10 and not using_catalog_sources:
            pass

        elif check and len(sources) > check_len and not using_catalog_sources:
            if not silent:
                log.info('    More sources found - trying again')

            m = fudge_factor
            check_len = len(sources)
            check = False

            continue

        # Last ditch attempt to recover a few more sources
        elif len(sources) < 5 and not check and not using_catalog_sources:
            if not silent:
                log.info('    Sources found but attempting to go lower')
            m = fudge_factor
            check_len = len(sources)
            check = True
            continue

        if not update_guess:
            if not silent:
                log.info('    Updating search FWHM value')

            new_fwhm_guess = []
            new_r_squared_guess = []
            src_tmp = sources.copy()
            no_sources = 150 if len(src_tmp.index.values) > 150 else len(src_tmp.index.values)
            # src_tmp = src_tmp.head(no_sources)

            for i in list(src_tmp.index.values):
                try:
                    idx = src_tmp.index.values[i]

                    stars_tbl = Table()
                    stars_tbl['x'] = src_tmp['xcentroid'].loc[[idx]]
                    stars_tbl['y'] = src_tmp['ycentroid'].loc[[idx]]
                    stars = extract_stars(nddata, stars_tbl, size=source_box_size)
                    stars = stars.data
                    stars[stars < 0] = 0.

                    snr_stars = np.nansum(stars) / np.sqrt(np.nanstd(stars) * source_box_size ** 2
                                                           + ((1. / 2.) ** 2) * source_box_size ** 2)

                    if np.nanmax(stars) >= sat_lim or np.isnan(np.max(stars)):
                        continue

                    model_func.set_param_hint('amp', value=np.nanmax(stars),
                                              min=1e-3, max=1.5 * np.nanmax(stars))
                    _, _, sky = sigma_clipped_stats(data=stars, sigma=3.)
                    model_func.set_param_hint('sky', value=sky)
                    fit_params = model_func.make_params()

                    result = model_func.fit(data=stars, x=xx, y=yy,
                                            method=fitting_method,
                                            params=fit_params,
                                            calc_covar=True, scale_covar=True,
                                            nan_policy='omit', max_nfev=100)

                    fwhm_fit = (2. * result.params['alpha']
                                * np.sqrt((2. ** (1. / result.params['beta'])) - 1.))
                    # or (result.params['amp'] >= sat_lim) \
                    if (max_good_fwhm <= fwhm_fit <= min_good_fwhm) \
                            or (result.params['fwhm'].value is None) \
                            or (snr_stars <= 0.):
                        new_fwhm_guess.append(np.nan)
                        continue

                    new_r_squared_guess.append(result.rsquared)
                    new_fwhm_guess.append(fwhm_fit)
                    m = 0

                    if len(new_fwhm_guess) == no_sources:
                        break

                    # print(new_fwhm_guess)
                except (Exception,):
                    pass

            new_r_squared_guess = np.array(new_r_squared_guess)

            new_fwhm_guess = np.array(new_fwhm_guess)
            all_nan_check = (np.where(new_fwhm_guess == np.nan, True, False)).all()
            if all_nan_check and not decrease_increment and not using_catalog_sources:
                log.warning('    No FWHM values - decreasing threshold')
                m = fudge_factor
                continue
            else:

                init_fwhm = np.nanpercentile(new_fwhm_guess, 50.,
                                             method='median_unbiased')

                if ~np.isnan(init_fwhm):  # and len(new_fwhm_guess[~np.isnan(new_fwhm_guess)]) >= 3:
                    if not silent:
                        log.info('    Updated guess for FWHM: %.1f pixels ' % init_fwhm)

                    init_r2 = np.nanpercentile(new_r_squared_guess, 50.,
                                               method='median_unbiased')  # 84.135

                    if not silent:
                        log.info('    Updated limit for R-squared: %.3f' % init_r2)

                    brightest = None
                    update_guess = True
                    continue
                if not using_catalog_sources:
                    log.warning('    Not enough FWHM values - decreasing threshold')
                    m = fudge_factor
                    continue

        if len(sources) > max_source_no and not using_catalog_sources:
            log.warning('    Too many sources - increasing threshold')
            if n == 0:
                threshold_value *= 2
            elif m != 0:
                decrease_increment = True
                n = fine_fudge_factor
                fudge_factor = fine_fudge_factor
            else:
                n = fudge_factor
            continue

        elif len(sources) > 5000 and m != 0 and not using_catalog_sources:
            log.warning('    Picking up noise - increasing threshold')
            fudge_factor = fine_fudge_factor
            n = fine_fudge_factor
            m = 0
            decrease_increment = True
            continue

        elif len(sources) < min_source_no and not decrease_increment and not using_catalog_sources:
            log.warning('    Too few sources - decreasing threshold')
            m = fudge_factor
            continue

        elif len(sources) == 0 and not using_catalog_sources:
            log.warning('    No sources - decreasing threshold')
            m = fudge_factor
            continue

        len_with_boundary = len(sources)
        hsize = (source_box_size - 1) // 2
        boundary_mask = ((sources['xcentroid'] > hsize) & (sources['xcentroid'] < (data.shape[1] - 1 - hsize)) &
                         (sources['ycentroid'] > hsize) & (sources['ycentroid'] < (data.shape[0] - 1 - hsize)))
        sources = sources[boundary_mask]
        n_near_boundary = len_with_boundary - len(sources)
        if n_near_boundary > 0:
            if not silent:
                log.info(f'    Sources removed near boundary: {n_near_boundary:d}')

        isolated_sources = remove_close_elements(sources, init_iso_dist)

        # src_positions = np.array(list(zip(sources['xcentroid'], sources['ycentroid'])))
        # iso_src_positions = np.array(list(zip(isolated_sources['xcentroid'], isolated_sources['ycentroid'])))
        # print(src_positions)
        # plt.figure()
        # plt.imshow(data)
        # plt.scatter(src_positions[:,0], src_positions[:,1], c='r', alpha=0.5)
        # plt.scatter(iso_src_positions[:,0], iso_src_positions[:,1], c='b', alpha=0.5)
        # plt.show()
        n_crowded = len(sources) - len(isolated_sources)
        if n_crowded > 0:
            log.info(f'    Crowded sources removed: {n_crowded:d}')

        del sources

        v = isolated_sources[['xcentroid', 'ycentroid']]
        dist = euclidean_distances(v, v)
        dist = np.floor(dist)
        minval = np.min(np.where(dist == 0., dist.max(), dist), axis=0)
        isolated_sources['min_sep'] = minval
        isolated_sources.reset_index(drop=True, inplace=True)

        if len(isolated_sources) < min_source_no:
            log.warning('    Less than min source available after isolating sources')
            m = fudge_factor
            continue

        if not silent:
            log.info(f'    Source for FWHM estimation: {len(isolated_sources.index)}')
            log.info('    Run fit (This may take a second.)')

        col_names = ['fwhm', 'fwhm_err', 'median']
        result_arr = np.empty((len(isolated_sources.index), len(col_names)))
        nan_arr = np.array([[np.nan] * 3])
        saturated_source = 0
        high_fwhm = 0
        for i in range(len(isolated_sources.index)):

            idx = isolated_sources.index.values[i]
            x0 = isolated_sources['xcentroid'].loc[[idx]]
            y0 = isolated_sources['ycentroid'].loc[[idx]]

            stars_tbl = Table()
            stars_tbl['x'] = x0
            stars_tbl['y'] = y0

            stars = extract_stars(nddata, stars_tbl, size=source_box_size)
            stars = stars.data
            stars[stars < 0] = 0.

            if np.nanmax(stars) >= sat_lim or np.isnan(np.max(stars)):
                result_arr[i] = nan_arr
                continue
            # plt.figure()
            # plt.imshow(stars)
            # plt.show()
            try:
                model_func.set_param_hint('amp', value=np.nanmax(stars),
                                          min=1e-3, max=1.5 * np.nanmax(stars))
                _, _, sky = sigma_clipped_stats(data=stars, sigma=3.)
                # print(sky)
                model_func.set_param_hint('sky', value=sky)

                alpha = init_fwhm / (2 * np.sqrt(2 ** (1 / default_moff_beta) - 1))
                model_func.set_param_hint('alpha', value=alpha)

                fit_params = model_func.make_params()

                result = model_func.fit(data=stars, x=xx, y=yy,
                                        method=fitting_method,
                                        params=fit_params, calc_covar=True,
                                        scale_covar=True,
                                        nan_policy='omit', max_nfev=100)

                fwhm_fit = 2. * result.params['alpha'] * np.sqrt((2. ** (1. / result.params['beta'])) - 1.)
                fwhm_fit_err = result.params['fwhm'].stderr

                A = result.params['amp'].value
                A_err = result.params['amp'].stderr
                x_fitted = result.params['xc'].value
                y_fitted = result.params['yc'].value
                bkg_approx = result.params['sky'].value
                # print(A, bkg_approx, fwhm_fit, fwhm_fit_err, result.rsquared, init_r2)
                to_add = nan_arr
                if fwhm_fit_err is not None:
                    corrected_x = x_fitted - source_box_size / 2 + x0
                    corrected_y = y_fitted - source_box_size / 2 + y0
                    if A > sat_lim:
                        # to_add = nan_arr
                        saturated_source += 1
                    elif max_good_fwhm - 1 <= fwhm_fit <= min_good_fwhm:
                        to_add = nan_arr
                        high_fwhm += 1
                    elif A <= A_err or A_err is None:
                        to_add = nan_arr
                    elif result.rsquared < 0.95 * init_r2:
                        to_add = nan_arr
                    else:
                        to_add = np.array([fwhm_fit, fwhm_fit_err, bkg_approx])
                        isolated_sources['xcentroid'] = isolated_sources['xcentroid'].replace([x0], corrected_x)
                        isolated_sources['ycentroid'] = isolated_sources['ycentroid'].replace([y0], corrected_y)

            except (Exception,):
                to_add = nan_arr

            result_arr[i] = to_add

        if saturated_source != 0:
            log.info(f'    Saturated sources removed: {saturated_source:d}')

        if high_fwhm != 0:
            log.info(f'    Sources with bad fwhm '
                     f'[limit: {min_good_fwhm:d},{max_good_fwhm:d} [pixels]: {high_fwhm:d}')

        isolated_sources = pd.concat(
            [
                isolated_sources,
                pd.DataFrame(
                    result_arr,
                    index=isolated_sources.index,
                    columns=col_names
                )
            ], axis=1
        )

        isolated_sources.reset_index(inplace=True, drop=True)
        if not using_catalog_sources:

            if len(isolated_sources['fwhm'].values) == 0:
                log.warning('    No sigma values taken')
                continue

            if len(isolated_sources) < min_source_no:
                log.warning('    Less than min source after sigma clipping: %d' % len(isolated_sources))
                threshold_value += m
                if n == 0:
                    decrease_increment = True
                    n = fine_fudge_factor
                    fudge_factor = fine_fudge_factor
                else:
                    n = fudge_factor
                m = 0
                continue

            FWHM_mask = sigma_clip(isolated_sources['fwhm'].values,
                                   sigma=sigmaclip_fwhm_sigma,
                                   masked=True,
                                   maxiters=10,
                                   cenfunc=np.nanmedian,
                                   stdfunc=mad_std)

            if np.sum(FWHM_mask.mask) == 0 or len(isolated_sources) < 5:
                isolated_sources['include_fwhm'] = [True] * len(isolated_sources)
                fwhm_array = isolated_sources['fwhm'].values
            else:
                fwhm_array = isolated_sources[~FWHM_mask.mask]['fwhm'].values
                isolated_sources['include_fwhm'] = ~FWHM_mask.mask
                # log.info('    Removed %d FWHM outliers' % (np.sum(FWHM_mask.mask)))

            median_mask = sigma_clip(isolated_sources['median'].values,
                                     sigma=3.,
                                     masked=True,
                                     maxiters=10,
                                     cenfunc=np.nanmedian,
                                     stdfunc=mad_std)

            if np.sum(median_mask) == 0 or np.sum(~median_mask.mask) < 5:
                isolated_sources['include_median'] = [True] * len(isolated_sources)
                pass

            else:
                isolated_sources['include_median'] = ~median_mask.mask
                # log.info('    Removed %d median outliers' % (np.sum(median_mask.mask)))

            if not silent:
                log.info('    Usable sources found [ %d sigma ]: %d' % (threshold_value,
                                                                        len(isolated_sources)))

            image_fwhm = np.nanmedian(fwhm_array)
            if len(fwhm_array) == 0:
                log.warning('    Less than min source after sigma clipping: %d' % len(isolated_sources))
                threshold_value -= n

                if m == 0:
                    decrease_increment = True
                    m = fine_fudge_factor
                    fudge_factor = fine_fudge_factor
                else:
                    m = fudge_factor

                n = 0
                continue

            if len(isolated_sources) > 3:
                too_close = (isolated_sources['min_sep'] <= isolate_sources_fwhm_sep * image_fwhm)
                isolated_sources = isolated_sources[~too_close]
                log.info(f'    Sources within minimum separation '
                         f'[ {(isolate_sources_fwhm_sep * image_fwhm):.0f} pixel ]: {too_close.sum():d}')
        break

    # isolated_sources = isolated_sources.dropna(subset=['fwhm', 'fwhm_err'], inplace=False)
    FWHM_mask = sigma_clip(isolated_sources['fwhm'].values,
                           sigma=sigmaclip_fwhm_sigma,
                           masked=True,
                           maxiters=10,
                           cenfunc=np.nanmedian,
                           stdfunc=np.nanstd)

    if np.sum(FWHM_mask.mask) == 0 or len(isolated_sources) < 5:
        isolated_sources['include_fwhm'] = [True] * len(isolated_sources)
        fwhm_array = isolated_sources['fwhm'].values
    else:
        fwhm_array = isolated_sources[~FWHM_mask.mask]['fwhm'].values
        isolated_sources['include_fwhm'] = ~FWHM_mask.mask
        log.info(f'    FWHM outliers removed: {np.sum(FWHM_mask.mask):d}')

    image_fwhm = np.nanmedian(fwhm_array)
    image_fwhm_err = np.nanstd(fwhm_array)

    if not silent:
        log.info(f'    FWHM: {image_fwhm:.3f} +/- {image_fwhm_err:.3f} [ pixels ]')
    isolated_sources['cat_id'] = np.arange(1, len(isolated_sources) + 1)

    return (image_fwhm, image_fwhm_err), isolated_sources, True, None


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


def clean_catalog_distance(in_cat: pd.DataFrame,
                           fwhm: float, r: float = 5.,
                           init_dist: float = None,
                           id_name: str = 'id') -> Table:
    """Remove sources that are close to each other"""

    if init_dist is None:
        radius = r * fwhm if init_dist is None else init_dist
    else:
        radius = init_dist

    coords = in_cat[['xcentroid', 'ycentroid']].to_numpy()
    distances = spsp.distance_matrix(coords, coords)
    in_cat['dist'] = distances.tolist()

    # CREATES d0, d1, d2, d3, ... COLUMNS
    dist_cols = ['d' + str(i) for i in range(len(in_cat['ycentroid']))]
    df = in_cat['dist'].apply(pd.Series)
    df.columns = df.columns.map(lambda x: dist_cols[x])
    df_ = pd.concat([in_cat, df], axis=1)

    # RESHAPE DATA LONG
    melted_df = df_.melt(id_vars=[id_name, 'xcentroid', 'ycentroid'],
                         value_vars=np.array(dist_cols),
                         var_name='dist', value_name='dist_val', ignore_index=False)

    # FILTER FOR DISTANCES (0, r)
    unwanted_indices = melted_df[melted_df['dist_val'].between(0.,
                                                               radius,
                                                               inclusive='neither')].index
    in_cat_cln = in_cat.drop(unwanted_indices, axis=0)

    del in_cat, unwanted_indices, df, df_
    gc.collect()

    in_cat_cln['min_sep'] = in_cat_cln.apply(lambda row:
                                             np.array(row['dist'])[pd.Index(np.array(row['dist']) > 0)].min(), axis=1)
    del in_cat_cln['dist']
    return in_cat_cln


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
        NDarray of science data for which the background needs to be computed
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
        background across the array.  If Background2D fails for any reason, a simpler
        sigma-clipped single-valued array will be computed instead.
    bkg_median:
        The median value (or single sigma-clipped value) of the computed background.
    bkg_rms:
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
        if not silent:
            log.info(f"    Percentile in use: {percentile}")

        try:

            # create a simple source mask
            threshold = detect_threshold(imgarr, nsigma=3., mask=mask,
                                         sigma_clip=SigmaClip(sigma=3.0, maxiters=None))

            segment_img = detect_sources(imgarr, threshold, mask=mask,
                                         npixels=9, connectivity=8)

            footprint = None
            src_mask = segment_img.make_source_mask(footprint=footprint)

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
            log.warning("     Background2D failure detected. "
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
    if cat_name in _base_conf.SUPPORTED_CATALOGS:
        cat_dict = _base_conf.SUPPORTED_CATALOGS[cat_name]
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


def dilate_mask(mask, tophat_size: int) -> np.ndarray:
    """ Take a mask and make the masked regions bigger."""

    area = np.pi * tophat_size ** 2.
    kernel = Tophat2DKernel(tophat_size)
    dilated_mask = convolve(mask, kernel) >= 1. / area

    del kernel, mask
    gc.collect()

    return dilated_mask


def extract_source_catalog(imgarr,
                           use_catalog=None,
                           known_fwhm=None,
                           vignette=-1, vignette_rectangular=-1.,
                           cutouts=None, only_rectangle=None,
                           silent=False, **config):
    """ Extract and source catalog using photutils.

    The catalog returned by this function includes sources found in the
    input image with the positions translated to the coordinate frame
    defined by the reference WCS `refwcs`.

    Parameters
    ----------
    imgarr: np.ndarray
        Input image as an astropy.io.fits HDUList.
    use_catalog:
    known_fwhm:
    config: dict
        Dictionary containing the configuration
    vignette: float, optional
        Cut off corners using a circle with radius (0. < vignette <= 2.). Defaults to -1.
    vignette_rectangular: float, optional
        Ignore a fraction of the image in the corner. Default: -1 = nothing ignored
        If the fraction < 1, the corresponding (1 - frac) percentage is ignored.
        Example: 0.9 ~ 10% ignored
    cutouts: list, or list of lists(s), None, optional
        Cut out rectangular regions of the image. Format: [(xstart, xend, ystart, yend)]
    only_rectangle: list, None, optional
        Use only_rectangle within image format: (xstart, xend, ystart, yend)
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
    source_box_size = config['SOURCE_BOX_SIZE']

    if config['telescope_keyword'] == 'CTIO 4.0-m telescope':
        imgarr_bkg_subtracted = imgarr
        bkg_rms_median = config['sky_noise']
    else:
        # Get default parameter
        box_size = config['BKG_BOX_SIZE']
        win_size = config['BKG_MED_WIN_SIZE']
        estimate_bkg = config['estimate_bkg']
        bkg_fname = config['bkg_fname']

        bkg_arr, _, _, bkg_rms_median = compute_2d_background(imgarr,
                                                              mask=img_mask,
                                                              box_size=box_size,
                                                              win_size=win_size,
                                                              estimate_bkg=estimate_bkg,
                                                              bkg_fname=bkg_fname,
                                                              silent=silent)

        # subtract background from the image
        imgarr_bkg_subtracted = imgarr - bkg_arr

    # apply bad pixel mask if present
    if img_mask is not None:
        imgarr_bkg_subtracted[img_mask] = bkg_rms_median

    # if not silent:
    #     log.info("  > Mask image")
    # imgarr_bkg_subtracted = mask_image(imgarr_bkg_subtracted,
    #                                    vignette=vignette,
    #                                    vignette_rectangular=vignette_rectangular,
    #                                    cutouts=cutouts,
    #                                    only_rectangle=only_rectangle)

    if not silent:
        log.info("  > Auto build source catalog")

    fwhm, source_cat, state, fail_msg = auto_build_source_catalog(data=imgarr_bkg_subtracted,
                                                                  img_std=bkg_rms_median,
                                                                  use_catalog=use_catalog,
                                                                  fwhm=known_fwhm,
                                                                  source_box_size=config['SOURCE_BOX_SIZE'],
                                                                  fwhm_init_guess=config['FWHM_INIT_GUESS'],
                                                                  threshold_value=config['THRESHOLD_VALUE'],
                                                                  min_source_no=config['SOURCE_MIN_NO'],
                                                                  max_source_no=config['SOURCE_MAX_NO'],
                                                                  fudge_factor=config['THRESHOLD_FUDGE_FACTOR'],
                                                                  fine_fudge_factor=config[
                                                                      'THRESHOLD_FINE_FUDGE_FACTOR'],
                                                                  max_iter=config['MAX_FUNC_ITER'],
                                                                  fitting_method=config['FITTING_METHOD'],
                                                                  lim_threshold_value=config['THRESHOLD_VALUE_LIM'],
                                                                  default_moff_beta=config['DEFAULT_MOFF_BETA'],
                                                                  min_good_fwhm=config['FWHM_LIM_MIN'],
                                                                  max_good_fwhm=config['FWHM_LIM_MAX'],
                                                                  sigmaclip_fwhm_sigma=config['SIGMACLIP_FWHM_SIGMA'],
                                                                  isolate_sources_fwhm_sep=config[
                                                                      'ISOLATE_SOURCES_FWHM_SEP'],
                                                                  init_iso_dist=config['ISOLATE_SOURCES_INIT_SEP'],
                                                                  sat_lim=config['sat_lim'])

    if not state or len(source_cat) == 0:
        del imgarr, imgarr_bkg_subtracted
        gc.collect()
        return None, None, None, None, None, False, fail_msg

    del imgarr, imgarr_bkg_subtracted
    gc.collect()

    return source_cat, None, None, None, fwhm, True, None


def get_reference_catalog_astro(ra, dec, sr: float = 0.5,
                                epoch: Time = None, catalog: str = 'GAIADR3',
                                mag_lim: float | int = -1, silent: bool = False):
    """ Extract reference catalog from VO web service.
    Queries the catalog available at the ``SERVICELOCATION`` specified
    for this module to get any available astrometric source catalog entries
    around the specified position in the sky based on a cone-search.

    todo: update this!!!

    Parameters
    ----------
    mag_lim
    ra: float
        Right Ascension (RA) of center of field-of-view (in decimal degrees)
    dec: float
        Declination (Dec) of center of field-of-view (in decimal degrees)
    sr: float, optional
        Search radius (in decimal degrees) from field-of-view center to use
        for sources from catalog. Default: 0.5 degrees
    epoch: float, optional
        Catalog positions returned for this field-of-view will have their
        proper motions applied to represent their positions at this date, if
        a value is specified at all, for catalogs with proper motions.
    catalog: str, optional
        Name of catalog to query, as defined by web-service.  Default: 'GSC242'
    silent: bool, optional
        Set to True to suppress most console output

    Returns
    -------
    csv: CSV object
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
            FROM {_base_conf.DEF_ASTROQUERY_CATALOGS[catalog.upper()]} AS g
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


def get_reference_catalog_phot(ra, dec, sr=0.1, epoch=None,
                               num_sources=None,
                               catalog='GSC243',
                               full_catalog=False, silent=False):
    """ Extract reference catalog from VO web service.
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
        CSV object of returned sources with all columns as provided by catalog
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

    url_chk = url_checker(f'{_base_conf.SERVICELOCATION}/{serviceType}')
    if not url_chk[0]:
        log.error(f"  {url_chk[1]}. ")
        sys.exit(1)
    if not silent:
        log.info(f"  {url_chk[1]}. Downloading data... This may take a while!!! Don't panic")

    serviceUrl = f'{_base_conf.SERVICELOCATION}/{serviceType}?{spec}'
    log.debug("Getting catalog using: \n    {}".format(serviceUrl))
    rawcat = requests.get(serviceUrl, headers=headers)

    # convert from bytes to a String
    r_contents = rawcat.content.decode()
    rstr = r_contents.split('\r\n')

    # remove initial line describing the number of sources returned
    # CRITICAL to proper interpretation of CSV data
    if rstr[0].startswith('Error'):
        # Try again without EPOCH
        serviceUrl = f'{_base_conf.SERVICELOCATION}/{serviceType}?{base_spec}'
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

    if catalog.upper() not in _base_conf.SUPPORTED_CATALOGS:
        log.warning(f"Given photometry catalog '{catalog}' NOT SUPPORTED. "
                    "Defaulting to GSC243")
        catalog = "GSC243"

    if config["_photo_ref_cat_fname"] is not None:
        photo_ref_cat_fname = config["_photo_ref_cat_fname"]

    # get the observation date
    # if ('time-obs'.upper() in hdr and 'telescop'.upper() in hdr and
    #         hdr['telescop'.upper()] != 'CTIO 4.0-m telescope'):
    #     time_string = f"{hdr['date-obs'.upper()]}T{hdr['time-obs'.upper()]}"
    # else:
    #     time_string = hdr['date-obs'.upper()]
    #
    # frmt = _base_conf.has_fractional_seconds(time_string)
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
    if config["_force_download"] or not chk_ref_cat_photo:
        read_ref_cat_photo = False

    # get photometric reference catalog
    if read_ref_cat_photo:
        log.info("> Load photometric reference source catalog from file")
        ref_tbl_photo, _, ref_catalog_photo = read_catalog(photo_ref_cat_fname)
    else:
        # get reference catalog
        ref_tbl_photo, ref_catalog_photo = \
            get_reference_catalog_phot(ra=ra, dec=dec, sr=fov_radius,
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
            log.info("> Save photometry reference catalog.")
        save_catalog(cat=ref_tbl_photo, wcsprm=wcsprm, out_name=photo_ref_cat_fname,
                     mode='ref_photo', catalog=ref_catalog_photo)

    # todo: make it so that if not enough stars remain after extraction in the original filter
    #  the alternative mag data are used.
    #  calculate conversion for all and separate those which have both mags, original and converted
    df, std_fkeys, mag_conv, exec_state, fail_msg = select_std_stars(ref_tbl_photo,
                                                                     ref_catalog_photo,
                                                                     config['_filter_val'],
                                                                     num_std_max=config['NUM_STD_MAX'],
                                                                     num_std_min=config['NUM_STD_MIN'])

    if not exec_state:
        return None, None, None, None, False, fail_msg

    src_tbl, _, _, _, kernel_fwhm, exec_state, fail_msg = \
        extract_source_catalog(imgarr=imgarr,
                               use_catalog=df,
                               known_fwhm=fwhm,
                               vignette=config["_vignette"],
                               vignette_rectangular=config["_vignette_rectangular"],
                               cutouts=config["_cutouts"],
                               silent=silent, **config)
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
    """Extract astrometric positions and photometric data for sources in the
            input images' field-of-view.

        """

    # Initialize logging for this user-callable function
    log.setLevel(logging.getLevelName(log.getEffectiveLevel()))

    kernel = None
    segmap = None
    segmap_thld = None

    # get the observation date
    if ('time-obs'.upper() in hdr and 'telescop'.upper() in hdr and
            hdr['telescop'.upper()] != 'CTIO 4.0-m telescope'):
        time_string = f"{hdr['date-obs'.upper()]}T{hdr['time-obs'.upper()]}"
    else:
        time_string = hdr['date-obs'.upper()]

    # frmt = _base_conf.has_fractional_seconds(time_string)
    t = pd.to_datetime(time_string,
                       format='ISO8601', utc=False)

    epoch = Time(t)

    # get RA and DEC value
    ra, dec = wcsprm.crval
    # set FoV
    fov_radius = config["_fov_radius"]

    # Convert pointing to string for catalog name
    coo = SkyCoord(ra=ra * u.degree, dec=dec * u.degree)
    ra_str = coo.ra.to_string(unit=u.hourangle, sep='', precision=2, pad=True)
    dec_str = coo.dec.to_string(sep='', precision=2, alwayssign=True, pad=True)

    # set default catalog name
    src_cat_fname = f'{loc}/{fname}_src_cat'
    astro_ref_cat_fname = f'{loc}/ref_cat_{ra_str}{dec_str}'

    # check input and overwrite the defaults if necessary
    if config["_src_cat_fname"] is not None:
        src_cat_fname = config["_src_cat_fname"]
    if config["_ref_cat_fname"] is not None:
        astro_ref_cat_fname = config["_ref_cat_fname"]

    # check for source catalog file. If present and not force extraction, use these catalogs
    chk_src_cat = os.path.isfile(src_cat_fname + '.cat')
    chk_ref_cat_astro = os.path.isfile(astro_ref_cat_fname + '.cat')

    read_src_cat = not (config["_force_extract"] or not chk_src_cat)
    read_ref_cat_astro = not (config["_force_download"] or not chk_ref_cat_astro)

    if read_src_cat:
        if not silent:
            log.info("> Load image source catalog from file")
        src_tbl, kernel_fwhm, _ = read_catalog(src_cat_fname)
    else:
        # detect sources in image and get positions
        src_tbl, segmap, segmap_thld, kernel, kernel_fwhm, state, fail_msg = \
            extract_source_catalog(imgarr=imgarr,
                                   vignette=config["_vignette"],
                                   vignette_rectangular=config["_vignette_rectangular"],
                                   cutouts=config["_cutouts"],
                                   silent=silent, **config)
        if not state:
            del imgarr, src_tbl, segmap, segmap_thld, kernel, kernel_fwhm
            gc.collect()
            return (None for _ in range(9)), False, fail_msg

        if not silent:
            log.info("> Save image source catalog.")
        save_catalog(cat=src_tbl, wcsprm=wcsprm, out_name=src_cat_fname,
                     kernel_fwhm=kernel_fwhm)

    # get astrometric reference catalog
    if read_ref_cat_astro:
        # get reference catalog for precise positions from file
        log.info("> Load astrometric reference source catalog from file")
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
            log.info("> Save astrometric reference catalog.")
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
        return (None for _ in range(9)), False, None

    return (src_tbl, ref_tbl_astro, ref_catalog_astro,
            src_cat_fname, astro_ref_cat_fname,
            kernel_fwhm, kernel, segmap, segmap_thld), True, None


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

    if band not in _base_conf.SUPPORTED_BANDS:
        log.error("Given or observed filter band are not supported. "
                  "Possible filter are: {}".format(", ".join(_base_conf.SUPPORTED_BANDS.keys())))
        sys.exit(1)

    if source != "auto":
        if source not in _base_conf.SUPPORTED_CATALOGS:
            log.error("Given catalog not supported. "
                      "Possible catalogs are: "
                      "{}".format(", ".join(_base_conf.SUPPORTED_CATALOGS.keys())))
            sys.exit(1)
        else:
            if source not in _base_conf.SUPPORTED_BANDS[band]:
                log.error("Given band is not supported for this catalog. "
                          "Possible catalog for "
                          "{} band: {}".format(band,
                                               ", ".join(_base_conf.SUPPORTED_BANDS[band])))
                sys.exit(1)
    if source == 'auto':
        catalog_name = _base_conf.SUPPORTED_BANDS[band][0]

    return catalog_name


def select_std_stars(ref_cat: pd.DataFrame,
                     catalog: str, band: str,
                     num_std_max: int = None,
                     num_std_min: int = 5,
                     silent: bool = False) -> tuple:
    """ Select standard stars for aperture photometry from photometric reference catalog

    Parameters
    ----------
    ref_cat:
        Standard star catalog data
    catalog:
        Standard star catalog name
    band:
        Filter used in the observation
    num_std_max:
        Maximum number of standard stars
    num_std_min:
        Minimum number of standard stars
    silent:
        Supress outputs

    Returns
    -------
    ref_cat_filtered:
        Catalog of selected standard stars
    """

    # check for error column
    has_mag_conv = False

    # exclude non-star sources from GSC2.4
    cat_srt = ref_cat.copy()

    if 'classification' in ref_cat:
        cat_srt = ref_cat.drop(ref_cat[ref_cat.classification > 0].index)
    cat_srt = cat_srt.reset_index()

    # convert fits filter to catalog filter + error (if available)
    filter_keys = _base_conf.CATALOG_FILTER_EXT[catalog][band]['Prim']
    alt_filter_key = _base_conf.CATALOG_FILTER_EXT[catalog][band]['Alt']

    if band != 'w':
        filter_keys = sum(filter_keys, [])
        # remove nan and err=0 values
        df = cat_srt.dropna(subset=filter_keys)
        df = df[df[filter_keys[1]] != 0.]
    else:
        flat_list = sum(filter_keys, [])
        df = cat_srt.dropna(subset=flat_list)
        df = df[(df[filter_keys[0][1]] != 0.) & (df[filter_keys[1][1]] != 0.)]

    n = len(df)
    if n < num_std_min and alt_filter_key is not None and band != 'w':
        if not silent:
            log.warning(f"  ==> No or less than {num_std_min} stars "
                        f"in {filter_keys[0]} band.")
            log.warning(f"      Using alternative band with known magnitude conversion.")
        alt_filter_keys = np.asarray(alt_filter_key, str)

        alt_cat = cat_srt.dropna(subset=alt_filter_keys.flatten())

        x1 = alt_cat[alt_filter_keys[0, :]].to_numpy()
        x2 = alt_cat[alt_filter_keys[1, :]].to_numpy()
        x3 = alt_cat[alt_filter_keys[2, :]].to_numpy()

        has_data = np.any(np.array([x1, x2, x3]), axis=1).all()
        if not has_data:
            log.critical("  Insufficient data for magnitude conversion.")
            del ref_cat, cat_srt, df, alt_cat
            gc.collect()
            fail_msg = ['StdStarMagError', 'Insufficient data for magnitude conversion', '']
            return None, filter_keys, has_mag_conv, False, fail_msg

        # convert the band from catalog to observation
        alt_mags, alt_mags_err = phot.convert_ssds_to_bvri(f=filter_keys[0],
                                                           x1=x1, x2=x2, x3=x3)

        new_df = pd.DataFrame({filter_keys[0]: alt_mags,
                               filter_keys[1]: alt_mags_err}, index=alt_cat.index)

        cat_srt = cat_srt.fillna(99999999).astype(np.int64, errors='ignore')
        cat_srt = cat_srt.replace(99999999, np.nan)

        cat_srt.update(new_df)
        has_mag_conv = True

    elif band == 'w':
        alt_filter_keys = np.asarray(filter_keys, str)
        alt_cat = cat_srt.dropna(subset=alt_filter_keys.flatten())

        x1 = alt_cat[alt_filter_keys[0, :]].to_numpy()
        x2 = alt_cat[alt_filter_keys[1, :]].to_numpy()

        has_data = np.any(np.array([x1, x2]), axis=1).all()
        if not has_data:
            log.critical("Insufficient data for magnitude conversion.")
            del ref_cat, cat_srt, df, alt_cat
            gc.collect()
            fail_msg = ['StdStarMagError', 'Insufficient data for magnitude conversion', '']
            return None, filter_keys, has_mag_conv, False, fail_msg

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
    df = cat_srt.sort_values(by=[filter_keys[0]], axis=0, ascending=True, inplace=False)
    df = df.dropna(subset=filter_keys, inplace=False)
    if df.empty:
        log.critical("  No sources with uncertainties remaining!!!")
        del ref_cat, cat_srt, df
        gc.collect()
        fail_msg = ['StdStarMagError', 'No sources with uncertainties remaining', '']
        return None, filter_keys, has_mag_conv, False, fail_msg

    # select the std stars by number
    if num_std_max is not None:
        idx = num_std_max
        df = df[:idx]

    # result column names
    # cols = ['objID', 'RA', 'DEC', 'xcentroid', 'ycentroid',
    #         'fwhm', 'fwhm_err', 'include_fwhm']
    cols = ['objID', 'RA', 'DEC', 'xcentroid', 'ycentroid']
    cols += filter_keys

    df = df[cols]

    del ref_cat, cat_srt
    gc.collect()

    return df, filter_keys, has_mag_conv, True, None


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
