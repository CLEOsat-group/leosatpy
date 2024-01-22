#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Name:         photometry.py
# Purpose:      Utilities for the extraction of photometry from the image.
#
#
#
#
# Author:       p4adch (cadam)
#
# Created:      05/09/2022
# Copyright:    (c) p4adch 2010-
#
# History:
#
# 09.05.2022
# - file created and basic methods
#
# -----------------------------------------------------------------------------

""" Modules """
import gc
import os
import inspect
import logging
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
import pandas as pd

try:
    import matplotlib
    from matplotlib import pyplot as plt
    import matplotlib.lines as mlines

except ImportError:
    plt = None
else:
    import matplotlib
    import matplotlib.gridspec as gridspec  # GRIDSPEC !
    from matplotlib.ticker import (AutoMinorLocator, LogLocator)
    from astropy.visualization import (LinearStretch, LogStretch, SqrtStretch)
    from astropy.visualization.mpl_normalize import ImageNormalize
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # matplotlib parameter
    matplotlib.use('Qt5Agg')
    matplotlib.rc("lines", linewidth=1.2)
    matplotlib.rc('figure', dpi=150, facecolor='w', edgecolor='k')
    matplotlib.rc('text.latex', preamble=r'\usepackage{sfmath}')

from astropy.stats import (sigma_clipped_stats, SigmaClip, mad_std)
from photutils.aperture import (aperture_photometry,
                                ApertureStats,
                                CircularAperture,
                                CircularAnnulus,
                                RectangularAperture,
                                RectangularAnnulus)

# pipeline-specific modules
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

__taskname__ = 'photometry'
# -----------------------------------------------------------------------------

""" Parameter used in the script """
log = logging.getLogger(__name__)

MODULE_PATH = os.path.dirname(inspect.getfile(inspect.currentframe()))


# -----------------------------------------------------------------------------

def convert_ssds_to_bvri(f, x1, x2, x3):
    """Convert SSDS magnitudes to BVRI magnitudes.
    Lupton et al.(2005) BibCode: 2005AAS...20713308L

    """

    coeff_dict = _base_conf.CONV_SSDS_BVRI

    y1 = x2[:, 0] + coeff_dict[f][0][0] * (x1[:, 0] - x2[:, 0]) + coeff_dict[f][0][1]
    e_y1 = np.sqrt((coeff_dict[f][0][0] * x1[:, 1]) ** 2 +
                   ((1 - coeff_dict[f][0][0]) * x2[:, 1]) ** 2 +
                   coeff_dict[f][0][2] ** 2)
    y2 = x2[:, 0] + coeff_dict[f][1][0] * (x2[:, 0] - x3[:, 0]) + coeff_dict[f][1][1]
    e_y2 = np.sqrt((coeff_dict[f][1][0] * x3[:, 1]) ** 2 +
                   ((1 - coeff_dict[f][1][0]) * x2[:, 1]) ** 2 +
                   coeff_dict[f][1][2] ** 2)
    if f in ['B', 'I']:
        y1 = x1[:, 0] + coeff_dict[f][0][0] * (x1[:, 0] - x2[:, 0]) + coeff_dict[f][0][1]
        e_y1 = np.sqrt((coeff_dict[f][0][0] * x2[:, 1]) ** 2 +
                       ((1 - coeff_dict[f][0][0]) * x1[:, 1]) ** 2 +
                       coeff_dict[f][0][2] ** 2)

    _mags = np.mean(list(zip(y1, y2)), axis=1)
    e_mags = np.mean(list(zip(e_y1, e_y2)), axis=1)

    del f, x1, x2, x3
    gc.collect()

    return _mags, e_mags


# Define the function that will be executed in parallel
def process_aper_photometry(r_aper, image, src_pos, mask, aper_mode, fwhm, exp_time,
                            r_in=1.9, r_out=2.2, width=500, theta=0., gain=1., rdnoise=0):
    if aper_mode == 'circ':

        phot_res = get_aper_photometry(image, src_pos, mask=mask,
                                       aper_mode=aper_mode,
                                       r_aper=r_aper * fwhm,
                                       r_in=r_in * fwhm,
                                       r_out=r_out * fwhm)
    else:
        height = r_aper * fwhm * 2.
        w_in = width + r_in * fwhm
        w_out = width + r_out * fwhm
        h_in = r_in * fwhm * 2.
        h_out = r_out * fwhm * 2.
        phot_res = get_aper_photometry(image, src_pos, mask=mask,
                                       height=height,
                                       width=width, w_in=w_in, w_out=w_out, h_in=h_in,
                                       h_out=h_out, theta=theta, aper_mode='rect')

    aper_flux_counts, _, _, aper_bkg_counts, _, area = phot_res

    aper_flux = aper_flux_counts / exp_time
    aper_bkg = aper_bkg_counts / exp_time

    # calculate the Signal-to-Noise ratio
    snr, snr_err = get_snr(flux_star=aper_flux, flux_bkg=aper_bkg,
                           t_exp=exp_time, area=area,
                           gain=gain, rdnoise=rdnoise, dc=0.)
    del image, mask
    return aper_flux, aper_bkg, snr, snr_err


def get_opt_aper_trail(image,
                       src_pos,
                       fwhm,
                       exp_time,
                       config: dict,
                       img_mask=None,
                       width: float = 500.,
                       theta: float = 0.,
                       gain: float = 1,
                       rdnoise: float = 0,
                       silent: bool = False):
    """"""
    # Initialize logging for this user-callable function
    log.setLevel(logging.getLevelName(log.getEffectiveLevel()))

    dstep = config['APER_STEP_SIZE']
    start = config['APER_START']
    stop = config['APER_STOP'] + 2.5
    rapers = np.linspace(start, stop, int((stop - start) / dstep + 1))

    output = np.zeros((len(rapers), len(src_pos), 4))
    output[output == 0] = np.nan

    r_in = config['RSKYIN'] * 1.10
    r_out = config['RSKYOUT'] * 1.10

    # Create the executor
    executor = ProcessPoolExecutor()

    try:
        # Create a partial function with the common parameters
        func = partial(process_aper_photometry, image=image, src_pos=src_pos,
                       mask=img_mask,
                       fwhm=fwhm, exp_time=exp_time, aper_mode='rect',
                       r_in=r_in, r_out=r_out,
                       width=width, theta=theta,
                       gain=gain, rdnoise=rdnoise)

        # Map the function over the range of rapers
        results = executor.map(func, rapers)

        # Store the results in the output array
        for i, result in enumerate(results):
            output[i, :, 0], output[i, :, 1], output[i, :, 2], output[i, :, 3] = result

    finally:
        # Ensure the executor is shut down properly
        executor.shutdown(wait=True)

    max_snr_idx = np.nanargmax(output[:, :, 2], axis=0)
    max_snr_aprad = float(rapers[max_snr_idx])
    optimum_aprad = max_snr_aprad * 1.25
    qlf_aptrail = True
    # if optimum_aprad * 2. > 6.:
    #     log.warning('Optimum width seems high [%.1f x FWHM] '
    #                 '- setting to %.1f x FWHM' % ((optimum_aprad * 2), (aper_height * 2)))
    #     optimum_aprad = aper_height
    #     qlf_aptrail = False

    if not silent:
        log.info('    ==> best-fit aperture height: %3.1f (FWHM)' % (max_snr_aprad * 2))
        log.info('    ==> optimum aperture height (h x 1.25): %3.1f (FWHM)' % (optimum_aprad * 2))

    del image
    return output, rapers, max_snr_aprad, optimum_aprad, qlf_aptrail


def get_optimum_aper_rad(image: np.ndarray,
                         std_cat: pd.DataFrame,
                         fwhm: float,
                         exp_time,
                         config: dict,
                         aper_rad: float = 1.7,
                         r_in: float = 1.9,
                         r_out: float = 2.2,
                         gain: float = 1,
                         rdnoise: float = 0,
                         silent: bool = False):
    """"""

    # Initialize logging for this user-callable function
    log.setLevel(logging.getLevelName(log.getEffectiveLevel()))

    # mask unwanted pixel
    mask = (image < 0)
    img_mask = config['img_mask']
    # if img_mask is not None:
    #     mask |= img_mask

    dstep = config['APER_STEP_SIZE']
    start = config['APER_START']
    stop = config['APER_STOP']
    rapers = np.linspace(start, stop, int((stop - start) / dstep + 1))

    # select only the 25 brightest sources
    std_cat = std_cat.head(25)

    # get positions
    src_pos = np.array(list(zip(std_cat['xcentroid'], std_cat['ycentroid'])))

    output = np.zeros((len(rapers), len(src_pos), 4))
    output[output == 0] = np.nan

    # Create the executor
    executor = ProcessPoolExecutor()

    try:
        # Create a partial function with the common parameters
        func = partial(process_aper_photometry, fwhm=fwhm,
                       exp_time=exp_time, gain=gain, rdnoise=rdnoise, image=image,
                       src_pos=src_pos, mask=img_mask, r_in=r_in, r_out=r_out,
                       aper_mode='circ')

        # Map the function over the range of rapers
        results = executor.map(func, rapers)

        # Store the results in the output array
        for i, result in enumerate(results):
            output[i, :, 0], output[i, :, 1], output[i, :, 2], output[i, :, 3] = result

    finally:
        # Ensure the executor is shut down properly
        executor.shutdown(wait=True)

    del executor
    # for i, r_aper in enumerate(rapers):
    #
    #     phot_res = get_aper_photometry(image, src_pos, mask=mask,
    #                                    r_aper=r_aper * fwhm,
    #                                    r_in=r_in * fwhm,
    #                                    r_out=r_out * fwhm)
    #     aper_flux_counts, _, _, aper_bkg_counts, _ = phot_res
    #
    #     aper_flux = aper_flux_counts / exp_time
    #     aper_bkg = aper_bkg_counts / exp_time
    #
    #     # calculate the Signal-to-Noise ratio
    #     snr, snr_err = get_snr(flux_star=aper_flux, flux_bkg=aper_bkg,
    #                            t_exp=exp_time,
    #                            r=r_aper * fwhm,
    #                            gain=gain, rdnoise=rdnoise, dc=0.)
    #
    #     output[i, :, 0] = aper_flux
    #     output[i, :, 1] = aper_bkg
    #     output[i, :, 2] = snr
    #     output[i, :, 3] = snr_err

    # if there are more than 5 sources
    if output.shape[1] > config['NUM_STD_MIN']:
        x = np.nanmax(output[:, :, 2], axis=0)
        idx = np.where(x >= np.nanmean(x))[0]
        if len(idx) <= config['NUM_STD_MIN']:
            output = output[:, :int(config['NUM_STD_MIN'] + 1), :]
        else:
            output = output[:, idx, :]

    max_snr_idx = np.nanargmax(np.nanmean(output[:, :, 2], axis=1))
    max_snr_aprad = float(rapers[max_snr_idx])
    optimum_aprad = max_snr_aprad * 1.25
    qlf_aprad = True
    if optimum_aprad > 3.:
        log.warning('Optimum radius seems high [%.1f x FWHM] '
                    '- setting to %.1f x FWHM' % (optimum_aprad, aper_rad))
        optimum_aprad = aper_rad
        qlf_aprad = False

    if not silent:
        log.info('    ==> best-fit aperture radius: %3.1f (FWHM)' % max_snr_aprad)
        log.info('    ==> optimum aperture radius (r x 1.25): %3.1f (FWHM)' % optimum_aprad)

    del image, std_cat
    gc.collect()
    # quit()
    return output, rapers, max_snr_aprad, optimum_aprad, qlf_aprad


def get_snr(flux_star: np.array, flux_bkg: np.array,
            t_exp: float, area: float,
            gain: float = 1., rdnoise: float = 0., dc: float = 0.):
    """"""

    counts_source = flux_star * t_exp
    sky_shot = flux_bkg * t_exp * area

    read_noise = ((rdnoise ** 2) + (gain / 2.) ** 2) * area
    dark_noise = dc * t_exp * area

    snr = counts_source / np.sqrt(counts_source + sky_shot + read_noise + dark_noise)

    del flux_star, flux_bkg
    gc.collect()

    SNR_cleaned = [i if i > 0 else 0 for i in snr]

    SNR_err = np.array([2.5 * np.log10(1. + (1. / snr)) if snr > 0 else np.nan for snr in SNR_cleaned])

    return SNR_cleaned, SNR_err


def get_aper_photometry(image: np.ndarray,
                        src_pos: np.ndarray,
                        mask: np.array = None,
                        aper_mode: str = 'circ',
                        r_aper: float = 1.7,
                        r_in: float = 1.9,
                        r_out: float = 2.2,
                        width: float = 500.,
                        height: float = 3.4,
                        w_in: float = 520., w_out: float = 530.,
                        h_in: float = 4., h_out: float = 6.,
                        theta: float = 0.,
                        gain: float = 1.):
    """Measure aperture photometry"""

    if aper_mode == 'circ':
        sum_method = 'exact'
        annulus_aperture = CircularAnnulus(src_pos, r_in=r_in, r_out=r_out)
        aperture = CircularAperture(src_pos, r=r_aper)
    else:
        sum_method = 'center'
        annulus_aperture = RectangularAnnulus(src_pos, w_in=w_in, w_out=w_out,
                                              h_in=h_in, h_out=h_out, theta=theta)
        aperture = RectangularAperture(src_pos,
                                       w=width,
                                       h=height,
                                       theta=theta)

    # get statistics with mask in mind
    bkg_stats = ApertureStats(image, annulus_aperture,
                              mask=mask,
                              # sigma_clip=sigclip,
                              sum_method=sum_method)

    # Background median and standard deviation of annulus
    bkg_median = bkg_stats.mean
    bkg_std = bkg_stats.std
    # print(bkg_std)
    # print(bkg_median, bkg_std)

    if np.any(image) < 0:
        error = None
    else:
        from photutils.datasets import make_noise_image
        image_masked = image * ~mask if mask is not None else image
        error = make_noise_image(image.shape,
                                 distribution='poisson',
                                 mean=np.nanmean(image_masked))

    # error = None
    # aper_stats = ApertureStats(image, aperture, mask=mask,
    #                            sigma_clip=None, sum_method='exact')
    # if aper_mode == 'rect':
    #
    #     fig = plt.figure(figsize=(10, 6))
    #
    #     gs = gridspec.GridSpec(1, 1)
    #     ax = fig.add_subplot(gs[0, 0])
    #     ax.imshow(image*~mask, origin='lower', interpolation='nearest')
    #     aperture.plot(axes=ax, **{'color': 'white', 'lw': 1.25, 'alpha': 0.75})
    #     annulus_aperture.plot(axes=ax, **{'color': 'red', 'lw': 1.25, 'alpha': 0.75})

    # Make sure to use the same area over which to perform the photometry
    aperture_area = aperture.area_overlap(data=image, mask=mask, method=sum_method)
    annulus_area = annulus_aperture.area_overlap(data=image, mask=mask, method=sum_method)
    # print(aperture_area, annulus_area)
    # do aperture photometry
    phot_table = aperture_photometry(image, aperture,
                                     mask=mask,
                                     error=error,
                                     method=sum_method)
    del mask, aperture, annulus_aperture

    # rename pix position to make it consistent
    names = ('xcenter', 'ycenter')
    new_names = ('xcentroid', 'ycentroid')
    phot_table.rename_columns(names, new_names)

    # convert to pandas dataframe
    phot_table = phot_table.to_pandas()

    # get total background
    total_bkg = bkg_median * aperture_area

    # subtract background
    phot_bkgsub = phot_table['aperture_sum'] - total_bkg
    # print(phot_bkgsub, total_bkg)
    # if telescope == 'CTIO 4.0-m telescope':
    #     phot_bkgsub = phot_table['aperture_sum']
    phot_table['aperture_sum_bkgsub'] = phot_bkgsub
    phot_table.loc[phot_table['aperture_sum_bkgsub'] <= 0] = 0

    # compute the flux error using DAOPHOT style
    flux_variance = phot_table['aperture_sum_bkgsub'].values
    if error is not None:
        flux_variance = phot_table['aperture_sum_err'].values ** 2

    bkg_var = (aperture_area * bkg_std ** 2.) * (1. + aperture_area / annulus_area)
    flux_error = (flux_variance / gain + bkg_var) ** 0.5
    phot_table['aperture_sum_bkgsub_err'] = flux_error

    del image, src_pos, bkg_stats, flux_variance
    gc.collect()

    return (phot_table['aperture_sum_bkgsub'].values,
            flux_error, phot_table, bkg_median, bkg_std, aperture_area)


def get_std_photometry(image, std_cat, src_pos, fwhm, config):
    """Perform aperture photometry on standard stars"""

    res_cat = std_cat.copy()
    aper_dict = dict(aper=config['APER_RAD_OPT'] * fwhm,
                     inf_aper=config['INF_APER_RAD_OPT'] * fwhm)
    # print(aper_dict)
    # mask unwanted pixel
    mask = (image < 0)
    img_mask = config['img_mask']
    if img_mask is not None:
        mask |= img_mask

    for key, val in aper_dict.items():
        phot_res = get_aper_photometry(image, src_pos,
                                       mask=img_mask,
                                       r_aper=val,
                                       r_in=val + config['RSKYIN_OPT'] * fwhm,
                                       r_out=val + config['RSKYOUT_OPT'] * fwhm
                                       )

        aper_flux_counts, aper_flux_count_err, _, aper_bkg_counts, _, _ = phot_res
        res_cat[f'flux_counts_{key}'] = aper_flux_counts
        res_cat[f'flux_counts_err_{key}'] = aper_flux_count_err
        del phot_res

    del src_pos, image
    gc.collect()

    return res_cat


def get_sat_photometry(image, img_mask, src_pos, fwhm, width, theta, config):
    """Perform aperture photometry on satellite"""

    res_cat = {}
    aper_dict = dict(aper=config['APER_RAD_RECT'],
                     inf_aper=config['INF_APER_RAD_RECT'])

    # mask unwanted pixel
    mask = (image < 0)
    # img_mask = config['img_mask']
    if img_mask is not None:
        mask |= img_mask

    w_in = width + config['RSKYIN_OPT_RECT'] * fwhm
    w_out = width + config['RSKYOUT_OPT_RECT'] * fwhm
    h_in = config['RSKYIN_OPT_RECT']
    h_out = config['RSKYOUT_OPT_RECT']

    for key, val in aper_dict.items():
        phot_res = get_aper_photometry(image, src_pos,
                                       mask=img_mask,
                                       height=val * 2. * fwhm,
                                       width=width,
                                       h_in=(val + h_in) * 2. * fwhm,
                                       h_out=(val + h_out) * 2. * fwhm,
                                       w_in=val + w_in,
                                       w_out=val + w_out,
                                       theta=theta,
                                       aper_mode='rect')
        # print(phot_res)
        aper_flux_counts, aper_flux_count_err, _, aper_bkg_counts, _, _ = phot_res
        res_cat[f'flux_counts_{key}'] = aper_flux_counts
        res_cat[f'flux_counts_err_{key}'] = aper_flux_count_err
        del phot_res

    del src_pos, image
    gc.collect()

    return res_cat


def get_glint_photometry(image, valid_mask, src_pos, width, height, theta, config,
                         sum_method='exact'):
    """"""
    gain = 1.
    res_cat = {}
    negative_mask = (image < 0)
    sat_lim_mask = (image > config['sat_lim'])

    # mask only invalid and negative values
    negative_mask |= valid_mask

    # mask negative and saturated pixel
    sat_lim_mask |= negative_mask
    sat_lim_mask |= valid_mask

    mask_dict = dict(with_sat_src=negative_mask, no_sat_src=sat_lim_mask)

    for key, mask in mask_dict.items():
        res_cat[key] = {}
        aperture = RectangularAperture(src_pos,
                                       w=width,
                                       h=height,
                                       theta=theta)

        rect_mask = aperture.to_mask(method='center')
        mask_img = rect_mask.to_image(image.shape)
        mask_img = np.where(mask_img == 1., True, False)

        bkg_mean, bkg_median, bkg_std = sigma_clipped_stats(image, mask_img)

        # Make sure to use the same area over which to perform the photometry
        aperture_area = aperture.area_overlap(data=image, mask=mask, method=sum_method)
        annulus_area = aperture.area_overlap(data=image, mask=mask_img, method=sum_method)

        # do aperture photometry
        phot_table = aperture_photometry(image, aperture,
                                         mask=mask,
                                         method=sum_method)
        del aperture

        # rename pix position to make it consistent
        names = ('xcenter', 'ycenter')
        new_names = ('xcentroid', 'ycentroid')
        phot_table.rename_columns(names, new_names)

        # convert to pandas dataframe
        phot_table = phot_table.to_pandas()

        # get total background
        total_bkg = bkg_median * aperture_area

        # subtract background
        phot_bkgsub = phot_table['aperture_sum'] - total_bkg
        # print(phot_bkgsub, total_bkg)
        # if telescope == 'CTIO 4.0-m telescope':
        #     phot_bkgsub = phot_table['aperture_sum']
        phot_table['aperture_sum_bkgsub'] = phot_bkgsub
        phot_table.loc[phot_table['aperture_sum_bkgsub'] <= 0] = 0

        # compute the flux error using DAOPHOT style
        flux_variance = phot_table['aperture_sum_bkgsub'].values
        bkg_var = (aperture_area * bkg_std ** 2.) * (1. + aperture_area / annulus_area)
        flux_error = (flux_variance / gain + bkg_var) ** 0.5

        res_cat[key]['flux_counts'] = phot_table['aperture_sum'].values[0]
        res_cat[key]['flux_counts_err'] = flux_error[0]
        res_cat[key]['area_pix'] = aperture_area

        del phot_table, mask_img
    del image, valid_mask
    return res_cat


def order_shift(x):
    """
    Get the order of magnitude of an array.
    Use to shift the array to between 0-10:
    This function uses the following equation to find the order of magnitude of an array.

    ..math::

       order\_of\_mag (x) = 10^{MAX(FLOOR(Log_{10}(x)))}

    :param x: Array of values
    :type x: float
    :return: Order of magnitude
    :rtype: float

    """

    idx = (np.isnan(x)) | (np.isinf(x)) | (x <= 0)

    order_of_mag = 10 ** np.nanmax(np.floor(np.log10(x[~idx])))

    return order_of_mag
