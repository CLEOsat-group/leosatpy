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

import astropy.table
from astropy.stats import sigma_clip
import numpy as np

import pandas as pd
import multiprocessing
from multiprocessing import Pool as ThreadPool
from functools import partial

try:
    import matplotlib
    from matplotlib import pyplot as plt
    import matplotlib.lines as mlines

except Exception:
    plt = None
else:
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

from astropy.coordinates import Angle
from astropy.stats import SigmaClip

from photutils.utils import calc_total_error
from photutils.centroids import (centroid_sources, centroid_2dg)
from photutils.aperture import (aperture_photometry,
                                ApertureStats,
                                CircularAperture,
                                CircularAnnulus,
                                RectangularAperture,
                                RectangularAnnulus)

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


def get_optimum_aper_rad(image: np.ndarray,
                         std_cat: astropy.table.Table,
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

    image[image < 0.] = 0.

    dstep = config['APER_STEP_SIZE']
    start = config['APER_START']
    stop = config['APER_STOP']
    rapers = np.linspace(start, stop, int((stop - start) / dstep + 1))

    # select only the 25 brightest sources
    n_sel = 5 if len(std_cat) <= 20 else int(round(len(std_cat) * 0.25))
    n_sel = 20 if n_sel >= 20 else n_sel
    std_cat = std_cat[:n_sel]

    # get positions
    src_pos = np.array(list(zip(std_cat['xcentroid'], std_cat['ycentroid'])))

    output = np.zeros((len(rapers), len(src_pos), 3))
    output[output == 0] = np.nan

    for i in range(len(rapers)):
        r_aper = rapers[i]
        phot_res = get_aper_photometry(image, src_pos,
                                       r_aper=r_aper * fwhm,
                                       r_in=r_in * fwhm,
                                       r_out=r_out * fwhm)
        aper_flux_counts, _, _, aper_bkg_counts, _ = phot_res

        aper_flux = aper_flux_counts / exp_time
        aper_bkg = aper_bkg_counts / exp_time

        # calculate Signal-to-Noise ratio
        snr = get_snr(flux_star=aper_flux, flux_bkg=aper_bkg, t_exp=exp_time,
                      r=r_aper * fwhm,
                      gain=gain, rdnoise=rdnoise, dc=0.)

        output[i, :, 0] = aper_flux
        output[i, :, 1] = aper_bkg
        output[i, :, 2] = snr

    max_snr_idx = np.nanargmax(np.nanmedian(output[:, :, 2], axis=1))
    max_snr_aprad = rapers[max_snr_idx]
    optimum_aprad = max_snr_aprad * 1.5

    if optimum_aprad > 3.:
        log.warning('Optimum radius seems high [%.1f x FWHM] '
                    '- setting to %.1f x FWHM' % (optimum_aprad, aper_rad))
        optimum_aprad = aper_rad

    if not silent:
        log.info('    ==> best-fit aperture radius: %3.1f (FWHM)' % max_snr_aprad)
        log.info('    ==> optimum aperture radius (r x 1.5): %3.1f (FWHM)' % optimum_aprad)

    del image, std_cat
    gc.collect()

    return output, rapers, max_snr_aprad, optimum_aprad


def get_snr(flux_star: np.array, flux_bkg: np.array,
            t_exp: float, r: float,
            gain: float = 1, rdnoise: float = 0, dc: float = 0):
    """"""

    area = np.pi * r ** 2

    counts_source = flux_star * t_exp
    sky_shot = flux_bkg * t_exp * area

    read_noise = ((rdnoise ** 2) + (gain / 2) ** 2) * area
    dark_noise = dc * t_exp * area

    snr = counts_source / np.sqrt(counts_source + sky_shot + read_noise + dark_noise)

    del flux_star, flux_bkg
    gc.collect()

    return snr


def get_aper_photometry(image: np.ndarray,
                        src_pos: np.ndarray,
                        mask: np.array = None,
                        bkg_sigma: float = 3.,
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

    sigclip = SigmaClip(sigma=bkg_sigma, maxiters=None, cenfunc=np.nanmedian)

    if aper_mode == 'circ':
        annulus_aperture = CircularAnnulus(src_pos, r_in=r_in, r_out=r_out)
        aperture = CircularAperture(src_pos, r=r_aper)
    else:

        annulus_aperture = RectangularAnnulus(src_pos, w_in=w_in, w_out=w_out,
                                              h_in=h_in, h_out=h_out, theta=theta)
        aperture = RectangularAperture(src_pos,
                                       w=width,
                                       h=height,
                                       theta=theta)

    # get statistics with mask in mind
    bkg_stats = ApertureStats(image, annulus_aperture, mask=mask,
                              sigma_clip=sigclip, sum_method='exact')
    aper_stats = ApertureStats(image, aperture, mask=mask,
                               sigma_clip=None, sum_method='exact')

    # Background median and standard deviation of annulus
    bkg_median = bkg_stats.median
    bkg_std = bkg_stats.std

    if np.any(image) < 0:
        error = None
    else:
        # mean = np.median(image, axis=None)
        error = np.sqrt(image)

    #
    aperture_area = aper_stats.sum_aper_area.value  # aperture.area
    annulus_area = bkg_stats.sum_aper_area.value

    phot_table = aperture_photometry(image, aperture, error=error, mask=mask)

    # rename pix position to make it consistent
    names = ('xcenter', 'ycenter')
    new_names = ('xcentroid', 'ycentroid')
    phot_table.rename_columns(names, new_names)

    phot_table = phot_table.to_pandas()
    total_bkg = bkg_median * aperture_area
    phot_bkgsub = phot_table['aperture_sum'] - total_bkg

    phot_table['aperture_sum_bkgsub'] = phot_bkgsub
    phot_table.loc[phot_table['aperture_sum_bkgsub'] <= 0] = 0
    # print(phot_table)

    flux_var = phot_table['aperture_sum_bkgsub'].values
    if error is not None:
        flux_var = phot_table['aperture_sum_err'].values ** 2

    bkg_var = (aperture_area * bkg_std ** 2.) * (1. + aperture_area / annulus_area)
    flux_error = np.sqrt(flux_var / gain + bkg_var)
    phot_table['aperture_sum_bkgsub_err'] = flux_error

    del image, src_pos, bkg_stats, flux_var
    gc.collect()

    return (phot_table['aperture_sum_bkgsub'].values,
            flux_error, phot_table, bkg_median, bkg_std)


def get_std_photometry(image, std_cat, src_pos, fwhm, config):
    """Perform aperture photometry on standard stars"""
    aper_dict = dict(aper=config['APER_RAD'] * fwhm,
                     inf_aper=config['INF_APER_RAD'] * fwhm)

    for key, val in aper_dict.items():
        phot_res = get_aper_photometry(image, src_pos,
                                       r_aper=val,
                                       r_in=config['RSKYIN'] * fwhm,
                                       r_out=config['RSKYOUT'] * fwhm)

        aper_flux_counts, aper_flux_count_err, _, aper_bkg_counts, _ = phot_res
        std_cat[f'flux_counts_{key}'] = aper_flux_counts
        std_cat[f'flux_counts_err_{key}'] = aper_flux_count_err

    del src_pos, image
    gc.collect()

    return std_cat
