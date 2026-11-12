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
import sys
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
    try:
        matplotlib.use('Qt5Agg')
    except ImportError:
        pass
    matplotlib.rc("lines", linewidth=1.2)
    matplotlib.rc('figure', dpi=150, facecolor='w', edgecolor='k')
    matplotlib.rc('text.latex', preamble=r'\usepackage{sfmath}')

from astropy.stats import sigma_clipped_stats, SigmaClip
from photutils.aperture import (aperture_photometry,
                                ApertureStats,
                                CircularAperture,
                                CircularAnnulus,
                                RectangularAperture,
                                RectangularAnnulus)

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

__taskname__ = 'photometry'
# -----------------------------------------------------------------------------

""" Parameter used in the script """
log = logging.getLogger(__name__)

MODULE_PATH = os.path.dirname(inspect.getfile(inspect.currentframe()))


# -----------------------------------------------------------------------------

def convert_sdss_to_BVRI_lupton(f, x1, x2, x3):
    """ Convert to BVRI magnitudes using SDSS magnitudes.

    Reference:
    Lupton et al.(2005) BibCode: 2005AAS...20713308L

    Parameters
    ----------
    f : str
        Filter to convert to. Options: 'B', 'V', 'R', 'I'.
    x1 : np.ndarray
        SDSS magnitude array for the first filter.
    x2 : np.ndarray
        SDSS magnitude array for the second filter.
    x3 : np.ndarray
        SDSS magnitude array for the third filter.

    Returns
    -------
    tuple
        Converted magnitudes and their errors.
    """
    coeff_dict = bc.CONV_SSDS_BVRI

    y1 = x2[:, 0] + coeff_dict[f][0][0] * (x1[:, 0] - x2[:, 0]) + coeff_dict[f][0][1]
    e_y1 = np.sqrt((coeff_dict[f][0][0] * x1[:, 1]) ** 2 +
                   ((1 - coeff_dict[f][0][0]) * x2[:, 1]) ** 2 +
                   coeff_dict[f][0][2] ** 2)
    y2 = x2[:, 0] + coeff_dict[f][1][0] * (x2[:, 0] - x3[:, 0]) + coeff_dict[f][1][1]
    # y2 = x2 + c (x2 - x3): d/dx2 = 1 + c, d/dx3 = -c
    e_y2 = np.sqrt((coeff_dict[f][1][0] * x3[:, 1]) ** 2 +
                   ((1 + coeff_dict[f][1][0]) * x2[:, 1]) ** 2 +
                   coeff_dict[f][1][2] ** 2)
    if f in ['B', 'I']:
        y1 = x1[:, 0] + coeff_dict[f][0][0] * (x1[:, 0] - x2[:, 0]) + coeff_dict[f][0][1]
        # y1 = x1 + c (x1 - x2): d/dx1 = 1 + c, d/dx2 = -c
        e_y1 = np.sqrt((coeff_dict[f][0][0] * x2[:, 1]) ** 2 +
                       ((1 + coeff_dict[f][0][0]) * x1[:, 1]) ** 2 +
                       coeff_dict[f][0][2] ** 2)

    _mags = np.mean(list(zip(y1, y2)), axis=1)
    e_mags = np.mean(list(zip(e_y1, e_y2)), axis=1)

    del f, x1, x2, x3
    gc.collect()

    return _mags, e_mags

def convert_sdss_to_U_jester(u, g, r):
    """ Convert to U magnitude using SDSS magnitudes.

    Reference:
        Jester et al.(2005)

    Parameters
    ----------
    u : np.ndarray
        SDSS u magnitude array.
    g : np.ndarray
        SDSS g magnitude array.
    r : np.ndarray
        SDSS r magnitude array.

    Returns
    -------
    tuple
        Converted U magnitudes and their errors.
    """
    B = g[:, 0] + 0.33 * (g[:, 0] - r[:, 0]) + 0.20
    U = 0.77 * (u[:, 0] - g[:, 0]) - 0.88 + B
    # U = 0.77 u + 0.56 g - 0.33 r + const
    e_U = np.sqrt((0.77 * u[:, 1])**2 +
                  (0.56 * g[:, 1])**2 +
                  (0.33 * r[:, 1])**2)

    return U, e_U

# Define the function that will be executed in parallel
def process_aper_photometry(r_aper, iteration, image, src_pos, mask, aper_mode, fwhm, exp_time,
                            r_in=1.9, r_out=2.2, width=500, theta=0., gain=1., rdnoise=0,
                            total=1, use_lock=False):
    """ Run aperture photometry for a given aperture radius.

    Parameters
    ----------
    r_aper : float
        Aperture radius in units of FWHM.
    iteration : int
        Current iteration number for progress tracking.
    image : np.ndarray
        The input image data.
    src_pos : np.ndarray
        Array of source positions (x, y).
    mask : np.ndarray
        Mask for the image (True for masked pixels).
    aper_mode : str
        Aperture mode ('circ' for circular, 'rect' for rectangular).
    fwhm : float
        Full Width at Half Maximum of the point spread function.
    exp_time : float
        Exposure time of the image.
    r_in : float, optional
        Inner radius of the sky annulus in units of FWHM. Default is 1.9.
    r_out : float, optional
        Outer radius of the sky annulus in units of FWHM. Default is 2.2.
    width : float, optional
        Width of the rectangular aperture in pixels. Default is 500.
    theta : float, optional
        Angle of the rectangular aperture in degrees. Default is 0.
    gain : float, optional
        Gain of the detector. Default is 1.
    rdnoise : float, optional
        Read noise of the detector. Default is 0.
    total : int, optional
        Total number of iterations for progress tracking. Default is 1.
    use_lock : bool, optional
        Whether to use a lock for thread-safe progress bar updates. Default is False.

    Returns
    -------
    aper_flux : np.ndarray
        Aperture flux in physical units.
    aper_bkg : np.ndarray
        Aperture background flux in physical units.
    snr : np.ndarray
        Signal-to-Noise ratio.
    snr_err : np.ndarray
        Error in Signal-to-Noise ratio.
    """
    # Keep the sky annulus outside the aperture for large apertures
    r_in_eff = max(r_in, r_aper + 1.)
    r_out_eff = r_in_eff + (r_out - r_in)

    if aper_mode == 'circ':
        phot_res = get_aper_photometry(image, src_pos, mask=mask,
                                       aper_mode=aper_mode,
                                       r_aper=r_aper * fwhm,
                                       r_in=r_in_eff * fwhm,
                                       r_out=r_out_eff * fwhm)
    else:
        height = r_aper * fwhm * 2.
        w_in = width + 2. * r_in_eff * fwhm
        w_out = width + 2. * r_out_eff * fwhm
        h_in = r_in_eff * fwhm * 2.
        h_out = r_out_eff * fwhm * 2.
        phot_res = get_aper_photometry(image, src_pos, mask=mask,
                                       height=height,
                                       width=width, w_in=w_in, w_out=w_out, h_in=h_in,
                                       h_out=h_out, theta=theta, aper_mode='rect')

    aper_flux_counts, aper_flux_err, _, aper_bkg_counts, _, area = phot_res

    aper_flux = aper_flux_counts / exp_time
    aper_bkg = aper_bkg_counts / exp_time

    # Signal-to-noise ratio from the measured flux error. The error contains the source Poisson term
    # and the pixel noise measured in the sky annulus (sky Poisson noise and read noise), so it is valid
    # also for background-subtracted images. The gain and read noise arguments are kept for compatibility.
    with np.errstate(divide='ignore', invalid='ignore'):
        snr = np.asarray(aper_flux_counts, dtype=float) / np.asarray(aper_flux_err, dtype=float)
    snr = np.where(np.isfinite(snr) & (snr > 0), snr, 0.)
    with np.errstate(divide='ignore'):
        snr_err = np.where(snr > 0, 2.5 * np.log10(1. + 1. / np.where(snr > 0, snr, 1.)), np.nan)
    del image, mask

    bc.print_progress_bar(iteration, total, length=80,
                          color=bc.BCOLORS.OKGREEN, use_lock=use_lock)

    return aper_flux, aper_bkg, snr, snr_err


def get_opt_aper_trail(image,
                       src_pos,
                       fwhm,
                       exp_time,
                       config: dict,
                       img_mask=None,
                       width: float = 500.,
                       theta: float = 0.,
                       gain: float = 1.,
                       rdnoise: float = 0.,
                       silent: bool = False):
    """ Get the optimum aperture height for trail photometry.

    Parameters
    ----------
    image : np.ndarray
        The input image data.
    src_pos : np.ndarray
        Array of source positions (x, y).
    fwhm : float
        Full Width at Half Maximum of the point spread function.
    exp_time : float
        Exposure time of the image.
    config : dict
        Configuration dictionary with parameters for aperture photometry.
    img_mask : np.ndarray, optional
        Mask for the image (True for masked pixels). Default is None.
    width : float, optional
        Width of the rectangular aperture in pixels. Default is 500.
    theta : float, optional
        Angle of the rectangular aperture in degrees. Default is 0.
    gain : float, optional
        Gain of the detector. Default is 1.
    rdnoise : float, optional
        Read noise of the detector. Default is 0.
    silent : bool, optional
        If True, suppresses logging output. Default is False.

    Returns
    -------
    output : np.ndarray
        Array containing photometry results for different aperture heights.
    rapers : np.ndarray
        Array of aperture heights tested.
    max_snr_aprad : float
        Aperture height that gives the maximum SNR.
    optimum_aprad : float
        Optimum aperture height (1.25 times max SNR height).
    qlf_aptrail : bool
        Quality flag indicating if the optimum aperture height is acceptable.
    """
    # Initialize logging for this user-callable function
    log.setLevel(logging.getLevelName(log.getEffectiveLevel()))

    dstep = config['TRAIL_APER_STEP_SIZE']
    start = config['TRAIL_APER_START']
    stop = config['TRAIL_APER_STOP']
    rapers = np.linspace(start, stop, int((stop - start) / dstep + 1))

    output = np.zeros((len(rapers), len(src_pos), 4))
    output[output == 0] = np.nan

    r_in = config['TRAIL_RSKYIN']
    r_out = config['TRAIL_RSKYOUT']

    total_rapers = len(rapers)
    try:
        # Create a partial function with the common parameters
        func = partial(process_aper_photometry, image=image, src_pos=src_pos,
                       mask=img_mask, fwhm=fwhm, exp_time=exp_time, aper_mode='rect',
                       r_in=r_in, r_out=r_out, width=width, theta=theta,
                       gain=gain, rdnoise=rdnoise, total=total_rapers, use_lock=True)

        # Map the function over the range of rapers and get the results
        with ProcessPoolExecutor() as executor:
            results = list(executor.map(func, rapers, range(1, total_rapers + 1)))

        # Store the results in the output array
        for i, result in enumerate(results):
            # Assuming result is of the form suitable to be assigned to the output array
            output[i, :, 0], output[i, :, 1], output[i, :, 2], output[i, :, 3] = result

        bc.print_progress_bar(total_rapers, total_rapers,  length=80,
                              color=bc.BCOLORS.OKGREEN, use_lock=True)

    finally:
        # Ensure the executor is shut down properly
        executor.shutdown(wait=True)
        sys.stdout.write('\n')

    del executor

    max_snr_idx = np.nanargmax(output[:, :, 2], axis=0)
    # Single trail position: .item() extracts the scalar (float() on a 1-d array fails with numpy >= 2.4)
    max_snr_aprad = rapers[max_snr_idx].item()
    optimum_aprad = max_snr_aprad * 1.25  # increased by 25%
    qlf_aptrail = True

    log.debug('    Best-fit aperture height: %3.1f (FWHM)' % (max_snr_aprad * 2))
    log.debug('    Optimum aperture height (h x 1.25): %3.1f (FWHM)' % (optimum_aprad * 2))

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
                         gain: float = 1.,
                         rdnoise: float = 0.,
                         silent: bool = False):
    """ Get the optimum aperture radius for photometry.

    Parameters
    ----------
    image : np.ndarray
        The input image data.
    std_cat : pd.DataFrame
        A DataFrame containing standard star catalog with 'xcentroid' and 'ycentroid' columns.
    fwhm : float
        Full Width at Half Maximum of the point spread function.
    exp_time : float
        Exposure time of the image.
    config : dict
        Configuration dictionary with parameters for aperture photometry.
    aper_rad : float, optional
        Initial aperture radius in units of FWHM. Default is 1.7.
    r_in : float, optional
        Inner radius of the sky annulus in units of FWHM. Default is 1.9.
    r_out : float, optional
        Outer radius of the sky annulus in units of FWHM. Default is 2.2.
    gain : float, optional
        Gain of the detector. Default is 1.
    rdnoise : float, optional
        Read noise of the detector. Default is 0.
    silent : bool, optional
        If True, suppresses logging output. Default is False.

    Returns
    -------
    output : np.ndarray
        Array containing photometry results for different aperture radii.
    rapers : np.ndarray
        Array of aperture radii tested.
    max_snr_aprad : float
        Aperture radius that gives the maximum SNR.
    optimum_aprad : float
        Optimum aperture radius (1.25 times max SNR radius).
    qlf_aprad : bool
        Quality flag indicating if the optimum aperture radius is acceptable.
    """
    # Initialize logging for this user-callable function
    log.setLevel(logging.getLevelName(log.getEffectiveLevel()))

    # Mask unwanted pixel
    img_mask = config['img_mask']

    dstep = config['APER_STEP_SIZE']
    start = config['APER_START']
    stop = config['APER_STOP']
    rapers = np.linspace(start, stop, int((stop - start) / dstep + 1))

    # Select only the 25 brightest sources
    std_cat = std_cat.head(25)

    # Get positions
    src_pos = np.array(list(zip(std_cat['xcentroid'], std_cat['ycentroid'])))

    output = np.zeros((len(rapers), len(src_pos), 4))
    output[output == 0] = np.nan

    total_rapers = len(rapers)
    try:
        # Create a partial function with the common parameters
        func = partial(process_aper_photometry, fwhm=fwhm,
                       exp_time=exp_time, gain=gain, rdnoise=rdnoise, image=image,
                       src_pos=src_pos, mask=img_mask, r_in=r_in, r_out=r_out,
                       aper_mode='circ', total=total_rapers, use_lock=True)

        # Map the function over the range of rapers and get the results
        with ProcessPoolExecutor() as executor:
            results = list(executor.map(func, rapers, range(1, total_rapers + 1)))

        # Store the results in the output array
        for i, result in enumerate(results):
            output[i, :, 0], output[i, :, 1], output[i, :, 2], output[i, :, 3] = result

        bc.print_progress_bar(total_rapers, total_rapers,  length=80,
                              color=bc.BCOLORS.OKGREEN, use_lock=True)
    finally:
        # Ensure the executor is shut down properly
        executor.shutdown(wait=True)
        sys.stdout.write('\n')

    del executor

    # If there are more than 5 sources
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
        log.warning('    Optimum aperture radius seems high [%.1f x FWHM]; '
                    'using %.1f x FWHM' % (optimum_aprad, aper_rad))
        optimum_aprad = aper_rad
        qlf_aprad = False

    log.debug('    Best-fit aperture radius: %3.1f (FWHM)' % max_snr_aprad)
    log.debug('    Optimum aperture radius (r x 1.25): %3.1f (FWHM)' % optimum_aprad)

    del image, std_cat
    gc.collect()

    return output, rapers, max_snr_aprad, optimum_aprad, qlf_aprad


def get_snr(flux_star: np.ndarray, flux_bkg: np.ndarray,
            t_exp: float, area: float,
            gain: float = 1., rdnoise: float = 0., dc: float = 0.):
    """ Calculate the Signal-to-Noise ratio.

    Parameters
    ----------
    flux_star : np.ndarray
        Flux from the star in physical units (e.g., electrons/sec).
    flux_bkg : np.ndarray
        Flux from the background in physical units (e.g., electrons/sec).
    t_exp : float
        Exposure time in seconds.
    area : float
        Area of the aperture in pixels.
    gain : float, optional
        Gain of the detector. Default is 1.
    rdnoise : float, optional
        Read noise of the detector. Default is 0.
    dc : float, optional
        Dark current of the detector in electrons/sec. Default is 0.

    Returns
    -------
    SNR_cleaned : np.array
        Cleaned Signal-to-Noise ratio (non-negative values).
    SNR_err : np.array
        Error in Signal-to-Noise ratio in magnitudes.
    """
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
                        mask: np.ndarray = None,
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
    """Measure aperture photometry.

    Parameters
    ----------
    image : np.ndarray
        The input image data.
    src_pos : np.ndarray
        Array of source positions (x, y).
    mask : np.ndarray, optional
        Mask for the image (True for masked pixels). Default is None.
    aper_mode : str, optional
        Aperture mode ('circ' for circular, 'rect' for rectangular). Default is 'circ'.
    r_aper : float, optional
        Aperture radius in fwhm for circular aperture. Default is 1.7.
    r_in : float, optional
        Inner radius of the sky annulus in fwhm for circular aperture. Default is 1.9.
    r_out : float, optional
        Outer radius of the sky annulus in fwhm for circular aperture. Default is 2.2.
    width : float, optional
        Width of the rectangular aperture in pixels. Default is 500.
    height : float, optional
        Height of the rectangular aperture in pixels. Default is 3.4.
    w_in : float, optional
        Inner width of the sky annulus in pixels for rectangular aperture. Default is 520.
    w_out : float, optional
        Outer width of the sky annulus in pixels for rectangular aperture. Default is 530.
    h_in : float, optional
        Inner height of the sky annulus in pixels for rectangular aperture. Default is 4.
    h_out : float, optional
        Outer height of the sky annulus in pixels for rectangular aperture. Default is 6.
    theta : float, optional
        Angle of the rectangular aperture in degrees. Default is 0.
    gain : float, optional
        Gain of the detector. Default is 1.

    Returns
    -------
    tuple
        Aperture flux, aperture flux error, photometry table,
        background median, background std, aperture area.
    """

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

    # Get statistics with mask in mind
    # Sigma clipping removes stars and other outliers in the annulus
    bkg_stats = ApertureStats(image, annulus_aperture,
                              mask=mask,
                              sigma_clip=SigmaClip(sigma=3., maxiters=10),
                              sum_method=sum_method)

    # Background (sigma-clipped median) and standard deviation of annulus
    bkg_median = np.asarray(bkg_stats.median)
    bkg_std = np.asarray(bkg_stats.std)
    # print(bkg_std)
    # print(bkg_median, bkg_std)

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

    # Run the aperture photometry
    phot_table = aperture_photometry(image, aperture,
                                     mask=mask,
                                     method=sum_method)
    del mask, aperture, annulus_aperture

    # Rename pix position to make it consistent
    names = ('xcenter', 'ycenter')
    new_names = ('xcentroid', 'ycentroid')
    phot_table.rename_columns(names, new_names)

    # Convert to pandas dataframe
    phot_table = phot_table.to_pandas()

    # Get total background
    total_bkg = bkg_median * aperture_area

    # Subtract background
    phot_bkgsub = phot_table['aperture_sum'] - total_bkg
    # print(phot_bkgsub, total_bkg)

    phot_table['aperture_sum_bkgsub'] = phot_bkgsub
    # phot_table.loc[phot_table['aperture_sum_bkgsub'] <= 0] = 0

    # Compute the flux error using DAOPHOT style: Poisson noise of the source counts
    # plus the sky noise measured in the annulus (which already contains the read noise).
    # Negative background-subtracted fluxes carry no source Poisson term.
    flux_variance = np.clip(phot_table['aperture_sum_bkgsub'].values, 0., None)

    bkg_var = (aperture_area * bkg_std ** 2.) * (1. + aperture_area / annulus_area)
    flux_error = (flux_variance / gain + bkg_var) ** 0.5
    phot_table['aperture_sum_bkgsub_err'] = flux_error

    del image, src_pos, bkg_stats, flux_variance
    gc.collect()

    return (phot_table['aperture_sum_bkgsub'].values,
            flux_error, phot_table, bkg_median, bkg_std, aperture_area)


def get_std_photometry(image, std_cat, src_pos, fwhm, config):
    """Perform aperture photometry on standard stars.

    Parameters
    ----------
    image : np.ndarray
        The input image data.
    std_cat : pd.DataFrame
        A DataFrame containing standard star catalog.
    src_pos : np.ndarray
        Array of source positions (x, y).
    fwhm : float
        Full Width at Half Maximum of the point spread function.
    config : dict
        Configuration dictionary with parameters for aperture photometry.

    Returns
    -------
    pd.DataFrame
        Aperture photometry results for standard stars.
    """
    res_cat = std_cat.copy()
    aper_dict = dict(aper=config['APER_RAD_OPT'] * fwhm,
                     inf_aper=config['INF_APER_RAD_OPT'] * fwhm)

    # Both apertures use the same sky annulus outside the large aperture.
    # RSKYIN_OPT and RSKYOUT_OPT are radii in units of the FWHM.
    r_in = config['RSKYIN_OPT'] * fwhm
    r_out = config['RSKYOUT_OPT'] * fwhm
    if r_in <= aper_dict['inf_aper']:
        raise ValueError('Sky annulus overlaps the large aperture')

    for key, val in aper_dict.items():
        phot_res = get_aper_photometry(image, src_pos,
                                       mask=config['img_mask'],
                                       r_aper=val,
                                       r_in=r_in,
                                       r_out=r_out
                                       )

        aper_flux_counts, aper_flux_count_err, _, aper_bkg_counts, _, _ = phot_res
        res_cat[f'flux_counts_{key}'] = aper_flux_counts
        res_cat[f'flux_counts_err_{key}'] = aper_flux_count_err
        del phot_res

    del src_pos, image
    gc.collect()

    return res_cat


def get_trail_fill_factor(image, mask, src_pos, width, height, theta, bkg_level=0.):
    """Profile-weighted fraction of the trail flux that falls on unmasked pixels of a rectangular aperture.

    Masked pixels (stars along the trail, saturated or bad pixels) are excluded from the aperture sum,
    so the summed trail flux is too low by this fraction. The mean profile perpendicular to the trail
    is measured from the unmasked pixels and used as a weight for each pixel of the aperture.

    Parameters
    ----------
    image : np.ndarray
        Image used for the photometry.
    mask : np.ndarray or None
        Boolean mask (True for masked pixels).
    src_pos : array-like
        Trail centre (x, y).
    width, height : float
        Length of the aperture along the trail and its full height in pixels.
    theta : float or astropy.coordinates.Angle
        Orientation of the trail.
    bkg_level : float
        Local background per pixel, subtracted before the profile is built.

    Returns
    -------
    tuple
        (fill factor f in (0, 1], masked pixel fraction, True if the trail core is masked along its
        full length so that the flux cannot be recovered)
    """
    theta_rad = float(getattr(theta, 'radian', theta))
    pos = np.asarray(src_pos, dtype=float).ravel()[:2]
    aperture = RectangularAperture(pos, w=width, h=height, theta=theta)
    aper_mask = aperture.to_mask(method='center')
    if isinstance(aper_mask, list):
        aper_mask = aper_mask[0]

    cut_img = aper_mask.cutout(image, fill_value=np.nan)
    if cut_img is None:
        return 1., 0., False
    if mask is not None:
        cut_msk = aper_mask.cutout(np.asarray(mask, dtype=float), fill_value=0.).astype(bool)
    else:
        cut_msk = np.zeros(cut_img.shape, dtype=bool)

    yy, xx = np.mgrid[aper_mask.bbox.iymin:aper_mask.bbox.iymax, aper_mask.bbox.ixmin:aper_mask.bbox.ixmax]
    dist = -(xx - pos[0]) * np.sin(theta_rad) + (yy - pos[1]) * np.cos(theta_rad)

    inside = (aper_mask.data > 0) & np.isfinite(cut_img)
    valid = inside & ~cut_msk
    n_inside = int(inside.sum())
    if n_inside == 0:
        return 1., 0., False
    masked_frac = 1. - valid.sum() / n_inside
    if masked_frac <= 0.:
        return 1., 0., False

    # Mean profile perpendicular to the trail in 1-pixel bins
    edges = np.arange(-height / 2., height / 2. + 1., 1.)
    centres = 0.5 * (edges[1:] + edges[:-1])
    idx_all = np.clip(np.digitize(dist, edges) - 1, 0, len(centres) - 1)
    prof = np.full(len(centres), np.nan)
    n_valid = np.zeros(len(centres), dtype=int)
    data = cut_img - bkg_level
    for k in range(len(centres)):
        sel = valid & (idx_all == k)
        n_valid[k] = int(sel.sum())
        if n_valid[k] >= 3:
            prof[k] = np.nanmedian(data[sel])

    core = np.abs(centres) <= max(height / 4., 1.)
    core_lost = bool(np.any(n_valid[core] < 3))

    good = np.isfinite(prof)
    if good.sum() < 2:
        return 1. - masked_frac, masked_frac, True
    prof = np.interp(centres, centres[good], prof[good])
    prof = np.clip(prof, 0., None)

    w = prof[idx_all]
    w_all = np.sum(w[inside])
    if w_all <= 0.:
        return 1. - masked_frac, masked_frac, core_lost
    f = float(np.sum(w[valid]) / w_all)
    f = min(max(f, 1.e-3), 1.)

    return f, float(masked_frac), core_lost


def get_trail_annulus(width, fwhm, r_in, r_out, h_offset=0.):
    """Rectangular sky annulus around a trail aperture.

    r_in and r_out are the half-heights of the inner and outer edge in units of the FWHM,
    measured from the trail centre line. The annulus extends by the same distance beyond the
    ends of the trail."""
    w_in = width + 2. * r_in * fwhm
    w_out = width + 2. * r_out * fwhm
    h_in = 2. * r_in * fwhm + h_offset
    h_out = 2. * r_out * fwhm + h_offset
    return w_in, w_out, h_in, h_out


def get_sat_photometry(image, img_mask, src_pos, fwhm, width, theta, config, h_offset=0.):
    """Perform aperture photometry on satellite.

    Both apertures (optimum and large) use the same sky annulus, which lies outside the large aperture.
    The fluxes are corrected for masked pixels with the profile-weighted fill factor.

    Parameters
    ----------
    image : np.ndarray
        The input image data.
    img_mask : np.ndarray
        Mask for the image (True for masked pixels).
    src_pos : np.ndarray
        Array of source positions (x, y).
    fwhm : float
        Full Width at Half Maximum of the point spread function.
    width : float
        Width of the rectangular aperture in pixels.
    theta : float
        Angle of the rectangular aperture in degrees.
    config : dict
        Configuration dictionary with parameters for aperture photometry.
        APER_RAD_RECT, INF_APER_RAD_RECT: half-heights of the apertures in FWHM.
        RSKYIN_OPT_RECT, RSKYOUT_OPT_RECT: half-heights of the annulus edges in FWHM.
    h_offset : float
        Additional height of the annulus in pixels.

    Returns
    -------
    dict
        Fluxes (flux_counts_<key>), errors, fill factors and masked fractions for the keys
        'aper' and 'inf_aper'. Fluxes are in the unit of the image (electrons).
    """
    res_cat = {}
    aper_dict = dict(aper=config['APER_RAD_RECT'],
                     inf_aper=config['INF_APER_RAD_RECT'])

    w_in, w_out, h_in, h_out = get_trail_annulus(width, fwhm,
                                                 config['RSKYIN_OPT_RECT'],
                                                 config['RSKYOUT_OPT_RECT'],
                                                 h_offset=h_offset)
    pos = np.atleast_2d(np.asarray(src_pos, dtype=float))

    for key, val in aper_dict.items():
        height = val * 2. * fwhm
        phot_res = get_aper_photometry(image, pos,
                                       mask=img_mask,
                                       height=height,
                                       width=width,
                                       h_in=h_in,
                                       h_out=h_out,
                                       w_in=w_in,
                                       w_out=w_out,
                                       theta=theta,
                                       aper_mode='rect')
        aper_flux_counts, aper_flux_count_err, phot_table, aper_bkg_counts, _, _ = phot_res

        fill, masked_frac, core_lost = get_trail_fill_factor(image, img_mask, pos[0], width, height, theta,
                                                             bkg_level=float(np.ravel(aper_bkg_counts)[0]))

        res_cat[f'flux_counts_{key}'] = np.asarray(aper_flux_counts, dtype=float) / fill
        res_cat[f'flux_counts_err_{key}'] = np.asarray(aper_flux_count_err, dtype=float) / fill
        res_cat[f'flux_counts_raw_{key}'] = np.asarray(aper_flux_counts, dtype=float)
        res_cat[f'fill_factor_{key}'] = fill
        res_cat[f'masked_frac_{key}'] = masked_frac
        res_cat[f'core_lost_{key}'] = core_lost
        res_cat[f'phot_table_{key}'] = phot_table
        del phot_res

    del src_pos, image
    gc.collect()

    return res_cat


def get_glint_photometry(image, valid_mask, src_pos, width, height, theta, config,
                         sum_method='exact'):
    """ Perform aperture photometry on glint event in the trail.

    Parameters
    ----------
    image : np.ndarray
        The input image data.
    valid_mask : np.ndarray
        Mask for invalid pixels (True for masked pixels).
    src_pos : np.ndarray
        Array of source positions (x, y).
    width : float
        Width of the rectangular aperture in pixels.
    height : float
        Height of the rectangular aperture in pixels.
    theta : float
        Angle of the rectangular aperture in degrees.
    config : dict
        Configuration dictionary with parameters for aperture photometry.
    sum_method : str, optional
        Method for summing pixel values ('exact' or 'center'). Default is 'exact'.

    Returns
    -------
    dict
        Aperture photometry results for glint event.
    """
    # todo: implement gain from config and update everything accordingly
    #  remove negative masking.
    gain = 1. # config['GAIN']
    res_cat = {}
    negative_mask = (image < 0)  #
    sat_lim_mask = (image >= config['sat_lim'])

    # Mask only invalid and negative values
    negative_mask |= valid_mask

    # Mask negative and saturated pixel
    # sat_lim_mask |= negative_mask
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

        # Run the aperture photometry
        phot_table = aperture_photometry(image, aperture,
                                         mask=mask,
                                         method=sum_method)
        del aperture

        # Rename pix position to make it consistent
        names = ('xcenter', 'ycenter')
        new_names = ('xcentroid', 'ycentroid')
        phot_table.rename_columns(names, new_names)

        # Convert to pandas dataframe
        phot_table = phot_table.to_pandas()

        # Get total background
        total_bkg = bkg_median * aperture_area

        # Subtract background
        phot_bkgsub = phot_table['aperture_sum'] - total_bkg
        # print(phot_bkgsub, total_bkg)

        phot_table['aperture_sum_bkgsub'] = phot_bkgsub
        phot_table.loc[phot_table['aperture_sum_bkgsub'] <= 0] = 0

        # Compute the flux error using DAOPHOT style
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

       order_of_mag (x) = 10^{MAX(FLOOR(Log_{10}(x)))}

    :param x: Array of values
    :type x: array-like
    :return: Order of magnitude
    :rtype: float

    """
    idx = (np.isnan(x)) | (np.isinf(x)) | (x <= 0)

    order_of_mag = 10 ** np.nanmax(np.floor(np.log10(x[~idx])))

    return order_of_mag
