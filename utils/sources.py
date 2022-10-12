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
from __future__ import annotations

import gc

""" Modules """
import os
import sys
from typing import Optional

import astropy.wcs
import numpy as np
import pandas as pd

import inspect
import logging
import requests

from astropy import units as u
from astropy.table import (
    Table, Column)
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.time import Time
from astropy.io import (
    ascii, fits)
from astropy import stats as astrostats
from astropy.stats import (
    gaussian_fwhm_to_sigma,
    gaussian_sigma_to_fwhm,
    sigma_clipped_stats)

from astropy.convolution import (
    convolve, Tophat2DKernel,
    Gaussian2DKernel)
from astropy.modeling.fitting import LevMarLSQFitter
from packaging.version import Version

import photutils  # needed to check the version

if Version(photutils.__version__) < Version('1.1.0'):
    OLD_PHOTUTILS = True
    from photutils.segmentation import (
        detect_sources,
        deblend_sources, make_source_mask)
    # noinspection PyPep8Naming
    from photutils.segmentation import source_properties as SourceCatalog
else:
    OLD_PHOTUTILS = False
    from photutils.segmentation import (
        detect_sources, SourceCatalog,
        deblend_sources, make_source_mask,
        SegmentationImage)

from photutils.detection import (
    DAOStarFinder, find_peaks)
from photutils.psf import (
    IntegratedGaussianPRF, DAOGroup,
    IterativelySubtractedPSFPhotometry)

from photutils import (
    Background2D,  # For estimating the background
    SExtractorBackground, StdBackgroundRMS, MMMBackground, MADStdBackgroundRMS,
    BkgZoomInterpolator,  # For interpolating background
    make_source_mask)

from skimage.draw import disk

try:
    import matplotlib

except ImportError:
    plt = None
else:
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
import config.base_conf as _base_conf
import utils.photometry as phot

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

__taskname__ = 'sources'
# -----------------------------------------------------------------------------

""" Parameter used in the script """
log = logging.getLogger(__name__)

MODULE_PATH = os.path.dirname(inspect.getfile(inspect.currentframe()))


# -----------------------------------------------------------------------------


class SourceMask:
    def __init__(self, img: np.ndarray, nsigma: float = 3., npixels: int = 3):
        """ Helper for making & dilating a source mask.

        See Photutils docs for make_source_mask.

        """
        self.mask = None
        self.img = img
        self.nsigma = nsigma
        self.npixels = npixels

    def single(self, filter_fwhm: float = 3.,
               tophat_size: int = 5,
               mask: np.ndarray = None) -> np.ndarray:
        """Mask on a single scale"""
        if mask is None:
            image = self.img
        else:
            image = self.img * (1 - mask)
        mask = make_source_mask(image, nsigma=self.nsigma,
                                npixels=self.npixels,
                                dilate_size=1, filter_fwhm=filter_fwhm)
        return dilate_mask(mask, tophat_size)

    # noinspection PyAugmentAssignment
    def multiple(self, filter_fwhm: list = None,
                 tophat_size: list = None,
                 mask: np.ndarray = None):
        """Mask repeatedly on different scales"""
        if tophat_size is None:
            tophat_size = [3]
        if filter_fwhm is None:
            filter_fwhm = [3.]
        if mask is None:
            self.mask = np.zeros(self.img.shape, dtype=np.bool)
        for fwhm, tophat in zip(filter_fwhm, tophat_size):
            smask = self.single(filter_fwhm=fwhm, tophat_size=tophat)
            self.mask = self.mask | smask  # Or the masks at each iteration
        return self.mask


def build_auto_kernel(imgarr, fwhm=4.0, threshold=None, source_box=7,
                      good_fwhm=None, num_fwhm=150,
                      isolation_size=11, saturation_limit=50000., silent=False):
    """Build kernel for use in source detection based on image PSF.

    This algorithm looks for an isolated point-source that is non-saturated to use as a template
    for the source detection kernel.
    Failing to find any suitable sources, it will return a
    Gaussian2DKernel based on the provided FWHM as a default.

    Parameters
    ----------

    imgarr: ndarray
        Image array (ndarray object) with sources to be identified
    fwhm: float
        Value of FWHM to use for creating a Gaussian2DKernel object in case no suitable source
        can be identified in the image.
    threshold: float
        Value from the image which serves as the limit for determining sources.
        If None, compute a default value of (background+5*rms(background)).
        If threshold < 0.0, use the absolute value as the scaling factor for default value.
    source_box: int
        Size of box (in pixels) which defines the minimum size of a valid source.
    isolation_size: int
        Separation (in pixels) to use to identify sources that are isolated from any other sources
        in the image.
    saturation_limit: float
        Flux in the image that represents the onset of saturation for a pixel.
    good_fwhm: list, optional
        List with start and end value for range of good fwhm.
    num_fwhm: int
        Number of tries to find a good source for psf extraction.
    silent: bool, optional
        Set to True to suppress most console output

    Notes
    -----

    Ideally, it would be best to determine the saturation_limit value from the data itself,
    perhaps by looking at the pixels flagged (in the DQ array) as saturated and selecting
    the value less than the minimum flux of all those pixels, or maximum pixel value in the
    image if non-were flagged as saturated (in the DQ array).
    """
    # Initialize logging for this user-callable function
    log.setLevel(logging.getLevelName(log.getEffectiveLevel()))

    if good_fwhm is None:
        good_fwhm = [1.0, 8.0]

    # Try to use PSF derived from the image as detection kernel
    # The kernel must be derived from well-isolated sources not near the edge of the image
    kern_img = imgarr.copy()
    kernel = None
    kernel_fwhm = None
    edge = source_box * 2
    kern_img[:edge, :] = 0.0
    kern_img[-edge:, :] = 0.0
    kern_img[:, :edge] = 0.0
    kern_img[:, -edge:] = 0.0
    kernel_psf = False

    # find peaks 3 times above the threshold
    peaks = find_peaks(kern_img, threshold=3. * threshold,
                       box_size=isolation_size, border_width=25)
    # print(peaks)
    if peaks is None or (peaks is not None and len(peaks) == 0):
        tmean = threshold.mean() if isinstance(threshold, np.ndarray) else threshold
        if tmean > kern_img.mean():
            kern_stats = sigma_clipped_stats(kern_img)
            threshold = kern_stats[2]
        peaks = find_peaks(kern_img, threshold=threshold, box_size=isolation_size)
    # print(peaks)
    del kern_img

    if peaks is not None:
        # Sort based on peak_value to identify the brightest sources for use as a kernel
        peaks.sort('peak_value', reverse=True)
        # print(peaks)
        if saturation_limit:
            sat_peaks = np.where(peaks['peak_value'] > saturation_limit)[0]
            sat_index = sat_peaks[-1] + 1 if len(sat_peaks) > 0 else 0
            peaks['peak_value'][:sat_index] = 0.
        # print(peaks)
        fwhm_attempts = 0
        # Identify position of brightest, non-saturated peak (in numpy index order)
        for peak_ctr in range(len(peaks)):
            kernel_pos = [peaks['y_peak'][peak_ctr], peaks['x_peak'][peak_ctr]]

            kernel = imgarr[kernel_pos[0] - source_box:kernel_pos[0] + source_box + 1,
                     kernel_pos[1] - source_box:kernel_pos[1] + source_box + 1].copy()

            if kernel is not None:
                kernel = np.clip(kernel, 0, None)  # insure background subtracted kernel has no negative pixels
                if kernel.sum() > 0.0:
                    kernel /= kernel.sum()  # Normalize the new kernel to a total flux of 1.0
                    kernel_fwhm = find_fwhm(kernel, fwhm)
                    fwhm_attempts += 1
                    if kernel_fwhm is None:
                        kernel = None
                    else:
                        log.debug("Determined FWHM from sample PSF of {:.2f}".format(kernel_fwhm))
                        log.debug("  based on good range of FWHM:  {:.1f} to {:.1f}".format(good_fwhm[0], good_fwhm[1]))

                        # This makes it hard to work with sub-sampled data (WFPC2?)
                        if good_fwhm[1] > kernel_fwhm > good_fwhm[0]:
                            fwhm = kernel_fwhm
                            kernel_psf = True
                            break
                        else:
                            kernel = None
                if fwhm_attempts == num_fwhm:
                    break
    else:
        kernel = None

    # kernel = None
    if kernel is None:
        num_peaks = len(peaks) if peaks else 0
        if not silent:
            log.warning(f"   Did not find a suitable PSF out of {num_peaks} possible sources...")
            log.warning("   Using a Gaussian 2D Kernel for source detection.")

        # Generate a default kernel using a simple 2D Gaussian
        kernel_fwhm = fwhm
        sigma = fwhm * gaussian_fwhm_to_sigma
        k = Gaussian2DKernel(sigma,
                             x_size=source_box,
                             y_size=source_box)
        k.normalize()
        kernel = k.array

    del imgarr
    gc.collect()

    return (kernel, kernel_psf), kernel_fwhm


def classify_sources(catalog, fwhm, sources=None):
    """ Convert moments_central attribute for source catalog into a star/cr flag.

    This algorithm interprets the central_moments from the source_properties
    generated for the sources as more-likely a star or a cosmic-ray.
    It is not intended or expected to be precise, merely a means of making a first cut at
    removing likely cosmic-rays or other artifacts.

    Parameters
    ----------
    catalog: `~photutils.segmentation.SourceCatalog`
        The photutils catalog for the image/chip.
    fwhm: float
        Full-width half-maximum (fwhm) of the PSF in pixels.
    sources: tuple, optional
        Range of objects from catalog to process as a tuple of (min, max).
        If None (default), all sources are processed.

    Returns
    -------
    srctype: ndarray
        An ndarray where a value of 1 indicates a likely valid, non-cosmic-ray
        source, and a value of 0 indicates a likely cosmic-ray.
    """
    moments = catalog.moments_central
    semiminor_axis = catalog.semiminor_sigma
    elon = catalog.elongation
    if sources is None:
        sources = (0, len(moments))
    num_sources = sources[1] - sources[0]
    srctype = np.zeros((num_sources,), np.int32)
    for src in range(sources[0], sources[1]):

        # Protect against spurious detections
        src_x = catalog[src].xcentroid
        src_y = catalog[src].ycentroid
        if np.isnan(src_x) or np.isnan(src_y):
            continue

        # This identifies moment of maximum value
        x, y = np.where(moments[src] == moments[src].max())
        valid_src = (x[0] > 1) and (y[0] > 1)

        # This looks for CR streaks (not delta CRs)
        valid_width = semiminor_axis[src].value < (0.75 * fwhm)  # skinny source
        valid_elon = elon[src].value > 2  # long source
        valid_streak = valid_width and valid_elon  # long and skinny...

        # If either a delta CR or a CR streak is identified, remove it
        if valid_src and not valid_streak:
            srctype[src] = 1
    return srctype


def clean_catalog_trail(imgarr, mask, catalog, fwhm, r=3.):
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

    df = catalog.copy()

    # loop over sources
    unwanted_indices = []
    for index, row in df.iterrows():
        src_ind = disk((row['xcentroid'], row['ycentroid']), radius)
        for rs, cs in zip(src_ind[0], src_ind[1]):
            dist = np.sqrt((rs - trl_ind[1]) ** 2 + (cs - trl_ind[0]) ** 2)
            # fixme: the distance can be empty,
            #  likely due to the trail being wrongly detected
            dist_min = dist[np.argmin(dist)]
            if dist_min <= radius:
                unwanted_indices.append(index)
                break
            if dist_min > radius:
                break

    desired_df = df.drop(unwanted_indices, axis=0)
    removed_df = df.iloc[unwanted_indices]

    del imgarr, df, trl_ind
    gc.collect()

    return desired_df, removed_df


def clean_catalog_distance(in_cat: pandas.DataFrame,
                           fwhm: float, r: float = 5) -> Table:
    """Remove sources that are close to each other"""

    radius = r * fwhm
    df = in_cat.copy()
    unwanted_indices = []
    # title = _base_conf.BCOLORS.OKGREEN + "[PROGRESS]" + _base_conf.BCOLORS.ENDC
    # with alive_bar(len(df), title=title) as bar:
    for index, row in df.iterrows():
        for index2, row2 in df.iterrows():
            if index2 != index:
                src_ind = disk((row2['xcentroid'], row2['ycentroid']), radius)
                dist = np.sqrt((row['xcentroid'] - src_ind[0]) ** 2 +
                               (row['ycentroid'] - src_ind[1]) ** 2)
                dist_min = dist[np.argmin(dist)]
                if dist_min < radius:
                    unwanted_indices.append(index)
                    break
            # bar()

    in_cat_cln = df.drop(unwanted_indices, axis=0)

    del df, unwanted_indices
    gc.collect()

    return in_cat_cln


def compute_2d_background(imgarr, box_size, win_size,
                          bkg_estimator=SExtractorBackground,
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

    if not silent:
        log.info("  > Estimate 2D background from image. (This may take a second.)")

    imgarr[np.isnan(imgarr)] = 0.

    bkg = None
    exclude_percentiles = [10, 25, 50, 75]
    for percentile in exclude_percentiles:
        if not silent:
            log.info(f"    Percentile in use: {percentile}")
        # estimate the background
        try:

            bkg = my_background(imgarr, box_size=(box_size, box_size),
                                filter_size=(win_size, win_size),
                                exclude_percentile=percentile, bkg_estimator=bkg_estimator,
                                bkgrms_estimator=rms_estimator, edge_method="pad")

            # make a deeper source mask
            if not silent:
                log.info("    Create a deeper source mask")
            sm = SourceMask(imgarr - bkg.background, nsigma=1.5)
            mask = sm.multiple(filter_fwhm=[2, 3, 5],
                               tophat_size=[4, 2, 1])

            del bkg

            if not silent:
                log.info("    Redo the background estimation with the new source mask")
            # redo the background estimation using the new mask
            # interpolator = BkgIDWInterpolator(n_neighbors=20, power=1, reg=30)
            bkg = my_background(imgarr,
                                # box_size=(0.6 * box_size, 0.6 * box_size),
                                # filter_size=(0.5 * win_size, 0.5 * win_size),
                                box_size=(7, 7),
                                filter_size=(5, 5),
                                mask=mask,
                                # interp=interpolator,
                                exclude_percentile=percentile,
                                bkg_estimator=bkg_estimator,
                                bkgrms_estimator=rms_estimator,
                                edge_method="pad")

        except Exception:
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
        mask = make_source_mask(imgarr, nsigma=2, npixels=5, dilate_size=11)
        sigcl_mean, sigcl_median, sigcl_std = sigma_clipped_stats(imgarr,
                                                                  sigma=3.0,
                                                                  mask=mask,
                                                                  maxiters=9)
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


def compute_radius(wcs: WCS, naxis1: int, naxis2: int) -> float:
    """Compute the radius from the center to the furthest edge of the WCS.

    Parameters
    -----------
    wcs:
        World coordinate system object describing translation between image and skycoord.
    naxis1:
        Axis length used to calculate the image footprint.
    naxis2:
        Axis length used to calculate the image footprint.

    Returns
    -------
    radius:
        Radius of field-of-view in arcmin.
    """

    ra, dec = wcs.wcs.crval

    img_center = SkyCoord(ra=ra * u.degree, dec=dec * u.degree)

    # calculate footprint
    wcs_foot = wcs.calc_footprint(axes=(naxis1, naxis2))

    # get corners
    img_corners = SkyCoord(ra=wcs_foot[:, 0] * u.degree,
                           dec=wcs_foot[:, 1] * u.degree)

    # make sure the radius is less than 1 deg because of GAIAdr3 search limit
    separations = img_center.separation(img_corners).value
    radius = separations.max() if separations.max() < 1. else 0.9

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
                newtimes = Time(table[old], format=dfmt).decimalyear
                timecol = Column(data=newtimes, name=new)
                table.add_column(timecol)
            else:
                # Otherwise, no format conversion needed, so simply rename it
                # since we already know the old column exists
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


def extract_source_catalog(imgarr, config, vignette=-1,
                           vignette_rectangular=-1., cutouts=None,
                           only_rectangle=None, fwhm=4.0,
                           saturation_limit=50000.0,
                           silent=False):
    """ Extract and source catalog using photutils.

    The catalog returned by this function includes sources found in the
    input image with the positions translated to the coordinate frame
    defined by the reference WCS `refwcs`.
    The sources will be
        * identified using photutils segmentation-based source finding code
        * classified as probable cosmic-rays (if enabled) using central_moments
    properties of each source, with these sources being removed from the
    catalog.

    Parameters
    ----------
    imgarr: np.ndarray
        Input image as an astropy.io.fits HDUList.
    config: dict
        Dictionary containing the configuration
    vignette: float, optional
        Cut off corners using a circle with radius (0. < vignette <= 2.). Default to -1.
    vignette_rectangular: float, optional
        Ignore a fraction of the image at the corner. Default: -1 = nothing ignored
        If fraction < 1, the corresponding (1 - frac) percentage is ignored.
        Example: 0.9 ~ 10% ignored
    cutouts: list, or list of lists(s), None, optional
        Cut out rectangular regions of the image. Format: [(xstart, xend, ystart, yend)]
    only_rectangle: list, None, optional
        Use only_rectangle within image format: (xstart, xend, ystart, yend)
    fwhm: float
        Full-width half-maximum (fwhm) of the PSF in pixels if no wcs is given. Default: 4 pixel
    saturation_limit: float, optional
        saturation limit for source selection
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

    # Get default parameter
    box_size = config['box_size']
    win_size = config['win_size']
    source_box = config['source_box']
    isolation_size = config['isolation_size']
    nsigma = config['nsigma']
    estimate_bkg = config['estimate_bkg']
    bkg_fname = config['bkg_fname']
    # print(bkg_fname)
    bkg_data = compute_2d_background(imgarr,
                                     box_size=box_size,
                                     win_size=win_size,
                                     estimate_bkg=estimate_bkg,
                                     bkg_fname=bkg_fname,
                                     silent=silent)

    bkg_ra, bkg_median, bkg_rms_ra, bkg_rms_median = bkg_data
    # set thresholds
    threshold = nsigma * bkg_rms_ra
    dao_threshold = nsigma * bkg_rms_median
    # print(threshold, dao_threshold)
    # subtract background from the image
    imgarr_bkg_subtracted = imgarr - bkg_ra

    if not silent:
        log.info("  > Mask image")
    imgarr_bkg_subtracted = mask_image(imgarr_bkg_subtracted, bkg_median,
                                       vignette=vignette, vignette_rectangular=vignette_rectangular,
                                       cutouts=cutouts, only_rectangle=only_rectangle)

    if not silent:
        log.info("  > Auto build kernel")
    (kernel, kernel_psf), kernel_fwhm = build_auto_kernel(imgarr_bkg_subtracted,
                                                          threshold=threshold,
                                                          fwhm=fwhm,
                                                          source_box=source_box,
                                                          isolation_size=isolation_size,
                                                          saturation_limit=saturation_limit,
                                                          silent=silent)

    if not silent:
        log.info(f"    Built kernel with FWHM = {kernel_fwhm}")

    # Build source catalog for entire image
    if not silent:
        log.info("> Build source catalog for entire image")
    source_cat, segmap, segmap_thld, state = extract_sources(imgarr_bkg_subtracted, kernel=kernel,
                                                             segment_threshold=threshold,
                                                             dao_threshold=dao_threshold,
                                                             fwhm=kernel_fwhm,
                                                             source_box=source_box,
                                                             # centering_mode=None,
                                                             nlargest=_base_conf.NUM_SOURCES_MAX,
                                                             MAX_AREA_LIMIT=_base_conf.MAX_AREA_LIMIT,
                                                             silent=silent)
    if not state:
        del imgarr, imgarr_bkg_subtracted, threshold, bkg_data
        gc.collect()
        return None, None, None, None, None, False

    # Remove sources above the saturation limit
    source_cat = source_cat.query("flux < " + str(saturation_limit))

    del imgarr, imgarr_bkg_subtracted, threshold, bkg_data
    gc.collect()

    return source_cat, segmap, segmap_thld, kernel, kernel_fwhm, True


def extract_sources(img, fwhm=3.0, kernel=None,
                    segment_threshold=None, dao_threshold=None,
                    dao_nsigma=3.0, source_box=5,
                    classify=True, centering_mode="starfind", nlargest=None,
                    MAX_AREA_LIMIT=1964,
                    deblend=False, silent=False):
    """Use photutils to find sources in image based on segmentation.

    Parameters
    ----------
    img: ndarray
        Numpy array of the science extension from the observations FITS file.
    fwhm: float
        Full-width half-maximum (fwhm) of the PSF in pixels.
    dao_threshold: float or None
        Value from the image which serves as the limit for determining sources.
        If None, compute a default value of (background+5*rms(background)).
        If threshold < 0.0, use absolute value as the scaling factor for default value.
    source_box: int
        Size of box (in pixels) which defines the minimum size of a valid source.
    classify: bool
        Specify whether to apply classification based on invariant moments
        of each source to determine whether a source is likely to be a
        cosmic-ray, and not include those sources in the final catalog.
    centering_mode: str
        "segmentation" or "starfind"
        Algorithm to use when computing the positions of the detected sources.
        Centering will only take place after `threshold` has been determined, and
        sources are identified using segmentation.
        Centering using `segmentation` will rely on `photutils.segmentation.source_properties` to generate the
        properties for the source catalog. Centering using `starfind` will use
        `photutils.detection.IRAFStarFinder` to characterize each source in the catalog.
    nlargest: int, None
        Number of the largest (brightest) sources in each chip/array to measure
        when using 'starfind' mode.
    deblend: bool, optional
        Specify whether to apply photutils deblending algorithm when
        evaluating each of the identified segments (sources).
    silent: bool, optional
        Set to True to suppress most console output

    """
    # Initialize logging for this user-callable function
    log.setLevel(logging.getLevelName(log.getEffectiveLevel()))

    imgarr = img.copy()

    if dao_threshold is None:
        dao_threshold, bkg = sigma_clipped_bkg(imgarr, sigma=3.0, nsigma=dao_nsigma, maxiters=3)

    if segment_threshold is None:
        segment_threshold = np.ones(imgarr.shape, imgarr.dtype) * dao_threshold

    segm = detect_sources(imgarr, segment_threshold, npixels=source_box,
                          kernel=kernel, connectivity=8)

    # photutils >= 0.7: segm=None; photutils < 0.7: segm.nlabels=0
    if segm is None or segm.nlabels == 0:
        if not silent:
            log.warning("No detected sources!")

        del img, imgarr, segm, dao_threshold, segment_threshold
        gc.collect()

        return None, None, None

    log.debug("Creating segmentation map.")
    if kernel is not None:
        kernel_area = ((kernel.shape[0] // 2) ** 2) * np.pi
        log.debug(f"   based on kernel shape of {kernel.shape}")
    else:
        kernel_area = ((source_box // 2) ** 2) * np.pi
        log.debug("   based on a default kernel.")

    num_brightest = 10 if len(segm.areas) > 10 else len(segm.areas)

    mean_area = np.mean(segm.areas)
    max_area = np.sort(segm.areas)[-1 * num_brightest:].mean()

    # This section looks for crowded fields where segments run into each other
    # By reducing the size of the kernel used for segment detection,
    # This can be minimized in crowded fields.
    # Also, the mean area is used to try to avoid this logic for fields with
    # several large extended sources in an otherwise empty field.
    if max_area > MAX_AREA_LIMIT and mean_area > (kernel_area / 2):  # largest > 25-pix radius source

        # reset kernel to only use the central 1/4 area and redefine the segment map
        kcenter = (kernel.shape[0] - 1) // 2
        koffset = (kcenter - 1) // 2
        kernel = kernel[kcenter - koffset: kcenter + koffset + 1, kcenter - koffset: kcenter + koffset + 1].copy()
        kernel /= kernel.sum()  # normalize to total sum == 1
        log.debug(f"Looking for crowded sources using smaller kernel with shape: {kernel.shape}")
        segm = detect_sources(imgarr, segment_threshold, npixels=source_box, kernel=kernel)

    if deblend:
        segm = deblend_sources(imgarr, segm, npixels=5,
                               kernel=kernel, nlevels=32,
                               contrast=1e-4)

    # If classify is turned on, it should modify the segmentation map
    if classify:
        log.debug('Removing suspected cosmic-rays from source catalog')
        cat = SourceCatalog(data=imgarr, segment_img=segm, kernel=kernel)

        # Remove likely cosmic-rays based on central_moments classification
        bad_srcs = np.where(classify_sources(cat, fwhm) == 0)[0] + 1

        segm.remove_labels(bad_srcs)

    if OLD_PHOTUTILS:
        flux_colname = 'source_sum'
        ferr_colname = 'source_sum_err'
    else:
        flux_colname = 'segment_flux'
        ferr_colname = 'segment_fluxerr'

    # convert segm to mask for daofind
    if centering_mode == 'starfind':
        src_table = None

        # Identify nbrightest/largest sources
        if nlargest is not None:
            if not silent and nlargest < len(segm.labels):
                log.info(f"  {len(segm.labels)} sources detected. Limit number to {nlargest}")
            nlargest = min(nlargest, len(segm.labels))

            # Look for the brightest sources by flux...
            src_fluxes = np.array([imgarr[src].max() for src in segm.slices])
            src_labels = np.array([lbl for lbl in segm.labels])
            src_brightest = np.flip(np.argsort(src_fluxes))
            large_labels = src_labels[src_brightest]
            log.debug(f"Brightest sources in segments: \n{large_labels}")
        else:
            src_brightest = np.arange(len(segm.labels))

        log.debug(f"Looking for sources in {len(segm.labels)} segments")

        for idx in src_brightest:
            segment = segm.segments[idx]

            if segment is None:
                continue

            # Get slice definition for the segment with this label
            seg_slice = segment.slices
            seg_yoffset = seg_slice[0].start
            seg_xoffset = seg_slice[1].start

            dao_threshold = segment_threshold[seg_slice].mean()
            daofind = DAOStarFinder(fwhm=fwhm, threshold=dao_threshold)
            log.debug(f"Setting up DAOStarFinder with: \n    fwhm={fwhm}  threshold={dao_threshold}")

            # Define raw data from this slice
            detection_img = img[seg_slice]

            # zero out any pixels which do not have this segment label
            detection_img[segm.data[seg_slice] == 0] = 0

            # Detect sources in this specific segment
            seg_table = daofind.find_stars(detection_img)

            # Pick out the brightest source only
            if src_table is None and seg_table:
                # Initialize final master source list catalog
                log.debug(f"Defining initial src_table based on: {seg_table.colnames}")
                src_table = Table(names=seg_table.colnames,
                                  dtype=[dt[1] for dt in seg_table.dtype.descr])

            if seg_table:
                # This logic will eliminate saturated sources, where the max pixel value is not
                # the center of the PSF (saturated and streaked along the Y axis)
                max_row = np.where(seg_table['peak'] == seg_table['peak'].max())[0][0]

                # Add logic to remove sources which have more than 3 pixels
                # within 10% of the max value in the source segment, a situation
                # which would indicate the presence of a saturated source
                if (detection_img > detection_img.max() * 0.9).sum() > 3:
                    # Revert to segmentation photometry for sat. source positions
                    if OLD_PHOTUTILS:
                        segment_properties = SourceCatalog(detection_img, segment.data,
                                                           kernel=kernel)
                    else:
                        segimg = SegmentationImage(segment.data)
                        segment_properties = SourceCatalog(detection_img, segimg,
                                                           kernel=kernel)

                    sat_table = segment_properties.to_table()
                    seg_table['flux'][max_row] = sat_table[flux_colname][0]
                    seg_table['peak'][max_row] = sat_table['max_value'][0]
                    if OLD_PHOTUTILS:
                        xcentroid = sat_table['xcentroid'][0].value
                        ycentroid = sat_table['ycentroid'][0].value
                        sky = sat_table['background_mean'][0].value
                    else:
                        xcentroid = sat_table['xcentroid'][0]
                        ycentroid = sat_table['ycentroid'][0]
                        sky = sat_table['local_background'][0]

                    seg_table['xcentroid'][max_row] = xcentroid
                    seg_table['ycentroid'][max_row] = ycentroid
                    seg_table['npix'][max_row] = sat_table['area'][0].value
                    seg_table['sky'][max_row] = sky if sky is not None and not np.isnan(sky) else 0.0
                    seg_table['mag'][max_row] = -2.5 * np.log10(sat_table[flux_colname][0])

                # Add row for the detected source to the master catalog
                # apply offset to slice to convert positions into full-frame coordinates
                seg_table['xcentroid'] += seg_xoffset
                seg_table['ycentroid'] += seg_yoffset
                src_table.add_row(seg_table[max_row])

            # If we have accumulated the desired number of sources, stop looking for more...
            if nlargest is not None and src_table is not None and len(src_table) == nlargest:
                break
    else:
        log.debug("Determining source properties as src_table...")
        cat = SourceCatalog(data=img, segment_img=segm, kernel=kernel)
        src_table = cat.to_table()

        # Make column names consistent with IRAFStarFinder column names
        src_table.rename_column(flux_colname, 'flux')
        src_table.rename_column(ferr_colname, 'flux_err')
        src_table.rename_column('max_value', 'peak')

    if src_table is not None and len(src_table) >= 3:
        log.info(f"    Total Number of detected sources: {len(src_table)}")
    elif src_table is not None and len(src_table) < 3:
        log.critical(f"Less than 3 sources detected. Skipping further steps")
        del img, imgarr, segm, dao_threshold, segment_threshold
        gc.collect()
        return None, None, None, False
    else:
        log.info("    No detected sources!")
        del img, imgarr, segm, dao_threshold, segment_threshold
        gc.collect()
        return None, None, None, False

    # Move 'id' column from first to last position
    # Makes it consistent for the remainder of code
    cnames = src_table.colnames
    cnames.append(cnames[0])
    del cnames[0]

    for col in src_table.colnames:
        src_table[col].info.format = '%.8g'  # for consistent table output

    tbl = src_table[cnames]
    # Insure all IDs are sequential and unique (at least in this catalog)
    tbl['cat_id'] = np.arange(1, len(tbl) + 1)
    del tbl['id']

    del img, imgarr, dao_threshold, src_table
    gc.collect()

    # return catalog table, segments map, and
    return tbl.to_pandas(), segm, segment_threshold, True


def find_fwhm(psf: np.ndarray, default_fwhm: float) -> float | None:
    """ Determine FWHM for auto-kernel PSF.

    This function iteratively fits a Gaussian model to the extracted PSF
    using `photutils.psf.IterativelySubtractedPSFPhotometry` to determine
    the FWHM of the PSF.

    Parameters
    ----------
    psf:
        Array (preferably a slice) containing the PSF to be measured.
    default_fwhm:
        Starting guess for the FWHM

    Returns
    -------
    fwhm: float
        Value of the computed Gaussian FWHM for the PSF
    """
    # Initialize logging for this user-callable function
    log.setLevel(logging.getLevelName(log.getEffectiveLevel()))

    daogroup = DAOGroup(crit_separation=8.)
    mmm_bkg = MMMBackground()
    iraffind = DAOStarFinder(threshold=2.5 * mmm_bkg(psf),
                             sigma_radius=2.5, exclude_border=True,
                             fwhm=default_fwhm)
    fitter = LevMarLSQFitter()
    sigma_psf = gaussian_fwhm_to_sigma * default_fwhm
    gaussian_prf = IntegratedGaussianPRF(sigma=sigma_psf)
    gaussian_prf.sigma.fixed = False
    itr_phot_obj = IterativelySubtractedPSFPhotometry(finder=iraffind,
                                                      group_maker=daogroup,
                                                      bkg_estimator=mmm_bkg,
                                                      psf_model=gaussian_prf,
                                                      fitter=fitter,
                                                      fitshape=(11, 11),
                                                      niters=2)
    phot_results = itr_phot_obj(psf)

    # Insure none of the fluxes determined by photutils is np.nan
    phot_results['flux_fit'] = np.nan_to_num(phot_results['flux_fit'].data, nan=0)

    if len(phot_results['flux_fit']) == 0:
        return None
    psf_row = np.where(phot_results['flux_fit'] == phot_results['flux_fit'].max())[0][0]
    sigma_fit = phot_results['sigma_fit'][psf_row]
    fwhm = gaussian_sigma_to_fwhm * sigma_fit

    log.debug(f"Found FWHM: {fwhm}")

    del psf, daogroup, mmm_bkg, iraffind, itr_phot_obj, phot_results, psf_row, fitter
    gc.collect()

    return fwhm


def find_worst_residual_near_center(resid: np.ndarray):
    """Find the pixel location of the worst residual, avoiding the edges"""
    yc, xc = resid.shape[0] / 2., resid.shape[1] / 2.
    radius = resid.shape[0] / 3.
    y, x = np.mgrid[0:resid.shape[0], 0:resid.shape[1]]
    mask = np.sqrt((y - yc) ** 2 + (x - xc) ** 2) < radius
    rmasked = resid * mask
    return np.unravel_index(np.argmax(rmasked), resid.shape)


def get_reference_catalog(ra, dec, sr=0.1, epoch=None, num_sources=None, catalog='GSC242',
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
        Name of catalog to query, as defined by web-service.  Default: 'GSC241'
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
    spec_str = 'RA={}&DEC={}&SR={}&FORMAT={}&CAT={}&MINDET=5'
    headers = {'Content-Type': 'text/csv'}
    fmt = 'CSV'
    epoch_str = '&EPOCH={:.3f}'

    base_spec = spec_str.format(ra, dec, sr, fmt, catalog)
    spec = base_spec + epoch_str.format(epoch) if epoch else base_spec
    if not silent:
        log.info("> Get reference catalog via VO web service")

    url_chk = url_checker(f'{_base_conf.SERVICELOCATION}/{serviceType}')
    if not url_chk[0]:
        log.error(f"  {url_chk[1]}. ")
        sys.exit(1)
    if not silent:
        log.info(f"  {url_chk[1]}. Downloading data...")

    serviceUrl = f'{_base_conf.SERVICELOCATION}/{serviceType}?{spec}'
    # print(serviceUrl)
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

    # If we still have an error returned by the web-service, report the exact error
    if rstr[0].startswith('Error'):
        log.warning(f"Astrometric catalog generation FAILED with: \n{rstr}")

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


def get_src_and_cat_info(fname, loc, imgarr, hdr, wcsprm,
                         catalog, has_trail=False, mode='astro',
                         silent=False, **config):
    """Extract astrometric positions and photometric data for sources in the
            input images' field-of-view.

        """

    # Initialize logging for this user-callable function
    log.setLevel(logging.getLevelName(log.getEffectiveLevel()))
    save_cat = True  # todo: remove this maybe?
    ref_tbl_photo = None
    ref_catalog_photo = None
    read_src_cat = read_ref_cat_astro = read_ref_cat_photo = True

    src_cat_fname = f'{loc}/{fname}_src_cat'
    astro_ref_cat_fname = f'{loc}/{fname}_ref_cat'
    photo_ref_cat_fname = f'{loc}/{fname}_trail_img_photo_ref_cat'

    # check catalog input
    if mode == 'astro' and catalog.upper() not in _base_conf.SUPPORTED_CATALOGS:
        log.warning(f"Given astrometry catalog '{catalog}' NOT SUPPORTED. "
                    "Defaulting to GAIAdr3")
        catalog = "GAIAedr3"
    elif mode == 'photo' and catalog.upper() not in _base_conf.SUPPORTED_CATALOGS:
        log.warning(f"Given photometry catalog '{catalog}' NOT SUPPORTED. "
                    "Defaulting to GSC242")
        catalog = "GSC242"

    # check output
    if config["_src_cat_fname"] is not None:
        src_cat_fname = config["_src_cat_fname"]
    if config["_ref_cat_fname"] is not None:
        astro_ref_cat_fname = config["_ref_cat_fname"]
    if config["_photo_ref_cat_fname"] is not None:
        photo_ref_cat_fname = config["_photo_ref_cat_fname"]

    # get the observation date
    t = pd.to_datetime(hdr['date-obs'], format=_base_conf.FRMT, utc=False)
    if 'time-obs' in hdr:
        t = pd.to_datetime(f"{hdr['date-obs']}T{hdr['time-obs']}",
                           format=_base_conf.FRMT, utc=False)
    epoch = Time(t).decimalyear

    # estimate the radius of the FoV for ref. catalog creation
    fov_radius = compute_radius(WCS(wcsprm.to_header()),
                                naxis1=hdr['NAXIS1'], naxis2=hdr['NAXIS2'])

    if mode == 'astro':
        fov_radius = config["_fov_radius"]

    fov_radius_deg = fov_radius
    ra, dec = wcsprm.crval

    # check for source catalog file. If present and not force extraction use these catalogs
    chk_src_cat = os.path.isfile(src_cat_fname + '.cat')
    chk_ref_cat_astro = os.path.isfile(astro_ref_cat_fname + '.cat')
    chk_ref_cat_photo = os.path.isfile(photo_ref_cat_fname + '.cat')

    if config["_force_extract"]:
        read_src_cat = read_ref_cat_astro = read_ref_cat_photo = False
    else:
        if not chk_src_cat:
            read_src_cat = False
        if not chk_ref_cat_astro:
            read_ref_cat_astro = False
        if not chk_ref_cat_photo:
            read_ref_cat_photo = False

    kernel = None
    segmap = None
    segmap_thld = None
    if read_src_cat:
        if not silent:
            log.info("> Load source catalog from file")
        src_tbl, kernel_fwhm, _ = read_catalog(src_cat_fname)

    else:
        saturation_limit = config["saturation_limit"]

        # detect sources in image and get positions
        src_tbl, segmap, segmap_thld, kernel, kernel_fwhm, state = \
            extract_source_catalog(imgarr=imgarr,
                                   config=config,
                                   vignette=config["_vignette"],
                                   vignette_rectangular=config["_vignette_rectangular"],
                                   cutouts=config["_cutouts"],
                                   fwhm=config["average_fwhm"],
                                   saturation_limit=saturation_limit,
                                   silent=silent)
        if not state:
            del imgarr, src_tbl, segmap, segmap_thld, kernel, kernel_fwhm
            gc.collect()
            return (None for _ in range(12)), False

        # add positions to table
        pos_on_sky = WCS(hdr).wcs_pix2world(src_tbl[["xcentroid", "ycentroid"]], 1)
        src_tbl["RA"] = pos_on_sky[:, 0]
        src_tbl["DEC"] = pos_on_sky[:, 1]
        if mode == 'photo' and save_cat:
            if not silent:
                log.info("> Save source catalog.")
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
            get_reference_catalog(ra=ra, dec=dec, sr=fov_radius_deg,
                                  epoch=epoch, num_sources=3 * _base_conf.NUM_SOURCES_MAX,
                                  catalog='GAIAedr3',
                                  silent=silent)
        # add positions to table
        pos_on_det = wcsprm.s2p(ref_tbl_astro[["RA", "DEC"]].values, 1)['pixcrd']
        ref_tbl_astro["xcentroid"] = pos_on_det[:, 0]
        ref_tbl_astro["ycentroid"] = pos_on_det[:, 1]
        if mode == 'photo' and save_cat:
            if not silent:
                log.info("> Save astrometric reference catalog.")
                save_catalog(cat=ref_tbl_astro, wcsprm=wcsprm, out_name=astro_ref_cat_fname,
                             mode='ref_astro', catalog=ref_catalog_astro)

    if mode == 'photo' and has_trail:
        # get photometric reference catalog
        if read_ref_cat_photo:
            log.info("> Load photometric reference source catalog from file")
            ref_tbl_photo, _, ref_catalog_photo = read_catalog(photo_ref_cat_fname)
        else:
            # get reference catalog
            ref_tbl_photo, ref_catalog_photo = \
                get_reference_catalog(ra=ra, dec=dec, sr=fov_radius_deg,
                                      epoch=epoch,  # num_sources=_base_conf.NUM_SOURCES_MAX,
                                      catalog=catalog,
                                      full_catalog=True, silent=silent)
            # add positions to table
            pos_on_det = wcsprm.s2p(ref_tbl_photo[["RA", "DEC"]].values, 1)['pixcrd']
            ref_tbl_photo["xcentroid"] = pos_on_det[:, 0]
            ref_tbl_photo["ycentroid"] = pos_on_det[:, 1]

            if not silent:
                log.info("> Save photometry reference catalog.")
            save_catalog(cat=ref_tbl_photo, wcsprm=wcsprm, out_name=photo_ref_cat_fname,
                         mode='ref_photo', catalog=ref_catalog_photo)

    return (src_tbl, ref_tbl_astro, ref_catalog_astro, ref_tbl_photo, ref_catalog_photo,
            src_cat_fname, astro_ref_cat_fname, photo_ref_cat_fname,
            kernel_fwhm, kernel, segmap, segmap_thld), True


def mask_image(image,
               bkg_median,
               vignette,
               vignette_rectangular,
               cutouts,
               only_rectangle, silent=False):
    """Mask image"""

    # Initialize logging for this user-callable function
    log.setLevel(logging.getLevelName(log.getEffectiveLevel()))

    imgarr = image.copy()

    # only search sources in a circle with radius <vignette>
    if (0. < vignette < 2.) & (vignette != -1.):
        sidelength = np.max(imgarr.shape)
        x = np.arange(0, imgarr.shape[1])
        y = np.arange(0, imgarr.shape[0])
        if not silent:
            log.info("    Only search sources in a circle "
                     "with radius {}px".format(vignette * sidelength / 2.))
        vignette = vignette * sidelength / 2.
        mask = (x[np.newaxis, :] - sidelength / 2) ** 2 + \
               (y[:, np.newaxis] - sidelength / 2) ** 2 < vignette ** 2
        imgarr[~mask] = bkg_median

    # ignore a fraction of the image at the corner
    if (0. < vignette_rectangular < 1.) & (vignette_rectangular != -1.):
        if not silent:
            log.info("    Ignore {0:0.1f}% of the image at the corner. ".format((1. - vignette_rectangular) * 100.))
        sidelength_x = imgarr.shape[1]
        sidelength_y = imgarr.shape[0]
        cutoff_left = (1. - vignette_rectangular) * sidelength_x
        cutoff_right = vignette_rectangular * sidelength_x
        cutoff_bottom = (1. - vignette_rectangular) * sidelength_y
        cutoff_top = vignette_rectangular * sidelength_y
        x = np.arange(0, imgarr.shape[1])
        y = np.arange(0, imgarr.shape[0])
        left = x[np.newaxis, :] > cutoff_left
        right = x[np.newaxis, :] < cutoff_right
        bottom = y[:, np.newaxis] > cutoff_bottom
        top = y[:, np.newaxis] < cutoff_top
        mask = (left * bottom) * (right * top)
        imgarr[~mask] = bkg_median

    # cut out rectangular regions of the image, [(xstart, xend, ystart, yend)]
    if cutouts is not None and all(isinstance(el, list) for el in cutouts):
        x = np.arange(0, imgarr.shape[1])
        y = np.arange(0, imgarr.shape[0])
        for cutout in cutouts:
            if not silent:
                log.info("    Cutting out rectangular region {} of image. "
                         "(xstart, xend, ystart, yend)".format(cutout))
            left = x[np.newaxis, :] > cutout[0]
            right = x[np.newaxis, :] < cutout[1]
            bottom = y[:, np.newaxis] > cutout[2]
            top = y[:, np.newaxis] < cutout[3]
            mask = (left * bottom) * (right * top)
            imgarr[mask] = bkg_median

    # use only_rectangle within image format: (xstart, xend, ystart, yend)
    if only_rectangle is not None and isinstance(only_rectangle, tuple):
        x = np.arange(0, imgarr.shape[1])
        y = np.arange(0, imgarr.shape[0])
        if not silent:
            log.info("    Use only rectangle {} within image. "
                     "(xstart, xend, ystart, yend)".format(only_rectangle))
        left = x[np.newaxis, :] > only_rectangle[0]
        right = x[np.newaxis, :] < only_rectangle[1]
        bottom = y[:, np.newaxis] > only_rectangle[2]
        top = y[:, np.newaxis] < only_rectangle[3]
        mask = (left * bottom) * (right * top)
        imgarr[~mask] = bkg_median

    return imgarr


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
                        sigma_clip=astrostats.SigmaClip(sigma=3.),
                        exclude_percentile=exclude_percentile,
                        mask=mask,
                        bkg_estimator=bkg_estimator(),
                        bkgrms_estimator=bkgrms_estimator(),
                        edge_method=edge_method,
                        interpolator=interp, filter_size=filter_size)


def plot_mask(scene, bkgd, mask, zmin, zmax, worst=None, smooth=0):
    """Make a three-panel plot of:
         * the mask for the whole image,
         * the scene times the mask
         * a zoomed-in region, with the mask shown as contours
    """
    if worst is None:
        y, x = find_worst_residual_near_center(bkgd)
    else:
        x, y = worst
    plt.figure(figsize=(20, 10))
    plt.subplot(131)
    plt.imshow(mask, vmin=0, vmax=1, cmap=plt.cm.get_cmap('gray'), origin='lower')
    plt.subplot(132)
    if smooth == 0:
        plt.imshow((scene - bkgd) * (1 - mask), vmin=zmin, vmax=zmax, origin='lower')
    else:
        smoothed = convolve((scene - bkgd) * (1 - mask), Gaussian2DKernel(smooth))
        plt.imshow(smoothed * (1 - mask), vmin=zmin / smooth, vmax=zmax / smooth,
                   origin='lower')
    plt.subplot(133)
    plt.imshow(scene - bkgd, vmin=zmin, vmax=zmax)
    plt.contour(mask, colors='red', alpha=0.2)
    plt.ylim(y - 100, y + 100)
    plt.xlim(x - 100, x + 100)
    return x, y


def rename_colname(table: Table, colname: str, newcol: str):
    """Convert column name in table to user-specified name"""
    # If table is missing a column, add a column with values of None
    if colname != '':
        table.rename_column(colname, newcol)
    else:
        empty_column = Column(data=[None] * len(table), name=newcol, dtype=np.float64)
        table.add_column(empty_column)


def read_catalog(cat_fname: str, tbl_format: str = "ecsv") -> tuple:
    """Load catalog from file"""

    cat = ascii.read(cat_fname + '.cat', format=tbl_format)

    kernel_fwhm = 4.
    catalog = 'GAIAedr3'

    if 'kernel_fwhm' in cat.meta.keys():
        kernel_fwhm = cat.meta['kernel_fwhm']

    if 'catalog' in cat.meta.keys():
        catalog = cat.meta['catalog']

    return cat.to_pandas(), kernel_fwhm, catalog


def save_catalog(cat: pandas.DataFrame,
                 wcsprm: astropy.wcs.Wcsprm,
                 out_name: str, kernel_fwhm: float = None,
                 catalog: str = None, mode: str = 'src', tbl_format: str = "ecsv"):
    """Save given catalog to .cat file"""

    # convert wcs parameters to wcs
    wcs = WCS(wcsprm.to_header())

    cat_out = cat.copy()

    if mode == 'src':
        # get position on sky
        pos_on_sky = wcs.wcs_pix2world(cat[["xcentroid", "ycentroid"]], 1)
        cat_out["RA"] = pos_on_sky[:, 0]
        cat_out["DEC"] = pos_on_sky[:, 1]
        cols = ['RA', 'DEC', 'xcentroid', 'ycentroid', 'mag', 'cat_id',
                'sharpness', 'roundness1', 'roundness2',
                'npix', 'sky', 'peak', 'flux']

    else:
        # get position on the detector
        pos_on_det = wcsprm.s2p(cat[["RA", "DEC"]].values, 1)['pixcrd']
        cat_out["xcentroid"] = pos_on_det[:, 0]
        cat_out["ycentroid"] = pos_on_det[:, 1]
        cols = ['RA', 'DEC', 'xcentroid', 'ycentroid', 'mag', 'objID']

    # todo: optional format maybe?
    # cat['xcentroid'].format = '.10f'
    # cat['ycentroid'].format = '.10f'
    # cat['flux'].format = '.10f'

    if mode == 'ref_astro':
        cat_out = cat_out[cols]

    # convert to astropy.Table and add meta info
    cat_out = Table.from_pandas(cat_out, index=True)
    cat_out.meta = {'kernel_fwhm': kernel_fwhm,
                    'catalog': catalog}

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
    """
    # todo: add preferable catalog and allow multiple catalogs for a specific band
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
                      "Possible catalogs are: {}".format(", ".join(_base_conf.SUPPORTED_CATALOGS.keys())))
            sys.exit(1)
        else:
            if source != _base_conf.SUPPORTED_BANDS[band]:
                log.error("Given band is not supported for this catalog. "
                          "Possible catalog for {} band: {}".format(band, _base_conf.SUPPORTED_BANDS[band]))
                sys.exit(1)
    if source == 'auto':
        catalog_name = _base_conf.SUPPORTED_BANDS[band]

    return catalog_name


def select_std_stars(ref_cat: pd.DataFrame,
                     catalog: str, band: str,
                     num_std_max: int = 10,
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

    # convert fits filter to catalog filter + error (if available)
    filter_keys = _base_conf.CATALOG_FILTER_EXT[catalog][band]['Prim']

    # result column names
    cols = ['objID', 'RA', 'DEC', 'xcentroid', 'ycentroid']
    cols += filter_keys

    # remove nan and err=0 values
    cat_srt = ref_cat.copy()
    df = cat_srt.dropna(subset=filter_keys)
    df = df[df[filter_keys[1]] != 0.]

    n = len(df)
    alt_filter_key = _base_conf.CATALOG_FILTER_EXT[catalog][band]['Alt']
    if n < num_std_min and alt_filter_key is not None:
        if not silent:
            log.warning(f"    ==> No or less than {num_std_min} stars "
                        f"in {filter_keys[0]} band.")
            log.warning(f"        Using alternative band with known magnitude conversion.")
        alt_filter_keys = np.asarray(alt_filter_key, str)

        alt_cat = ref_cat.dropna(subset=alt_filter_keys.flatten())

        x1 = alt_cat[alt_filter_keys[0, :]]
        x1 = x1.to_numpy()
        x2 = alt_cat[alt_filter_keys[1, :]].to_numpy()
        x3 = alt_cat[alt_filter_keys[2, :]].to_numpy()

        has_data = np.any(np.array([x1, x2, x3]), axis=1).all()
        if not has_data:
            log.critical("Insufficient data for magnitude conversion.")
            del ref_cat, cat_srt, df, alt_cat
            gc.collect()
            return None, filter_keys, has_mag_conv

        # convert band from catalog to observation
        alt_mags, alt_mags_err = phot.convert_ssds_to_bvri(f=filter_keys[0],
                                                           x1=x1, x2=x2, x3=x3)

        new_df = pd.DataFrame({filter_keys[0]: alt_mags,
                               filter_keys[1]: alt_mags_err}, index=alt_cat.index)

        cat_srt.update(new_df)
        has_mag_conv = True

    # sort table by magnitude, brightest to fainter
    df = cat_srt.sort_values(by=[filter_keys[0]], axis=0, ascending=True, inplace=False)
    df.dropna(subset=filter_keys, inplace=True)

    # select the std stars by number
    if num_std_max is not None:
        idx = num_std_max
        df = df[:idx]

    df = df[cols]

    del ref_cat, cat_srt
    gc.collect()

    return df, filter_keys, has_mag_conv


def sigma_clipped_bkg(arr: np.array, sigma: float = 3.0,
                      nsigma: float = 4., maxiters: float = None) -> tuple:
    """Compute the sigma clipped background."""

    # Account for input being blank
    if arr.max() == 0:
        return 0.0, [0.0, 0.0, 0.0]
    if maxiters is None:
        maxiters = int(np.log10(arr.max() / 2.) + 0.5)

    # Use a simple constant background to avoid problems with nebulosity
    bkg = sigma_clipped_stats(arr, sigma=sigma, maxiters=maxiters)

    # total background = mean + 4 * sigma
    bkg_total = bkg[0] + nsigma * bkg[2]

    del arr

    return bkg_total, bkg


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
