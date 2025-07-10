#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Name:         reduceSatObs.py
# Purpose:      Pipeline to reduce LEOSat observations from different
#               telescopes
#
#
#
# Author:       p4adch (cadam)
#
# Created:      09/29/2021
# Copyright:    (c) p4adch 2010-
#
# History:
#
# 29.09.2021
# - file created and basic methods
#
# -----------------------------------------------------------------------------

""" Modules """
from __future__ import annotations

import argparse
import collections
import gc
import logging
import os

import re
import sys
import time
from copy import deepcopy
from datetime import datetime
from datetime import timedelta
from pathlib import Path

import ccdproc

import numpy as np
import pandas as pd

import photutils

from astropy import units as u
from astropy.io import fits
from astropy.stats import (sigma_clip, mad_std, SigmaClip, sigma_clipped_stats)

from ccdproc import CCDData
from ccdproc import ImageFileCollection
from photutils.background import (Background2D, SExtractorBackground, StdBackgroundRMS)
from photutils.segmentation import (detect_sources, detect_threshold)

# Project modules
try:
    import leosatpy
except ModuleNotFoundError:
    from utils import arguments
    from utils import dataset
    from utils import tables
    from utils.version import __version__
    from utils import base_conf as bc
else:
    from leosatpy.utils import arguments
    from leosatpy.utils import dataset
    from leosatpy.utils import tables
    from leosatpy.utils.version import __version__
    from leosatpy.utils import base_conf as bc

# -----------------------------------------------------------------------------

""" Meta-info """
__author__ = "Christian Adam"
__copyright__ = 'Copyright 2021-2023, CLEOSat group'
__credits__ = ["Eduardo Unda-Sanzana, Jeremy Tregloan-Reed, Christian Adam"]
__license__ = "GPL-3.0 license"
__maintainer__ = "Christian Adam"
__email__ = "christian.adam84@gmail.com"
__status__ = "Production"

__taskname__ = 'reduceSatObs'
# -----------------------------------------------------------------------------

""" Parameter used in the script """
# Logging and console output
logging.root.handlers = []
_log = logging.getLogger()
_log.setLevel(bc.LOG_LEVEL)
stream = logging.StreamHandler()
stream.setFormatter(bc.FORMATTER)
_log.addHandler(stream)
_log_level = _log.level


# -----------------------------------------------------------------------------


class ReduceSatObs(object):
    """ Class to reduce satellite observations from different telescopes."""

    def __init__(self,
                 input_path: str,
                 args: argparse.Namespace = None,
                 silent: bool = False, verbose: bool = False,
                 log: logging.Logger = _log, log_level: int = _log_level):
        """ Constructor with default values """

        ignore_warnings = args.ignore_warnings
        if silent:
            verbose = False
            ignore_warnings = True

        if verbose:
            log = logging.getLogger()
            log.setLevel("debug".upper())
            log_level = log.level

        if ignore_warnings:
            bc.load_warnings()

        # Set variables
        self._config = collections.OrderedDict()
        self._dataset_object = None
        self._input_path = input_path
        self._log = log
        self._log_level = log_level
        self._root_dir = bc.ROOT_DIR
        self._master_calib_path = None
        self._master_bias_dict = {}
        self._master_dark_dict = {}
        self._master_flat_dict = {}
        self._dark_exptimes = []
        self._force_reduction = args.force_reduction
        # self._make_mbias = False
        # self._make_mdark = False
        # self._make_mflat = False
        # self._make_light = False
        self._ccd_mask_fname = None
        self._instrument = None
        self._telescope = None
        self._obsparams = None
        self._tmp_path = None
        self._red_path = None
        self._sci_root_path = None
        self._suffix = None
        self.silent = silent
        self.verbose = verbose

        self._obsTable = None
        self._bin_str = None
        self._binxy = [1, 1]

        # Run full reduction
        self.run_full_reduction(silent=silent, verbose=verbose)

    def run_full_reduction(self, silent=False, verbose=False):
        """Run full-reduction routine on the input path.

        Prepare science files, find calibration files,
        i.e., master_bias, master_dark, and master_flat
        and run reduction for a given set of data.
        """
        starttime = time.perf_counter()
        self._log.info('====> Science image reduction init <====')

        if not silent:
            self._log.info("> Search input and prepare data")
        if verbose:
            self._log.debug("  > Check input argument(s)")

        # Prepare dataset from input argument
        ds = dataset.DataSet(input_args=self._input_path,
                             prog_typ='reduceSatObs',
                             prog_sub_typ=['light'],
                             log=self._log, log_level=self._log_level)

        # Load configuration file
        ds.load_config()
        self._config = ds.config

        # Load observation result table
        self._obsTable = tables.ObsTables(config=self._config)
        self._obsTable.load_obs_table()

        # Set variables for use
        inst_list = ds.instruments_list
        inst_data = ds.instruments

        n_inst = len(inst_list)
        for i in range(n_inst):
            inst = inst_list[i]
            self._instrument = inst
            self._telescope = inst_data[inst]['telescope']
            self._obsparams = inst_data[inst]['obsparams']
            ds.get_valid_sci_observations(inst, prog_typ="reduceSatObs")
            obsfile_list = ds.valid_sci_obs
            self._dataset_object = obsfile_list

            # Loop over groups and run reduction for each group
            for src_path, files in obsfile_list:
                if not silent:
                    self._log.info('====> Science image reduction run <====')
                    self._log.info(f"> Reduce {len(files)} datasets from instrument {inst} "
                                   f"at the {self._telescope} "
                                   "telescope in folder:")
                    self._log.info(f"  {src_path}")

                # Reduce the individual dataset
                self.run_single_reduction(src_path, files)

        endtime = time.perf_counter()
        dt = endtime - starttime
        td = timedelta(seconds=dt)

        if not silent:
            self._log.info(f"Program execution time: {td}")
        self._log.info('====> Science image reduction finished <====')

    def run_single_reduction(self, sci_src_path, fits_files_dict):
        """Run the full reduction process on a given dataset.

        This method creates the necessary folder structure, copies science and
        calibration files, and runs the entire data reduction procedure.

        Parameters
        ----------
        sci_src_path : str
            Path to the source directory containing the science data files to be processed.
        fits_files_dict : dict
            Dictionary containing metadata about the FITS files, including the paths
            to science files and their suffixes.

        Returns
        -------
        None
            This method performs in-place operations on the file system, including
            file creation, copying, and directory management.

        """
        if self.verbose:
            self._log.debug("  > Create folder")

        # Make a folder for the reduction results
        red_path = Path(sci_src_path, 'reduced')
        if not red_path.exists():
            red_path.mkdir(exist_ok=True)

        # Create a temporary working directory
        tmp_path = Path(f"{sci_src_path}/atmp/")

        # Remove old temp folder and make a new one
        os.system(f"rm -rf {tmp_path}")
        tmp_path.mkdir(exist_ok=True)

        # Get the file suffix
        suffix = np.unique(fits_files_dict['suffix'].values[0])[0]

        # Update needed variables
        self._sci_root_path = sci_src_path
        self._red_path = red_path
        self._tmp_path = tmp_path
        self._suffix = suffix

        # Copy the found science files to the tmp folder
        new_file_paths = []
        verbose = 'v' if self.verbose else ''
        if not self.silent:
            self._log.info("> Copy raw science images.")

        for row_index, row in fits_files_dict.iterrows():
            abs_file_path = row['input']
            file_name = row['file_name']
            file_name_suffix = row['suffix']

            # Copy science files
            os.system(f"cp -r{verbose} {abs_file_path} {tmp_path}")
            new_fname = os.path.join(tmp_path, file_name + file_name_suffix)
            new_file_paths.append(new_fname)

        # Prepare the science files
        if not self.silent:
            self._log.info(f"> Prepare {len(new_file_paths)} ``SCIENCE`` files for reduction.")
        obs_date, filters, binnings = self.prepare_fits_files(new_file_paths,
                                                              self._telescope,
                                                              self._obsparams, 'science')
        # print(obs_date, filters, binnings)

        # Loop over binnings and prepare the calibration files for each binning
        for binxy in binnings:

            self._binxy = binxy

            # Find and prepare calibration files
            if not self.silent:
                self._log.info(f"> Find calibration files for binning: {binxy[0]}x{binxy[1]}.")
            self.find_calibrations(self._telescope, self._obsparams, obs_date, filters, binxy)

            # Run ccdproc to create reduced science images
            self.run_ccdproc(self._obsparams, filters, binxy)

        # Clean temporary folder
        if not self.silent:
            self._log.info("> Cleaning up")
        os.system(f"rm -rf {tmp_path}")

    def check_closest_exposure(self, ic_all,
                               obsparams, tolerance=0.5):
        """
        Find the closest matching dark exposure times for science images.

        This function checks the exposure times of light and flat field images
        in the dataset and finds the closest matching dark exposure time within
        a specified tolerance. It returns a list of exposure times for each image type,
        paired with the nearest matching dark exposure time.

        Parameters
        ----------
        ic_all : ImageFileCollection
            Collection of FITS files including science (light) and flat field images,
            as well as available dark frames.
        obsparams : dict
            Dictionary containing observational parameters, specifically requiring
            the 'exptime' key that indicates the exposure time header keyword.
        tolerance : float, optional
            Maximum allowable difference between the image and dark exposure times
            to be considered a match. Defaults to 0.5 seconds.

        Returns
        -------
        exptimes_list : list of dict
            List of two dictionaries. Each dictionary corresponds to an image type
            (light and flat). Each entry in the dictionary has the image exposure
            time as the key and a list containing the closest dark exposure time
            and a boolean indicating whether a suitable dark frame was found within
            the tolerance.
        exptimes : list of float
            List of unique exposure times found in the light and flat images.

        """
        dark_exposures = np.array(list(self._dark_exptimes))
        exptimes_list = [{}, {}]

        if not list(dark_exposures):
            return exptimes_list

        exptime_check_img_types = [bc.IMAGETYP_LIGHT, bc.IMAGETYP_FLAT]
        exptimes = []

        for i in range(len(exptime_check_img_types)):
            # Filter data
            dfilter = {'imagetyp': f'{exptime_check_img_types[i]}'}
            ic_exptimes = ic_all.filter(regex_match=False, **dfilter)

            if ic_exptimes.summary is not None:
                # Get unique exposure times
                expt = list(np.unique(ic_exptimes.summary[obsparams['exptime'].lower()]))
                exptimes += expt
                for t in expt:
                    abs_diff = np.abs(dark_exposures - t)
                    idx = np.argmin(abs_diff)
                    closest_dark_exposure = dark_exposures[idx]
                    has_nearest = True
                    if np.abs(closest_dark_exposure - t) > tolerance:
                        continue
                    exptimes_list[i].update({t: [closest_dark_exposure, has_nearest]})
                del ic_exptimes

        del ic_all

        return exptimes_list, exptimes

    def run_ccdproc(self, obsparams, filters, binnings):
        """Run ccdproc on dataset.

        Parameters
        ----------
        obsparams : dict
            A Dictionary containing telescope and observational parameters.
        filters : list
            A list with filter identified in the dataset.
        binnings : tuple
            Binning of the current dataset, typically specified as "XxY" (e.g., "2x2").

        """
        _bin_str = f'{binnings[0]}x{binnings[1]}'
        self._bin_str = _bin_str

        # Get master_calib files
        dbias = self._master_bias_dict
        ddark = self._master_dark_dict
        dflat = self._master_flat_dict

        # Reload files in temp folder
        ic_all = ImageFileCollection(location=self._tmp_path)

        # Get the number of amplifiers and trim, and overscan sections
        has_multiple_amps = obsparams['multiple_amps']

        # Get science and flat exposure times
        exptimes_list, img_exptimes = self.check_closest_exposure(ic_all=ic_all,
                                                                  obsparams=obsparams,
                                                                  tolerance=1)

        # Process files by filter
        for filt in filters:
            img_count_by_filter = len(ic_all.files_filtered(imagetyp=bc.IMAGETYP_LIGHT,
                                                            filter=filt[0]))
            if img_count_by_filter == 0:
                continue

            if not self.silent:
                self._log.info(f"> Process filter {filt[0]}, binning {_bin_str}.")

            # Get gain and readout noise
            gain = None
            if 'gain' in ic_all.summary.keys():
                gain = np.array(ic_all.summary['gain'].data)
                gain = list(np.unique(gain[gain.nonzero()]))[0]
            readnoise = None
            if 'ron' in ic_all.summary.keys():
                readnoise = np.array(ic_all.summary['ron'].data)
                readnoise = list(np.unique(readnoise[readnoise.nonzero()]))[0]

            # Create master bias file for each filter. Only executed once per dataset
            ccd_master_bias = None
            if self._config['BIAS_CORRECT']:
                master_bias = dbias[0]
                create_bias = dbias[1]
                if create_bias:
                    ccd_master_bias = self.create_master_bias(files_list=ic_all,
                                                              obsparams=obsparams,
                                                              mbias_fname=master_bias,
                                                              multiple_amps=has_multiple_amps,
                                                              gain=gain,
                                                              readnoise=readnoise,
                                                              error=self._config['EST_UNCERTAINTY'],
                                                              method=self._config['COMBINE_METHOD_BIAS'])
                else:
                    if not self.silent:
                        self._log.info(f'  Loading existing master bias file: {os.path.basename(master_bias)}')

                    ccd_master_bias = self.convert_fits_to_ccd(master_bias, single=True)
                    readnoise = ccd_master_bias.header['ron']

            # Create master dark file for each exposure time. Only executed once per dataset
            ccd_master_dark = {'sci_dark': {}, 'flat_dark': {}}
            if self._config['DARK_CORRECT']:
                for i in range(2):
                    exptimes = exptimes_list[i]
                    for dexpt in img_exptimes:

                        if dexpt not in exptimes:
                            continue
                        dark_exptime = exptimes[dexpt][0]
                        has_nearest = exptimes[dexpt][1]
                        master_dark = ddark[dark_exptime][0]
                        create_dark = ddark[dark_exptime][1]
                        bias_corr = False
                        mbias = None
                        if create_dark:
                            if not has_nearest:
                                bias_corr = True
                                mbias = ccd_master_bias
                            ccd_mdark = self.create_master_dark(files_list=ic_all, obsparams=obsparams,
                                                                mdark_fname=master_dark,
                                                                exptime=dark_exptime,
                                                                mbias=mbias,
                                                                multiple_amps=has_multiple_amps,
                                                                gain=gain, readnoise=readnoise,
                                                                error=self._config['EST_UNCERTAINTY'],
                                                                method=self._config['COMBINE_METHOD_DARK'])
                        else:
                            if not self.silent:
                                self._log.info(f'  Loading existing master dark file: {os.path.basename(master_dark)}')
                            ccd_mdark = self.convert_fits_to_ccd(master_dark, single=True)
                            bias_corr = ccd_mdark.header['biascorr'] if 'biascorr' in ccd_mdark.header else False
                        if i == 0:
                            ccd_master_dark['sci_dark'][dexpt] = ccd_mdark, bias_corr, has_nearest
                        else:
                            ccd_master_dark['flat_dark'][dexpt] = ccd_mdark, bias_corr, has_nearest
                            self._config['FLATDARK_CORRECT'] = True

            if not ccd_master_dark['sci_dark']:
                ccd_master_dark['sci_dark'] = None, None, None
                self._config['DARK_CORRECT'] = False

            if not ccd_master_dark['flat_dark']:
                ccd_master_dark['flat_dark'] = None, None, None
                self._config['FLATDARK_CORRECT'] = False

            # Create master flat file for each filter. Only executed once per dataset
            ccd_master_flat = {}
            if self._config['FLAT_CORRECT']:
                master_flat = dflat[filt[0]][0]
                create_flat = dflat[filt[0]][1]
                if create_flat:
                    ccd_master_flat[filt[0]] = self.create_master_flat(files_list=ic_all,
                                                                       obsparams=obsparams,
                                                                       mflat_fname=master_flat,
                                                                       flat_filter=filt[0],
                                                                       mbias=ccd_master_bias,
                                                                       mdark=ccd_master_dark['flat_dark'],
                                                                       multiple_amps=has_multiple_amps,
                                                                       gain=gain, readnoise=readnoise,
                                                                       error=self._config['EST_UNCERTAINTY'],
                                                                       method=self._config['COMBINE_METHOD_FLAT'])
                else:
                    if not self.silent:
                        self._log.info(f'  Loading existing master flat file: {os.path.basename(master_flat)}')
                    ccd_master_flat[filt[0]] = self.convert_fits_to_ccd(master_flat, single=True)
            if not ccd_master_flat or not ccd_master_flat[filt[0]]:
                ccd_master_flat = None
                self._ccd_mask_fname = None

            # Reduce science images
            self.ccdproc_sci_images(files_list=ic_all,
                                    obsparams=obsparams,
                                    image_filter=filt[0],
                                    master_bias=ccd_master_bias,
                                    master_dark=ccd_master_dark['sci_dark'],
                                    master_flat=ccd_master_flat,
                                    gain=gain, readnoise=readnoise,
                                    multiple_amps=has_multiple_amps,
                                    cosmic=self._config['CORRECT_COSMIC'],
                                    error=self._config['EST_UNCERTAINTY'])
        del ic_all

        gc.collect()

    def ccdproc_sci_images(self, files_list,
                           obsparams, image_filter=None,
                           master_bias= None,
                           master_dark= None,
                           master_flat= None,
                           multiple_amps=False,
                           gain=None, readnoise=None,
                           error=False, cosmic=False,
                           mbox=15, rbox=15, gbox=11,
                           cleantype="medmask",
                           cosmic_method='lacosmic',
                           sigclip=5, key_filter='filter',
                           dfilter=None, key_find='find',
                           invert_find=False):
        """
        Process science images by applying bias, dark, flat correction, and optional
        cosmic ray cleaning using CCD processing techniques.

        Parameters
        ----------
        files_list : ImageFileCollection
            Collection of science FITS files to be processed.
        obsparams : dict
            Dictionary containing observation parameters, including exposure time,
            binning, and instrument specifics.
        image_filter : str or None, optional
            Filter name used in the science observations, used to match calibration frames.
            Defaults to None.
        master_bias : str or None, optional
            Path to the master bias frame. If None, no bias correction is applied.
            Defaults to None.
        master_dark : dict or None, optional
            Dictionary containing paths to master dark frames, matched by exposure time.
            If None, no dark correction is applied. Defaults to None.
        master_flat : str or None, optional
            Path to the master flat frame. If None, no flat correction is applied.
            Defaults to None.
        multiple_amps : bool, optional
            If True, processes each amplifier separately for multi-amp detectors.
            Defaults to False.
        gain : float or None, optional
            Gain of the detector in electrons per ADU. Required if `error=True`.
            Defaults to None.
        readnoise : float or None, optional
            Readout noise of the detector in electrons. Required if `error=True`.
            Defaults to None.
        error : bool, optional
            If True, generates an uncertainty map alongside the processed image.
            Requires `gain` and `readnoise`. Defaults to False.
        cosmic : bool, optional
            If True, applies cosmic ray cleaning. Defaults to False.
        mbox : int, optional
            Median box size for the 'median' cosmic ray cleaning method. Defaults to 15.
        rbox : int, optional
            Replacement box size for the 'median' cosmic ray cleaning method. Defaults to 15.
        gbox : int, optional
            Growing box size for the 'median' cosmic ray cleaning method. Defaults to 11.
        cleantype : str, optional
            Cleaning type for cosmic ray removal when using the 'lacosmic' method.
            Defaults to "medmask".
        cosmic_method : {'lacosmic', 'median'}, optional
            Method for cosmic ray cleaning: 'lacosmic' for L.A. Cosmic or 'median'
            for a median filter. Defaults to 'lacosmic'.
        sigclip : int, optional
            Sigma-clipping threshold for cosmic ray detection with 'lacosmic' method.
            Defaults to 5.
        key_filter : str, optional
            FITS header keyword used to identify the filter for each image. Defaults to 'filter'.
        dfilter : dict or None, optional
            Dictionary specifying filter criteria for selecting images from `files_list`.
            Defaults to None.
        key_find : str, optional
            Keyword used for finding specific images within the collection based on criteria.
            Defaults to 'find'.
        invert_find : bool, optional
            If True, inverts the criteria specified by `key_find` to exclude matching files.
            Defaults to False.

        Returns
        -------
        processed_images : list of CCDData
            List of processed CCDData objects with corrections applied.

        """
        # Check inputs
        if error and (gain is None or readnoise is None):
            self._log.warning('You need to provide "gain" and "readnoise" to compute the error!')
            error = False
        if gain is not None and not isinstance(gain, u.Quantity):
            gain = gain * u.Unit("electron") / u.Unit("adu")
        if readnoise is not None and not isinstance(readnoise, u.Quantity):
            readnoise = readnoise * u.Unit("electron")

        dfilter = {'imagetyp': bc.IMAGETYP_LIGHT,
                   'binning': self._bin_str} if dfilter is None else dfilter
        if dfilter is not None and key_filter is not None and image_filter is not None:
            dfilter = add_keys_to_dict(dfilter, {key_filter: image_filter})
        files_list = self.get_file_list(files_list, dfilter,
                                        key_find=key_find, invert_find=invert_find,
                                        abspath=False)

        mbias_ccd = None
        mflat_ccd = None

        ccd_mask_fname = os.path.join(self._master_calib_path,
                                      'mask_from_ccdmask_%s_%s.fits' % (image_filter,
                                                                        self._bin_str))
        if self._telescope == 'DDOTI 28-cm f/2.2':
            ccd_mask_fname = os.path.join(self._master_calib_path,
                                          'mask_from_ccdmask_%s_%s_%s.fits' % (self._instrument,
                                                                               image_filter,
                                                                               self._bin_str))
        if os.path.isfile(ccd_mask_fname):
            self._ccd_mask_fname = ccd_mask_fname
        else:
            self._ccd_mask_fname = None

        for filename in files_list:
            if not self.silent:
                self._log.info(f"  >>> ccdproc is processing: {filename}")

            # Read image into a CCD object
            fname = os.path.join(self._tmp_path, filename)
            try:
                ccd = CCDData.read(fname, unit=u.Unit("adu"))
            except (Exception,):
                if not self.silent:
                    self._log.warning(f"{' ':<6}Failed to load FITS file. Skipping file.")
                continue

            ccd_hdr = ccd.header
            ccd_expt = ccd_hdr['exptime']

            ccd_is_cropped = False
            crop_key = obsparams['cropsec']
            if (not multiple_amps
                    and crop_key is not None
                    and (ccd_hdr.get('LTV1', 0) < 0
                         or ccd_hdr.get('LTV2', 0) < 0)):
                ccd_is_cropped = True

            # Process the ccd, trim image and subtract overscan if needed
            ccd = self.process_ccd(img_ccd=ccd, obsparams=obsparams, multiple_amps=multiple_amps)

            # Create an uncertainty map
            if error:
                ccd = ccdproc.create_deviation(ccd, gain=gain,
                                               readnoise=readnoise,
                                               disregard_nan=True)
            # Gain correct
            if gain is not None and self._config['CORRECT_GAIN']:
                ccd = ccdproc.gain_correct(ccd, gain)

            # Cosmic ray correction
            if cosmic and gain is not None and self._config['CORRECT_GAIN']:
                ccd = self.clean_cosmic_ray(ccd, ccd_mask_fname,
                                            mbox=mbox, rbox=rbox, gbox=gbox, sigclip=sigclip,
                                            cleantype=cleantype, cosmic_method=cosmic_method)

            # Mask bad pixel
            ccd_mask = None
            if self._ccd_mask_fname is not None:
                mask_ccdmask = CCDData.read(self._ccd_mask_fname,
                                            unit=u.dimensionless_unscaled)
                mask_ccdmask.data = mask_ccdmask.data.astype('bool')

                if ccd_is_cropped:
                    ccd_mask = self.crop_image(img_ccd=mask_ccdmask,
                                               hdr=ccd_hdr,
                                               crop_section_unbinned=ccd_hdr[crop_key])
                else:
                    ccd_mask = mask_ccdmask

                ccd, ccd_mask = check_shape(ccd, ccd_mask)

                ccd_mask = ccd_mask.data

            ccd.mask = ccd_mask

            # Bias or dark frame subtract
            dark_corrected = False
            if master_dark is not None and self._config['DARK_CORRECT']:
                mdark = master_dark[ccd_expt][0]
                is_nearest = master_dark[ccd_expt][2]
                scale = False
                if not is_nearest:
                    scale = True
                    mbias = master_bias
                    if mbias is not None and self._config['BIAS_CORRECT']:
                        if isinstance(mbias, str):
                            mbias = self.convert_fits_to_ccd(mbias, single=True)
                        if mbias_ccd is None:
                            mbias_ccd = mbias

                        if ccd_is_cropped:
                            mbias = self.crop_image(img_ccd=mbias,
                                                    hdr=ccd_hdr,
                                                    crop_section_unbinned=ccd_hdr[crop_key])
                        ccd, mbias = check_shape(ccd, mbias)
                        ccd = ccdproc.subtract_bias(ccd, mbias)

                if isinstance(mdark, str):
                    mdark = self.convert_fits_to_ccd(mdark, single=True)

                # ccd, mdark = check_shape(ccd, mdark_ccd)
                if ccd_is_cropped:
                    mdark = self.crop_image(img_ccd=mdark,
                                            hdr=ccd_hdr,
                                            crop_section_unbinned=ccd_hdr[crop_key])
                ccd, mdark = check_shape(ccd, mdark)
                ccd = ccdproc.subtract_dark(ccd, mdark,
                                            exposure_time='exptime',
                                            exposure_unit=u.Unit("second"),
                                            scale=scale)
                add_key_to_hdr(ccd.header, 'MDARK', get_filename(mdark))
                dark_corrected = True

            # Bias subtraction
            bias_corrected = False
            if master_bias is not None and self._config['BIAS_CORRECT'] and not dark_corrected:
                mbias = master_bias
                if isinstance(master_bias, str):
                    mbias = self.convert_fits_to_ccd(master_bias, single=True)
                if mbias_ccd is None:
                    mbias_ccd = mbias

                if ccd_is_cropped:
                    mbias = self.crop_image(img_ccd=mbias,
                                            hdr=ccd_hdr,
                                            crop_section_unbinned=ccd_hdr[crop_key])

                ccd, mbias = check_shape(ccd, mbias)
                ccd = ccdproc.subtract_bias(ccd, mbias)
                add_key_to_hdr(ccd.header, 'MBIAS', get_filename(mbias))
                bias_corrected = True

            # Flat correction
            flat_corrected = False
            if master_flat is not None and self._config['FLAT_CORRECT']:
                mflat = master_flat[image_filter]
                if isinstance(mflat, str):
                    mflat = self.convert_fits_to_ccd(mflat, single=True)
                if mflat_ccd is None:
                    mflat_ccd = mflat

                if ccd_is_cropped:
                    mflat = self.crop_image(img_ccd=mflat,
                                            hdr=ccd_hdr,
                                            crop_section_unbinned=ccd_hdr[crop_key])
                ccd, mflat = check_shape(ccd, mflat)

                ccd = ccdproc.flat_correct(ccd, mflat)
                add_key_to_hdr(ccd.header, 'MFLAT', get_filename(mflat))
                flat_corrected = True

            # Create bad pixel map if non exists
            if self._ccd_mask_fname is None:
                ccd_mask = self.create_bad_pixel_map(ccd.header, obsparams,
                                                     mflat_ccd, mbias_ccd,
                                                     ccd_mask_fname,
                                                     vignette=self._config['VIGNETTE_FRAC'])
                if ccd_is_cropped:
                    ccd_mask = self.crop_image(img_ccd=ccd_mask,
                                               hdr=ccd_hdr,
                                               crop_section_unbinned=ccd_hdr[crop_key])
                ccd.mask = ccd_mask

            # Store results
            fbase = Path(filename).stem
            suffix = Path(filename).suffix
            filename = f'{fbase}_red{suffix}'
            filename = join_path(filename, self._red_path)

            ccd.header['FILENAME'] = os.path.basename(filename)
            ccd.header['combined'] = True
            ccd.header = ammend_hdr(ccd.header)

            # Change to float32 to keep the file size under control
            ccd.data = ccd.data.astype('float32')

            ccd.uncertainty = None
            hdr = ccd.header
            ccd.write(filename, overwrite=True)

            # Make sure that the RA and DEC are consistent and rounded to the same decimal
            # Reformat ra and dec if the values are in degree to avoid float <-> str conversion errors
            if obsparams['radec_separator'] == 'XXX':
                hdr[obsparams['ra']] = round(hdr[obsparams['ra']],
                                             bc.ROUND_DECIMAL)
                hdr[obsparams['dec']] = round(hdr[obsparams['dec']],
                                              bc.ROUND_DECIMAL)

            # Update result table
            hdr_dict = {}
            for k in obsparams['obs_keywords']:
                hdr_dict.setdefault(k.upper(), ccd.header.get(k))
            hdr_dict['bias_cor'] = 'T' if bias_corrected else 'F'
            hdr_dict['dark_cor'] = 'T' if dark_corrected else 'F'
            hdr_dict['flat_cor'] = 'T' if flat_corrected else 'F'

            self._obsTable.update_obs_table(fbase, hdr_dict, obsparams)

            del ccd
        del files_list
        gc.collect()

    def clean_cosmic_ray(self, ccd, ccd_mask_fname,
                         mbox=15, rbox=15, gbox=11, sigclip=5,
                         cleantype="medmask",
                         cosmic_method='lacosmic'):
        """ Clean cosmic rays from CCD image data.

        This method removes cosmic rays from a CCD image.
        The following two methods are supported: 'lacosmic' and 'median'.

        Parameters
        ----------
        ccd : astropy.nddata.CCDData
            The CCD data to be cleaned of cosmic rays. This should be an `astropy.nddata.CCDData`
            object containing the image data to be processed.
        ccd_mask_fname : str
            Filename of the FITS file containing the mask. If the file exists, its data will be used
            as a mask to mark regions in the CCD data where cosmic ray detection should be ignored.
        mbox : int, optional
            Median box size for the 'median' method. Defines the size of the box used to replace
            pixels marked as cosmic rays with the median value of neighboring pixels. Defaults to 15.
        rbox : int, optional
            Replacement box size for the 'median' method. Determines the area size around each cosmic
            ray pixel that will be replaced with the median of nearby pixels. Defaults to 15.
        gbox : int, optional
            Growing box size for the 'median' method. Specifies the size of the box used to grow
            the area around detected cosmic ray pixels. Defaults to 11.
        sigclip : float, optional
            Sigma clipping threshold for the 'lacosmic' method. This value determines the sensitivity
            to cosmic rays based on the deviation from the mean. Defaults to 5.
        cleantype : str, optional
            Specifies the cleaning type to use with the 'lacosmic' method. Default is 'medmask', which
            uses a median filter with a mask to clean cosmic ray pixels.
        cosmic_method : {'lacosmic', 'median'}, optional
            The method to use for cosmic ray cleaning. Options are 'lacosmic' for the L.A. Cosmic
            algorithm or 'median' for a median filtering approach.

        Returns
        -------
        ccd : astropy.nddata.CCDData
            The CCD data with cosmic rays removed. A `CCDData` object of the same type as the input
            `ccd` is returned with a 'COSMIC' header keyword added, indicating the cleaning method used.
        """
        ctype = cosmic_method.lower().strip()
        ctypes = ['lacosmic', 'median']
        if ctype not in ctypes:
            self._log.warning('> Cosmic ray type "%s" NOT available [%s]' % (ctype, ' | '.join(ctypes)))
            return

        ccd_mask_fname = ccd_mask_fname if os.path.exists(ccd_mask_fname) else None

        if ccd_mask_fname is not None and isinstance(ccd_mask_fname, str):
            ccd_mask = self.convert_fits_to_ccd(ccd_mask_fname, single=True)
            ccd.mask = ccd_mask.data

        if ctype == 'lacosmic':
            ccd = ccdproc.cosmicray_lacosmic(ccd,
                                             sigclip=sigclip,
                                             cleantype=cleantype, gain_apply=False)
        elif ctype == 'median':
            ccd = ccdproc.cosmicray_median(ccd, mbox=mbox, rbox=rbox, gbox=gbox)
        if isinstance(ccd, CCDData):
            ccd.header['COSMIC'] = ctype.upper()
        return ccd

    def create_master_flat(self, files_list, obsparams, mflat_fname, flat_filter=None,
                           multiple_amps=False, mbias=None, mdark=None,
                           gain=None, method='average', readnoise=None, error=False,
                           key_filter='filter', dfilter=None,
                           key_find='find', invert_find=False, sjoin=','):
        """Create a master flat field frame.

        Parameters
        ----------
        files_list : list
            A list of file paths to individual flat field FITS files.
        obsparams : dict
            Dictionary containing observational parameters, such as exposure time
            and binning settings.
        mflat_fname : str
            Path and filename for saving the output master flat frame.
        flat_filter : str or None, optional
            Filter name used for the flat frames, to ensure the master flat is specific
            to a certain wavelength range. Defaults to None.
        multiple_amps : bool, optional
            If True, handles data from detectors with multiple amplifiers by processing
            each amplifier's data separately before combining. Defaults to False.
        mbias : str or None, optional
            Path to a master bias frame. If provided, applies bias correction to the
            individual flat frames before combining. Defaults to None.
        mdark : dict or None, optional
            Dictionary with paths to master dark frames, matched by exposure time.
            If provided, applies dark correction to the individual flat frames.
            Defaults to None.
        gain : float or None, optional
            Detector gain in electrons per ADU. Required if `error=True`.
            Defaults to None.
        method : {'average', 'median'}, optional
            Method to combine the individual flat frames: 'average' computes the mean,
            while 'median' computes the median. Defaults to 'average'.
        readnoise : float or None, optional
            Detector readout noise in electrons. Required if `error=True`.
            Defaults to None.
        error : bool, optional
            If True, generates an uncertainty map with the master flat frame, requiring
            `gain` and `readnoise`. Defaults to False.
        key_filter : str, optional
            Keyword used to identify the filter in the FITS headers of the flat frames.
            Defaults to 'filter'.
        dfilter : dict or None, optional
            Dictionary to filter the files in `files_list` based on specific criteria,
            such as selecting only frames with a particular exposure time. Defaults to None.
        key_find : str, optional
            Key used to locate specific files in `files_list` based on criteria.
            Defaults to 'find'.
        invert_find : bool, optional
            If True, inverts the criteria specified by `key_find`, excluding matching files.
            Defaults to False.
        sjoin : str, optional
            String used to join the names of individual flat files in the FITS header
            of the master flat. Defaults to ','.

        Returns
        -------
        master_flat : CCDData
            Combined master flat field frame as a CCDData object, with optional uncertainty
            map if `error=True`.

        """
        # Check inputs
        if error and (gain is None or readnoise is None):
            self._log.warning('You need to provide "gain" and "readnoise" to compute the error!')
            error = False
        if gain is not None and not isinstance(gain, u.Quantity):
            gain = gain * u.Unit("electron") / u.Unit("adu")
        if readnoise is not None and not isinstance(readnoise, u.Quantity):
            readnoise = readnoise * u.Unit("electron")
        dfilter = {'imagetyp': bc.IMAGETYP_FLAT,
                   'binning': self._bin_str} if dfilter is None else dfilter
        if dfilter is not None and key_filter is not None and flat_filter is not None:
            dfilter = add_keys_to_dict(dfilter, {key_filter: flat_filter})

        files_list = self.get_file_list(files_list, dfilter,
                                        key_find=key_find, invert_find=invert_find,
                                        abspath=False)

        if not files_list.size > 0:
            return []

        lflat = []
        mbias_ccd = None
        for filename in files_list:

            # Read image to a ccd object
            fname = os.path.join(self._tmp_path, filename)
            ccd = CCDData.read(fname, unit=u.Unit("adu"))
            ccd_expt = ccd.header['exptime']

            # Trim image and update header
            ccd = self.process_ccd(img_ccd=ccd,
                                   obsparams=obsparams,
                                   multiple_amps=multiple_amps)

            crop_key = obsparams['cropsec']
            if crop_key is not None and not multiple_amps:
                ccd = self.crop_image(img_ccd=ccd,
                                      hdr=ccd.header,
                                      crop_section_unbinned=ccd.header[crop_key])

            # Create an uncertainty map
            if error:
                ccd = ccdproc.create_deviation(ccd, gain=gain,
                                               readnoise=readnoise)
            # Gain correct
            if gain is not None and self._config['CORRECT_GAIN']:
                ccd = ccdproc.gain_correct(ccd, gain)

            dark_corrected = False
            if mdark is not None and self._config['FLATDARK_CORRECT']:
                mdark_ccd = mdark[ccd_expt][0]
                is_nearest = mdark[ccd_expt][2]
                scale = False
                if not is_nearest:
                    scale = True
                    if mbias is not None and self._config['BIAS_CORRECT']:
                        if isinstance(mbias, str):
                            mbias = self.convert_fits_to_ccd(mbias, single=True)
                        if crop_key is not None and not multiple_amps:
                            mbias = self.crop_image(img_ccd=mbias,
                                                    hdr=ccd.header,
                                                    crop_section_unbinned=ccd.header[crop_key])
                        if mbias_ccd is None:
                            mbias_ccd = mbias
                        ccd = ccdproc.subtract_bias(ccd, mbias)

                if isinstance(mdark_ccd, str):
                    mdark_ccd = self.convert_fits_to_ccd(mdark_ccd, single=True)
                if crop_key is not None and not multiple_amps:
                    mdark_ccd = self.crop_image(img_ccd=mdark_ccd,
                                                hdr=ccd.header,
                                                crop_section_unbinned=ccd.header[crop_key])
                ccd = ccdproc.subtract_dark(ccd, mdark_ccd,
                                            exposure_time='exptime',
                                            exposure_unit=u.Unit("second"),
                                            scale=scale)
                add_key_to_hdr(ccd.header, 'MDARK', get_filename(mdark_ccd))
                dark_corrected = True

            if mbias is not None and self._config['BIAS_CORRECT'] and not dark_corrected:
                if isinstance(mbias, str):
                    mbias = self.convert_fits_to_ccd(mbias, single=True)

                if crop_key is not None and not multiple_amps:
                    mbias = self.crop_image(img_ccd=mbias,
                                            hdr=ccd.header,
                                            crop_section_unbinned=ccd.header[crop_key])
                if mbias_ccd is None:
                    mbias_ccd = mbias

                # Check shape
                ccd, mbias = check_shape(ccd, mbias)
                ccd = ccdproc.subtract_bias(ccd, mbias)

            # Append the result to a file list for combining
            lflat.append(ccd)

            del ccd

        if not self.silent:
            self._log.info(f'  Creating master flat file: {os.path.basename(mflat_fname)}')

        # Combine flat ccds
        combine = ccdproc.combine(lflat, method=method,
                                  mem_limit=self._config['MEM_LIMIT_COMBINE'],
                                  scale=inv_median,
                                  minmax_clip=True,
                                  minmax_clip_min=0.9,
                                  minmax_clip_max=1.05,
                                  sigma_clip=True, sigma_clip_low_thresh=3,
                                  sigma_clip_high_thresh=3,
                                  sigma_clip_func=np.ma.median,
                                  sigma_clip_dev_func=mad_std,
                                  dtype='float32')

        ccd_mask_fname = os.path.join(self._master_calib_path,
                                      f'mask_from_ccdmask_{flat_filter}_{self._bin_str}.fits')
        if self._telescope == 'DDOTI 28-cm f/2.2':
            ccd_mask_fname = os.path.join(self._master_calib_path,
                                          'mask_from_ccdmask_%s_%s_%s.fits' % (self._instrument,
                                                                               flat_filter,
                                                                               self._bin_str))

        self.create_bad_pixel_map(combine.header, obsparams, combine, mbias_ccd,
                                  ccd_mask_fname, vignette=self._config['VIGNETTE_FRAC'])
        self._ccd_mask_fname = ccd_mask_fname

        # Update fits header
        if gain is not None and 'GAIN' not in combine.header:
            combine.header.set('GAIN', gain.value, gain.unit)
        if readnoise is not None and 'RON' not in combine.header:
            combine.header.set('RON', readnoise.value, readnoise.unit)
        combine.header['combined'] = True
        combine.header['CGAIN'] = True if gain is not None and self._config['CORRECT_GAIN'] else False
        combine.header['IMAGETYP'] = 'FLAT'
        combine.header['CMETHOD'] = method
        if sjoin is not None:
            combine.header['LFLAT'] = sjoin.join([os.path.basename(f) for f in files_list])
        combine.header['NFLAT'] = len(files_list)

        # Remove mask and the uncertainty extension
        combine.mask = None
        combine.uncertainty = None

        # Change dtype to float32 to keep the file size under control
        combine.data = combine.data.astype('float32')

        # Save master flat
        combine.header['FILENAME'] = os.path.basename(mflat_fname)
        combine.header = ammend_hdr(combine.header)
        combine.write(mflat_fname, overwrite=True)

        return combine

    def create_master_dark(self, files_list, obsparams, mdark_fname,
                           exptime=None, mbias=None,
                           multiple_amps=False,
                           gain=None, method='average', readnoise=None, error=False,
                           dfilter=None,
                           key_find='find', invert_find=False, sjoin=','):
        """Create a master dark frame.

        Parameters
        ----------
        files_list : list
            A list of file paths to individual dark frame FITS files.
        obsparams : dict
            Dictionary containing observational parameters, such as exposure time
            and binning settings.
        mdark_fname : str
            Path and filename for saving the output master dark frame.
        exptime : float or None, optional
            Exposure time of the dark frames, used to match files with the correct
            integration time. Defaults to None.
        mbias : str or None, optional
            Path to a master bias frame. If provided, applies bias correction to each
            individual dark frame before combining. Defaults to None.
        multiple_amps : bool, optional
            If True, handles data from detectors with multiple amplifiers by processing
            each amplifier's data separately before combining. Defaults to False.
        gain : float or None, optional
            Detector gain in electrons per ADU. Required if `error=True`.
            Defaults to None.
        method : {'average', 'median'}, optional
            Method used to combine the individual dark frames: 'average' computes the mean,
            while 'median' computes the median. Defaults to 'average'.
        readnoise : float or None, optional
            Detector readout noise in electrons. Required if `error=True`.
            Defaults to None.
        error : bool, optional
            If True, generates an uncertainty map with the master dark frame, requiring
            `gain` and `readnoise`. Defaults to False.
        dfilter : dict or None, optional
            Dictionary to filter the files in `files_list` based on specific criteria,
            such as selecting only frames with a particular exposure time. Defaults to None.
        key_find : str, optional
            Keyword used to locate specific files in `files_list` based on criteria.
            Defaults to 'find'.
        invert_find : bool, optional
            If True, inverts the criteria specified by `key_find`, excluding matching files.
            Defaults to False.
        sjoin : str, optional
            String used to join the names of individual dark files in the FITS header
            of the master dark frame. Defaults to ','.

        Returns
        -------
        master_dark : CCDData
            Combined master dark frame as a CCDData object, with optional uncertainty
            map if `error=True`.

        """
        if not self.silent:
            self._log.info(f'  Creating master dark file: {os.path.basename(mdark_fname)}')

        # Check inputs
        if error and (gain is None or readnoise is None):
            self._log.warning('You need to provide "gain" and "readnoise" to compute the error!')
            error = False
        if gain is not None and not isinstance(gain, u.Quantity):
            gain = gain * u.Unit("electron") / u.Unit("adu")
        if readnoise is not None and not isinstance(readnoise, u.Quantity):
            readnoise = readnoise * u.Unit("electron")
        dfilter = {'imagetyp': bc.IMAGETYP_DARK} if dfilter is None else dfilter

        # Get the list with files to reduce
        dfilter = add_keys_to_dict(dfilter, dkeys={'exptime': exptime,
                                                   'binning': self._bin_str})
        files_list = self.get_file_list(files_list, dfilter,
                                        key_find=key_find, invert_find=invert_find,
                                        abspath=False)
        if not files_list.size > 0:
            return []

        ldark = []
        for filename in files_list:

            # Read image to ccd object
            fname = os.path.join(self._tmp_path, filename)
            ccd = CCDData.read(fname, unit=u.Unit("adu"))

            # Trim image and update header
            ccd = self.process_ccd(img_ccd=ccd, obsparams=obsparams, multiple_amps=multiple_amps)

            crop_key = obsparams['cropsec']
            if crop_key is not None and not multiple_amps:
                ccd = self.crop_image(img_ccd=ccd,
                                      hdr=ccd.header,
                                      crop_section_unbinned=ccd.header[crop_key])

            # Create an uncertainty map
            if error:
                ccd = ccdproc.create_deviation(ccd, gain=gain,
                                               readnoise=readnoise)
            # Gain correct
            if gain is not None and self._config['CORRECT_GAIN']:
                ccd = ccdproc.gain_correct(ccd, gain)

            if mbias is not None and self._config['BIAS_CORRECT']:
                if isinstance(mbias, str):
                    mbias = self.convert_fits_to_ccd(mbias, single=True)

                if crop_key is not None and not multiple_amps:
                    mbias = self.crop_image(img_ccd=mbias,
                                            hdr=ccd.header,
                                            crop_section_unbinned=ccd.header[crop_key])

                ccd = ccdproc.subtract_bias(ccd, mbias)

            # Append the result to a file list for combining
            ldark.append(ccd)
            del ccd

        # Combine dark ccds
        combine = ccdproc.combine(ldark, method=method,
                                  mem_limit=self._config['MEM_LIMIT_COMBINE'],
                                  sigma_clip=True,
                                  sigma_clip_low_thresh=5,
                                  sigma_clip_high_thresh=5,
                                  sigma_clip_func=np.ma.median,
                                  sigma_clip_dev_func=mad_std,
                                  dtype='float32')

        # Update fits header
        if gain is not None and 'GAIN' not in combine.header:
            combine.header.set('GAIN', gain.value, gain.unit)
        if readnoise is not None and 'RON' not in combine.header:
            combine.header.set('RON', readnoise.value, readnoise.unit)
        combine.header['combined'] = True
        combine.header['BIASCORR'] = True if mbias is not None else False
        combine.header['CGAIN'] = True if gain is not None and self._config['CORRECT_GAIN'] else False
        combine.header['IMAGETYP'] = 'DARK'
        combine.header['CMETHOD'] = method

        if sjoin is not None:
            combine.header['LDARK'] = sjoin.join([os.path.basename(f) for f in files_list])
        combine.header['NDARK'] = len(files_list)

        # Remove mask and the uncertainty extension
        combine.mask = None
        combine.uncertainty = None

        # change dtype to float32 to keep the file size under control
        combine.data = combine.data.astype('float32')

        # Save master dark
        combine.header['FILENAME'] = os.path.basename(mdark_fname)
        combine.header = ammend_hdr(combine.header)
        combine.write(mdark_fname, overwrite=True)

        # Return result
        return combine

    def create_master_bias(self, files_list, obsparams, mbias_fname,
                           multiple_amps=False,
                           gain=None, method='average', readnoise=None, error=False,
                           dfilter=None, key_find='find', invert_find=False, sjoin=','):
        """Create a master bias frame.

        Parameters
        ----------
        files_list : list
            A list containing the full paths to individual FITS files of bias frames.
        obsparams : dict
            Dictionary containing observational parameters such as binning, exposure
            time, and other instrument details.
        mbias_fname : str or None, optional
            Absolute path and filename for saving the master bias frame. If None,
            the master bias frame is not saved. Defaults to None.
        multiple_amps : bool, optional
            If True, indicates that the detector has multiple amplifiers, and the function
            will handle each amplifier’s data separately before combining. Defaults to False.
        gain : float or None, optional
            Detector gain in electrons per ADU. If provided, used to scale the bias
            frames accordingly. Required if `error=True`. Defaults to None.
        method : {'average', 'median'}, optional
            Method used to combine the individual bias frames. Options are 'average'
            to compute the mean, or 'median' to compute the median. Defaults to 'average'.
        readnoise : float or None, optional
            Detector readout noise in electrons. If None, readout noise is estimated
            from the standard deviation of a bias frame. Required if `error=True`.
            Defaults to None.
        error : bool, optional
            If True, generates an uncertainty map with the master bias frame. Requires
            both `gain` and `readnoise` to be provided. Defaults to False.
        dfilter : dict or None, optional
            Dictionary specifying filtering criteria for selecting bias files based
            on their metadata or header values. Defaults to None.
        key_find : str, optional
            Keyword used to locate specific files within `files_list` based on criteria.
            Defaults to 'find'.
        invert_find : bool, optional
            If True, inverts the criteria specified by `key_find` to exclude files
            that match the criteria. Defaults to False.
        sjoin : str, optional
            String used to join the names of individual bias files in the FITS header
            of the master bias frame. Defaults to ','.

        Returns
        -------
        master_bias : CCDData
            Combined master bias frame as a CCDData object, with an optional uncertainty
            map if `error=True`.

        """
        if not self.silent:
            self._log.info(f'  Creating master bias file: {os.path.basename(mbias_fname)}')

        # Check inputs
        if error and (gain is None or readnoise is None):
            self._log.warning('You need to provide "gain" and "readnoise" to compute the error!')
            error = False
        if gain is not None and not isinstance(gain, u.Quantity):
            gain = gain * u.Unit("electron") / u.Unit("adu")
        if readnoise is not None and not isinstance(readnoise, u.Quantity):
            readnoise = readnoise * u.Unit("electron")

        dfilter = {'imagetyp': bc.IMAGETYP_BIAS,
                   'binning': self._bin_str} if dfilter is None else dfilter

        # Get the list with files
        files_list = self.get_file_list(files_list, dfilter,
                                        key_find=key_find, invert_find=invert_find,
                                        abspath=False)
        if not files_list.size > 0:
            return []

        lbias = []
        for filename in files_list:

            # Read image to ccd object
            fname = os.path.join(self._tmp_path, filename)
            ccd = CCDData.read(fname, unit=u.Unit("adu"))

            # Trim image and update header
            ccd = self.process_ccd(img_ccd=ccd, obsparams=obsparams, multiple_amps=multiple_amps)

            crop_key = obsparams['cropsec']
            if crop_key is not None and not multiple_amps:
                ccd = self.crop_image(img_ccd=ccd,
                                      hdr=ccd.header,
                                      crop_section_unbinned=ccd.header[crop_key])

            # Create an uncertainty map
            if error:
                ccd = ccdproc.create_deviation(ccd, gain=gain,
                                               readnoise=readnoise)
            # Gain correct
            if gain is not None and self._config['CORRECT_GAIN']:
                ccd = ccdproc.gain_correct(ccd, gain)

            # Append the result to a file list for combining
            lbias.append(ccd)
            del ccd

        # Combine bias ccds
        combine = ccdproc.combine(lbias, method=method,
                                  mem_limit=self._config['MEM_LIMIT_COMBINE'],
                                  sigma_clip=True,
                                  sigma_clip_low_thresh=3,
                                  sigma_clip_high_thresh=3,
                                  # sigma_clip_func=np.ma.median,
                                  # sigma_clip_dev_func=mad_std,
                                  dtype='float32')

        # Update fits header
        if gain is not None and 'GAIN' not in combine.header:
            combine.header.set('GAIN', gain.value, gain.unit)
        if readnoise is not None and 'RON' not in combine.header:
            combine.header.set('RON', readnoise.value, readnoise.unit)
        combine.header['combined'] = True
        combine.header['CGAIN'] = True if gain is not None and self._config['CORRECT_GAIN'] else False
        combine.header['IMAGETYP'] = 'BIAS'
        combine.header['CMETHOD'] = method
        if sjoin is not None:
            combine.header['LBIAS'] = sjoin.join([os.path.basename(f) for f in files_list])
        combine.header['NBIAS'] = len(files_list)

        # Remove mask and the uncertainty extension
        combine.mask = None
        combine.uncertainty = None

        # change dtype to float32 to keep the file size under control
        combine.data = combine.data.astype('float32')

        # Save master bias
        combine.header['FILENAME'] = os.path.basename(mbias_fname)
        combine.header = ammend_hdr(combine.header)
        combine.write(mbias_fname, overwrite=True)

        # Return result
        return combine

    def create_bad_pixel_map(self, hdr, obsparams,
                             mflat_ccd, mbias_ccd, ccd_mask_fname, vignette=0.975, silent=False):
        """
        Create a bad pixel map from master flat and optionally master bias frames.

        Parameters
        ----------
        hdr : astropy.io.fits.Header
            FITS header containing the metadata about the image.
        obsparams : dict
            Dictionary containing observation parameters.
        mflat_ccd : CCDData or None
            The master flat frame data. If provided, it is used for masking.
        mbias_ccd : CCDData or None
            The master bias frame data. If provided, it is also used for masking.
        ccd_mask_fname : str
            The filename to save the bad pixel mask.
        vignette : float, optional

        silent : bool, optional
            If True, suppress log messages. Default is False.

        Returns
        -------
        str
            The filename of the created bad pixel mask.
        """
        mask_as_ccd = None
        ccd_mask = None
        crop_key = obsparams['cropsec']

        if mflat_ccd is not None:

            vignette_mask = None
            if hdr['FILTER'] in ['U'] and self._telescope == 'DK-1.54':
                vignette_mask = self.create_vignette_mask(original_size=mflat_ccd.data.shape,
                                                          vignette=vignette)

                vignette_mask = self.crop_image(img_ccd=vignette_mask,
                                                hdr=mflat_ccd.header,
                                                crop_section_unbinned=mflat_ccd.header[crop_key])

            # Apply sigma clipping to the master flat frame
            flat_mask = sigma_clip(mflat_ccd, masked=True,
                                   sigma=5,
                                   maxiters=None,
                                   cenfunc=np.nanmedian,
                                   stdfunc=mad_std).mask

            if vignette_mask is not None:
                combined_mask = flat_mask & vignette_mask
            else:
                combined_mask = flat_mask

            combined_mask = CCDData(data=combined_mask, unit=u.dimensionless_unscaled)
            ccd_mask = combined_mask.data

        # Apply sigma clipping to the master bias frame, if provided
        if mbias_ccd is not None:
            bias_mask = sigma_clip(mbias_ccd, masked=True,
                                   sigma=5,
                                   maxiters=None,
                                   cenfunc=np.nanmedian,
                                   stdfunc=mad_std).mask

            bias_mask = CCDData(data=bias_mask, unit=u.dimensionless_unscaled)

            if crop_key is not None:
                bias_mask = self.crop_image(img_ccd=bias_mask,
                                            hdr=mflat_ccd.header,
                                            crop_section_unbinned=mflat_ccd.header[crop_key])
            # Check shape
            ccd_mask = CCDData(data=ccd_mask, unit=u.dimensionless_unscaled)
            ccd_mask, bias_mask = check_shape(ccd_mask, bias_mask)

            ccd_mask = ccd_mask.data
            bias_mask = bias_mask.data

            if ccd_mask is None:
                ccd_mask = bias_mask
            else:

                ccd_mask |= bias_mask

        if ccd_mask is not None:

            # Log message if not in silent mode
            if not silent:
                self._log.info(f'  Creating bad pixel map file: {os.path.basename(ccd_mask_fname)}')

            # Convert mask to CCDData and set header
            mask_as_ccd = CCDData(data=ccd_mask, unit=u.dimensionless_unscaled)
            mask_as_ccd.data = mask_as_ccd.data.astype('uint8')
            mask_as_ccd.header['imagetyp'] = 'flat mask'

            # Write the mask to file
            mask_as_ccd.write(ccd_mask_fname, overwrite=True)

        # Return the created mask
        return mask_as_ccd

    def process_ccd(self, img_ccd, obsparams, multiple_amps):
        """
        Processes a CCD image by applying overscan correction, trimming, and reconstructing the full CCD if multiple amplifiers are used.

        Parameters
        ----------
        img_ccd : ccdproc.CCDData
            The input CCD image data to be processed.
        obsparams : dict
            Dictionary containing observation parameters, including overscan and trim section information.
        multiple_amps : bool
            Flag indicating whether the CCD has multiple amplifiers.

        Returns
        -------
        ccdproc.CCDData
            The processed CCD image with overscan corrected and trimmed, or reconstructed if multiple amplifiers are used.

        Notes
        -----
        - If `multiple_amps` is False, the function applies overscan correction and trimming to the entire image based on the provided parameters.
        - If `multiple_amps` is True, the function processes each amplifier separately, applying overscan correction, trimming, and reconstructing the full CCD image.
        - Specific rows and columns are further adjusted for each amplifier to ensure proper alignment.
        """

        nccd = img_ccd.copy()
        hdr = nccd.header
        if not multiple_amps:
            oscan_key = obsparams['oscansec']
            if oscan_key is not None:
                oscan_section = hdr[oscan_key]
                oscan_slice = self.parse_section(oscan_section)
                oscan_arr = nccd[oscan_slice]
                nccd = ccdproc.subtract_overscan(nccd,
                                                 overscan=oscan_arr,
                                                 add_keyword={'oscan_cor': True},
                                                 median=True, overscan_axis=None)

            trim_key = obsparams['trimsec']
            if trim_key is not None:
                trim_section = hdr[trim_key]
                trim_slice = self.parse_section(trim_section)
                nccd = ccdproc.trim_image(nccd[trim_slice],
                                          add_keyword={'trimmed': True})
        else:
            # Get the list of amplifiers from the header
            amp_list = hdr[obsparams['amplist']].split()
            ny, nx = map(int, hdr[obsparams['namps_yx']].split())

            # Determine the full CCD size from the amplifier sections
            xsize = 0
            ysize = 0
            ccd_list = []
            ni = 0
            for yi in range(ny):
                for xi in range(nx):
                    amp_id = amp_list[ni]

                    tsec_key = f"{obsparams['trimsec']}{amp_id}"
                    trim_slice = self.parse_section(hdr[tsec_key])
                    amp_ccd = ccdproc.trim_image(nccd[trim_slice])

                    bsec_key = f"{obsparams['oscansec']}{amp_id}"
                    oscan_slice_y, oscan_slice_x = self.parse_section(hdr[bsec_key])
                    oscan_data = ccdproc.trim_image(nccd[oscan_slice_y, oscan_slice_x])
                    amp_ccd = ccdproc.subtract_overscan(amp_ccd,
                                                        overscan=oscan_data,
                                                        median=True,
                                                        overscan_axis=1)

                    # Further adjust amplifier-specific slices by removing a row and column
                    if amp_id == '11':  # Lower Left
                        amp_ccd = amp_ccd[:-1, 1:]
                    elif amp_id == '12':  # Lower Right
                        amp_ccd = amp_ccd[:-1, :-1]
                    elif amp_id == '21':  # Upper Left
                        amp_ccd = amp_ccd[1:, 1:]
                    elif amp_id == '22':  # Upper Right
                        amp_ccd = amp_ccd[1:, :-1]

                    # Update x and y sizes
                    xc = np.array([0, amp_ccd.shape[1]], int)
                    yc = np.array([0, amp_ccd.shape[0]], int)
                    if yi == 0:
                        xsize += amp_ccd.shape[1]
                    else:
                        yc += amp_ccd.shape[0]
                    if xi == 0:
                        ysize += amp_ccd.shape[0]
                    else:
                        xc += amp_ccd.shape[1]

                    ccd_list.append([(yc, xc), amp_ccd])

                    ni += 1
            # Create an empty array for the full CCD
            data = np.zeros((ysize, xsize))

            # Place each amplifier section into the full CCD
            for i in range(len(ccd_list)):
                y1 = ccd_list[i][0][0][0]
                y2 = ccd_list[i][0][0][1]
                x1 = ccd_list[i][0][1][0]
                x2 = ccd_list[i][0][1][1]
                data[y1:y2, x1:x2] = ccd_list[i][1].data

            nccd = ccdproc.CCDData(data, unit=nccd.unit)
            nccd.header = img_ccd.header

        return nccd

    def crop_image(self, img_ccd, hdr, crop_section_unbinned):
        """
        Crops an image based on the provided unbinned crop section and adjusts the crop for the binning factor.

        Parameters
        ----------
        img_ccd : ccdproc.CCDData
            The input CCD image to be cropped.
        hdr : astropy.io.fits.Header
            FITS header containing metadata about the image.
        crop_section_unbinned : str
            A string representing the crop section in the unbinned image (e.g., '[1:1024, 1:1024]').

        Returns
        -------
        ccdproc.CCDData
            The cropped CCD image.

        """
        nccd = img_ccd.copy()

        ltv1, ltv2 = hdr.get('LTV1', 0), hdr.get('LTV2', 0)
        if ltv1 < 0 or ltv2 < 0:
            bin_x, bin_y = map(int, hdr['CCDSUM'].split())
            crop_slice_y, crop_slice_X = self.parse_section(crop_section_unbinned, bin_x, bin_y)
            crop_slice_y = slice(crop_slice_y.start, crop_slice_y.stop - 1)
            crop_slice_X = slice(crop_slice_X.start, crop_slice_X.stop - 1)
            nccd = ccdproc.trim_image(nccd[crop_slice_y, crop_slice_X],
                                      add_keyword={'cropped': True})
        del img_ccd

        return nccd

    def convert_fits_to_ccd(self, lfits,
                            key_unit='BUNIT',
                            key_file='FILENAME',
                            unit=None, single=False):
        """Convert fits file to ccd object.

        Parameters
        ----------
        lfits : list, dict
            Dictionary or list of dictionaries for conversion to fits-files
        key_unit : str, optional
            Keyword for unit of fits-image stored in the header. Default is 'BUNIT'.
        key_file : str, optional
            Keyword for file name stored in the header. Default is 'FILENAME'
        unit : str, None, optional
            Fits image unit. Default is None.
        single : bool, optional
            If True, the input is treated as a single image, else as a list. Defaults to False.

        Returns
        -------
        lccd : list, ccdproc.CCDData
            A list of image ccd objects.
        """
        lccd = []
        if not isinstance(lfits, (tuple, list)):
            lfits = [lfits]
        for fn in lfits:
            fits_unit = unit
            if os.path.exists(fn):
                hdr = fits.getheader(fn)
            else:
                self._log.warning(f'>>> File "{os.path.basename(fn)}" NOT found')
                continue
            if key_unit is not None and key_unit in hdr:
                try:
                    fits_unit = eval('u.%s' % hdr[key_unit])
                except (Exception,):
                    pass
            if fits_unit is None:
                if key_unit is not None:
                    sys.exit('>>> Units NOT found in header ("%s") of image "%s". '
                             'Specify one in "unit" variable' % (
                                 key_unit, os.path.basename(fn)))
                else:
                    self._log.warning('>>> "key_unit" not specified')
            self._log.setLevel("warning".upper())
            # ccd = CCDData.read(f, unit=fits_unit)
            ccd = CCDData.read(fn)
            self._log.setLevel("info".upper())
            if key_file is not None and key_file not in ccd.header:
                ccd.header[key_file] = os.path.basename(fn)
            lccd.append(ccd)
            del ccd
        if len(lccd) == 0:
            self._log.warning('>>> NO files found!')
            return
        if single and len(lccd) == 1:
            lccd = lccd[0]
        return lccd

    def find_calibrations(self, telescope, obsparams,
                          date, filters,
                          binnings):
        """
        Find calibration files and create master calibration file names.

        This method is used to search for the bias, dark and flat files, if they are required,
        copies them to the tmp reduction folder and applies the appropriate reduction method.

        Parameters
        ----------
        telescope : str
            Telescope identifier.
        obsparams : dict
            Dictionary with telescope configuration.
        date : datetime.date
            Observation date.
        filters : list
            The list of filters used in the dataset.
        binnings : tuple
            A tuple containing the binning factor in x, and y direction.

        """
        sci_root_path = self._sci_root_path
        tmp_path = self._tmp_path
        suffix = self._suffix
        sci_bin_str = f'{binnings[0]}x{binnings[1]}'

        inst = None
        if self._telescope == 'DDOTI 28-cm f/2.2':
            inst = self._instrument

        # Create a range of days before and after the observation night to search for calibrations
        dt_list = list(range(-1 * self._config['TIMEDELTA_DAYS'], self._config['TIMEDELTA_DAYS']))
        date_range = np.array([str(date + timedelta(days=i)) for i in dt_list])

        # Get base folder for search for calib files
        # fixme: this is not working as intended
        #  The idea is to search for calibration files in the sci folder and if no suitable
        #  calib files are found search for other dates.
        #  For this to work i will have to redo this part.
        #
        # search YYYY-MM-DD, and YYYYMMDD
        dsplit = str(date).replace('-', '')
        str_comp = re.compile(f'({str(date)}|{dsplit}).?/')
        match = re.search(str_comp, sci_root_path)

        search_base_path = None
        if match is not None:
            search_base_path = Path(sci_root_path[:match.end()])
        else:
            for d in date_range:
                dsplit = str(d).replace('-', '')
                str_comp = re.compile(f'({str(d)}|{dsplit}).?/')
                match = re.search(str_comp, sci_root_path)
                if match is not None:
                    search_base_path = Path(sci_root_path[:match.end()])
                    break

        # Create a master calibration folder
        calib_out = Path(f"{search_base_path}/master_calibs/")
        calib_out.mkdir(exist_ok=True)
        self._master_calib_path = calib_out

        # Find all images
        fits_list = Path(search_base_path).rglob(f"*{suffix}")
        fits_list = [f for f in fits_list if '/atmp/' not in str(f) and not f.name.startswith('.')]

        # Create an image file collection from the file list
        ic_all = ImageFileCollection(filenames=fits_list)

        # Find master calibration files and check if the file exists
        dbias, ddark, dflat, dark_exptimes = self.create_master_fnames(ic_all,
                                                                       obsparams, filters,
                                                                       sci_bin_str)

        # Set variables
        self._master_bias_dict = dbias
        self._master_dark_dict = ddark
        self._master_flat_dict = dflat
        self._dark_exptimes = dark_exptimes

        # Filter available files according to their image typ
        create_bias = dbias[1]
        if create_bias:

            # Create filter
            dfilter, add_filters, regexp = self.create_dfilter(inst, obsparams, 'bias')
            # Filter for bias files
            bias_files = self.filter_calib_files(fits_list, obsparams,
                                                 dfilter, add_filters,
                                                 binnings, date, regexp=regexp)

            # Copy bias files if no master_bias is available
            if list(bias_files) and self._config['BIAS_CORRECT']:
                new_bias_path = []
                if not self.silent:
                    self._log.info("> Copy raw bias images.")
                for bfile in bias_files:
                    os.system(f"cp -r {bfile} {tmp_path}")
                    file_name = Path(bfile).stem
                    new_bias_path.append(os.path.join(tmp_path, file_name + suffix))

                # Prepare bias files
                if not self.silent:
                    self._log.info(f"> Preparing {len(new_bias_path)} ``BIAS`` files for reduction.")
                self.prepare_fits_files(new_bias_path, telescope, obsparams, 'bias')
            else:
                self._config['BIAS_CORRECT'] = False
            del bias_files

        create_dark = np.any(np.array([v[1] for _, v in ddark.items()])) if ddark else False
        if create_dark:
            dfilter, add_filters, regexp = self.create_dfilter(inst, obsparams, 'dark')
            # Filter for dark files
            dark_files = self.filter_calib_files(fits_list, obsparams,
                                                 dfilter, add_filters,
                                                 binnings, date, regexp=regexp)

            # Copy dark files if no master_dark is available
            if list(dark_files) and self._config['DARK_CORRECT']:
                new_dark_path = []
                if not self.silent:
                    self._log.info("> Copy raw dark images.")
                for dfile in dark_files:
                    os.system(f"cp -r {dfile} {tmp_path}")
                    file_name = Path(dfile).stem
                    new_dark_path.append(os.path.join(tmp_path, file_name + suffix))

                # Prepare dark files
                if not self.silent:
                    self._log.info(f"> Preparing {len(new_dark_path)} ``DARK`` files for reduction.")
                self.prepare_fits_files(new_dark_path, telescope, obsparams, 'dark')
            else:
                self._config['DARK_CORRECT'] = False
            del dark_files

        # Create_flat = np.any(np.array([v[1] for _, v in dflat.items()]))
        # print(create_flat)
        for filt in filters:
            create_flat = dflat[filt[0]][1]
            if create_flat:
                dfilter, add_filters, regexp = self.create_dfilter(inst, obsparams, 'flat')
                dfilter[obsparams['filter'].lower()] = filt[1]
                flat_files = self.filter_calib_files(fits_list, obsparams,
                                                     dfilter, add_filters,
                                                     binnings, date, regexp=regexp)

                # Copy flat files if no master_flat is available
                if list(flat_files) and self._config['FLAT_CORRECT']:
                    new_flat_path = []
                    if not self.silent:
                        self._log.info("> Copy raw flat images.")
                    for ffile in flat_files:
                        os.system(f"cp -r {ffile} {tmp_path}")
                        file_name = Path(ffile).stem
                        new_flat_path.append(os.path.join(tmp_path, file_name + suffix))

                    # Prepare the flat files
                    if not self.silent:
                        self._log.info(f"> Preparing {len(new_flat_path)} ``FLAT`` files in"
                                       f" {filt[0]} band for reduction.")
                    self.prepare_fits_files(new_flat_path, telescope, obsparams, 'flat')

                del flat_files

        # self._make_mbias = create_bias
        # self._make_mdark = create_dark
        # self._make_mflat = create_flat

        self._obsparams = obsparams

        del fits_list, ic_all
        gc.collect()

    def filter_calib_files(self, fits_list, obsparams,
                           dfilter, add_filters,
                           binnings, obs_date=None, regexp=False):
        """ Filter files according to their binning and observation date.

        If the first try with a binning keyword fails, use the unbinned image size
        known from a telescope to determine the binning factor

        Parameters
        ----------
        fits_list : list
            The file list containing all the images to be filtered.
        obsparams : dict
            A dictionary containing observation parameters.
        dfilter : dict
            Dictionary specifying filter criteria for the image files.
        add_filters : dict
            Additional filters to apply, such as binning keywords.
        binnings : tuple
            A tuple specifying the binning in the x and y directions (e.g., (2, 2)).
        obs_date : datetime or None, optional
            The observation date to filter files by. Files from the same date are prioritized.
            If None, no date-based filtering is applied. Defaults to None.
        regexp : bool, optional
            If True, applies regular expressions to the filtering process. Defaults to False.

        Returns
        -------
        files : list of str
            A list of file paths that match the specified binning and observation date criteria.

        """
        files_filtered = [f for f in fits_list if '/atmp/' not in str(f)
                          and 'reduced' not in str(f)
                          and 'master_calibs' not in str(f)]


        # Create an image file collection from the file list
        ic_all = ImageFileCollection(filenames=files_filtered)

        # Filter with binning keyword
        add_filters_tmp = add_filters.copy()
        if isinstance(obsparams['binning'][0], str) and isinstance(obsparams['binning'][1], str):
            add_filters_tmp[obsparams['binning'][0]] = binnings[0]
            add_filters_tmp[obsparams['binning'][1]] = binnings[1]
        if not add_filters_tmp:
            add_filters = None

        files = self.get_file_list(ic_all, dfilter, key_find='find', regexp=regexp,
                                   invert_find=False, dkeys=add_filters_tmp)

        if not list(files.data):
            files = []
            if not add_filters_tmp:
                add_filters = None
            files_tmp = self.get_file_list(ic_all, dfilter, key_find='find', regexp=regexp,
                                           invert_find=False, dkeys=add_filters)
            files_tmp = files_tmp.data

            for file_path in files_tmp:
                # Load fits file
                hdul = fits.open(file_path, mode='readonly', ignore_missing_end=True)
                hdul.verify('fix')
                prim_hdr = hdul[0].header
                naxis1 = prim_hdr['NAXIS1']
                naxis2 = prim_hdr['NAXIS2']

                binning_x = obsparams['image_size_1x1'][0] // naxis1
                binning_y = obsparams['image_size_1x1'][1] // naxis2

                if binning_x == int(binnings[0]) and binning_y == int(binnings[1]):
                    files.append(file_path)

        same_date = []
        diff_date = []
        for file_path in files:

            # Load fits file
            hdul = fits.open(file_path, mode='readonly', ignore_missing_end=True)
            hdul.verify('fix')
            prim_hdr = hdul[0].header

            self._log.debug("  Check Obs-Date")
            if 'time-obs'.upper() in prim_hdr:
                time_string = f"{prim_hdr['date-obs'.upper()]}T{prim_hdr['time-obs'.upper()]}"

            else:
                time_string = prim_hdr['date-obs'.upper()]

            t = pd.to_datetime(time_string,
                               format='ISO8601', utc=False)

            t_short = t.strftime('%Y-%m-%d')
            if t_short == obs_date.strftime('%Y-%m-%d'):
                same_date.append(file_path)
            else:
                diff_date.append(file_path)

        if not same_date:
            files = diff_date
        else:
            files = same_date

        del same_date, diff_date, ic_all

        return files

    def create_master_fnames(self, ic_all, obsparams, filters, binning_ext):
        """ Create file names for master-bias, dark, and flat.

        Parameters
        ----------
        ic_all: 
            Image file collection
        obsparams:
            Dictionary with telescope configuration.
        filters:
            List of filters used in the dataset.
        binning_ext:
            String with binning used in science image.

        Returns
        -------

        """
        # Create names for master calibration files
        ddark = {}
        dflat = {}
        dark_exptimes = []
        out_path = self._master_calib_path
        suffix = self._suffix

        # Create name master bias
        master_bias = 'master_bias_%s%s' % (binning_ext, suffix)
        if self._telescope == 'DDOTI 28-cm f/2.2':
            master_bias = 'master_bias_%s_%s%s' % (self._instrument, binning_ext, suffix)
        mbias_path = join_path(master_bias, out_path)

        dbias = [mbias_path, False]
        if self._config['BIAS_CORRECT']:
            create_mbias = False if os.path.exists(mbias_path) and not self._force_reduction else True
            dbias = [mbias_path, create_mbias]

        # Create master dark names
        if self._config['DARK_CORRECT']:
            image_typ = obsparams['imagetyp_dark']
            regex_match = False
            if obsparams['add_imagetyp_keys'] and 'dark' in obsparams['add_imagetyp']:
                image_typ = f"{obsparams['imagetyp_dark']}|" \
                            f"{obsparams['add_imagetyp']['dark']['imagetyp']}"
                regex_match = True
            ic_filtered = ic_all.filter(regex_match=regex_match,
                                        imagetyp=image_typ)
            if ic_filtered.summary is not None:

                dark_exptimes = list(np.unique(ic_filtered.summary[obsparams['exptime'].lower()]))

                for dexpt in dark_exptimes:
                    master_dark = 'master_dark_%s_%s%s' % (binning_ext,
                                                           str.replace(str(dexpt), '.', 'p'),
                                                           suffix)
                    if self._telescope == 'DDOTI 28-cm f/2.2':
                        master_dark = 'master_dark_%s_%s_%s%s' % (self._instrument, binning_ext,
                                                                  str.replace(str(dexpt), '.', 'p'),
                                                                  suffix)
                    mdark_path = join_path(master_dark, out_path)
                    create_mdark = False if os.path.exists(mdark_path) and not self._force_reduction else True
                    ddark[dexpt] = [mdark_path, create_mdark]
            else:
                self._config['DARK_CORRECT'] = False

        for filt in filters:
            # Create master flat names
            if self._config['FLAT_CORRECT']:
                master_flat = 'master_flat_%s_%s%s' % (filt[0], binning_ext, suffix)
                if self._telescope == 'DDOTI 28-cm f/2.2':
                    master_flat = 'master_flat_%s_%s_%s%s' % (self._instrument,
                                                              filt[0], binning_ext, suffix)
                mflat_path = join_path(master_flat, out_path)
                create_mflat = False if os.path.exists(mflat_path) and not self._force_reduction else True
                dflat[filt[0]] = [mflat_path, create_mflat]

        del ic_all, obsparams

        return dbias, ddark, dflat, dark_exptimes

    def get_file_list(self, file_list, dfilter=None, regexp=False, key_find='find',
                      invert_find=False, dkeys=None, abspath=False):
        """Wrapper for filter_img_file_collection.

        See filter_img_file_collection() for parameter description.
        """
        if isinstance(file_list, ImageFileCollection):
            file_list = self.filter_img_file_collection(file_list, dfilter,
                                                        abspath=abspath, regexp=regexp,
                                                        key_find=key_find, invert_find=invert_find,
                                                        dkeys=dkeys)

        return file_list

    def prepare_fits_files(self, file_names, telescope, obsparams, imagetyp):
        """ Create uniform data structure.

        Create uniform data structure using the image type,
        telescope and known parameter to update the header values.

        """
        dates = []
        filters = []
        binnings = []

        file_counter = 0
        for file_path in file_names:
            fname = Path(file_path).stem
            fname_suffix = Path(file_path).suffix
            if self.verbose:
                self._log.debug(f"> Preparing {fname}")

            # Load fits file
            hdul = fits.open(file_path, mode='readonly', ignore_missing_end=True)
            hdul.verify('fix')
            prim_hdr = hdul[0].header
            new_hdr = prim_hdr.copy()

            # Get observation date from header
            match = re.search(r'\d{4}-\d{2}-\d{2}', prim_hdr[obsparams['date_keyword']])
            date = datetime.strptime(match.group(), '%Y-%m-%d').date()
            if date not in dates:
                dates.append(date)

            # Setup indices for multi-chip data
            nhdus = len(hdul)
            istart = 0
            if nhdus > 1:
                istart = 1

            # Loop from first-data to last HDU
            hdu_indices = list(range(nhdus))[istart:]
            for i in hdu_indices:
                hdul_subset = hdul[i]
                hdr_subset = hdul_subset.header
                data_subset = hdul_subset.data

                for k, v in hdr_subset.items():
                    if obsparams['keys_to_exclude'] is not None \
                            and k in obsparams['keys_to_exclude']:
                        continue
                    new_hdr[k] = v

                # Check the object key in header and set if needed
                if 'OBJECT' not in new_hdr:
                    new_hdr['OBJECT'] = new_hdr[obsparams['object']].strip()

                # Check the imagetyp key in header and set
                new_hdr['IMAGETYP'] = imagetyp

                # Special case GROND NIR obs
                if i == 0 and telescope == 'GROND_IRIM':
                    filt_arr = np.array(obsparams['cut_order'])

                    for filt in filt_arr:
                        filt_hdr = new_hdr.copy()
                        idx = np.where(filt == filt_arr, True, False)
                        for k in filt_arr[~idx]:
                            [filt_hdr.remove(key) for key, val in filt_hdr.items()
                             if f"{k}_" in key]

                        # Update filter keyword
                        filt_key = filt_arr[idx][0]
                        filt_trnsl = obsparams['filter_translations'][filt_key]
                        if filt_trnsl is None:
                            filt_trnsl = 'clear'
                        filt_hdr['FILTER'] = filt_trnsl
                        if filt_trnsl not in filters:
                            filters.append(filt_trnsl)

                        # Cut image amplifier regions
                        cut_idx = obsparams['image_cutx'][filt]
                        data_subset2 = data_subset * -1.
                        new_img = data_subset2[:, cut_idx[0]:cut_idx[1]]

                        # Update gain and readout noise
                        filt_hdr, file_counter = update_gain_readnoise(new_img, filt_hdr, obsparams,
                                                                       file_counter, filt_trnsl)

                        # Update binning
                        filt_hdr = update_binning(filt_hdr, obsparams, binnings)

                        # Save result to separate fits file
                        new_fname = f"{fname}_{filt_hdr['FILTER']}{fname_suffix}"
                        new_hdu = fits.PrimaryHDU(data=new_img, header=filt_hdr)
                        new_hdu.writeto(os.path.join(self._tmp_path, new_fname),
                                        output_verify='ignore', overwrite=True)
                    # Close header
                    hdul.close()

                    # Remove the original file
                    os.system(f"rm -f {file_path}")
                    break

                # Update filter
                filt_trnsl = None
                filt_key = None
                if obsparams['filter'] in new_hdr:
                    filt_key = new_hdr[obsparams['filter']].strip()
                    filt_trnsl = obsparams['filter_translations'][filt_key]
                if filt_trnsl is None:
                    filt_trnsl = 'clear'
                new_hdr['FILTER'] = filt_trnsl
                if (filt_trnsl, filt_key) not in filters:
                    filters.append((filt_trnsl, filt_key))

                # Update binning
                new_hdr = update_binning(new_hdr, obsparams, binnings)

                if i > 0:
                    # Update gain and readout noise
                    new_hdr, file_counter = update_gain_readnoise(data_subset, new_hdr,
                                                                  obsparams, file_counter)
                    # Remove extension name
                    new_hdr.remove('EXTNAME')

                    # Save result to separate fits file
                    new_fname = f"{fname}_{new_hdr['FILTER']}{fname_suffix}"
                    new_hdu = fits.PrimaryHDU(data=data_subset, header=new_hdr)
                    new_hdu.writeto(os.path.join(self._tmp_path, new_fname),
                                    output_verify='ignore', overwrite=True)

                    # Remove the original file if done
                    if i == hdu_indices[-1]:
                        os.system(f"rm -f {file_path}")
                        hdul.close()
                else:
                    # Update gain and readout noise
                    if telescope != 'GROND_OIMG':
                        new_hdr, file_counter = update_gain_readnoise(data_subset, new_hdr,
                                                                      obsparams, file_counter)

                    # Save result to separate fits file
                    new_hdu = fits.PrimaryHDU(data=data_subset, header=new_hdr)

                    # os.system(f"ls -la1 {self._tmp_path}")
                    new_hdu.writeto(file_path, output_verify='ignore', overwrite=True)

                    hdul.close()

        return dates[0], filters, binnings

    @staticmethod
    def create_dfilter(instrument, obsparams, mode):
        """ Create a filter for ccdproc"""
        key = 'imagetyp' if obsparams['imagetyp'] == 'IMAGETYP' else obsparams['imagetyp']
        def_key = f'imagetyp_{mode}'

        # Create data filter for imagetyp and instrument
        dfilter = {key: obsparams[def_key]}
        if instrument is not None:
            dfilter[obsparams['instrume'].lower()] = instrument

        add_filters = {}
        regexp = False
        if obsparams['add_imagetyp_keys'] and mode in obsparams['add_imagetyp']:
            add_filters = {k: v for k, v in obsparams['add_imagetyp'][mode].items() if
                           k != 'imagetyp'}
            x = obsparams['add_imagetyp'][mode]
            if 'imagetyp' in x:
                dfilter[key] = f"({obsparams[def_key]}|" \
                               f"{x['imagetyp']})"
                regexp = True

        return dfilter, add_filters, regexp

    @staticmethod
    def create_vignette_mask(original_size, vignette=-1.):
        """

        """
        mask = None

        # Only search sources in a circle with radius <vignette>
        if (0. < vignette <= 1.) & (vignette != -1.):
            # Create the vignette mask for a 1x1 binning
            original_sidelength = np.min(original_size)
            x = np.arange(0, original_size[1])
            y = np.arange(0, original_size[0])
            vignette_radius = vignette * original_sidelength / 2.

            # Generate the mask for 1x1 binning
            mask = (x[np.newaxis, :] - original_size[1] / 2) ** 2 + \
                   (y[:, np.newaxis] - original_size[0] / 2) ** 2 >= vignette_radius ** 2

        return mask

    @staticmethod
    def parse_section(section_str, bin_x=None, bin_y=None):
        """
        Parses a section string from the FITS header to extract the slice indices for the given section.

        Parameters
        ----------
        section_str : str
            A string representing the section in the FITS header (e.g., '[1:1024, 1:1024]').
        bin_x : int, optional
            Binning factor in the x-direction (default is None).
        bin_y : int, optional
            Binning factor in the y-direction (default is None).

        Returns
        -------
        tuple of slice
            A tuple of two slice objects representing the y and x slices, respectively.

        Notes
        -----
        The function converts the 1-based FITS indexing (inclusive) to Python's 0-based indexing (exclusive).
        If binning is provided, the slice indices are adjusted accordingly.
        """
        section_str = section_str.strip('[]')
        x_part, y_part = section_str.split(',')
        x1, x2 = [int(x) for x in x_part.split(':')]
        y1, y2 = [int(y) for y in y_part.split(':')]
        if bin_x is not None and bin_y is not None:
            if bin_x > 1 and bin_y > 1:
                return slice((y1 - 1) // bin_y, y2 // bin_y), slice((x1 - 1) // bin_x, x2 // bin_x)
        return slice(y1 - 1, y2), slice(x1 - 1, x2)

    @staticmethod
    def filter_img_file_collection(image_file_collection, ldfilter=None,
                                   abspath=False, regexp=False,
                                   key_find='find', invert_find=False,
                                   return_mask=False, key='file',
                                   dkeys=None, copy=True):
        """Filter image file collection object.

        Parameters
        ----------
        image_file_collection: ccdproc.ImageFileCollection
            An ImageFileCollection object containing the images to be filtered.
        ldfilter: dict, list, tuple, or None, optional
            Dictionary or list of dictionaries containing filter criteria.
            If None, no filtering is applied. Defaults to None.
        abspath: bool, optional
            If True, returns absolute paths for the filtered file list.
            Defaults to False.
        regexp: bool, optional
            If True, uses regular expressions for matching file names.
            Defaults to False.
        key_find: str, optional
            Key used to find specific entries in the filter dictionary.
            Defaults to 'find'.
        invert_find: bool, optional
            If True, inverts the matching criteria for the `key_find` value.
            Defaults to False.
        return_mask: bool, optional
            If True, returns the mask used for filtering along with the file list.
            Defaults to False.
        key: str, optional
            The key used to identify files in the ImageFileCollection summary.
            Defaults to 'file'.
        dkeys: dict or None, optional
            Dictionary of additional keys to add to the filter dictionary.
            Defaults to None.
        copy: bool, optional
            If True, makes a copy of the filter dictionary when adding keys.
            Defaults to True.

        Returns
        -------
        file_list: numpy.ndarray
            Array containing the filtered list of file names.
        mask: numpy.ndarray, optional
            Boolean mask used for filtering, returned if `return_mask` is True.

        """
        if ldfilter is None:
            ldfilter = dict()
        ifc = image_file_collection

        if not isinstance(ldfilter, (tuple, list)):
            ldfilter = [ldfilter]
        if dkeys is not None:
            ldfilter = add_keys_to_dict(ldfilter, dkeys, copy=copy)
        mask = None
        lmask = []
        for dfilter in ldfilter:
            if dfilter is None or len(dfilter) == 0:
                continue
            lfmask = []
            if key_find in dfilter:
                key_string = dfilter.pop(key_find)
                lfiles_find_mask = np.char.find(ifc.summary[key].data, key_string) > -1
                if invert_find:
                    lfiles_find_mask = np.invert(lfiles_find_mask)
                lfmask.append(lfiles_find_mask)
            # print(dfilter, key_find, lfmask, regexp, abspath)
            lfiles = ifc.files_filtered(regex_match=regexp, include_path=abspath, **dfilter)
            # print(lfiles)
            lfiles_mask = np.isin(ifc.summary['file'].flatten(), lfiles)

            lfmask.append(lfiles_mask)
            fmask = np.logical_and.reduce(lfmask)
            lmask.append(fmask)
            del lfiles_mask

        if len(lmask) > 0:
            mask = np.logical_or.reduce(lmask)
            file_list = ifc.summary['file'][mask]
        else:
            file_list = ifc.summary['file']

        del image_file_collection, ifc, lmask

        if return_mask:
            return file_list, mask
        else:
            return file_list


def check_shape(ccd1, ccd2):
    """

    Parameters
    ----------
    ccd1 : astropy.nddata.CCDData
        The first CCDData object to be checked.
    ccd2 : astropy.nddata.CCDData
        The second CCDData object to be checked.

    Returns
    -------
    tuple :
        A tuple containing the trimmed CCDData objects (trimmed_ccd1, trimmed_ccd2).
    """
    # Get the shapes of both CCDData objects
    shape1 = ccd1.shape
    shape2 = ccd2.shape

    # Check if trimming is necessary
    if shape1 == shape2:
        return ccd1, ccd2

    # Find the minimum dimensions for trimming
    min_rows = min(shape1[0], shape2[0])
    min_cols = min(shape1[1], shape2[1])

    # Define the trimming slice
    trim_slice = f"[1:{min_cols}, 1:{min_rows}]"

    # Trim both CCDData objects to the same shape using trim_image
    trimmed_ccd1 = ccdproc.trim_image(ccd=ccd1, fits_section=trim_slice)
    trimmed_ccd2 = ccdproc.trim_image(ccd=ccd2, fits_section=trim_slice)

    return trimmed_ccd1, trimmed_ccd2


def compute_2d_background_simple(imgarr, box_size, win_size,
                                 bkg_estimator=SExtractorBackground,
                                 rms_estimator=StdBackgroundRMS):
    """Compute a 2D background for the input array.
    This function uses `~photutils.background.Background2D` to determine
    an adaptive background that takes into account variations in flux
    across the image.

    Parameters
    ----------
    imgarr: ndarray
        Science data for which the background needs to be computed
    box_size: integer
        The box_size along each axis for Background2D to use.
    win_size: integer
        The window size of the 2D median filter to apply to the low-resolution map as the
        `filter_size` parameter in Background2D.
    bkg_estimator: function
        The name of the function to use as the estimator of the background.
    rms_estimator: function
        The name of the function to use for estimating the RMS in the background.

    Returns
    -------
    bkg_background:
        An ND-array of the same shape as the input image array which contains the determined
        background across the array. If Background2D fails for any reason, a simpler
        sigma-clipped single-valued array will be computed instead.
    bkg_median:
        The median value (or single sigma-clipped value) of the computed background.
    bkg_rms:
        ND-array the same shape as the input image array which contains the RMS of the
        background across the array. If Background2D fails for any reason, a simpler
        sigma-clipped single-valued array will be computed instead.
    bkg_rms_median:
        The median value (or single sigma-clipped value) of the background RMS.

    """
    # SExtractorBackground and StdBackgroundRMS are the defaults
    bkg = None
    bkg_background = None
    bkg_median = None
    bkg_rms = None
    bkg_rms_median = None

    # exclude_percentiles = [5, 10, 15, 25, 50, 75]
    exclude_percentiles = [10]
    for percentile in exclude_percentiles:
        try:
            bkg = Background2D(imgarr, (box_size, box_size),
                               filter_size=(win_size, win_size),
                               bkg_estimator=bkg_estimator(),
                               bkgrms_estimator=rms_estimator(),
                               exclude_percentile=percentile, edge_method="pad")
        except (Exception,):
            bkg = None
            continue

        if bkg is not None:
            bkg_background = bkg.background
            bkg_median = bkg.background_median
            bkg_rms = bkg.background_rms
            bkg_rms_median = bkg.background_rms_median
            break

    # If Background2D does not work at all, define default scalar values for
    # the background to be used in source identification
    if bkg is None:
        # Detect the sources
        threshold = detect_threshold(imgarr, nsigma=2.0,
                                     sigma_clip=SigmaClip(sigma=3.0, maxiters=10))
        segment_img = detect_sources(imgarr, threshold, npixels=5)
        src_mask = segment_img.make_source_mask(footprint=None)
        sigcl_mean, sigcl_median, sigcl_std = sigma_clipped_stats(imgarr,
                                                                  sigma=3.0,
                                                                  mask=src_mask,
                                                                  maxiters=10)
        # sigcl_mean, sigcl_median, sigcl_std = sigma_clipped_stats(imgarr, sigma=3.0, mask=mask, maxiters=11)
        bkg_median = max(0.0, sigcl_median)
        bkg_rms_median = sigcl_std
        # Create background frame shaped like imgarr populated with sigma-clipped median value
        bkg_background = np.full_like(imgarr, bkg_median)
        # Create background frame shaped like imgarr populated with sigma-clipped standard deviation value
        bkg_rms = np.full_like(imgarr, sigcl_std)

    return bkg_background, bkg_median, bkg_rms, bkg_rms_median


def join_path(fname, directory=None):
    """Join the given file name with the given directory path.

    Parameters
    ----------
    fname: str
        The file name to be joined with the directory.
    directory: str or None, optional
        The directory path to join with the file name. If None, only the file name is returned.

    Returns
    -------
    str
        The full path formed by joining the directory and the file name. If no directory is provided,
        the original file name is returned.
    """
    if directory is not None:
        fname = os.path.join(directory, os.path.basename(fname))
    return fname


def update_gain_readnoise(img, hdr, obsparams, file_counter, filt_key=None):
    """ Update gain and readout noise.

    """
    # Update gain
    if 'GAIN' not in hdr and obsparams['gain'] is not None:
        if isinstance(obsparams['gain'], str):
            if filt_key is None:
                hdr['GAIN'] = (hdr[obsparams['gain']], 'CCD Gain (e-/ADU)')
            else:
                hdr['GAIN'] = (hdr[obsparams['gain'][filt_key]], 'CCD Gain (e-/ADU)')
        elif isinstance(obsparams['gain'], float):
            hdr['GAIN'] = (obsparams['gain'], 'CCD Gain (e-/ADU)')

    # Update readout noise
    if 'RON' not in hdr:
        if filt_key is None:
            ron = obsparams['readnoise']
        else:
            ron = obsparams['readnoise'][filt_key]
        if ron is not None:
            if isinstance(ron, str):
                hdr['RON'] = (hdr[ron], 'CCD Read Out Noise (e-)')
            else:
                hdr['RON'] = (ron, 'CCD Read Out Noise (e-)')
        if 'GAIN' in hdr and ron is None and \
                hdr['IMAGETYP'] == bc.IMAGETYP_BIAS and \
                file_counter == 0:
            bkg_background, bkg_median, bkg_rms, bkg_rms_median = \
                compute_2d_background_simple(img, box_size=11, win_size=5)
            readnoise = bkg_rms_median * hdr['GAIN'] / np.sqrt(2)
            hdr['RON'] = (readnoise, 'CCD Read Out Noise (e-)')
            # file_counter = file_counter + 1
    return hdr, file_counter


def update_binning(hdr, obsparams, binnings: list):
    """Update the FITS header with new binning information.

    Parameters
    ----------
    hdr : FITS header
        The header of the FITS image, where binning information may be updated.
    obsparams : dict
        Dictionary containing observational parameters, specifically requiring 
        the keys for binning in X and Y dimensions as defined in `obsparams['binning']`.
    binnings : list of tuples
        List of binning configurations encountered so far. If the current 
        binning is not in the list, it is appended.

    Returns
    -------
    hdr : FITS header
        Updated FITS header with binning information.
        
    """
    binning = get_binning(hdr, obsparams)
    if binning not in binnings:
        binnings.append(binning)
    _bin_str = f'{binning[0]}x{binning[1]}'

    if not (bc.BINNING_DKEYS[0] in hdr and bc.BINNING_DKEYS[1] in hdr):
        hdr[bc.BINNING_DKEYS[0]] = binning[0]
        hdr[bc.BINNING_DKEYS[1]] = binning[1]
    if 'BINNING' not in hdr:
        hdr['binning'] = _bin_str

    return hdr


def get_binning(header, obsparams):
    """ Derive binning from image header.

    Use obsparams['binning'] keywords, unless both keywords are set to 1.

    Parameters
    ----------
    header : FITS header
        The header of the FITS image, which contains metadata including
        binning information.
    obsparams : dict
        Dictionary containing observational parameters. Specifically requires
        `obsparams['binning']` to specify the header keywords for the binning
        in X and Y dimensions.

    Returns
    -------
    tuple of int
        A tuple `(binning_x, binning_y)` representing the binning in the X
        and Y directions as derived from the header.

    """
    binning_x = None
    binning_y = None

    param_bin_x = obsparams['binning'][0]
    param_bin_y = obsparams['binning'][1]

    if not (param_bin_x in header and param_bin_y in header):
        if obsparams['telescope_keyword'] in ['CBNUO-JC', 'CTIO 0.9 meter telescope']:
            binning_x = obsparams['image_size_1x1'][0] // header['NAXIS1']
            binning_y = obsparams['image_size_1x1'][1] // header['NAXIS2']
    else:
        if (isinstance(param_bin_x, int) and
                isinstance(param_bin_y, int)):
            binning_x = param_bin_x
            binning_y = param_bin_y
        elif '#' in param_bin_x:
            if '#blank' in param_bin_x:
                binning_x = float(header[param_bin_x.
                                  split('#')[0]].split()[0])
                binning_y = float(header[param_bin_y.
                                  split('#')[0]].split()[1])
            elif '#x' in param_bin_x:
                binning_x = float(header[param_bin_x.
                                  split('#')[0]].split('x')[0])
                binning_y = float(header[param_bin_y.
                                  split('#')[0]].split('x')[1])
            elif '#_' in param_bin_x:
                binning_x = float(header[param_bin_x.
                                  split('#')[0]].split('_')[0])
                binning_y = float(header[param_bin_y.
                                  split('#')[0]].split('_')[1])
            elif '#CH#' in param_bin_x:
                # Only for RATIR
                channel = header['INSTRUME'].strip()[1]
                binning_x = float(header[param_bin_x.
                                  replace('#CH#', channel)])
                binning_y = float(header[param_bin_y.
                                  replace('#CH#', channel)])
        else:
            binning_x = header[param_bin_x]
            binning_y = header[param_bin_y]

    return binning_x, binning_y


def get_filename(ccd_file, key='FILENAME'):
    """ Get file name from string or ccd object.

    Parameters
    ----------
    ccd_file : str, astropy.nddata.CCDData
        String or CCDData object of which the name should be found.
    key : str, optional
        Header keyword to identify file name.
        Only used if the ccd_file is a CCDData object. Default is ``FILENAME``.

    Returns
    -------
    name_file : str
        Name of file
    """
    name_file = None
    if isinstance(ccd_file, str):
        name_file = ccd_file
    elif isinstance(ccd_file, CCDData):
        if key in ccd_file.header:
            name_file = ccd_file.header[key]
    return name_file


def add_key_to_hdr(hdr, key, value):
    """ Add key to header.

    Parameters
    ----------
    hdr : astropy.io.fits.Header
        The FITS header object.
    key : str
        Keyword to be updated.
    value : object
        Value to be set.

    Returns
    -------
    hdr : astropy.io.fits.Header
        Updated header.
    """
    if value is not None and key is not None:
        hdr[key] = value
    return hdr


def ammend_hdr(header):
    """Remove trailing blank from header.

    Parameters
    ----------
    header : astropy.io.fits.Header
        A FITS header object.

    Returns
    -------
    header : astropy.io.fits.Header
        The same fits header object with trailing blanks removed.
    """
    if '' in header:
        del header['']
    return header


def add_keys_to_dict(ldict, dkeys, copy=True, force=False):
    """Add dictionary content to the list of dictionaries.

    Parameters
    ----------
    ldict : list, dict
        List of dictionary to which the dict of keys is added.
    dkeys : dict
        Dictionary of keys and values to add to the given list of dictionary.
    copy : bool
        Make a deepcopy of input list of dictionary.
    force : bool, optional
        Force replacement of key in dictionary if True.
        Default is True.

    Returns
    -------
    ldict : dict
        List of dictionaries with added data
    """
    if ldict is not None and dkeys is not None:
        if not isinstance(ldict, (list, tuple)):
            ldict = [ldict]
        if copy:
            ldict = deepcopy(ldict)
        for i, v in enumerate(ldict):
            if v is not None and isinstance(v, dict):
                for key in dkeys:
                    if key is not None:
                        if key not in v or (key in v and force):
                            ldict[i][key] = dkeys[key]
    return ldict


def inv_median(a):
    return 1. / np.median(a)


def main():
    """ Main procedure """
    pargs = arguments.ParseArguments(prog_typ='reduce_sat_obs')
    args = pargs.args_parsed
    main.__doc__ = pargs.args_doc

    # Version check
    bc.check_version(_log)

    ReduceSatObs(input_path=args.input, args=args, silent=args.silent, verbose=args.verbose)


# -----------------------------------------------------------------------------


# Standard boilerplate to set 'main' as starting function
if __name__ == '__main__':
    main()
# -----------------------------------------------------------------------------
