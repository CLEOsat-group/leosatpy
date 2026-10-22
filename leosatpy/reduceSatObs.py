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
import shutil
from dataclasses import dataclass, field

import re
import sys
import time
from copy import deepcopy
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from pathlib import Path
from typing import Any, Optional

import ccdproc

import numpy as np
import pandas as pd

# import photutils

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


@dataclass
class CalibrationPrepResult:
    ok: bool
    dataset_failed: bool
    status: str
    reason: str = ""
    severity: str = "info"
    dataset_path: Optional[Path] = None
    binning: Optional[str] = None
    selected_source: Optional[Path] = None
    master_output: Optional[Path] = None
    failed_products: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    searched_sources: list[Path] = field(default_factory=list)
    calib_state: Optional[dict[str, Any]] = None
    master_darks: Optional[dict[str, Any]] = None


class CalibrationInternalError(RuntimeError):
    pass


class MasterBuildError(RuntimeError):
    pass


ALLOWED_RESULT_STATUSES = {
    'ok',
    'dataset_failed_missing_requested_bias',
    'dataset_failed_missing_requested_flat',
    'dataset_failed_no_bias_or_dark',
    'dataset_failed_master_build_error',
    'dataset_failed_validation_error',
    'dataset_failed_internal_error',
}

ALLOWED_MASTER_STATUSES = {
    'loaded_existing_science_date_master',
    'adopted_from_fallback_master',
    'overwritten_by_adopted_fallback_master',
    'created_from_raw',
    'overwritten_from_raw',
    'skipped_by_preference',
    'skipped_optional_unavailable_bias_fallback',
    'skipped_unavailable_dark_baseline',
    'failed_build',
    'failed_validation',
}

FAILURE_STATUS_TEXT = {
    'dataset_failed_missing_requested_bias': 'no usable master bias',
    'dataset_failed_missing_requested_flat': 'no usable master flat',
    'dataset_failed_no_bias_or_dark': 'neither a usable bias nor a usable dark',
    'dataset_failed_master_build_error': 'building a master calibration failed',
    'dataset_failed_validation_error': 'calibration check failed',
    'dataset_failed_internal_error': 'internal error (run with -v for details)',
}

EXCEPTION_RESULT_STATUSES = {
    MasterBuildError: 'dataset_failed_master_build_error',
    CalibrationInternalError: 'dataset_failed_internal_error',
}


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
        self._master_bias_info = {}
        self._master_dark_info = {}
        self._master_flat_info = {}
        self._dark_exptimes = []
        self._selected_calibration_source_folder = None
        self._force_reduction = args.force_reduction

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
        self._calib_state = None  # calibration availability; set by find_and_create_calibrations
        self._master_darks = None  # loaded master darks for science reduction; assigned from CalibrationPrepResult
        self._reduction_failures = []
        self._failure_log_path = None

        # Run full reduction
        self.run_full_reduction(silent=silent)

    def run_full_reduction(self, silent=False):
        """Run full-reduction routine on the input path.

        Prepare science files, find calibration files,
        i.e., master_bias, master_dark, and master_flat
        and run reduction for a given set of data.
        """
        starttime = time.perf_counter()
        self._log.info(f"{'=' * 80}")
        self._log.info(f"INITIALIZING IMAGE REDUCTION")
        self._log.info(f"{'=' * 80}")

        if not silent:
            self._log.info("> Check input argument(s) and search for data")

        # Initialise data set from input argument
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
        if not silent:
            self._log.info('')

        # Loop over instruments and run reduction for each instrument
        for i, inst in enumerate(inst_list):
            if not silent:
                self._log.info(f"{'=' * 80}")
                self._log.info(f"INSTRUMENT: {inst}")
                self._log.info(f"TELESCOPE: {inst_data[inst]['telescope']}")
                self._log.info(f"{'=' * 80}")
            self._instrument = inst
            self._telescope = inst_data[inst]['telescope']
            self._obsparams = inst_data[inst]['obsparams']

            # Get valid science observations from data set
            ds.get_valid_sci_observations(inst, prog_typ="reduceSatObs")
            obsfiles_grouped = ds.valid_sci_obs
            self._dataset_object = obsfiles_grouped
            n_groups = sum(1 for _ in obsfiles_grouped)

            # Loop over groups and run reduction for each group
            group_index = 0
            for src_path, files in obsfiles_grouped:
                if not silent:
                    # self._log.info(f"{' ':<2}{'=' * 78}")
                    self._log.info(f"> Current Data Set: {group_index + 1}/{n_groups}")
                    # self._log.info(f"{' ':<2}{'=' * 78}")
                    self._log.info(f"  Folder: {src_path}")
                    self._log.info(f"  Number of images: {len(files)}")
                # Reduce the individual data set
                self.run_single_reduction(src_path, files)

                if not silent:
                    self._log.info(f"  ====> Reduction [{group_index + 1}/{n_groups}] finished <====")
                    self._log.info(f"{'=' * 80}")
                group_index += 1

        endtime = time.perf_counter()
        dt = endtime - starttime
        td = timedelta(seconds=dt)

        self._log.info(f"Program execution time: {td}")
        if self._reduction_failures:
            self._log.warning(f"Skipped dataset/binning combinations: {len(self._reduction_failures)}")
            self._log.warning(f"Details in: {self._failure_log_path}")
        self._log.info('====> IMAGE REDUCTION FINISHED <====')

    @staticmethod
    def _make_failure_result(status, reason, dataset_path=None, binning=None,
                             selected_source=None, master_output=None,
                             failed_products=None, warnings=None,
                             searched_sources=None, severity='error'):
        if status not in ALLOWED_RESULT_STATUSES:
            raise CalibrationInternalError(f'Unsupported controlled failure status: {status}')
        return CalibrationPrepResult(
            ok=False,
            dataset_failed=True,
            status=status,
            reason=str(reason),
            severity=severity,
            dataset_path=Path(dataset_path).resolve() if dataset_path else None,
            binning=binning,
            selected_source=Path(selected_source).resolve() if selected_source else None,
            master_output=Path(master_output).resolve() if master_output else None,
            failed_products=list(failed_products or []),
            warnings=list(warnings or []),
            searched_sources=[Path(p).resolve() for p in (searched_sources or [])],
            calib_state=None,
            master_darks=None,
        )

    def _result_from_exception(self, exc, dataset_path=None, binning=None,
                               selected_source=None, master_output=None,
                               searched_sources=None):
        if isinstance(exc, MasterBuildError):
            status = 'dataset_failed_master_build_error'
        else:
            status = 'dataset_failed_internal_error'
        return self._make_failure_result(
            status=status,
            reason=f'{type(exc).__name__}: {exc}',
            dataset_path=dataset_path,
            binning=binning,
            selected_source=selected_source,
            master_output=master_output,
            searched_sources=searched_sources,
        )

    def _record_reduction_failure(self, result):
        if not isinstance(result, CalibrationPrepResult):
            raise CalibrationInternalError('Failure recorder requires CalibrationPrepResult input.')

        record = {
            'dataset_path': str(result.dataset_path) if result.dataset_path else '',
            'binning': result.binning or '',
            'filters': ','.join(sorted((result.calib_state or {}).get('flat', {}).keys())),
            'status': result.status,
            'reason': result.reason,
            'failed_products': '; '.join(sorted(
                f"{item.get('product', 'unknown')}:{item.get('reason', '')}" for item in result.failed_products
            )),
            'searched_sources': '; '.join(str(p) for p in result.searched_sources),
            'selected_source': str(result.selected_source) if result.selected_source else '',
            'master_output': str(result.master_output) if result.master_output else '',
        }
        self._reduction_failures.append(record)

        if self._failure_log_path is None:
            timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
            self._failure_log_path = Path(self._red_path or self._sci_root_path,
                                          f'fails_reduceSatObs_{timestamp}.log').resolve()

        header = ('dataset_path\tbinning\tfilters\tstatus\treason\tfailed_products\t'
                  'searched_sources\tselected_source\tmaster_output\n')
        file_exists = self._failure_log_path.exists()
        with open(self._failure_log_path, 'a', encoding='utf-8') as fh:
            if not file_exists:
                fh.write(header)
            fh.write(
                f"{record['dataset_path']}\t{record['binning']}\t{record['filters']}\t{record['status']}\t"
                f"{record['reason']}\t{record['failed_products']}\t{record['searched_sources']}\t"
                f"{record['selected_source']}\t{record['master_output']}\n"
            )

    def run_single_reduction(self, sci_src_path, fits_files_dict):
        """Run the full reduction process on a given data set.

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
        if tmp_path.name == 'atmp' and tmp_path != tmp_path.anchor:
            shutil.rmtree(tmp_path, ignore_errors=True)
        tmp_path.mkdir(exist_ok=True)

        # Get the file suffix
        unique_suffixes = np.unique(fits_files_dict['suffix'].values)
        if len(unique_suffixes) > 1:
            self._log.warning(f"{' ':<2}Multiple file suffixes found {list(unique_suffixes)}; "
                              f"using '{unique_suffixes[0]}'.")
        suffix = unique_suffixes[0]

        # Update needed variables
        self._sci_root_path = sci_src_path
        self._red_path = red_path
        self._tmp_path = tmp_path
        self._suffix = suffix

        # Copy the found science files to the tmp folder
        new_file_paths = []
        if not self.silent:
            self._log.info(f"> Register ``SCIENCE`` images")
            # self._log.info(f"{' ':<2}{'-' * 15}{' Register ``SCIENCE`` images '}{'-' * 15}")
            self._log.info(f"{' ':<2}> Create copy of raw images in temporary folder.")

        fits_df = pd.DataFrame(fits_files_dict)
        for row_index, row in fits_df.iterrows():
            abs_file_path = row['input']
            file_name = row['file_name']
            file_name_suffix = row['suffix']

            # Copy science files
            shutil.copy2(abs_file_path, tmp_path)
            new_fname = os.path.join(tmp_path, file_name + file_name_suffix)
            new_file_paths.append(new_fname)
            if not self.silent:
                bc.print_progress_bar(row_index + 1, len(fits_df), length=77,
                                      color=bc.BCOLORS.OKGREEN, prefix=' ' * 2)

        # Prepare/Normalize the science files to get metadata
        if not self.silent:
            # Add a new line after the progress bar
            sys.stdout.write('\n')
            self._log.info(f"{' ':<2}> Perform header normalization and index images for reduction.")

        obs_date, filters, binnings = self.prepare_fits_files(new_file_paths,
                                                              self._telescope,
                                                              self._obsparams,
                                                              'science')
        # print(obs_date, filters, binnings)
        if not self.silent:
            # Extract names for clean printing
            filter_names = ", ".join([f[0] for f in filters])
            binning_names = ", ".join([f"{b[0]}x{b[1]}" for b in binnings])

            # Determine singular/plural labels
            filter_label = "Filter" if len(filters) == 1 else "Filters"
            binning_label = "Binning" if len(binnings) == 1 else "Binnings"

            self._log.info(f"{' ':<2}{'-' * 78}")
            self._log.info(f"{' ':<2}Summary:")
            self._log.info(f"{' ':<2}{'-' * 78}")
            self._log.info(f"{' ':<2}{'Observation date':<20}: {obs_date}")
            self._log.info(f"{' ':<2}{binning_label:<20}: {binning_names}")
            self._log.info(f"{' ':<2}{filter_label:<20}: {filter_names}")
            self._log.info(f"{' ':<2}{'-' * 78}")

        # Loop over binnings and prepare the calibration files for each binning
        for binxy in binnings:
            self._binxy = binxy
            self._bin_str = f"{binxy[0]}x{binxy[1]}"
            if not self.silent:
                self._log.info("")
                self._log.info(f"{'-' * 45}")
                self._log.info(f"  PROCESSING BINNING: {self._bin_str}")
                self._log.info(f"{'-' * 45}")

            # Find and prepare calibration files
            self._calib_state = None
            self._master_darks = None
            try:
                result = self.find_and_create_calibrations(self._telescope,
                                                           self._obsparams,
                                                           filters,
                                                           binxy,
                                                           obs_date=obs_date)
            except (CalibrationInternalError, MasterBuildError) as exc:
                result = self._result_from_exception(exc,
                                                     dataset_path=self._sci_root_path,
                                                     binning=self._bin_str)
            except Exception as exc:
                result = self._result_from_exception(
                    CalibrationInternalError(str(exc)),
                    dataset_path=self._sci_root_path,
                    binning=self._bin_str,
                )

            if not isinstance(result, CalibrationPrepResult):
                result = self._make_failure_result(
                    status='dataset_failed_internal_error',
                    reason='Calibration preparation did not return CalibrationPrepResult.',
                    dataset_path=self._sci_root_path,
                    binning=self._bin_str,
                )

            # In silent mode the dataset header is not shown, so name the folder in the result line
            result_label = f"binning {self._bin_str}" + (f", {self._sci_root_path}" if self.silent else '')
            if not result.ok:
                self._record_reduction_failure(result)
                products = ', '.join(sorted({str(item.get('reason', '')) for item in result.failed_products}))
                self._log.warning(
                    f"> Skipped science reduction ({result_label}): "
                    f"{FAILURE_STATUS_TEXT.get(result.status, result.status)}"
                    + (f" [{products}]" if products else '')
                )
                self._log.debug(f"{' ':<4}Failure details: {result.reason}")
                continue

            self._calib_state = result.calib_state
            self._master_darks = result.master_darks

            # Run the CCD processing on the science images
            self.run_ccdproc(self._obsparams, filters, binxy)
            self._log.info(f"> Science reduction done ({result_label})")

        # Clean temporary folder
        self._log.debug(f"{' ':<2}> Cleaning up")
        if tmp_path.name == 'atmp' and tmp_path != tmp_path.anchor:
            shutil.rmtree(tmp_path, ignore_errors=True)


    def run_ccdproc(self, obsparams, filters, binnings):
        """Run ccdproc on the prepared data set.

        Consumes the calib_state and master_darks map prepared by
        ``find_and_create_calibrations``.  Does NOT re-discover, re-create, or
        re-stage any calibration products, and does NOT mutate ``self._config``
        to reflect ephemeral availability.

        Behaviour:
           * If ``self._master_darks['sci_dark']`` has an entry for a science
             frame's exposure time → apply dark (only when DARK_CORRECT=True).
           * Else if master bias is available and BIAS_CORRECT=True → bias-only
             fallback.
           * Else → no thermal/bias correction; header flags record reality.

        Parameters
        ----------
        obsparams : dict
            A Dictionary containing telescope and observational parameters.
        filters : list
            A list with filter identified in the data set.
        binnings : tuple
            Binning of the current data set, typically specified as "XxY" (e.g., "2x2").
        """

        # Calibration file preparation must have run first
        if self._calib_state is None or self._master_darks is None:
            raise RuntimeError(
                'run_ccdproc called before find_and_create_calibrations.'
            )

        # Initialize ImageFileCollection
        ic_all = ImageFileCollection(location=self._tmp_path)
        _bin_str = f'{binnings[0]}x{binnings[1]}'
        self._bin_str = _bin_str

        has_multiple_amps = obsparams['multiple_amps']

        # Extract gain / readnoise from staged science frames
        gain, readnoise = self._extract_gain_readnoise(ic_all)

        # Load master bias
        calib_state = self._calib_state
        ccd_master_bias = None
        if self._config['BIAS_CORRECT'] and calib_state['bias'].get('available'):
            master_bias_path = calib_state['bias'].get('master_path')
            if master_bias_path is not None:
                self._log.debug(f"{' ':<4}Loading master bias: {os.path.basename(str(master_bias_path))}")
                ccd_master_bias = self.convert_fits_to_ccd(str(master_bias_path), single=True)
                if (readnoise is None
                        and ccd_master_bias is not None
                        and 'ron' in ccd_master_bias.header):
                    readnoise = ccd_master_bias.header['ron']

        # Load master darks
        master_darks = self._master_darks  # {'sci_dark': {...}, 'flat_dark': {...}}
        sci_dark_count = len(master_darks['sci_dark'])
        flat_dark_count = len(master_darks['flat_dark'])

        self._log.debug(f"{' ':<4}Science-reduction darks: {sci_dark_count} sci_dark exposure(s), "
                        f"{flat_dark_count} flat_dark exposure(s)")

        # Process files by filter
        for filt in filters:
            filt_name = filt[0]
            sci_img_count_by_filter = len(
                ic_all.files_filtered(imagetyp=bc.IMAGETYP_LIGHT, filter=filt_name)
            )
            if sci_img_count_by_filter == 0:
                continue

            if not self.silent:
                self._log.info(f"> Reduce {sci_img_count_by_filter} science image(s) in filter {filt_name}")

            # Load master flat
            ccd_master_flat = {}
            if self._config['FLAT_CORRECT']:
                flat_state = calib_state['flat'].get(filt_name, {})
                flat_path = flat_state.get('master_path')
                if flat_path is not None and flat_state.get('available'):
                    self._log.debug(f"{' ':<4}Loading master flat: {os.path.basename(str(flat_path))}")
                    loaded = self.convert_fits_to_ccd(str(flat_path), single=True)
                    if loaded:
                        ccd_master_flat[filt_name] = loaded

            if not ccd_master_flat:
                ccd_master_flat = None
                self._ccd_mask_fname = None

            # Reduce science images
            self.ccdproc_sci_images(
                files_list=ic_all,
                obsparams=obsparams,
                image_filter=filt_name,
                master_bias=ccd_master_bias,
                master_dark=master_darks['sci_dark'],
                master_flat=ccd_master_flat,
                gain=gain,
                readnoise=readnoise,
                multiple_amps=has_multiple_amps,
                cosmic=self._config['CORRECT_COSMIC'],
            )

        del ic_all

        gc.collect()

    def ccdproc_sci_images(self, files_list,
                           obsparams, image_filter=None,
                           master_bias=None,
                           master_dark=None,
                           master_flat=None,
                           multiple_amps=False,
                           gain=None, readnoise=None,
                           cosmic=False,
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
            Gain of the detector in electrons per ADU.
            Defaults to None.
        readnoise : float or None, optional
            Readout noise of the detector in electrons.
            Defaults to None.
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
        if gain is not None and not isinstance(gain, u.Quantity):
            gain = gain * u.Unit("electron") / u.Unit("adu")
        if readnoise is not None and not isinstance(readnoise, u.Quantity):
            readnoise = readnoise * u.Unit("electron")

        # Create default filter if none provided, add additional filters if given, and get file list
        dfilter = {'imagetyp': bc.IMAGETYP_LIGHT,
                   'binning': self._bin_str} if dfilter is None else dfilter
        if dfilter is not None and key_filter is not None and image_filter is not None:
            dfilter = add_keys_to_dict(dfilter, {key_filter: image_filter})
        files_list = self.get_file_list(files_list, dfilter,
                                        key_find=key_find, invert_find=invert_find,
                                        abspath=False)

        mbias_ccd = master_bias
        mflat_ccd = None

        # Get ccd mask file if exists
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

        # Loop over files and process them individually
        files_list = list(files_list)
        n_files = len(files_list)
        for file_index, filename in enumerate(files_list):
            self._log.debug(f"{' ':<4}Reducing: {filename}")
            if not self.silent and not self.verbose:
                bc.print_progress_bar(file_index + 1, n_files, length=77,
                                      color=bc.BCOLORS.OKGREEN, prefix=' ' * 2)

            # Read image into a CCD object
            fname = os.path.join(self._tmp_path, filename)
            try:
                ccd = CCDData.read(fname, unit=u.Unit("adu"))
            except (Exception,):
                self._log.warning(f"{' ':<4}Failed to load {filename}. Skipping file.")
                continue

            # Extract header info and exposure time
            ccd_hdr = ccd.header
            ccd_expt = float(ccd_hdr['exptime'])

            # Determine if the ccd is cropped
            ccd_is_cropped = False
            crop_key = obsparams['cropsec']
            if (not multiple_amps
                    and crop_key is not None
                    and (ccd_hdr.get('LTV1', 0) < 0
                         or ccd_hdr.get('LTV2', 0) < 0)):
                ccd_is_cropped = True

            # Process the ccd, trim image and subtract overscan if needed
            ccd = self.process_ccd(img_ccd=ccd, obsparams=obsparams, multiple_amps=multiple_amps)

            # Gain correct
            gain_corrected = False
            if gain is not None and self._config['CORRECT_GAIN']:
                ccd = ccdproc.gain_correct(ccd, gain)
                gain_corrected = True

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

            # Add ccd mask to ccd object
            ccd.mask = ccd_mask

            # Bias or dark frame subtract
            dark_corrected = False
            bias_corrected = False
            flat_corrected = False

            if master_dark is not None and self._config['DARK_CORRECT'] and ccd_expt in master_dark:
                mdark, dark_is_pure, is_nearest = master_dark[ccd_expt]
                # 1. If Dark is "Pure" (Bias-subtracted), we MUST bias-subtract the Science frame now
                if dark_is_pure and self._config['BIAS_CORRECT']:
                    if isinstance(mbias_ccd, str):
                        mbias_ccd = self.convert_fits_to_ccd(mbias_ccd, single=True)

                    temp_mbias = mbias_ccd
                    if ccd_is_cropped:
                        temp_mbias = self.crop_image(img_ccd=temp_mbias, hdr=ccd_hdr,
                                                     crop_section_unbinned=ccd_hdr[crop_key])

                    ccd, temp_mbias = check_shape(ccd, temp_mbias)
                    ccd = ccdproc.subtract_bias(ccd, temp_mbias)
                    bias_corrected = True
                    add_key_to_hdr(ccd.header, 'MBIAS', get_filename(mbias_ccd))

                # 2. Apply Master Dark
                if isinstance(mdark, str):
                    mdark = self.convert_fits_to_ccd(mdark, single=True)

                if ccd_is_cropped:
                    mdark = self.crop_image(img_ccd=mdark, hdr=ccd_hdr,
                                            crop_section_unbinned=ccd_hdr[crop_key])

                ccd, mdark = check_shape(ccd, mdark)

                # Only ever downscale (scale=True if not an exact match)
                ccd = ccdproc.subtract_dark(ccd, mdark, exposure_time='exptime',
                                            exposure_unit=u.Unit("second"),
                                            scale=not is_nearest)

                add_key_to_hdr(ccd.header, 'MDARK', get_filename(mdark))
                dark_corrected = True

                if not dark_is_pure and is_nearest:
                    # An impure dark still carries the bias pedestal, so subtracting it at a
                    # matching exposure also removes the science bias. Record that for provenance
                    # (no separate master bias / MBIAS keyword was applied).
                    bias_corrected = True

            # 3. Fallback: Standard Bias Subtraction if no dark was applied
            if not dark_corrected and mbias_ccd is not None and self._config['BIAS_CORRECT']:
                if isinstance(mbias_ccd, str):
                    mbias_ccd = self.convert_fits_to_ccd(mbias_ccd, single=True)

                temp_mbias = mbias_ccd
                if ccd_is_cropped:
                    temp_mbias = self.crop_image(img_ccd=temp_mbias, hdr=ccd_hdr,
                                                 crop_section_unbinned=ccd_hdr[crop_key])

                ccd, temp_mbias = check_shape(ccd, temp_mbias)
                ccd = ccdproc.subtract_bias(ccd, temp_mbias)
                bias_corrected = True
                add_key_to_hdr(ccd.header, 'MBIAS', get_filename(mbias_ccd))

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
                # Check shape
                ccd, mflat = check_shape(ccd, mflat)
                # Flat correction
                ccd = ccdproc.flat_correct(ccd, mflat)
                # Add master flat file name to header
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

            # Cosmic ray correction
            if cosmic and gain is not None and self._config['CORRECT_GAIN']:
                ccd = self.clean_cosmic_ray(ccd, ccd_mask_fname,
                                            mbox=mbox, rbox=rbox, gbox=gbox, sigclip=sigclip,
                                            cleantype=cleantype, cosmic_method=cosmic_method)

            # Update header information and save the reduced file
            fbase = Path(filename).stem
            suffix = Path(filename).suffix
            filename = f'{fbase}_red{suffix}'
            filename = join_path(filename, self._red_path)

            ccd.header['FILENAME'] = os.path.basename(filename)
            ccd.header['bias_cor'] = 'T' if bias_corrected else 'F'
            ccd.header['dark_cor'] = 'T' if dark_corrected else 'F'
            ccd.header['flat_cor'] = 'T' if flat_corrected else 'F'
            ccd.header['combined'] = True
            ccd.header['CGAIN'] = True if gain_corrected else False
            ccd.header = ammend_hdr(ccd.header)

            # Change to float32 to keep the file size under control
            ccd.data = ccd.data.astype('float32')

            ccd.uncertainty = None  # Remove uncertainty to save space
            hdr = ccd.header  # Get header for updating the observation data table
            ccd.write(filename, overwrite=True)  # Save the reduced file

            # Make sure that the RA and DEC are consistent and rounded to the same decimal
            # Reformat ra and dec if the values are in degree to avoid float <-> str conversion errors
            if obsparams['radec_separator'] == 'XXX':
                hdr[obsparams['ra']] = round(hdr[obsparams['ra']],
                                             bc.ROUND_DECIMAL)
                hdr[obsparams['dec']] = round(hdr[obsparams['dec']],
                                              bc.ROUND_DECIMAL)

            # Update the result table
            hdr_dict = {}
            for k in obsparams['obs_keywords']:
                hdr_dict.setdefault(k.upper(), hdr.get(k))
            hdr_dict['bias_cor'] = 'T' if bias_corrected else 'F'
            hdr_dict['dark_cor'] = 'T' if dark_corrected else 'F'
            hdr_dict['flat_cor'] = 'T' if flat_corrected else 'F'
            hdr_dict['gain_cor'] = 'T' if gain_corrected else 'F'

            self._obsTable.update_obs_table(fbase, hdr_dict, obsparams)

            del ccd
        if not self.silent and not self.verbose and n_files:
            # Add a new line after the progress bar
            sys.stdout.write('\n')
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
            return ccd

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
                           gain=None, method='average', readnoise=None,
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
            Detector gain in electrons per ADU.
            Defaults to None.
        method : {'average', 'median'}, optional
            Method to combine the individual flat frames: 'average' computes the mean,
            while 'median' computes the median. Defaults to 'average'.
        readnoise : float or None, optional
            Detector readout noise in electrons.
            Defaults to None.
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
            Combined master flat field frame as a CCDData object.

        """
        # Check inputs
        if gain is not None and not isinstance(gain, u.Quantity):
            gain = gain * u.Unit("electron") / u.Unit("adu")
        if readnoise is not None and not isinstance(readnoise, u.Quantity):
            readnoise = readnoise * u.Unit("electron")

        # Prepare file filter
        dfilter = {'imagetyp': bc.IMAGETYP_FLAT,
                   'binning': self._bin_str} if dfilter is None else dfilter
        if dfilter is not None and key_filter is not None and flat_filter is not None:
            dfilter = add_keys_to_dict(dfilter, {key_filter: flat_filter})
        # Filter files
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
            ccd_expt = float(ccd.header['exptime'])

            # Trim image and update header
            ccd = self.process_ccd(img_ccd=ccd,
                                   obsparams=obsparams,
                                   multiple_amps=multiple_amps)

            crop_key = obsparams['cropsec']
            if crop_key is not None and not multiple_amps:
                ccd = self.crop_image(img_ccd=ccd,
                                      hdr=ccd.header,
                                      crop_section_unbinned=ccd.header[crop_key])

            # Gain correct
            if gain is not None and self._config['CORRECT_GAIN']:
                ccd = ccdproc.gain_correct(ccd, gain)

            dark_corrected = False
            if mdark is not None and self._config['FLATDARK_CORRECT'] and ccd_expt in mdark:
                mdark_ccd, dark_is_pure, is_nearest = mdark[ccd_expt]

                # 1. If Dark is "Pure" (Bias-subtracted), we MUST bias-subtract the Flat now
                if dark_is_pure and self._config['BIAS_CORRECT']:
                    if isinstance(mbias, str):
                        mbias = self.convert_fits_to_ccd(mbias, single=True)

                    temp_mbias = mbias
                    if crop_key is not None and not multiple_amps:
                        temp_mbias = self.crop_image(img_ccd=temp_mbias,
                                                     hdr=ccd.header,
                                                     crop_section_unbinned=ccd.header[crop_key])

                    if mbias_ccd is None:
                        mbias_ccd = temp_mbias

                    ccd, temp_mbias = check_shape(ccd, temp_mbias)
                    ccd = ccdproc.subtract_bias(ccd, temp_mbias)

                # 2. Prepare and apply Dark
                if isinstance(mdark_ccd, str):
                    mdark_ccd = self.convert_fits_to_ccd(mdark_ccd, single=True)

                if crop_key is not None and not multiple_amps:
                    mdark_ccd = self.crop_image(img_ccd=mdark_ccd,
                                                hdr=ccd.header,
                                                crop_section_unbinned=ccd.header[crop_key])

                scale_required = not is_nearest
                ccd, mdark_ccd = check_shape(ccd, mdark_ccd)
                ccd = ccdproc.subtract_dark(ccd, mdark_ccd,
                                            exposure_time='exptime',
                                            exposure_unit=u.Unit("second"),
                                            scale=scale_required)

                add_key_to_hdr(ccd.header, 'MDARK', get_filename(mdark_ccd))
                dark_corrected = True

            # 3. Fallback: Bias subtraction only if no dark was applied
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
                add_key_to_hdr(ccd.header, 'MBIAS', get_filename(mbias_ccd))

            # Append the result to a file list for combining
            lflat.append(ccd)

            del ccd

        if not self.silent:
            self._log.info(f'    Creating master flat file: {os.path.basename(mflat_fname)}')

        # Combine flat ccds
        combine = ccdproc.combine(lflat, method=method,
                                  mem_limit=self._config['MEM_LIMIT_COMBINE'],
                                  scale=inv_median,
                                  minmax_clip=False,
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

        # Update the FITS header
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
                           gain=None, method='average', readnoise=None,
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
            Detector gain in electrons per ADU.
            Defaults to None.
        method : {'average', 'median'}, optional
            Method used to combine the individual dark frames: 'average' computes the mean,
            while 'median' computes the median. Defaults to 'average'.
        readnoise : float or None, optional
            Detector readout noise in electrons.
            Defaults to None.
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
            Combined master dark frame as a CCDData object.

        """
        if not self.silent:
            self._log.info(f'    Creating master dark file: {os.path.basename(mdark_fname)}')

        # Check inputs
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
                ccd, mbias = check_shape(ccd, mbias)
                ccd = ccdproc.subtract_bias(ccd, mbias)

            # Append the result to a file list for combining
            ldark.append(ccd)
            del ccd

        # Combine dark ccds
        combine = ccdproc.combine(ldark, method=method,
                                  mem_limit=self._config['MEM_LIMIT_COMBINE'],
                                  sigma_clip=True,
                                  sigma_clip_low_thresh=3,
                                  sigma_clip_high_thresh=3,
                                  sigma_clip_func=np.ma.median,
                                  sigma_clip_dev_func=mad_std,
                                  dtype='float32')

        # Update fits header
        if gain is not None and 'GAIN' not in combine.header:
            combine.header.set('GAIN', gain.value, gain.unit)
        if readnoise is not None and 'RON' not in combine.header:
            combine.header.set('RON', readnoise.value, readnoise.unit)
        combine.header['combined'] = True
        combine.header['BIASCORR'] = bool(mbias is not None and self._config['BIAS_CORRECT'])
        combine.header['CGAIN'] = bool(gain is not None and self._config['CORRECT_GAIN'])
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
                           gain=None, method='average', readnoise=None,
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
            will handle each amplifier’s data separately. Defaults to False.
        gain : float or None, optional
            Detector gain in electrons per ADU. If provided, used to scale the bias
            frames accordingly. Defaults to None.
        method : {'average', 'median'}, optional
            Method used to combine the individual bias frames. Options are 'average'
            to compute the mean, or 'median' to compute the median. Defaults to 'average'.
        readnoise : float or None, optional
            Detector readout noise in electrons. If None, readout noise is estimated
            from the standard deviation of a bias frame.
            Defaults to None.
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
            Combined master bias frame as a CCDData object.

        """
        if not self.silent:
            self._log.info(f'    Creating master bias file: {os.path.basename(mbias_fname)}')

        # Check inputs
        if gain is not None and not isinstance(gain, u.Quantity):
            gain = gain * u.Unit("electron") / u.Unit("adu")
        if readnoise is not None and not isinstance(readnoise, u.Quantity):
            readnoise = readnoise * u.Unit("electron")

        if dfilter is None:
            dfilter = {'imagetyp': bc.IMAGETYP_BIAS,
                       'binning': self._bin_str}
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
                                  sigma_clip_func=np.ma.median,
                                  sigma_clip_dev_func=mad_std,
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
                self._log.info(f'    Creating bad pixel map file: {os.path.basename(ccd_mask_fname)}')

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
                                                 median=True, overscan_axis=None)
                nccd.header['oscan_cor'] = True

            trim_key = obsparams['trimsec']
            if trim_key is not None:
                trim_section = hdr[trim_key]
                trim_slice = self.parse_section(trim_section)
                nccd = ccdproc.trim_image(nccd[trim_slice])
                nccd.header['trimmed'] = True
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
            nccd = ccdproc.trim_image(nccd[crop_slice_y, crop_slice_X])
            nccd.header['cropped'] = True
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
                self._log.warning(f'File "{os.path.basename(fn)}" NOT found')
                continue
            if key_unit is not None and key_unit in hdr:
                try:
                    fits_unit = u.Unit(hdr[key_unit])
                except (Exception,):
                    pass
            if fits_unit is None:
                if key_unit is not None:
                    raise RuntimeError(
                        'Units NOT found in header ("%s") of image "%s". '
                        'Specify one in "unit" variable' % (
                            key_unit, os.path.basename(fn)))
                else:
                    self._log.debug('"key_unit" not specified')
            # Silence astropy's INFO chatter during the read, then restore the caller's level
            prev_level = self._log.level
            self._log.setLevel(max(prev_level, logging.WARNING))
            try:
                ccd = CCDData.read(fn)
            finally:
                self._log.setLevel(prev_level)
            if key_file is not None and key_file not in ccd.header:
                ccd.header[key_file] = os.path.basename(fn)
            lccd.append(ccd)
            del ccd
        if len(lccd) == 0:
            self._log.warning('No files found!')
            return []
        if single and len(lccd) == 1:
            lccd = lccd[0]
        return lccd

    def _check_calibration_files_exist(self, search_path, suffix, obsparams, binnings, filters, inst=None):
        """
        Check if calibration files exist in the specified path.

        Parameters
        ----------
        search_path : Path
            Path to search for calibration files.
        suffix : str
            File suffix (e.g., '.fits', '.fit').
        obsparams : dict
            Dictionary with telescope configuration.
        binnings : tuple
            A tuple containing the binning factor in x, and y direction.
        filters : list
            The list of filters used in the data set.
        inst : str, optional
            Instrument identifier for DDOTI telescope.

        Returns
        -------
        bool
            True if calibration files (bias, dark, or flat) are found, False otherwise.

        """
        # Find all FITS files in the search path
        fits_files = list(Path(search_path).rglob(f"*{suffix}"))
        fits_files = [f for f in fits_files if '/atmp/' not in str(f)
                      and 'reduced' not in str(f)
                      and 'master_calibs' not in str(f)
                      and not f.name.startswith('.')]

        if not fits_files:
            return False

        try:
            # Create an image file collection to check for calibration types
            ic = ImageFileCollection(filenames=fits_files)

            if ic.summary is None or len(ic.summary) == 0:
                return False

            # Check for bias files
            if self._config.get('BIAS_CORRECT', False):
                dfilter, _, regexp = self.create_dfilter(inst, obsparams, 'bias')
                bias_files = self.get_file_list(ic, dfilter, key_find='find', regexp=regexp,
                                                invert_find=False, dkeys=None)
                if bias_files is not None and len(bias_files) > 0:
                    return True

            # Check for dark files
            if self._config.get('DARK_CORRECT', False):
                dfilter, _, regexp = self.create_dfilter(inst, obsparams, 'dark')
                dark_files = self.get_file_list(ic, dfilter, key_find='find', regexp=regexp,
                                                invert_find=False, dkeys=None)
                if dark_files is not None and len(dark_files) > 0:
                    return True

            # Check for flat files
            if self._config.get('FLAT_CORRECT', False):
                dfilter, _, regexp = self.create_dfilter(inst, obsparams, 'flat')
                for filt in filters:
                    dfilter[obsparams['filter'].lower()] = filt[1]
                    flat_files = self.get_file_list(ic, dfilter, key_find='find', regexp=regexp,
                                                    invert_find=False, dkeys=None)
                    if flat_files is not None and len(flat_files) > 0:
                        return True

            del ic
            return False

        except Exception as e:
            self._log.debug(f"{' ':<4}> Error checking calibration files in {search_path}: {e}")
            return False

    @staticmethod
    def _parse_obs_date_from_text(text):
        """Parse YYYY-MM-DD or YYYYMMDD date tokens from text."""
        if text is None:
            return None

        token_match = re.search(r'(\d{4}-\d{2}-\d{2}|\d{8})', str(text))
        if token_match is None:
            return None

        token = token_match.group(1)
        fmt = '%Y-%m-%d' if '-' in token else '%Y%m%d'
        try:
            return datetime.strptime(token, fmt).date()
        except ValueError:
            return None

    @staticmethod
    def _build_master_bias_path(out_path, suffix, binning_ext, telescope, instrument=None):
        master_bias = f"master_bias_{binning_ext}{suffix}"
        if telescope == 'DDOTI 28-cm f/2.2':
            master_bias = f"master_bias_{instrument}_{binning_ext}{suffix}"
        return Path(join_path(master_bias, out_path))

    @staticmethod
    def _build_master_dark_glob(suffix, binning_ext, telescope, instrument=None):
        if telescope == 'DDOTI 28-cm f/2.2':
            return f"master_dark_{instrument}_{binning_ext}_*{suffix}"
        return f"master_dark_{binning_ext}_*{suffix}"

    @staticmethod
    def _build_master_flat_path(out_path, suffix, filter_name, binning_ext, telescope, instrument=None):
        master_flat = f"master_flat_{filter_name}_{binning_ext}{suffix}"
        if telescope == 'DDOTI 28-cm f/2.2':
            master_flat = f"master_flat_{instrument}_{filter_name}_{binning_ext}{suffix}"
        return Path(join_path(master_flat, out_path))

    @staticmethod
    def _build_master_dark_path(out_path, suffix, dark_exposure, binning_ext, telescope, instrument=None):
        exposure_token = str(dark_exposure).replace('.', 'p')
        master_dark = f"master_dark_{binning_ext}_{exposure_token}{suffix}"
        if telescope == 'DDOTI 28-cm f/2.2':
            master_dark = f"master_dark_{instrument}_{binning_ext}_{exposure_token}{suffix}"
        return Path(join_path(master_dark, out_path))

    def find_and_create_calibrations(self, telescope, obsparams,
                                     filters, binnings, obs_date=None):
        """ Find calibration files and create master calibration files if required.

        This method is used to search for the bias, dark and flat files, if they are required,
        copies them to the tmp reduction folder and applies the appropriate reduction method.
        It first looks for calibration files in the date folder of the science data,
        if it finds them it uses those, otherwise it expands the search to a date range around the observation night.
        If no calibration files are found in the date range, raise an error.

        Parameters
        ----------
        telescope : str
            Telescope identifier.
        obsparams : dict
            Dictionary with telescope configuration.
        filters : list
            The list of filters used in the data set.
        binnings : tuple
            A tuple containing the binning factor in x, and y direction.
        obs_date : datetime.date or None
            Observation date derived from science headers, used as fallback if path-based date
            resolution fails.
        """

        sci_root_path = Path(self._sci_root_path).resolve()
        suffix = self._suffix
        sci_bin_str = self._bin_str if self._bin_str else f'{binnings[0]}x{binnings[1]}'

        inst = None
        if self._telescope == 'DDOTI 28-cm f/2.2':
            inst = self._instrument

        if not self.silent:
            self._log.info("> Prepare calibrations")

        # Resolve date
        self._log.debug(f"{' ':<2}> Resolve science dataset context")
        try:
            resolved_date, date_route, same_night_path, parent_search_root = self._resolve_date_context(
                sci_root_path,
                obs_date=obs_date,
            )
        except Exception as exc:
            return self._make_failure_result(
                status='dataset_failed_validation_error',
                reason=f'Observation date resolution failed: {exc}',
                dataset_path=sci_root_path,
                binning=sci_bin_str,
            )

        # Derive dataset date root and master output directory
        dataset_date_root, date_root_route = self._derive_dataset_date_root(sci_root_path, same_night_path)
        master_out = dataset_date_root / 'master_calibs'
        master_out.mkdir(exist_ok=True, parents=True)

        try:
            requirement = self._build_science_calibration_requirement(obsparams=obsparams,
                                                                      filters=filters,
                                                                      obs_date=resolved_date,
                                                                      master_out=master_out)
        except Exception as exc:
            return self._make_failure_result(
                status='dataset_failed_validation_error',
                reason=f'Science calibration requirement build failed: {exc}',
                dataset_path=sci_root_path,
                binning=sci_bin_str,
                master_output=master_out,
            )

        # Initialize calibration state from user preferences
        calib_state = self._build_calib_state_template(requirement)
        prefs = requirement['user_preferences']
        self._log.debug(f"{' ':<4}User correction preferences: " + ', '.join(
            f"{flag_name}={prefs.get(flag_name)}"
            for flag_name in ['BIAS_CORRECT', 'FLAT_CORRECT', 'DARK_CORRECT',
                              'FLATDARK_CORRECT', 'ALLOW_DARK_SCALING', 'DARK_TOLERANCE']))
        self._log.debug(f"{' ':<4}Date resolution route : {date_route}")
        self._log.debug(f"{' ':<4}Dataset date root     : {dataset_date_root}  [{date_root_route}]")
        self._log.debug(f"{' ':<4}Date-resolved path    : {same_night_path}")
        self._log.debug(f"{' ':<4}Master output root    : {master_out}")
        self._log.debug(f"{' ':<4}Science exposures     : {requirement['science_exposure_times']}")
        self._log.debug(f"{' ':<4}Science filters       : {[f[0] for f in requirement['filters']]}")

        # Probe same-night candidate
        selected_probe = None  # type: Optional[dict]
        fallback_deltas = []
        same_night_candidate = Path(dataset_date_root)

        self._log.debug(f"{' ':<2}> Evaluate same-night calibration candidate")

        same_night_probe = self._probe_candidate(search_path=same_night_candidate,
                                                 resolved_date=resolved_date,
                                                 master_out=master_out,
                                                 requirement=requirement,
                                                 obsparams=obsparams,
                                                 filters=requirement['filters'],
                                                 binnings=binnings,
                                                 suffix=suffix,
                                                 inst=inst,
                                                 sci_bin_str=sci_bin_str)
        probe_log = [same_night_probe]

        if same_night_probe['complete']:
            selected_probe = same_night_probe
            self._log.debug(f"{' ':<4}Fallback search       : skipped; same-night candidate complete")
        else:
            if not self.silent:
                self._log.info(f"{' ':<4}Same-night calibrations incomplete; searching "
                               f"±{self._config['TIMEDELTA_DAYS']} days under {parent_search_root}")

            fallback_candidates, fallback_deltas = self._collect_fallback_candidates(parent_search_root, resolved_date)
            fallback_candidates = [
                p for p in fallback_candidates
                if p.resolve() != Path(dataset_date_root).resolve()
            ]

            self._log.debug(f"{' ':<4}Fallback candidates   : {len(fallback_candidates)}")
            self._log.debug(f"{' ':<4}Fallback source order : {[str(p) for p in fallback_candidates]}")

            for i, cpath in enumerate(fallback_candidates, start=1):
                self._log.debug(f"{' ':<2}> Evaluate fallback candidate {i}/{len(fallback_candidates)}")

                probe = self._probe_candidate(search_path=cpath,
                                              resolved_date=resolved_date,
                                              master_out=master_out,
                                              requirement=requirement,
                                              obsparams=obsparams,
                                              filters=requirement['filters'],
                                              binnings=binnings,
                                              suffix=suffix,
                                              inst=inst,
                                              sci_bin_str=sci_bin_str)
                probe_log.append(probe)
                if probe['complete']:
                    selected_probe = probe
                    break
                self._log.debug(f"{' ':<4}Candidate result      : incomplete; trying next fallback candidate")

        if selected_probe is None:
            unresolved_union = sorted({item for p in probe_log for item in p['unresolved']})
            searched = [p['search_path'] for p in probe_log]
            fallback_range = [int(d) for d in fallback_deltas]
            msg = (
                f"Calibration preparation failed for binning {sci_bin_str}. "
                f"Unresolved calibration types: {unresolved_union}. "
                f"Searched candidate folders: {[str(p) for p in searched]}. "
                f"Date resolution route: {date_route}. "
                f"Fallback day offsets used: {fallback_range}. "
                f"Master output root: {master_out}."
            )
            failure_items = []
            for token in unresolved_union:
                product = 'flat' if token.startswith(('flat:', 'flat-baseline:')) else token
                failure_items.append({'product': product, 'reason': token})
            if any(str(item['reason']).startswith(('dark-baseline:', 'flat-baseline:')) for item in failure_items):
                status = 'dataset_failed_no_bias_or_dark'
            elif any(i['product'] == 'flat' for i in failure_items):
                status = 'dataset_failed_missing_requested_flat'
            else:
                status = 'dataset_failed_missing_requested_bias'
            return self._make_failure_result(
                status=status,
                reason=msg,
                dataset_path=sci_root_path,
                binning=sci_bin_str,
                master_output=master_out,
                failed_products=failure_items,
                searched_sources=searched,
            )

        # Select calibration source
        selected_search_path = Path(selected_probe['search_path']).resolve()
        selected_stage_plan = selected_probe.get('stage_plan', {'bias': [], 'dark': [], 'flat': {}})
        selected_adoption_plan = selected_probe.get('adoption_plan', {'bias': None, 'dark': {}, 'flat': {}})
        adoption_needed = bool(
            selected_adoption_plan.get('bias')
            or selected_adoption_plan.get('dark')
            or selected_adoption_plan.get('flat')
        )
        raw_staging_needed = bool(
            selected_stage_plan.get('bias')
            or selected_stage_plan.get('dark')
            or any(selected_stage_plan.get('flat', {}).values())
        )
        is_same_night_candidate = selected_search_path == Path(dataset_date_root).resolve()
        if is_same_night_candidate:
            candidate_role = 'same-night candidate'
        elif adoption_needed and raw_staging_needed:
            candidate_role = 'fallback mixed master adoption/raw creation'
        elif adoption_needed:
            candidate_role = 'fallback master adoption'
        elif raw_staging_needed:
            candidate_role = 'fallback raw creation'
        else:
            candidate_role = 'fallback candidate already satisfied'
        self._log.debug(f"{' ':<2}> Select calibration source")
        self._log.debug(f"{' ':<4}Selected candidate root : {selected_probe['search_path']}")
        self._log.debug(f"{' ':<4}Candidate role          : {candidate_role}")
        self._master_calib_path = master_out
        self._selected_calibration_source_folder = str(selected_search_path)

        # Merge winning probe's calib_state_preview into the live
        # calib_state — the only place it is set from the probe
        preview = selected_probe['calib_state_preview']
        calib_state['bias'].update(preview['bias'])
        for filt_name, filt_data in preview['flat'].items():
            if filt_name in calib_state['flat']:
                calib_state['flat'][filt_name].update(filt_data)
            else:
                calib_state['flat'][filt_name] = filt_data
        # Dark entries: replace entirely (probe has the authoritative availability result)
        calib_state['dark']['science'].update(preview['dark']['science'])
        calib_state['dark']['flat'].update(preview['dark']['flat'])
        self._log.debug(f"{' ':<4}Calibration state after selection — "
                        f"bias.available={calib_state['bias']['available']}, "
                        f"flat={[(f, v['available']) for f, v in calib_state['flat'].items()]}, "
                        f"sci_darks={[(k, v['available']) for k, v in calib_state['dark']['science'].items()]}")

        # Resolve master calibration filenames
        dbias = selected_probe['master_bias_info']
        ddark = selected_probe['master_dark_info']
        dflat = selected_probe['master_flat_info']
        dark_exptimes = sorted(ddark.keys())
        self._master_bias_info = dbias
        self._master_dark_info = ddark
        self._master_flat_info = dflat
        self._dark_exptimes = dark_exptimes
        def _format_target(value):
            path, create = value
            action = 'create' if create else 'load/use'
            return f'{Path(path).name} [{action}]'

        if dbias:
            bias_summary = _format_target(dbias)
        elif requirement['user_preferences']['BIAS_CORRECT']:
            bias_summary = 'unavailable (science darks used)'
        else:
            bias_summary = 'not requested'
        dark_summary = ', '.join(
            f'{float(exp):g}s={_format_target(info)}'
            for exp, info in sorted(ddark.items())
        ) if ddark else 'none'
        flat_summary = ', '.join(
            f'{filt}={_format_target(info)}'
            for filt, info in sorted(dflat.items())
        ) if dflat else 'none'
        self._log.debug(f"{' ':<4}Bias targets : {bias_summary}")
        self._log.debug(f"{' ':<4}Dark targets : {dark_summary}")
        self._log.debug(f"{' ':<4}Flat targets : {flat_summary}")

        # Adopt fallback masters before raw staging
        if adoption_needed:
            self._log.debug(f"{' ':<2}> Apply fallback-master adoption plan")
            self._adopt_fallback_master_products(adoption_plan=selected_adoption_plan,
                                                 requirement=requirement,
                                                 calib_state=calib_state)

        # Stage and prepare raw calibration files
        self._log.debug(f"{' ':<2}> Stage raw calibration files if needed")
        staged_seen = set()
        # Use calib_state (not self._config) to decide what needs staging
        if calib_state['bias']['requested']:
            if not dbias:
                self._log.debug(f"{' ':<4}Bias raw staging skipped: no bias available")
            elif dbias[1]:
                self._stage_and_prepare(selected_stage_plan.get('bias', []), 'bias', staged_seen,
                                        telescope, obsparams)
            else:
                self._log.debug(f"{' ':<4}Bias raw staging skipped: master already available")

        if requirement['user_preferences']['DARK_CORRECT']:
            create_dark = np.any(np.array([v[1] for _, v in ddark.items()])) if ddark else False
            if create_dark:
                self._stage_and_prepare(selected_stage_plan.get('dark', []), 'dark', staged_seen,
                                        telescope, obsparams)
            else:
                self._log.debug(f"{' ':<4}Dark raw staging skipped: masters already available")

        for filt in filters:
            filt_name = filt[0]
            if not calib_state['flat'].get(filt_name, {}).get('requested', False):
                continue
            if filt_name in dflat and dflat[filt_name][1]:
                flat_stage_map = selected_stage_plan.get('flat', {})
                self._stage_and_prepare(flat_stage_map.get(filt_name, []), 'flat', staged_seen,
                                        telescope, obsparams, label=f'flat ({filt_name})')
            else:
                self._log.debug(f"{' ':<4}Flat raw staging skipped for {filt_name}: master already available")

        # Create or load master products
        self._log.debug(f"{' ':<2}> Build master calibration files")
        try:
            master_darks = self._build_master_products(requirement=requirement,
                                                       calib_state=calib_state,
                                                       obsparams=obsparams,
                                                       filters=requirement['filters'])
        except MasterBuildError:
            raise
        except Exception as exc:
            raise MasterBuildError(str(exc)) from exc

        # Validate master products
        self._log.debug(f"{' ':<2}> Validate master calibration outputs")
        fatal_missing, optional_missing = self._validate_master_products(requirement=requirement,
                                                                         calib_state=calib_state,
                                                                         master_out=master_out)
        if fatal_missing:
            searched = [p['search_path'] for p in probe_log]
            reason = (
                f"Calibration validation failed. Missing hard products: {fatal_missing}. "
                f"Optional unavailable: {optional_missing}. "
                f"Master output root: {master_out}. "
                f"Date resolution route: {date_route}. "
                f"Fallback offsets: {[int(d) for d in fallback_deltas]}. "
                f"Searched folders: {[str(p) for p in searched]}."
            )
            failed_products = []
            for item in fatal_missing:
                product = 'flat' if str(item).startswith('flat:') else str(item).split(':', 1)[0]
                failed_products.append({'product': product, 'reason': str(item)})
            return self._make_failure_result(
                status='dataset_failed_validation_error',
                reason=reason,
                dataset_path=sci_root_path,
                binning=sci_bin_str,
                selected_source=selected_search_path,
                master_output=master_out,
                failed_products=failed_products,
                searched_sources=searched,
            )

        # Controlled gate: correction baseline unavailable (no usable bias + no science master dark)
        if requirement['user_preferences']['DARK_CORRECT']:
            bias_available = bool(calib_state['bias'].get('available'))
            dark_available = bool(master_darks and master_darks.get('sci_dark'))
            if not bias_available and not dark_available:
                return self._make_failure_result(
                    status='dataset_failed_no_bias_or_dark',
                    reason='No usable science bias or science dark available.',
                    dataset_path=sci_root_path,
                    binning=sci_bin_str,
                    selected_source=selected_search_path,
                    master_output=master_out,
                    searched_sources=[p['search_path'] for p in probe_log],
                )

        self._obsparams = obsparams

        if not self.silent:
            self._log_calibration_summary(requirement=requirement,
                                          calib_state=calib_state,
                                          source=None if is_same_night_candidate else selected_search_path)

        return CalibrationPrepResult(
            ok=True,
            dataset_failed=False,
            status='ok',
            dataset_path=sci_root_path,
            binning=sci_bin_str,
            selected_source=selected_search_path,
            master_output=master_out,
            searched_sources=[Path(p['search_path']).resolve() for p in probe_log],
            calib_state=calib_state,
            master_darks=master_darks,
        )

    def _log_calibration_summary(self, requirement, calib_state, source=None):
        """Log a compact per-binning overview of the calibrations used for science reduction.

        ``source`` is the fallback folder the calibrations came from, or None for same-night.
        """
        status_words = {
            'loaded_existing_science_date_master': 'loaded',
            'adopted_from_fallback_master': 'adopted',
            'overwritten_by_adopted_fallback_master': 'adopted',
            'created_from_raw': 'created',
            'overwritten_from_raw': 're-created',
        }
        prefs = requirement['user_preferences']

        def _darks_line(entries):
            parts = []
            for target, info in sorted(entries.items()):
                if not info.get('available'):
                    parts.append(f"{target:g}s missing" + (' (bias only)' if info.get('fallback') == 'bias' else ''))
                    continue
                text = f"{target:g}s {status_words.get(info.get('master_status'), 'loaded')}"
                actual = info.get('actual_exposure')
                if not info.get('is_exact') and actual is not None and float(actual) != float(target):
                    text += f" (scaled from {float(actual):g}s)"
                parts.append(text)
            return ', '.join(parts) if parts else 'none'

        bias = calib_state['bias']
        if not prefs['BIAS_CORRECT']:
            bias_line = 'off'
        elif bias.get('master_status') == 'skipped_unavailable_dark_baseline':
            bias_line = 'none (science darks used instead)'
        else:
            bias_name = Path(bias['master_path']).name if bias.get('master_path') else ''
            bias_line = f"{status_words.get(bias.get('master_status'), 'loaded'):<10} {bias_name}".rstrip()

        dark_line = _darks_line(calib_state['dark']['science']) if prefs['DARK_CORRECT'] else 'off'

        if prefs['FLAT_CORRECT']:
            flat_line = ', '.join(
                f"{filt} {status_words.get(info.get('master_status'), 'loaded')}"
                for filt, info in sorted(calib_state['flat'].items())
            ) or 'none'
        else:
            flat_line = 'off'

        source_text = 'same-night' if source is None else f'fallback {Path(source).name}'
        self._log.info(f"> Calibrations (binning {self._bin_str}) — source: {source_text}")
        self._log.info(f"{' ':<4}Bias      : {bias_line}")
        self._log.info(f"{' ':<4}Darks     : {dark_line}")
        if prefs['FLATDARK_CORRECT']:
            self._log.info(f"{' ':<4}Flat-darks: {_darks_line(calib_state['dark']['flat'])}")
        self._log.info(f"{' ':<4}Flats     : {flat_line}")

    def _list_fits(self, search_path, suffix):
        fits_files = list(Path(search_path).rglob(f"*{suffix}"))
        result = [f for f in fits_files if '/atmp/' not in str(f)
                  and 'reduced' not in str(f)
                  and 'master_calibs' not in str(f)
                  and not f.name.startswith('.')]
        self._log.debug(f"{' ':<4}[list_fits] {search_path}: {len(result)} FITS file(s) found")
        return result

    def _resolve_date_context(self, sci_root_path, obs_date=None):
        sci_root_path = Path(sci_root_path).resolve()
        self._log.debug(f"{' ':<4}[date_ctx] Trying folder-name date from: {sci_root_path.name}")
        folder_date = self._parse_obs_date_from_text(sci_root_path.name)
        if folder_date is not None:
            self._log.debug(f"{' ':<4}[date_ctx] Resolved via folder name: {folder_date}")
            return folder_date, 'science_root_folder_name', sci_root_path, sci_root_path.parent

        for idx in range(len(sci_root_path.parts) - 1, -1, -1):
            part = sci_root_path.parts[idx]
            route_date = self._parse_obs_date_from_text(part)
            if route_date is not None:
                date_folder = Path(*sci_root_path.parts[:idx + 1])
                self._log.debug(
                    f"{' ':<4}[date_ctx] Resolved via path segment: {route_date}, date folder: {date_folder}")
                return route_date, 'nearest_date_segment_in_science_path', date_folder, date_folder.parent

        if obs_date is not None:
            candidate = self._find_date_folder_candidate(sci_root_path.parent, obs_date)
            if candidate is not None:
                self._log.debug(f"{' ':<4}[date_ctx] Using obs_date folder fallback: {candidate}")
                return obs_date, 'science_header_date_folder_search', candidate, candidate.parent
            self._log.debug(f"{' ':<4}[date_ctx] Using obs_date fallback: {obs_date}")
            return obs_date, 'configured_date_range_fallback', sci_root_path, sci_root_path.parent

        raise RuntimeError(
            'Could not resolve observation date from science path and no obs_date fallback was provided.')

    @staticmethod
    def _looks_like_date_root(folder_path):
        folder_path = Path(folder_path)
        try:
            child_dirs = [d for d in folder_path.iterdir() if d.is_dir()]
        except (OSError, PermissionError):
            return False

        child_names = {d.name.lower() for d in child_dirs}
        calib_markers = {'bias', 'biases', 'dark', 'darks', 'flat', 'flats', 'sci', 'master_calibs'}
        if child_names.intersection(calib_markers):
            return True

        # DATE/TARGET/raw layout support.
        for child in child_dirs:
            try:
                if (child / 'raw').is_dir():
                    return True
            except (OSError, PermissionError):
                continue

        return False

    @staticmethod
    def _find_date_folder_candidate(search_root, target_date):
        search_root = Path(search_root).resolve()
        # Date-root-only policy: do not probe arbitrary nested folders.
        # Search the current root and direct child date roots while walking up ancestors.
        for anchor in [search_root, *search_root.parents]:
            if ReduceSatObs._parse_obs_date_from_text(anchor.name) == target_date \
                    and ReduceSatObs._looks_like_date_root(anchor):
                return anchor.resolve()

            try:
                child_dirs = [d for d in anchor.iterdir() if d.is_dir()]
            except (OSError, PermissionError):
                continue

            direct_date_roots = [d.resolve() for d in child_dirs
                                 if ReduceSatObs._parse_obs_date_from_text(d.name) == target_date
                                 and ReduceSatObs._looks_like_date_root(d)]
            if direct_date_roots:
                return sorted(direct_date_roots, key=str)[0]

        return None

    def _collect_fallback_candidates(self, parent_root, base_date):
        parent_root = Path(parent_root).resolve()
        dt_days = self._config['TIMEDELTA_DAYS']
        deltas = list(range(-1 * dt_days, dt_days + 1))
        deltas = [d for d in sorted(deltas, key=lambda x: (abs(x), x)) if d != 0]

        target_dates = {base_date + timedelta(days=d): d for d in deltas}
        by_date = {d: set() for d in target_dates.keys()}

        direct_child_dirs = [d for d in parent_root.iterdir() if d.is_dir()]
        for folder in direct_child_dirs:
            dval = self._parse_obs_date_from_text(folder.name)
            if dval in by_date:
                by_date[dval].add(str(folder))

        ordered = []
        for delta in deltas:
            date_key = base_date + timedelta(days=delta)
            for item in sorted(by_date.get(date_key, set())):
                ordered.append(Path(item).resolve())

        self._log.debug(f"{' ':<4}[fallback] {len(ordered)} date-root fallback candidate(s) found under {parent_root}")
        return ordered, deltas

    def _build_science_calibration_requirement(self, obsparams, filters, obs_date, master_out):
        """Build requirement model from science data and user config.
        
        This requirement model is created once per binning pass and represents:
        - observation parameters
        - science data characteristics
        - user correction preferences
        - correction policy settings (tolerance, scaling)
        
        Note: user_preferences are NEVER overwritten during probing/build.
        Actual availability is tracked separately in calib_state.
        """
        ic_all = ImageFileCollection(location=self._tmp_path)
        ic_science = ic_all.filter(imagetyp=bc.IMAGETYP_LIGHT, binning=self._bin_str)
        if ic_science.summary is None or len(ic_science.summary) == 0:
            raise RuntimeError(f'No normalized science files found for binning {self._bin_str}.')

        science_filters = []
        for filt in filters:
            n_files = len(ic_all.files_filtered(imagetyp=bc.IMAGETYP_LIGHT,
                                                filter=filt[0],
                                                binning=self._bin_str))
            if n_files > 0:
                science_filters.append(filt)

        science_exptimes = np.unique(ic_science.summary[obsparams['exptime'].lower()]).astype(float).tolist()
        science_files = list(self.get_file_list(ic_all,
                                                {'imagetyp': bc.IMAGETYP_LIGHT,
                                                 'binning': self._bin_str},
                                                abspath=True))
        representative_science_file = self._resolve_representative_science_file(science_files,
                                                                                ic_science)

        science_shape = self._derive_processed_science_shape(representative_science_file, obsparams)

        # Build user-preferences snapshot from config
        user_preferences = {
            'BIAS_CORRECT': bool(self._config.get('BIAS_CORRECT', False)),
            'FLAT_CORRECT': bool(self._config.get('FLAT_CORRECT', False)),
            'DARK_CORRECT': bool(self._config.get('DARK_CORRECT', False)),
            'FLATDARK_CORRECT': bool(self._config.get('FLATDARK_CORRECT', False)),
            'ALLOW_DARK_SCALING': bool(self._config.get('ALLOW_DARK_SCALING', False)),
            'DARK_TOLERANCE': float(self._config.get('DARK_TOLERANCE', 0)),
        }

        return {
            'observation_date': obs_date,
            'instrument': self._instrument,
            'telescope': self._telescope,
            'suffix': self._suffix,
            'binning': tuple(self._binxy),
            'binning_str': self._bin_str,
            'filters': science_filters,
            'science_exposure_times': sorted(science_exptimes),
            'science_image_shape': science_shape,
            'user_preferences': user_preferences,
            'force_reduction': bool(self._force_reduction),
            'master_output_root': Path(master_out).resolve(),
        }

    @staticmethod
    def _build_calib_state_template(requirement):
        """Initialize the calibration state (calib_state).
        
        calib_state tracks what is actually available/created at runtime,
        separate from user correction preferences. This model is updated during
        probing/build but user_preferences are never mutated.
        """
        user_preferences = requirement['user_preferences']
        filters = requirement['filters']

        # Initialize calib_state per user preferences
        calib_state = {
            'bias': {
                'requested': user_preferences['BIAS_CORRECT'],
                'available': False,
                'created': False,
                'master_path': None,
                'blocking': user_preferences['BIAS_CORRECT'],
                'master_status': 'skipped_by_preference' if not user_preferences['BIAS_CORRECT'] else None,
            },
            'flat': {
                filt[0]: {
                    'requested': user_preferences['FLAT_CORRECT'],
                    'available': False,
                    'created': False,
                    'master_path': None,
                    'blocking': user_preferences['FLAT_CORRECT'],
                    'master_status': 'skipped_by_preference' if not user_preferences['FLAT_CORRECT'] else None,
                }
                for filt in filters
            },
            'dark': {
                'science': {
                    # Will be populated: exposure -> {requested, available, created, blocking=False, fallback}
                },
                'flat': {
                    # Will be populated: exposure -> {requested, available, created, blocking=False, fallback}
                },
            },
            'warnings': [],
        }

        # Initialize science dark entries for all science exposure times
        if user_preferences['DARK_CORRECT']:
            for exp in requirement['science_exposure_times']:
                calib_state['dark']['science'][float(exp)] = {
                    'requested': True,
                    'available': False,
                    'created': False,
                    'blocking': False,  # Non-blocking optional product
                    'fallback': 'bias',  # Fallback to bias if missing
                    'actual_exposure': None,
                    'is_exact': False,
                    'master_status': None,
                }
        else:
            # Even if not requested, track for diagnostic purposes
            for exp in requirement['science_exposure_times']:
                calib_state['dark']['science'][float(exp)] = {
                    'requested': False,
                    'available': False,
                    'created': False,
                    'blocking': False,
                    'fallback': None,
                    'actual_exposure': None,
                    'is_exact': False,
                    'master_status': 'skipped_by_preference',
                }

        # Flat-dark entries will be populated during flat probing
        if user_preferences['FLATDARK_CORRECT']:
            # Placeholder for flat-darks (will be populated when flat exposure times are known)
            pass

        return calib_state

    def _derive_processed_science_shape(self, science_file, obsparams):
        ccd = CCDData.read(science_file, unit=u.Unit('adu'))
        ccd = self.process_ccd(img_ccd=ccd,
                               obsparams=obsparams,
                               multiple_amps=obsparams['multiple_amps'])
        crop_key = obsparams['cropsec']
        if crop_key is not None and not obsparams['multiple_amps'] and crop_key in ccd.header:
            ccd = self.crop_image(img_ccd=ccd,
                                  hdr=ccd.header,
                                  crop_section_unbinned=ccd.header[crop_key])
        return tuple(ccd.data.shape)

    def _normalize_ifc_file_entry(self, file_entry):
        if file_entry is None:
            return None

        if isinstance(file_entry, bytes):
            file_entry = file_entry.decode('utf-8', errors='ignore')

        candidate = Path(str(file_entry).strip())
        if candidate.name.startswith("b'") and candidate.name.endswith("'"):
            candidate = Path(candidate.name[2:-1])

        if not candidate.is_absolute():
            candidate = Path(self._tmp_path, candidate)

        return candidate.resolve()

    def _resolve_representative_science_file(self, science_files, ic_science):
        for file_entry in science_files:
            candidate = self._normalize_ifc_file_entry(file_entry)
            if candidate is not None and candidate.exists():
                return candidate

        if 'file' in ic_science.summary.colnames:
            for file_entry in ic_science.summary['file']:
                candidate = self._normalize_ifc_file_entry(file_entry)
                if candidate is not None and candidate.exists():
                    return candidate

        # Final bounded fallback: scan only inside tmp for a matching science file.
        tmp_root = Path(self._tmp_path)
        for file_path in sorted(tmp_root.glob(f"*{self._suffix}")):
            try:
                with fits.open(file_path, mode='readonly', ignore_missing_end=True) as hdul:
                    hdul.verify('fix')
                    hdr = hdul[0].header
                    imagetyp = str(self._get_header_value(hdr, 'IMAGETYP', '')).strip().upper()
                    if imagetyp != bc.IMAGETYP_LIGHT.upper():
                        continue
                    binning = str(self._get_header_value(hdr, 'BINNING', '')).strip()
                    if binning == self._bin_str:
                        return file_path.resolve()
            except Exception:
                continue

        raise RuntimeError(
            f"No representative science file could be resolved for binning {self._bin_str} in {self._tmp_path}."
        )

    @staticmethod
    def _get_header_value(header, key, default=None):
        if key in header:
            return header[key]
        if key.upper() in header:
            return header[key.upper()]
        if key.lower() in header:
            return header[key.lower()]
        return default

    def _read_fits_exposure(self, file_path, obsparams):
        with fits.open(file_path, mode='readonly', ignore_missing_end=True) as hdul:
            hdul.verify('fix')
            exptime = self._get_header_value(hdul[0].header, obsparams['exptime'])
        return None if exptime is None else float(exptime)

    def _group_files_by_exposure(self, file_paths, obsparams):
        grouped = collections.defaultdict(list)
        for file_path in file_paths:
            exptime = self._read_fits_exposure(file_path, obsparams)
            if exptime is None:
                continue
            grouped[float(exptime)].append(file_path)
        return {k: sorted(v) for k, v in grouped.items()}

    def _is_master_usable(self, master_path, expected_shape):
        master_path = Path(master_path)
        if not master_path.exists():
            return False
        try:
            with fits.open(master_path, mode='readonly', ignore_missing_end=True) as hdul:
                hdul.verify('fix')
                if hdul[0].data is None:
                    return False
                if expected_shape is not None and tuple(hdul[0].data.shape) != tuple(expected_shape):
                    return False
        except Exception as exc:
            self._log.debug(f"{' ':<4}[master] rejecting unreadable master {master_path}: {exc}")
            return False
        return True

    @staticmethod
    def _derive_candidate_master_root(search_path):
        return (Path(search_path) / 'master_calibs').resolve()

    @staticmethod
    def _is_same_master_root(left_root, right_root):
        try:
            return Path(left_root).resolve() == Path(right_root).resolve()
        except Exception:
            return False

    def _scan_existing_dark_masters(self, master_root, requirement, obsparams, inst=None):
        usable = {}
        dark_glob = self._build_master_dark_glob(requirement['suffix'],
                                                 requirement['binning_str'],
                                                 requirement['telescope'],
                                                 inst)
        for dark_path in Path(master_root).glob(dark_glob):
            if not self._is_master_usable(dark_path, requirement['science_image_shape']):
                continue
            try:
                exptime = self._read_fits_exposure(dark_path, obsparams)
            except Exception:
                continue
            if exptime is not None:
                usable[float(exptime)] = Path(dark_path)
        return usable

    @staticmethod
    def _select_exposure_matches(required_exposures, available_exposures, tolerance, allow_scaling):
        # Policy note: a match within DARK_TOLERANCE is intentionally treated as exact
        # (is_exact=True, scale_required=False) — i.e. a near dark is subtracted WITHOUT
        # scaling. Only matches outside tolerance are scaled, and only when allow_scaling.
        # 'is_exact' therefore means "usable without scaling", not "identical exposure".
        matches = {}
        available = sorted({float(v) for v in available_exposures})
        for target in sorted({float(v) for v in required_exposures}):
            if not available:
                matches[target] = None
                continue
            diffs = sorted([(abs(exp - target), exp) for exp in available], key=lambda item: (item[0], item[1]))
            diff, selected = diffs[0]
            if diff <= tolerance:
                matches[target] = {
                    'actual_exposure': float(selected),
                    'is_exact': True,
                    'scale_required': False,
                }
            elif allow_scaling:
                matches[target] = {
                    'actual_exposure': float(selected),
                    'is_exact': False,
                    'scale_required': True,
                }
            else:
                matches[target] = None
        return matches

    @staticmethod
    def _dark_needs_bias_removal(info):
        """Return True if this dark must be bias-subtracted (made "pure") before use.

        A dark is used *scaled* whenever its exposure does not match the target within
        tolerance (``is_exact`` is False). Scaling multiplies the whole frame, so an
        impure dark's bias pedestal would be scaled together with the thermal signal and
        corrupt the subtraction — the dark must therefore be bias-free first. An
        exact/unscaled dark may remain impure, because its bias cancels in the science
        subtraction. The direction of the mismatch is irrelevant: both scale-up
        (``actual < target``) and scale-down (``actual > target``) require a bias-free
        dark. (Valid scaling additionally presupposes that a master bias is available;
        without one a scaled subtraction cannot be made bias-correct.)
        """
        return bool(
            info.get('available')
            and info.get('actual_exposure') is not None
            and not info.get('is_exact', True)
        )

    def _probe_candidate(self, search_path, resolved_date, master_out, requirement,
                         obsparams, filters, binnings, suffix, inst=None,
                         sci_bin_str=None):
        search_path = Path(search_path).resolve()
        science_master_root = Path(master_out).resolve()
        candidate_master_root = self._derive_candidate_master_root(search_path)
        fallback_master_allowed = not self._is_same_master_root(science_master_root, candidate_master_root)

        self._log.debug(f"{' ':<4}Candidate probe       : {search_path}")

        stage_plan = {'bias': [], 'dark': [], 'flat': {}}
        adoption_plan = {'bias': None, 'dark': {}, 'flat': {}}
        optional_unavailable = []
        has_any_calib = None
        fits_list = None
        raw_screen_reasons = []

        def ensure_raw_calib_screen(reason):
            """Run the coarse raw-calib screen only when raw creation is actually considered."""
            nonlocal has_any_calib, fits_list, raw_screen_reasons
            if reason and reason not in raw_screen_reasons:
                raw_screen_reasons.append(reason)
            if has_any_calib is None:
                has_any_calib = self._check_calibration_files_exist(search_path, suffix, obsparams,
                                                                    binnings, filters, inst)
                fits_list = self._list_fits(search_path, suffix) if has_any_calib else []
            return has_any_calib, fits_list

        flat_status = {}
        dark_status = {'science': {}, 'flat': {}}

        force_make = requirement['force_reduction']
        expected_shape = requirement['science_image_shape']
        allowed_available_status = {
            'science_date_master_usable',
            'fallback_master_adoptable',
            'raw_creatable',
        }

        master_bias_path = self._build_master_bias_path(science_master_root, suffix, sci_bin_str,
                                                        self._telescope, inst)
        if requirement['user_preferences']['BIAS_CORRECT']:
            local_exists = Path(master_bias_path).exists()
            local_usable = self._is_master_usable(master_bias_path, expected_shape)
            if local_usable and not force_make:
                bias_status = {'status': 'science_date_master_usable', 'path': master_bias_path,
                               'source': master_bias_path}
            elif local_usable and force_make:
                bias_status = {'status': 'science_date_master_forced_stale', 'path': master_bias_path,
                               'source': master_bias_path}
            elif local_exists:
                bias_status = {'status': 'science_date_master_rejected_unsuitable', 'path': master_bias_path,
                               'source': master_bias_path}
            else:
                bias_status = {'status': 'science_date_master_missing', 'path': master_bias_path, 'source': None}

            if bias_status['status'] not in allowed_available_status and fallback_master_allowed:
                fallback_bias = self._build_master_bias_path(candidate_master_root, suffix, sci_bin_str,
                                                             self._telescope, inst)
                fallback_exists = Path(fallback_bias).exists()
                fallback_usable = self._is_master_usable(fallback_bias, expected_shape)
                if fallback_usable:
                    bias_status = {'status': 'fallback_master_adoptable', 'path': master_bias_path,
                                   'source': fallback_bias}
                    adoption_plan['bias'] = {
                        'source_path': Path(fallback_bias),
                        'target_path': Path(master_bias_path),
                    }
                elif fallback_exists:
                    bias_status = {'status': 'fallback_master_rejected_unsuitable', 'path': master_bias_path,
                                   'source': fallback_bias}
                else:
                    bias_status = {'status': 'fallback_master_missing', 'path': master_bias_path, 'source': None}

            if bias_status['status'] not in allowed_available_status:
                raw_available, raw_fits_list = ensure_raw_calib_screen('bias')
                if raw_available and raw_fits_list:
                    dfilter, add_filters, regexp = self.create_dfilter(inst, obsparams, 'bias')
                    bias_files = list(self.filter_calib_files(raw_fits_list, obsparams, dfilter, add_filters,
                                                              binnings, resolved_date, regexp=regexp))
                    if bias_files:
                        stage_plan['bias'] = bias_files
                        bias_status = {'status': 'raw_creatable', 'path': master_bias_path, 'source': None}
                    else:
                        bias_status = {'status': 'unavailable', 'path': master_bias_path, 'source': None}
                else:
                    bias_status = {'status': 'unavailable', 'path': master_bias_path, 'source': None}
        else:
            bias_status = {'status': 'not_requested', 'path': master_bias_path, 'source': None}

        flat_raw_exposure_requirements = {}
        if requirement['user_preferences']['FLAT_CORRECT']:
            for filt in filters:
                filt_name = filt[0]
                master_flat_path = self._build_master_flat_path(science_master_root, suffix, filt_name, sci_bin_str,
                                                                self._telescope, inst)
                local_exists = Path(master_flat_path).exists()
                local_usable = self._is_master_usable(master_flat_path, expected_shape)
                if local_usable and not force_make:
                    flat_status[filt_name] = {'status': 'science_date_master_usable', 'path': master_flat_path,
                                              'source': master_flat_path, 'exposures': []}
                elif local_usable and force_make:
                    flat_status[filt_name] = {'status': 'science_date_master_forced_stale', 'path': master_flat_path,
                                              'source': master_flat_path, 'exposures': []}
                elif local_exists:
                    flat_status[filt_name] = {'status': 'science_date_master_rejected_unsuitable',
                                              'path': master_flat_path, 'source': master_flat_path, 'exposures': []}
                else:
                    flat_status[filt_name] = {'status': 'science_date_master_missing', 'path': master_flat_path,
                                              'source': None, 'exposures': []}

                if flat_status[filt_name]['status'] not in allowed_available_status and fallback_master_allowed:
                    fallback_flat = self._build_master_flat_path(candidate_master_root, suffix, filt_name, sci_bin_str,
                                                                 self._telescope, inst)
                    fallback_exists = Path(fallback_flat).exists()
                    fallback_usable = self._is_master_usable(fallback_flat, expected_shape)
                    if fallback_usable:
                        flat_status[filt_name] = {'status': 'fallback_master_adoptable', 'path': master_flat_path,
                                                  'source': fallback_flat, 'exposures': []}
                        adoption_plan['flat'][filt_name] = {
                            'source_path': Path(fallback_flat),
                            'target_path': Path(master_flat_path),
                        }
                    elif fallback_exists:
                        flat_status[filt_name] = {'status': 'fallback_master_rejected_unsuitable',
                                                  'path': master_flat_path, 'source': fallback_flat, 'exposures': []}
                    else:
                        flat_status[filt_name] = {'status': 'fallback_master_missing', 'path': master_flat_path,
                                                  'source': None, 'exposures': []}

                if flat_status[filt_name]['status'] not in allowed_available_status:
                    raw_available, raw_fits_list = ensure_raw_calib_screen(f'flat:{filt_name}')
                    if raw_available and raw_fits_list:
                        dfilter, add_filters, regexp = self.create_dfilter(inst, obsparams, 'flat')
                        dfilter[obsparams['filter'].lower()] = filt[1]
                        flat_files = list(self.filter_calib_files(raw_fits_list, obsparams, dfilter, add_filters,
                                                                  binnings, resolved_date, regexp=regexp))
                        if flat_files:
                            stage_plan['flat'][filt_name] = flat_files
                            flat_raw_exposure_requirements[filt_name] = sorted(
                                self._group_files_by_exposure(flat_files, obsparams).keys()
                            )
                            flat_status[filt_name] = {
                                'status': 'raw_creatable',
                                'path': master_flat_path,
                                'source': None,
                                'exposures': flat_raw_exposure_requirements[filt_name],
                            }
                        else:
                            flat_status[filt_name] = {'status': 'unavailable', 'path': master_flat_path,
                                                      'source': None, 'exposures': []}
                    else:
                        flat_status[filt_name] = {'status': 'unavailable', 'path': master_flat_path,
                                                  'source': None, 'exposures': []}
        else:
            for filt in filters:
                filt_name = filt[0]
                master_flat_path = self._build_master_flat_path(science_master_root, suffix, filt_name, sci_bin_str,
                                                                self._telescope, inst)
                flat_status[filt_name] = {'status': 'not_requested', 'path': master_flat_path, 'source': None,
                                          'exposures': []}

        science_dark_requirements = requirement['science_exposure_times'] if requirement['user_preferences'][
            'DARK_CORRECT'] else []
        flat_dark_requirements = sorted({float(exp)
                                         for _, exps in flat_raw_exposure_requirements.items()
                                         for exp in exps})

        dark_products = {}
        dark_required = bool(
            requirement['user_preferences']['DARK_CORRECT'] or requirement['user_preferences']['FLATDARK_CORRECT'])
        if dark_required:
            local_dark_masters = {} if force_make else self._scan_existing_dark_masters(science_master_root,
                                                                                        requirement,
                                                                                        obsparams,
                                                                                        inst=inst)
            all_targets = science_dark_requirements + flat_dark_requirements
            local_matches = self._select_exposure_matches(required_exposures=all_targets,
                                                          available_exposures=local_dark_masters.keys(),
                                                          tolerance=requirement['user_preferences']['DARK_TOLERANCE'],
                                                          allow_scaling=requirement['user_preferences'][
                                                              'ALLOW_DARK_SCALING'])

            unresolved_targets = [t for t, m in local_matches.items() if m is None]
            fallback_dark_masters = {}
            fallback_matches = {}
            if fallback_master_allowed and unresolved_targets:
                fallback_dark_masters = self._scan_existing_dark_masters(candidate_master_root,
                                                                         requirement,
                                                                         obsparams,
                                                                         inst=inst)
                fallback_matches = self._select_exposure_matches(required_exposures=unresolved_targets,
                                                                 available_exposures=fallback_dark_masters.keys(),
                                                                 tolerance=requirement['user_preferences'][
                                                                     'DARK_TOLERANCE'],
                                                                 allow_scaling=requirement['user_preferences'][
                                                                     'ALLOW_DARK_SCALING'])

            unresolved_targets = [t for t in unresolved_targets if fallback_matches.get(t) is None]
            raw_dark_groups = {}
            raw_dark_matches = {}
            if unresolved_targets:
                raw_available, raw_fits_list = ensure_raw_calib_screen('dark')
            else:
                raw_available, raw_fits_list = False, []
            if unresolved_targets and raw_available and raw_fits_list:
                dfilter, add_filters, regexp = self.create_dfilter(inst, obsparams, 'dark')
                raw_dark_files = list(self.filter_calib_files(raw_fits_list, obsparams, dfilter, add_filters,
                                                              binnings, resolved_date, regexp=regexp))
                raw_dark_groups = self._group_files_by_exposure(raw_dark_files, obsparams)
                raw_dark_matches = self._select_exposure_matches(required_exposures=unresolved_targets,
                                                                 available_exposures=raw_dark_groups.keys(),
                                                                 tolerance=requirement['user_preferences'][
                                                                     'DARK_TOLERANCE'],
                                                                 allow_scaling=requirement['user_preferences'][
                                                                     'ALLOW_DARK_SCALING'])

            bias_available_for_fallback = bias_status['status'] in allowed_available_status
            for consumer, targets in [('science', science_dark_requirements), ('flat', flat_dark_requirements)]:
                for target in sorted(set(float(v) for v in targets)):
                    match = local_matches.get(target)
                    status = 'science_date_master_usable' if match is not None else None
                    source_master = local_dark_masters.get(
                        float(match['actual_exposure'])) if match is not None else None

                    if status is None and fallback_master_allowed:
                        match = fallback_matches.get(target)
                        if match is not None:
                            status = 'fallback_master_adoptable'
                            source_master = fallback_dark_masters.get(float(match['actual_exposure']))

                    if status is None:
                        match = raw_dark_matches.get(target)
                        if match is not None:
                            status = 'raw_creatable'

                    if status is None:
                        status = 'optional_unavailable_bias_fallback' if bias_available_for_fallback else 'unavailable'
                        dark_status[consumer][target] = {
                            'status': status,
                            'target_exposure': float(target),
                            'actual_exposure': None,
                            'is_exact': False,
                            'source': None,
                            'requested': True,
                            'wanted': True,
                            'available': False,
                            'blocking': False,
                            'fallback': 'bias' if bias_available_for_fallback else 'none',
                            'master_status': 'skipped_optional_unavailable_bias_fallback'
                            if bias_available_for_fallback else None,
                        }
                        if status == 'optional_unavailable_bias_fallback':
                            optional_unavailable.append(f'{consumer}-dark:{target:g}')
                        continue

                    actual_exp = float(match['actual_exposure'])
                    target_dark_path = self._build_master_dark_path(science_master_root, suffix, actual_exp,
                                                                    sci_bin_str, self._telescope, inst)
                    dark_status[consumer][target] = {
                        'status': status,
                        'target_exposure': float(target),
                        'actual_exposure': actual_exp,
                        'is_exact': bool(match['is_exact']),
                        'source': 'science_master' if status == 'science_date_master_usable' else (
                            'fallback_master' if status == 'fallback_master_adoptable' else 'raw_candidate'
                        ),
                        'requested': True,
                        'wanted': True,
                        'available': True,
                        'blocking': False,
                        'fallback': 'none',
                        'master_status': None,
                    }
                    if status == 'fallback_master_adoptable' and source_master is not None:
                        adoption_plan['dark'][actual_exp] = {
                            'source_path': Path(source_master),
                            'target_path': Path(target_dark_path),
                        }
                    if status == 'raw_creatable':
                        stage_plan['dark'].extend(raw_dark_groups.get(actual_exp, []))
                    dark_products[actual_exp] = {
                        'path': Path(target_dark_path),
                        'create': status == 'raw_creatable',
                    }

            stage_plan['dark'] = sorted(set(stage_plan['dark']))

        product_blocking = []
        if requirement['user_preferences']['FLAT_CORRECT']:
            for filt in filters:
                filt_name = filt[0]
                if flat_status[filt_name]['status'] == 'unavailable':
                    product_blocking.append(f'flat:{filt_name}')
                elif flat_status[filt_name]['status'] == 'raw_creatable':
                    # A flat built from raw frames needs a bias or unscaled flat-darks for every
                    # flat exposure; otherwise the bias level stays in the flat.
                    bias_usable = bias_status['status'] in allowed_available_status
                    flat_darks_cover = requirement['user_preferences']['FLATDARK_CORRECT'] and all(
                        dark_status['flat'].get(float(exp), {}).get('status') in allowed_available_status
                        and dark_status['flat'].get(float(exp), {}).get('is_exact')
                        for exp in flat_status[filt_name]['exposures']
                    )
                    if not bias_usable and not flat_darks_cover:
                        product_blocking.append(f'flat-baseline:{filt_name}')

        if requirement['user_preferences']['BIAS_CORRECT'] and bias_status['status'] == 'unavailable':
            # Darks can only replace the bias when used unscaled: scaling an impure
            # dark would scale its bias pedestal too (see _dark_needs_bias_removal).
            all_sci_dark_covered = bool(science_dark_requirements) and all(
                dark_status['science'].get(float(exp), {}).get('status') in allowed_available_status
                and dark_status['science'].get(float(exp), {}).get('is_exact')
                for exp in science_dark_requirements
            )
            if not all_sci_dark_covered:
                product_blocking.append('bias')

        # Darks are used when no bias is available.
        if requirement['user_preferences']['DARK_CORRECT'] and not bias_status['status'] in allowed_available_status:
            missing_science_dark_targets = [
                float(exp) for exp in science_dark_requirements
                if dark_status['science'].get(float(exp), {}).get('status') == 'unavailable'
            ]
            if missing_science_dark_targets:
                product_blocking.extend(
                    [f'dark-baseline:{exp:g}' for exp in sorted(set(missing_science_dark_targets))]
                )

        complete = len(product_blocking) == 0
        unresolved = list(product_blocking)

        # If None = no bias will be used, else path to bias master (existing or to be created)
        use_bias = bias_status['status'] in allowed_available_status
        master_bias_info = [Path(master_bias_path),
                            bias_status['status'] == 'raw_creatable'] if use_bias else None
        master_flat_info = {}
        for filt in filters:
            filt_name = filt[0]
            status_info = flat_status[filt_name]
            master_flat_info[filt_name] = [Path(status_info['path']),
                                           status_info['status'] == 'raw_creatable']
        master_dark_info = {exp: [info['path'], bool(info['create'])] for exp, info in sorted(dark_products.items())}

        calib_state_preview = {
            'bias': {
                'requested': requirement['user_preferences']['BIAS_CORRECT'],
                'available': use_bias,
                'created': False,
                'master_path': Path(master_bias_path) if use_bias else None,
                'blocking': requirement['user_preferences']['BIAS_CORRECT'] and use_bias,
                'probe_status': bias_status['status'],
                'master_status': 'skipped_by_preference' if not requirement['user_preferences'][
                    'BIAS_CORRECT'] else None,
            },
            'flat': {
                filt[0]: {
                    'requested': requirement['user_preferences']['FLAT_CORRECT'],
                    'available': flat_status[filt[0]]['status'] in allowed_available_status,
                    'created': False,
                    'master_path': Path(flat_status[filt[0]]['path']),
                    'blocking': requirement['user_preferences']['FLAT_CORRECT'],
                    'probe_status': flat_status[filt[0]]['status'],
                    'master_status': 'skipped_by_preference' if not requirement['user_preferences'][
                        'FLAT_CORRECT'] else None,
                }
                for filt in filters
            },
            'dark': {
                'science': {k: dict(v) for k, v in dark_status['science'].items()},
                'flat': {k: dict(v) for k, v in dark_status['flat'].items()},
            },
            'warnings': [],
        }

        science_master_items = []
        if bias_status.get('status') == 'science_date_master_usable':
            science_master_items.append('bias')
        science_master_items.extend(
            f'flat:{filt_name}'
            for filt_name in sorted(flat_status.keys())
            if flat_status[filt_name].get('status') == 'science_date_master_usable'
        )
        science_master_dark_exposures = set()
        for consumer_status in dark_status.values():
            for dark_info in consumer_status.values():
                if dark_info.get('status') == 'science_date_master_usable':
                    actual_exposure = dark_info.get('actual_exposure')
                    if actual_exposure is not None:
                        science_master_dark_exposures.add(float(actual_exposure))
        science_master_items.extend(
            f'dark:{actual_exp:g}s'
            for actual_exp in sorted(science_master_dark_exposures)
        )
        if science_master_items:
            self._log.debug(
                f"{' ':<6}Science-date master check: PASS "
                f"(usable master_calibs found: {', '.join(science_master_items)})"
            )
            self._log.debug(f"{' ':<6}Science-date master root : {science_master_root}")
            self._log.debug(f"{' ':<6}Science-date master action: use in place; no adoption/copy needed")

        adoption_items = []
        if adoption_plan.get('bias'):
            adoption_items.append('bias')
        adoption_items.extend(
            f'flat:{filt_name}' for filt_name in sorted(adoption_plan.get('flat', {}).keys())
        )
        adoption_items.extend(
            f'dark:{float(actual_exp):g}s'
            for actual_exp in sorted(adoption_plan.get('dark', {}).keys(), key=lambda value: float(value))
        )
        if adoption_items:
            self._log.debug(
                f"{' ':<6}Fallback master check: PASS "
                f"(usable master_calibs found: {', '.join(adoption_items)})"
            )
            self._log.debug(f"{' ':<6}Fallback master root : {candidate_master_root}")
            self._log.debug(f"{' ':<6}Fallback master action: adopt/copy to {science_master_root}")

        if has_any_calib is not None:
            raw_screen_label = 'present' if has_any_calib else 'none found'
            raw_reason_text = ', '.join(raw_screen_reasons) if raw_screen_reasons else 'unspecified raw creation need'
            self._log.debug(
                f"{' ':<6}Raw-calib screen: {raw_screen_label} "
                f"(raw creation considered for: {raw_reason_text})"
            )

        self._log.debug(
            f"{' ':<6}Probe result: {'COMPLETE' if complete else f'INCOMPLETE – unresolved hard products: {unresolved}'}")
        return {
            'search_path': Path(search_path),
            'science_master_root': science_master_root,
            'candidate_master_root': candidate_master_root,
            'fits_list': fits_list or [],
            'coarse': bool(has_any_calib),
            'stage_plan': stage_plan,
            'adoption_plan': adoption_plan,
            'unresolved': unresolved,
            'optional_unavailable': optional_unavailable,
            'complete': complete,
            'calib_state_preview': calib_state_preview,
            'science_dark_requirements': science_dark_requirements,
            'flat_dark_requirements': flat_dark_requirements,
            'master_bias_info': master_bias_info,
            'master_dark_info': master_dark_info,
            'master_flat_info': master_flat_info,
            'diagnostics': {
                'bias_status': bias_status,
                'flat_status': flat_status,
            },
        }

    def _adopt_fallback_master_products(self, adoption_plan, requirement, calib_state):
        if not adoption_plan:
            return
        if not (adoption_plan.get("bias") or adoption_plan.get("dark") or adoption_plan.get("flat")):
            return


        expected_shape = requirement['science_image_shape']
        force_make = requirement['force_reduction']

        def _copy_and_verify(source_path, target_path):
            source_path = Path(source_path)
            target_path = Path(target_path)
            if not source_path.exists():
                raise MasterBuildError(f'Fallback master missing for adoption: {source_path}')
            target_path.parent.mkdir(exist_ok=True, parents=True)
            if target_path.exists() and not force_make and self._is_master_usable(target_path, expected_shape):
                return 'loaded_existing_science_date_master'
            was_existing = target_path.exists()
            try:
                shutil.copy2(source_path, target_path)
            except Exception as exc:
                raise MasterBuildError(
                    f'Failed to copy fallback master {source_path} -> {target_path}: {exc}'
                ) from exc
            if not self._is_master_usable(target_path, expected_shape):
                raise MasterBuildError(f'Adopted master is unreadable/unsuitable: {target_path}')
            if was_existing:
                return 'overwritten_by_adopted_fallback_master'
            return 'adopted_from_fallback_master'

        bias_adoption = adoption_plan.get('bias')
        if bias_adoption:
            mat_status = _copy_and_verify(bias_adoption['source_path'], bias_adoption['target_path'])
            calib_state['bias']['master_path'] = Path(bias_adoption['target_path'])
            calib_state['bias']['created'] = False
            calib_state['bias']['master_status'] = mat_status
            self._log.debug(
                f"{' ':<4}Adopted fallback bias master: {bias_adoption['source_path']} -> {bias_adoption['target_path']}"
            )

        for filt_name, mapping in adoption_plan.get('flat', {}).items():
            mat_status = _copy_and_verify(mapping['source_path'], mapping['target_path'])
            if filt_name in calib_state['flat']:
                calib_state['flat'][filt_name]['master_path'] = Path(mapping['target_path'])
                calib_state['flat'][filt_name]['created'] = False
                calib_state['flat'][filt_name]['master_status'] = mat_status
            self._log.debug(
                f"{' ':<4}Adopted fallback flat master [{filt_name}]: {mapping['source_path']} -> {mapping['target_path']}"
            )

        for actual_exp, mapping in adoption_plan.get('dark', {}).items():
            mat_status = _copy_and_verify(mapping['source_path'], mapping['target_path'])
            for consumer in ('science', 'flat'):
                for _, info in calib_state['dark'][consumer].items():
                    if info.get('actual_exposure') == float(actual_exp):
                        info['master_path'] = Path(mapping['target_path'])
                        info['created'] = False
                        info['master_status'] = mat_status
            self._log.debug(
                f"{' ':<4}Adopted fallback dark master [{actual_exp:g}s]: {mapping['source_path']} -> {mapping['target_path']}"
            )

    def _stage_and_prepare(self, file_list, imagetyp, seen, telescope, obsparams, label=None):
        """Copy raw calibration files to tmp_path and prepare them.

        Only stages files when needed (master doesn't exist or needs creation).
        Tracks already-staged files to avoid duplicate processing.
        """
        if not file_list:
            self._log.debug(f"{' ':<4}[stage:{imagetyp}] No files to stage (master already exists or none found)")
            return
        tmp_path = Path(self._tmp_path)
        suffix = self._suffix
        new_paths = []
        if not self.silent:
            self._log.info(f"{' ':<4}Copying {len(file_list)} raw {label or imagetyp} frame(s)")
        self._log.debug(f"{' ':<4}[stage:{imagetyp}] Target folder: {tmp_path}")
        for i, fpath in enumerate(file_list):
            fpath_str = str(fpath)
            if fpath_str in seen:
                self._log.debug(f"{' ':<6}Already staged, skipping: {fpath_str}")
                continue
            seen.add(fpath_str)
            shutil.copy2(fpath_str, tmp_path)
            new_paths.append(os.path.join(tmp_path, Path(fpath_str).stem + suffix))
            if not self.silent:
                bc.print_progress_bar(i + 1, len(file_list), length=60,
                                      color=bc.BCOLORS.OKGREEN, prefix=' ' * 2)
        if not self.silent:
            sys.stdout.write('\n')
        if new_paths:
            self._log.debug(f"{' ':<4}[stage:{imagetyp}] Normalising headers for {len(new_paths)} copied file(s)")
            self.prepare_fits_files(new_paths, telescope, obsparams, imagetyp)
        else:
            self._log.debug(f"{' ':<4}[stage:{imagetyp}] All files were already staged, nothing new to prepare")

    @staticmethod
    def _derive_dataset_date_root(path, fallback):
        p = Path(path).resolve()
        parts = p.parts
        for i, part in enumerate(parts):
            if part.lower() == 'sci':
                candidate = Path(*parts[:i])
                return candidate.resolve(), 'science_segment_parent'
        for i in range(len(parts) - 1, -1, -1):
            if ReduceSatObs._parse_obs_date_from_text(parts[i]) is not None:
                candidate = Path(*parts[:i + 1])
                return candidate.resolve(), 'nearest_date_segment'
        return Path(fallback).resolve(), 'same_night_path'

    @staticmethod
    def _extract_gain_readnoise(ic_all):
        gain = None
        readnoise = None

        if 'gain' in ic_all.summary.keys():
            g_vals = np.array(ic_all.summary['gain'].data)
            g_valid = g_vals[g_vals.nonzero()]
            if g_valid.size > 0:
                gain = np.unique(g_valid)[0]

        if 'ron' in ic_all.summary.keys():
            r_vals = np.array(ic_all.summary['ron'].data)
            r_valid = r_vals[r_vals.nonzero()]
            if r_valid.size > 0:
                readnoise = np.unique(r_valid)[0]

        return gain, readnoise

    def _build_master_products(self, requirement, calib_state, obsparams, filters):
        """Create or load master calibration products.

        Reads product availability from ``calib_state`` (the single runtime
        truth) and writes back ``available``, ``created``, and ``master_path``
        after each product is successfully created or loaded.

        Key invariants:
        1. Only darks with calib_state['dark'][consumer][exp]['available']=True
           are built and entered into the master_darks map.
        2. Missing/optional darks are absent from the master_darks map — not None.
        3. FLATDARK_CORRECT is stored in calib_state, never in self._config.
        4. The master_darks map is returned for the science-reduction handoff.
        """
        ic_all = ImageFileCollection(location=self._tmp_path)
        if ic_all.summary is None or len(ic_all.summary) == 0:
            raise RuntimeError('No files available in temporary directory for building master calibrations.')

        gain, readnoise = self._extract_gain_readnoise(ic_all)
        has_multiple_amps = obsparams['multiple_amps']
        dbias = self._master_bias_info
        ddark = self._master_dark_info
        dflat = self._master_flat_info
        dark_state = calib_state['dark']  # shorthand — reads/writes in-place

        # Create or load master bias
        ccd_master_bias = None
        if requirement['user_preferences']['BIAS_CORRECT'] and not dbias:
            # Bias requested but unavailable; the probe only allows this when the
            # science darks cover every science exposure.
            self._log.debug(f"{' ':<4}No master bias available; science darks are the correction baseline")
            calib_state['bias']['available'] = False
            calib_state['bias']['created'] = False
            calib_state['bias']['master_path'] = None
            calib_state['bias']['master_status'] = 'skipped_unavailable_dark_baseline'
        elif requirement['user_preferences']['BIAS_CORRECT']:
            master_bias = dbias[0]
            create_bias = dbias[1]
            try:
                if create_bias:
                    status = 'overwritten_from_raw' if Path(master_bias).exists() else 'created_from_raw'
                    ccd_master_bias = self.create_master_bias(files_list=ic_all,
                                                              obsparams=obsparams,
                                                              mbias_fname=master_bias,
                                                              multiple_amps=has_multiple_amps,
                                                              gain=gain,
                                                              readnoise=readnoise,
                                                              method=self._config['COMBINE_METHOD_BIAS'])
                else:
                    # Keep an adoption status set by _adopt_fallback_master_products
                    status = calib_state['bias'].get('master_status') or 'loaded_existing_science_date_master'
                    self._log.debug(f"{' ':<4}Loading master bias file: {os.path.basename(master_bias)}")
                    ccd_master_bias = self.convert_fits_to_ccd(master_bias, single=True)
                if ccd_master_bias is None:
                    raise RuntimeError(f'master bias could not be read: {master_bias}')
            except Exception as exc:
                calib_state['bias']['master_status'] = 'failed_build'
                raise MasterBuildError(f'Bias build failed: {exc}') from exc
            # Write back into calib_state
            calib_state['bias']['available'] = ccd_master_bias is not None
            calib_state['bias']['created'] = create_bias
            calib_state['bias']['master_path'] = Path(master_bias)
            calib_state['bias']['master_status'] = status

        # Identify pure-dark exposures (need bias removal)
        # A dark that will be SCALED (exposure mismatch beyond tolerance, in EITHER
        # direction) must be bias-free before scaling; unscaled (exact) darks may stay
        # impure since their bias cancels in the science subtraction. See
        # _dark_needs_bias_removal. (C1 fix: previously only actual>target was flagged,
        # so scale-up darks were subtracted with their bias pedestal scaled — incorrect.)
        pure_dark_exposures = {
            info['actual_exposure']
            for consumer in ('science', 'flat')
            for info in dark_state[consumer].values()
            if self._dark_needs_bias_removal(info)
        }

        # Build only darks referenced by available entries
        actual_dark_ccds = {}
        dark_preexisting_targets = {}
        for dark_exptime, dval in ddark.items():
            master_dark = dval[0]
            create_dark = dval[1]
            dark_preexisting_targets[dark_exptime] = Path(master_dark).exists()
            is_referenced = any(
                info.get('available') and info.get('actual_exposure') == dark_exptime
                for consumer in ('science', 'flat')
                for info in dark_state[consumer].values()
            )
            if not is_referenced and dark_exptime not in pure_dark_exposures:
                self._log.debug(f"{' ':<4}Dark {dark_exptime:g}s: not referenced, skipping")
                continue
            mbias_for_dark = ccd_master_bias if dark_exptime in pure_dark_exposures else None
            if create_dark:
                try:
                    actual_dark_ccds[dark_exptime] = self.create_master_dark(
                        files_list=ic_all, obsparams=obsparams,
                        mdark_fname=master_dark, exptime=dark_exptime,
                        mbias=mbias_for_dark, multiple_amps=has_multiple_amps,
                        gain=gain, readnoise=readnoise,
                        method=self._config['COMBINE_METHOD_DARK'])
                except Exception as exc:
                    raise MasterBuildError(
                        f'Dark build failed for exposure {dark_exptime:g}s: {exc}'
                    ) from exc
            else:
                self._log.debug(f"{' ':<4}Loading master dark file: {os.path.basename(master_dark)}")
                try:
                    actual_dark_ccds[dark_exptime] = self.convert_fits_to_ccd(master_dark, single=True)
                except Exception as exc:
                    raise MasterBuildError(
                        f'Dark load failed for exposure {dark_exptime:g}s: {exc}'
                    ) from exc

        # Build master_darks map; write back into calib_state
        master_darks = {'sci_dark': {}, 'flat_dark': {}}
        consumer_to_key = {'science': 'sci_dark', 'flat': 'flat_dark'}
        for consumer, dark_key in consumer_to_key.items():
            for target, info in dark_state[consumer].items():
                if not info.get('available'):
                    self._log.debug(f"{' ':<4}Dark {consumer}:{target:g}: unavailable, not in master_darks")
                    if info.get('fallback') == 'bias':
                        info['master_status'] = 'skipped_optional_unavailable_bias_fallback'
                    continue
                actual_exp = info.get('actual_exposure')
                if actual_exp is None or actual_exp not in actual_dark_ccds:
                    self._log.warning(f"{' ':<4}Dark {consumer}:{target:g}: available but not built")
                    # Downgrade availability in calib_state
                    info['available'] = False
                    continue
                ccd_mdark = actual_dark_ccds[actual_exp]
                bias_corr_in_dark = ccd_mdark.header.get('BIASCORR', False)
                master_darks[dark_key][float(target)] = (ccd_mdark, bias_corr_in_dark, bool(info['is_exact']))
                # Write back: built = True
                info['created'] = ddark[actual_exp][1] if actual_exp in ddark else False
                info['master_path'] = Path(ddark[actual_exp][0]) if actual_exp in ddark else None
                if info['created']:
                    info['master_status'] = (
                        'overwritten_from_raw' if dark_preexisting_targets.get(actual_exp,
                                                                               False) else 'created_from_raw'
                    )
                elif info.get('master_status') is None:
                    info['master_status'] = 'loaded_existing_science_date_master'
                self._log.debug(f"{' ':<4}Dark {consumer}:{target:g} → {actual_exp:g}s added to master_darks")

        # Derive flat-dark availability into calib_state
        # Do NOT touch self._config here.
        # Store into calib_state so science reduction reads it from there.
        flat_dark_available = bool(master_darks['flat_dark'])
        calib_state['flat_dark_available'] = flat_dark_available
        self._log.debug(f"{' ':<4}Flat-dark available: {flat_dark_available}")
        if flat_dark_available:
            self._log.debug(f"{' ':<6}Available flat-dark exposures: {sorted(master_darks['flat_dark'].keys())}")
        unavailable = [t for t, i in dark_state['flat'].items() if not i.get('available')]
        if unavailable:
            self._log.debug(f"{' ':<6}Flat-darks requested but unavailable: {unavailable}")

        # Create or load master flats with available-only dark map
        for filt in filters:
            filt_name = filt[0]
            if filt_name not in dflat or not requirement['user_preferences']['FLAT_CORRECT']:
                continue
            master_flat = dflat[filt_name][0]
            create_flat = dflat[filt_name][1]
            if create_flat:
                flat_status = 'overwritten_from_raw' if Path(master_flat).exists() else 'created_from_raw'
                try:
                    self.create_master_flat(files_list=ic_all, obsparams=obsparams,
                                            mflat_fname=master_flat, flat_filter=filt_name,
                                            mbias=ccd_master_bias,
                                            mdark=master_darks['flat_dark'],
                                            multiple_amps=has_multiple_amps,
                                            gain=gain, readnoise=readnoise,
                                            method=self._config['COMBINE_METHOD_FLAT'])
                except Exception as exc:
                    calib_state['flat'][filt_name]['master_status'] = 'failed_build'
                    raise MasterBuildError(
                        f'Flat build failed for filter {filt_name}: {exc}'
                    ) from exc
            else:
                # Keep an adoption status set by _adopt_fallback_master_products
                flat_status = (calib_state['flat'][filt_name].get('master_status')
                               or 'loaded_existing_science_date_master')
                self._log.debug(f"{' ':<4}Loading master flat file: {os.path.basename(master_flat)}")
                try:
                    self.convert_fits_to_ccd(master_flat, single=True)
                except Exception as exc:
                    calib_state['flat'][filt_name]['master_status'] = 'failed_build'
                    raise MasterBuildError(
                        f'Flat load failed for filter {filt_name}: {exc}'
                    ) from exc
            # Write back into calib_state
            calib_state['flat'][filt_name]['available'] = True
            calib_state['flat'][filt_name]['created'] = create_flat
            calib_state['flat'][filt_name]['master_path'] = Path(master_flat)
            calib_state['flat'][filt_name]['master_status'] = flat_status

        return master_darks

    def _validate_master_products(self, requirement, calib_state, master_out):
        """Validation gate v2.

        Reads exclusively from ``calib_state`` — the single runtime truth.

        Fatal (hard) products:
        - BIAS_CORRECT=True → no master bias file on disk after the master build
        - FLAT_CORRECT=True → no master flat for a required science filter
        - Any hard product path outside locked master_out

        Optional (soft) products — logged, non-fatal:
        - DARK_CORRECT=True and science dark entry has available=False
        - FLATDARK_CORRECT=True and flat-dark entry has available=False
        """
        master_out = Path(master_out).resolve()
        fatal_missing = []
        optional_missing = []

        dbias = self._master_bias_info
        ddark = self._master_dark_info
        dflat = self._master_flat_info

        # Hard: bias — or, when no bias is used, a science dark for every exposure
        if requirement['user_preferences']['BIAS_CORRECT'] and not dbias:
            sci_darks = calib_state['dark']['science']
            uncovered = [
                float(exp) for exp in requirement['science_exposure_times']
                if not sci_darks.get(float(exp), {}).get('available')
            ]
            if uncovered:
                fatal_missing.extend(f'dark-baseline:{exp:g}' for exp in sorted(set(uncovered)))
        elif requirement['user_preferences']['BIAS_CORRECT']:
            bias_path = calib_state['bias'].get('master_path') or Path(dbias[0])
            bias_path = Path(bias_path)
            if not bias_path.exists():
                fatal_missing.append('bias')
            elif master_out not in [bias_path.parent, *bias_path.parents]:
                fatal_missing.append(f'output-root-mismatch:bias:{bias_path}')

        # Hard: flats
        if requirement['user_preferences']['FLAT_CORRECT']:
            for filt in requirement['filters']:
                filt_name = filt[0]
                flat_info = calib_state['flat'].get(filt_name, {})
                flat_path = flat_info.get('master_path') or Path(dflat[filt_name][0])
                flat_path = Path(flat_path)
                if not flat_path.exists():
                    fatal_missing.append(f'flat:{filt_name}')
                elif master_out not in [flat_path.parent, *flat_path.parents]:
                    fatal_missing.append(f'output-root-mismatch:flat:{filt_name}:{flat_path}')

        # Soft: science darks
        if requirement['user_preferences']['DARK_CORRECT']:
            for target, info in calib_state['dark']['science'].items():
                if not info.get('available'):
                    optional_missing.append(f'science-dark:{target:g}')
                else:
                    actual = info.get('actual_exposure')
                    if actual is not None and actual in ddark:
                        dark_path = Path(ddark[actual][0])
                        if not dark_path.exists():
                            optional_missing.append(f'science-dark:{target:g}')

        # Soft: flat-darks
        if requirement['user_preferences']['FLATDARK_CORRECT']:
            for target, info in calib_state['dark']['flat'].items():
                if not info.get('available'):
                    optional_missing.append(f'flat-dark:{target:g}')
                else:
                    actual = info.get('actual_exposure')
                    if actual is not None and actual in ddark:
                        dark_path = Path(ddark[actual][0])
                        if not dark_path.exists():
                            optional_missing.append(f'flat-dark:{target:g}')

        # Output-root drift check for all built dark masters
        for actual_exp, dval in ddark.items():
            dark_path = Path(dval[0])
            if dark_path.exists() and master_out not in [dark_path.parent, *dark_path.parents]:
                fatal_missing.append(f'output-root-mismatch:dark:{dark_path}')

        # Gate I: enforce master status vocabulary.
        status_records = [('bias', calib_state.get('bias', {}).get('master_status'))]
        for filt_name, item in calib_state.get('flat', {}).items():
            status_records.append((f'flat:{filt_name}', item.get('master_status')))
        for consumer in ('science', 'flat'):
            for target, item in calib_state.get('dark', {}).get(consumer, {}).items():
                status_records.append((f'dark:{consumer}:{target:g}', item.get('master_status')))

        invalid_status = [
            f'{label}:{status}'
            for label, status in status_records
            if status is not None and status not in ALLOWED_MASTER_STATUSES
        ]
        if invalid_status:
            fatal_missing.extend([f'invalid-master-status:{item}' for item in invalid_status])

        if fatal_missing:
            return fatal_missing, optional_missing

        self._log.debug(f"{' ':<4}Final validation result: PASS")
        if optional_missing:
            self._log.debug(f"{' ':<4}Optional unavailable products (non-fatal): {optional_missing}")
        return [], optional_missing


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
        for row_index, file_path in enumerate(file_names):
            fname = Path(file_path).stem
            fname_suffix = Path(file_path).suffix
            if self.verbose:
                self._log.debug(f"> Preparing {fname}")

            # Load fits file
            hdul = fits.open(file_path, mode='readonly', ignore_missing_end=True)
            hdul.verify('fix')
            prim_hdr = hdul[0].header
            new_hdr = prim_hdr.copy()

            # Get observation date from header (accepts YYYY-MM-DD or YYYYMMDD)
            date_raw = prim_hdr.get(obsparams['date_keyword'])
            date = self._parse_obs_date_from_text(date_raw)
            if date is None:
                raise RuntimeError(
                    f'Could not parse observation date from header keyword '
                    f'"{obsparams["date_keyword"]}" (value: {date_raw!r}) in {Path(file_path).name}.')
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
                        filt_trnsl = obsparams['filter_translations'].get(filt_key)
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
                    Path(file_path).unlink(missing_ok=True)
                    break

                # Update filter
                filt_trnsl = None
                filt_key = None
                if obsparams['filter'] in new_hdr:
                    filt_key = new_hdr[obsparams['filter']].strip()
                    filt_trnsl = obsparams['filter_translations'].get(filt_key)
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
                        Path(file_path).unlink(missing_ok=True)
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

            if not self.silent:
                bc.print_progress_bar(row_index + 1, len(file_names), length=77,
                                      color=bc.BCOLORS.OKGREEN, prefix=' ' * 2)
        if not self.silent:
            sys.stdout.write('\n')

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
            Boolean mask used to filtering, returned if `return_mask` is True.

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
