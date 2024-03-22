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
from astropy.stats import (
    mad_std, SigmaClip, sigma_clipped_stats)
from ccdproc import CCDData
from ccdproc import ImageFileCollection
from photutils.background import (
    Background2D, SExtractorBackground, StdBackgroundRMS)
from photutils.segmentation import (detect_sources,
                                    detect_threshold)

# Project modules
try:
    import leosatpy
except ModuleNotFoundError:
    from utils import arguments
    from utils import dataset
    from utils import tables
    from utils import version
    from utils import base_conf as _base_conf
else:
    from leosatpy.utils import arguments
    from leosatpy.utils import dataset
    from leosatpy.utils import tables
    from leosatpy.utils import version
    from leosatpy.utils import base_conf as _base_conf

# -----------------------------------------------------------------------------

""" Meta-info """
__author__ = "Christian Adam"
__copyright__ = 'Copyright 2021-2023, CLEOSat group'
__credits__ = ["Eduardo Unda-Sanzana, Jeremy Tregloan-Reed, Christian Adam"]
__license__ = "GPL-3.0 license"
__version__ = version.__version__
__maintainer__ = "Christian Adam"
__email__ = "christian.adam84@gmail.com"
__status__ = "Production"

__taskname__ = 'reduceSatObs'
# -----------------------------------------------------------------------------

""" Parameter used in the script """
# Logging and console output
logging.root.handlers = []
_log = logging.getLogger()
_log.setLevel(_base_conf.LOG_LEVEL)
stream = logging.StreamHandler()
stream.setFormatter(_base_conf.FORMATTER)
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
            _base_conf.load_warnings()

        # set variables
        self._config = collections.OrderedDict()
        self._dataset_object = None
        self._input_path = input_path
        self._log = log
        self._log_level = log_level
        self._root_dir = _base_conf.ROOT_DIR
        self._master_calib_path = None
        self._master_bias_dict = {}
        self._master_dark_dict = {}
        self._master_flat_dict = {}
        self._make_mbias = False
        self._make_mdark = False
        self._make_mflat = False
        self._make_light = False
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
        self.reduc = None

        # run full reduction
        self._run_full_reduction(silent=silent, verbose=verbose)

    def _run_full_reduction(self, silent: bool = False, verbose: bool = False):
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

        # prepare dataset from input argument
        ds = dataset.DataSet(input_args=self._input_path,
                             prog_typ='reduceSatObs',
                             prog_sub_typ=['light'],
                             log=self._log, log_level=self._log_level)

        # load configuration
        ds.load_config()
        self._config = ds.config

        # load observation result table
        self._obsTable = tables.ObsTables(config=self._config)
        self._obsTable.load_obs_table()

        # set variables for use
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

            # loop over groups and run reduction for each group
            for src_path, files in obsfile_list:
                if not silent:
                    self._log.info('====> Science image reduction run <====')
                    self._log.info(f"> Reduce {len(files)} datasets from instrument {inst} "
                                   f"at the {self._telescope} "
                                   "telescope in folder:")
                    self._log.info(f"  {src_path}")

                # reduce dataset
                self._run_single_reduction(src_path, files)

        endtime = time.perf_counter()
        dt = endtime - starttime
        td = timedelta(seconds=dt)

        if not silent:
            self._log.info(f"Program execution time: {td}")
        self._log.info('====> Science image reduction finished <====')

    def _run_single_reduction(self, sci_src_path: str, fits_files_dict: dict):
        """Run reduction on a given dataset.

        Create the required folder, copy science and calibration files and
        run the full-reduction procedure.

        """
        if self.verbose:
            self._log.debug("  > Create folder")

        # make folder for reduction results
        red_path = Path(sci_src_path, 'reduced')
        if not red_path.exists():
            red_path.mkdir(exist_ok=True)

        # create temporary working directory
        tmp_path = Path(f"{sci_src_path}/atmp/")

        # remove old temp folder and make a new one
        os.system(f"rm -rf {tmp_path}")
        tmp_path.mkdir(exist_ok=True)

        # get file suffix
        suffix = np.unique(fits_files_dict['suffix'].values[0])[0]

        # update variables
        self._sci_root_path = sci_src_path
        self._red_path = red_path
        self._tmp_path = tmp_path
        self._suffix = suffix

        # copy science files to tmp folder
        new_file_paths = []
        verbose = 'v' if self.verbose else ''
        if not self.silent:
            self._log.info("> Copy raw science images.")
        for row_index, row in fits_files_dict.iterrows():
            abs_file_path = row['input']
            file_name = row['file_name']
            file_name_suffix = row['suffix']

            # copy science files
            os.system(f"cp -r{verbose} {abs_file_path} {tmp_path}")
            new_fname = os.path.join(tmp_path, file_name + file_name_suffix)
            new_file_paths.append(new_fname)

        # prepare science file
        if not self.silent:
            self._log.info(f"> Prepare {len(new_file_paths)} ``SCIENCE`` files for reduction.")
        obs_date, filters, binnings = self._prepare_fits_files(new_file_paths,
                                                               self._telescope,
                                                               self._obsparams, 'science')
        # print(obs_date, filters, binnings)
        # loop binnings and prepare calibration files
        for binxy in binnings:

            # find and prepare calibration files
            if not self.silent:
                self._log.info(f"> Find calibration files for binning: {binxy[0]}x{binxy[1]}.")
            self._find_calibrations(self._telescope, self._obsparams, obs_date, filters, binxy)

            # run ccdproc
            self._run_ccdproc(self._obsparams, filters, binxy)

        # clean temporary folder
        if not self.silent:
            self._log.info("> Cleaning up")
        os.system(f"rm -rf {tmp_path}")

    def _check_closest_exposure(self, ic_all: ImageFileCollection,
                                obsparams: dict, tolerance: float = 0.5):
        """"""

        dark_exposures = np.array(list(self._dark_exptimes))
        exptimes_list = [{}, {}]

        if not list(dark_exposures):
            return exptimes_list

        exptime_check_imgtypes = [_base_conf.IMAGETYP_LIGHT, _base_conf.IMAGETYP_FLAT]
        expts = []
        ic_exptimes_list = []
        for i in range(len(exptime_check_imgtypes)):
            # filter data
            dfilter = {'imagetyp': f'{exptime_check_imgtypes[i]}'}
            ic_exptimes = ic_all.filter(regex_match=False, **dfilter)
            if ic_exptimes.summary is not None:
                # get unique exposure times
                expt = list(np.unique(ic_exptimes.summary[obsparams['exptime'].lower()]))
                expts += expt
                ic_exptimes_list.append(ic_exptimes)

        for i in range(len(exptime_check_imgtypes)):

            # filter data
            dfilter = {'imagetyp': f'{exptime_check_imgtypes[i]}'}
            ic_exptimes = ic_all.filter(regex_match=False, **dfilter)

            if ic_exptimes.summary is not None:
                # get unique exposure times
                expt = list(np.unique(ic_exptimes.summary[obsparams['exptime'].lower()]))
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

        return exptimes_list, expts

    def _run_ccdproc(self, obsparams: dict, filters: list, binnings: str):
        """Run ccdproc on science file

        Parameters
        ----------
        obsparams:
            Dictionary with telescope parameter
        filters:
            List with filter identified in the dataset
        binnings:
            Binning of the current dataset

        Returns
        -------

        """

        _bin_str = f'{binnings[0]}x{binnings[1]}'
        self._bin_str = _bin_str

        # get master_calib files
        dbias = self._master_bias_dict
        ddark = self._master_dark_dict
        dflat = self._master_flat_dict
        # dark_exptimes = self._dark_exptimes

        # reload files in temp folder
        ic_all = ImageFileCollection(location=self._tmp_path)

        # get the number of amplifiers and trim, and overscan sections
        namps = obsparams['n_amps']
        trim_section = None
        if obsparams['trimsec'][_bin_str] is not None:
            trim_section = obsparams['trimsec'][_bin_str]['11'] if namps == 1 else obsparams['trimsec'][_bin_str]
        oscan_section = None
        if obsparams['oscansec'][_bin_str] is not None and self._config['OVERSCAN_CORRECT']:
            oscan_section = obsparams['oscansec'][_bin_str]['11'] if namps == 1 else obsparams['oscansec'][_bin_str]

        # get science and flat exposure times
        exptimes_list, img_exptimes = self._check_closest_exposure(ic_all=ic_all,
                                                                   obsparams=obsparams,
                                                                   tolerance=1)

        # process files by filter
        for filt in filters:
            img_count_by_filter = len(ic_all.files_filtered(imagetyp=_base_conf.IMAGETYP_LIGHT,
                                                            filter=filt[0]))
            if img_count_by_filter == 0:
                continue

            if not self.silent:
                self._log.info(f"> Process filter {filt[0]}, binning {_bin_str}.")

            # get gain and readout noise
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
                    ccd_master_bias = self._create_master_bias(files_list=ic_all,
                                                               obsparams=obsparams,
                                                               mbias_fname=master_bias,
                                                               namps=namps,
                                                               trim_section=trim_section,
                                                               oscan_section=oscan_section,
                                                               gain=gain,
                                                               readnoise=readnoise,
                                                               error=self._config['EST_UNCERTAINTY'],
                                                               method=self._config['COMBINE_METHOD_BIAS'])
                else:
                    if not self.silent:
                        self._log.info('  Loading existing master bias file: %s' % os.path.basename(master_bias))

                    ccd_master_bias = self._convert_fits_to_ccd(master_bias, single=True)
                    readnoise = ccd_master_bias.header['ron']

            # create master dark file for each exposure time. Only executed once per dataset
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
                            ccd_mdark = self._create_master_dark(files_list=ic_all, obsparams=obsparams,
                                                                 mdark_fname=master_dark,
                                                                 exptime=dark_exptime,
                                                                 mbias=mbias,
                                                                 namps=namps,
                                                                 trim_section=trim_section,
                                                                 oscan_section=oscan_section,
                                                                 gain=gain, readnoise=readnoise,
                                                                 error=self._config['EST_UNCERTAINTY'],
                                                                 method=self._config['COMBINE_METHOD_DARK'])
                        else:
                            if not self.silent:
                                self._log.info(f'  Loading existing master dark file: {os.path.basename(master_dark)}')
                            ccd_mdark = self._convert_fits_to_ccd(master_dark, single=True)
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

            # create master flat file for each filter. Only executed once per dataset
            ccd_master_flat = {}
            if self._config['FLAT_CORRECT']:
                master_flat = dflat[filt[0]][0]
                create_flat = dflat[filt[0]][1]
                if create_flat:
                    ccd_master_flat[filt[0]] = self._create_master_flat(files_list=ic_all, obsparams=obsparams,
                                                                        flat_filter=filt[0],
                                                                        mflat_fname=master_flat,
                                                                        mbias=ccd_master_bias,
                                                                        mdark=ccd_master_dark['flat_dark'],
                                                                        namps=namps,
                                                                        trim_section=trim_section,
                                                                        oscan_section=oscan_section,
                                                                        gain=gain, readnoise=readnoise,
                                                                        error=self._config['EST_UNCERTAINTY'],
                                                                        method=self._config['COMBINE_METHOD_FLAT'])
                else:
                    if not self.silent:
                        self._log.info('  Loading existing master flat file: %s' % os.path.basename(master_flat))
                    ccd_master_flat[filt[0]] = self._convert_fits_to_ccd(master_flat, single=True)
            if not ccd_master_flat or not ccd_master_flat[filt[0]]:
                ccd_master_flat = None
                self._ccd_mask_fname = None

            # reduce science images
            self._ccdproc_sci_images(files_list=ic_all, obsparams=obsparams,
                                     image_filter=filt[0],
                                     master_bias=ccd_master_bias,
                                     master_dark=ccd_master_dark['sci_dark'],
                                     master_flat=ccd_master_flat,
                                     namps=namps,
                                     trim_section=trim_section,
                                     oscan_section=oscan_section,
                                     gain=gain, readnoise=readnoise,
                                     cosmic=self._config['CORRECT_COSMIC'],
                                     error=self._config['EST_UNCERTAINTY'])
        del ic_all

        gc.collect()

    def _ccdproc_sci_images(self, files_list: ImageFileCollection,
                            obsparams: dict, image_filter=None,
                            master_bias: str = None, master_dark: dict = None,
                            master_flat: str = None, trim_section: dict = None,
                            oscan_section: dict = None,
                            gain: float = None, readnoise: float = None,
                            error: bool = False, cosmic: bool = False,
                            mbox: int = 15, rbox: int = 15, gbox: int = 11,
                            cleantype: str = "medmask",
                            cosmic_method: str = 'lacosmic',
                            sigclip: int = 5, key_filter: str = 'filter',
                            dfilter: dict = None, namps: int = 1, key_find: str = 'find',
                            invert_find: bool = False):
        """
        Process science images separately.

        Parameters
        ----------

        files_list: 
        obsparams: 
        image_filter:
        master_bias:
        master_dark:
        master_flat:
        trim_section:
        gain:
        readnoise:
        error:
        cosmic:
        mbox:
        rbox:
        gbox:
        cleantype:
        cosmic_method:
        sigclip:
        key_filter:
        dfilter:
        namps:
        key_find:
        invert_find:

        Returns
        -------

        """

        # check inputs
        if error and (gain is None or readnoise is None):
            self._log.warning('You need to provide "gain" and "readnoise" to compute the error!')
            error = False
        if gain is not None and not isinstance(gain, u.Quantity):
            gain = gain * u.Unit("electron") / u.Unit("adu")
        if readnoise is not None and not isinstance(readnoise, u.Quantity):
            readnoise = readnoise * u.Unit("electron")

        dfilter = {'imagetyp': _base_conf.IMAGETYP_LIGHT,
                   'binning': self._bin_str} if dfilter is None else dfilter
        if dfilter is not None and key_filter is not None and image_filter is not None:
            dfilter = add_keys_to_dict(dfilter, {key_filter: image_filter})
        files_list = self._get_file_list(files_list, dfilter,
                                         key_find=key_find, invert_find=invert_find,
                                         abspath=False)

        ccd_mask_fname = os.path.join(self._master_calib_path,
                                      'mask_from_ccdmask_%s_%s.fits' % (image_filter, self._bin_str))
        if self._telescope == 'DDOTI 28-cm f/2.2':
            ccd_mask_fname = os.path.join(self._master_calib_path,
                                          'mask_from_ccdmask_%s_%s_%s.fits' % (self._instrument,
                                                                               image_filter, self._bin_str))
        if os.path.isfile(ccd_mask_fname):
            self._ccd_mask_fname = ccd_mask_fname

        for filename in files_list:
            if not self.silent:
                self._log.info(f"  >>> ccdproc is working for: {filename}")

            # read image to a ccd object
            fname = os.path.join(self._tmp_path, filename)
            ccd = CCDData.read(fname, unit=u.Unit("adu"))
            ccd_expt = ccd.header['exptime']

            # trim image and update header
            trimmed = True if trim_section is not None else False
            oscan_corrected = True if oscan_section is not None else False
            ccd = self._trim_image(img_ccd=ccd, obsparams=obsparams,
                                   namps=namps, oscansec=oscan_section,
                                   trimsec=trim_section,
                                   trimmed=trimmed, oscan_corrected=oscan_corrected)

            # create an uncertainty map
            if error:
                ccd = ccdproc.create_deviation(ccd, gain=gain,
                                               readnoise=readnoise,
                                               disregard_nan=True)
            # gain correct
            if gain is not None and self._config['CORRECT_GAIN']:
                ccd = ccdproc.gain_correct(ccd, gain)

            # cosmic ray correction
            if cosmic and gain is not None and self._config['CORRECT_GAIN']:
                ccd = self._clean_cosmic_ray(ccd, ccd_mask_fname,
                                             mbox=mbox, rbox=rbox, gbox=gbox, sigclip=sigclip,
                                             cleantype=cleantype, cosmic_method=cosmic_method)

            # mask bad pixel
            if 'ccd_mask' in obsparams and obsparams['ccd_mask'][self._bin_str] is not None:
                ccd_mask_list = obsparams['ccd_mask'][self._bin_str]
                for yx in ccd_mask_list:
                    ccd.data[yx[0]:yx[1], yx[2]:yx[3]] = 0

            if self._ccd_mask_fname is not None:
                mask_ccdmask = CCDData.read(self._ccd_mask_fname,
                                            unit=u.dimensionless_unscaled)
                ccd.mask = mask_ccdmask.data.astype('bool')
            else:
                # remove mask and the uncertainty extension
                ccd.mask = None
                mask_ccdmask = None

            # bias or dark frame subtract
            dark_corrected = False
            if self._config['DARK_CORRECT'] and master_dark is not None:
                mdark_ccd = master_dark[ccd_expt][0]
                is_nearest = master_dark[ccd_expt][2]
                scale = False
                if not is_nearest:
                    scale = True
                    mbias = master_bias
                    if mbias is not None and self._config['BIAS_CORRECT']:
                        if isinstance(mbias, str):
                            mbias = self._convert_fits_to_ccd(mbias, single=True)
                        ccd = ccdproc.subtract_bias(ccd, mbias)

                if isinstance(mdark_ccd, str):
                    mdark_ccd = self._convert_fits_to_ccd(mdark_ccd, single=True)
                ccd = ccdproc.subtract_dark(ccd, mdark_ccd,
                                            exposure_time='exptime',
                                            exposure_unit=u.Unit("second"),
                                            scale=scale)
                add_key_to_hdr(ccd.header, 'MDARK', get_filename(mdark_ccd))
                dark_corrected = True

            bias_corrected = False
            if self._config['BIAS_CORRECT'] and master_bias is not None and not dark_corrected:
                mbias = master_bias
                if isinstance(master_bias, str):
                    mbias = self._convert_fits_to_ccd(master_bias, single=True)
                ccd = ccdproc.subtract_bias(ccd, mbias)
                add_key_to_hdr(ccd.header, 'MBIAS', get_filename(mbias))
                bias_corrected = True

            # flat correction
            flat_corrected = False
            if self._config['FLAT_CORRECT'] and master_flat is not None:
                mflat = master_flat[image_filter]
                if isinstance(mflat, str):
                    mflat = self._convert_fits_to_ccd(mflat, single=True)
                    mflat.mask = None if mask_ccdmask is None else mask_ccdmask.data.astype('bool')
                ccd = ccdproc.flat_correct(ccd, mflat)
                add_key_to_hdr(ccd.header, 'MFLAT', get_filename(mflat))
                flat_corrected = True

            fbase = Path(filename).stem
            suffix = Path(filename).suffix
            filename = '%s_red%s' % (fbase, suffix)
            # dccd[filename] = ccd
            filename = join_path(filename, self._red_path)
            ccd.header['FILENAME'] = os.path.basename(filename)
            ccd.header['combined'] = True
            ccd.header = ammend_hdr(ccd.header)

            # change to float32 to keep the file size under control
            ccd.data = ccd.data.astype('float32')

            ccd.uncertainty = None
            hdr = ccd.header
            ccd.write(filename, overwrite=True)

            # reformat ra and dec if the values are in degree
            # to avoid float <-> str conversion errors
            if obsparams['radec_separator'] == 'XXX':
                hdr[obsparams['ra']] = round(hdr[obsparams['ra']],
                                             _base_conf.ROUND_DECIMAL)
                hdr[obsparams['dec']] = round(hdr[obsparams['dec']],
                                              _base_conf.ROUND_DECIMAL)

            # update result table
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

    def _clean_cosmic_ray(self, ccd, ccd_mask_fname,
                          mbox=15, rbox=15, gbox=11, sigclip=5,
                          cleantype="medmask",
                          cosmic_method='lacosmic'):
        """ Clean cosmic rays.

        Parameters
        ----------
        ccd: 
        ccd_mask_fname:
        mbox: 
        rbox: 
        gbox: 
        sigclip: 
        cleantype: 
        cosmic_method: 

        Returns
        -------
        ccd: astropy.nddata.CCDData
            An object of the same type as the ccd is returned.
        """

        ctype = cosmic_method.lower().strip()
        ctypes = ['lacosmic', 'median']
        if ctype not in ctypes:
            self._log.warning('> Cosmic ray type "%s" NOT available [%s]' % (ctype, ' | '.join(ctypes)))
            return

        ccd_mask_fname = ccd_mask_fname if os.path.exists(ccd_mask_fname) else None

        if ccd_mask_fname is not None and isinstance(ccd_mask_fname, str):
            ccd_mask = self._convert_fits_to_ccd(ccd_mask_fname, single=True)
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

    def _create_master_flat(self, files_list, obsparams, flat_filter=None, mflat_fname=None,
                            trim_section=None, oscan_section=None, namps=1, mbias=None, mdark=None,
                            gain=None, method='average', readnoise=None, error=False,
                            key_filter='filter', dfilter=None,
                            key_find='find', invert_find=False, sjoin=','):
        """Create master flat.

        Parameters
        ----------
        files_list: 
        obsparams: 
        flat_filter: 
        mdark: 
        trim_section:
        namps: 
        mbias: 
        gain: 
        method: 
        readnoise: 
        error: 
        key_filter: 
        dfilter: 
        key_find: 
        invert_find: 
        sjoin: 

        Returns
        -------

        """

        # check inputs
        if error and (gain is None or readnoise is None):
            self._log.warning('You need to provide "gain" and "readnoise" to compute the error!')
            error = False
        if gain is not None and not isinstance(gain, u.Quantity):
            gain = gain * u.Unit("electron") / u.Unit("adu")
        if readnoise is not None and not isinstance(readnoise, u.Quantity):
            readnoise = readnoise * u.Unit("electron")
        dfilter = {'imagetyp': _base_conf.IMAGETYP_FLAT} if dfilter is None else dfilter
        if dfilter is not None and key_filter is not None and flat_filter is not None:
            dfilter = add_keys_to_dict(dfilter, {key_filter: flat_filter})

        files_list = self._get_file_list(files_list, dfilter,
                                         key_find=key_find, invert_find=invert_find,
                                         abspath=False)

        if not files_list.size > 0:
            return []

        if not self.silent:
            self._log.info('  Creating master flat file: %s' % os.path.basename(mflat_fname))

        lflat = []
        for filename in files_list:

            # read image to a ccd object
            fname = os.path.join(self._tmp_path, filename)
            ccd = CCDData.read(fname, unit=u.Unit("adu"))
            ccd_expt = ccd.header['exptime']

            # trim image and update header
            trimmed = True if trim_section is not None else False
            oscan_corrected = True if oscan_section is not None else False
            ccd = self._trim_image(img_ccd=ccd, obsparams=obsparams,
                                   namps=namps, oscansec=oscan_section, trimsec=trim_section,
                                   trimmed=trimmed, oscan_corrected=oscan_corrected)

            # create an uncertainty map
            if error:
                ccd = ccdproc.create_deviation(ccd, gain=gain,
                                               readnoise=readnoise)
            # gain correct
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
                            mbias = self._convert_fits_to_ccd(mbias, single=True)
                        ccd = ccdproc.subtract_bias(ccd, mbias)

                if isinstance(mdark_ccd, str):
                    mdark_ccd = self._convert_fits_to_ccd(mdark_ccd, single=True)
                ccd = ccdproc.subtract_dark(ccd, mdark_ccd,
                                            exposure_time='exptime',
                                            exposure_unit=u.Unit("second"),
                                            scale=scale)
                add_key_to_hdr(ccd.header, 'MDARK', get_filename(mdark_ccd))
                dark_corrected = True

            if mbias is not None and self._config['BIAS_CORRECT'] and not dark_corrected:
                if isinstance(mbias, str):
                    mbias = self._convert_fits_to_ccd(mbias, single=True)
                ccd = ccdproc.subtract_bias(ccd, mbias)

            # change to float32 to keep the file size under control
            # ccd.data = ccd.data.astype('float32')

            # append the result to a file list for combining
            lflat.append(ccd)

            del ccd

        ccd_mask_fname = os.path.join(self._master_calib_path,
                                      'mask_from_ccdmask_%s_%s.fits' % (flat_filter, self._bin_str))
        if self._telescope == 'DDOTI 28-cm f/2.2':
            ccd_mask_fname = os.path.join(self._master_calib_path,
                                          'mask_from_ccdmask_%s_%s_%s.fits' % (self._instrument,
                                                                               flat_filter, self._bin_str))

        if not self.silent:
            self._log.info('> Creating bad pixel map file: %s' % os.path.basename(ccd_mask_fname))

        ccd_mask = ccdproc.ccdmask(lflat[0],
                                   findbadcolumns=False,
                                   byblocks=False)
        mask_as_ccd = CCDData(data=ccd_mask.astype('uint8'), unit=u.dimensionless_unscaled)
        mask_as_ccd.header['imagetyp'] = 'flat mask'

        mask_as_ccd.write(ccd_mask_fname, overwrite=True)
        self._ccd_mask_fname = ccd_mask_fname

        # combine flat ccds
        combine = ccdproc.combine(lflat, method=method,
                                  mem_limit=_base_conf.MEM_LIMIT_COMBINE,
                                  scale=inv_median,
                                  minmax_clip=True,
                                  minmax_clip_min=0.9,
                                  minmax_clip_max=1.05,
                                  sigma_clip=True, sigma_clip_low_thresh=3,
                                  sigma_clip_high_thresh=3,
                                  sigma_clip_func=np.ma.median,
                                  sigma_clip_dev_func=mad_std,
                                  dtype='float32')

        # update fits header
        if gain is not None and 'GAIN' not in combine.header:
            combine.header.set('GAIN', gain.value, gain.unit)
        if readnoise is not None and 'RON' not in combine.header:
            combine.header.set('RON', readnoise.value, readnoise.unit)
        combine.header['combined'] = True
        combine.header['CGAIN'] = True if gain is not None and self._config['CORRECT_GAIN'] else False
        combine.header['IMAGETYP'] = 'FLAT'
        combine.header['CMETHOD'] = method
        # combine.header['CCDVER'] = VERSION
        if sjoin is not None:
            combine.header['LFLAT'] = sjoin.join([os.path.basename(f) for f in files_list])
        combine.header['NFLAT'] = len(files_list)

        # remove mask and the uncertainty extension
        combine.mask = None
        combine.uncertainty = None

        # change dtype to float32 to keep the file size under control
        combine.data = combine.data.astype('float32')

        # save master bias
        if mflat_fname is not None:
            combine.header['FILENAME'] = os.path.basename(mflat_fname)
            combine.header = ammend_hdr(combine.header)
            combine.write(mflat_fname, overwrite=True)

        # return result
        return combine

    def _create_master_dark(self, files_list, obsparams, mdark_fname=None,
                            exptime=None, mbias=None,
                            trim_section=None, oscan_section=None, namps=1,
                            gain=None, method='average', readnoise=None, error=False,
                            dfilter=None,
                            key_find='find', invert_find=False, sjoin=','):
        """
        Create master dark.

        Parameters
        ----------
        files_list: 
        obsparams: 
        mdark_fname: 
        exptime: 
        trim_section:
        namps: 
        gain:
        method: 
        readnoise: 
        error: 
        dfilter: 
        key_find: 
        invert_find: 
        sjoin: 
        Returns
        -------

        """

        if not self.silent:
            self._log.info('  Creating master dark file: %s' % os.path.basename(mdark_fname))

        # check inputs
        if error and (gain is None or readnoise is None):
            self._log.warning('You need to provide "gain" and "readnoise" to compute the error!')
            error = False
        if gain is not None and not isinstance(gain, u.Quantity):
            gain = gain * u.Unit("electron") / u.Unit("adu")
        if readnoise is not None and not isinstance(readnoise, u.Quantity):
            readnoise = readnoise * u.Unit("electron")
        dfilter = {'imagetyp': _base_conf.IMAGETYP_DARK} if dfilter is None else dfilter

        # get the list with files to reduce
        dfilter = add_keys_to_dict(dfilter, {'exptime': exptime})
        files_list = self._get_file_list(files_list, dfilter,
                                         key_find=key_find, invert_find=invert_find,
                                         abspath=False)
        if not files_list.size > 0:
            return []

        ldark = []
        for filename in files_list:

            # read image to ccd object
            fname = os.path.join(self._tmp_path, filename)
            ccd = CCDData.read(fname, unit=u.Unit("adu"))

            # trim image and update header
            trimmed = True if trim_section is not None else False
            oscan_corrected = True if oscan_section is not None else False
            ccd = self._trim_image(img_ccd=ccd, obsparams=obsparams,
                                   namps=namps, oscansec=oscan_section, trimsec=trim_section,
                                   trimmed=trimmed, oscan_corrected=oscan_corrected)

            # create an uncertainty map
            if error:
                ccd = ccdproc.create_deviation(ccd, gain=gain,
                                               readnoise=readnoise)
            # gain correct
            if gain is not None and self._config['CORRECT_GAIN']:
                ccd = ccdproc.gain_correct(ccd, gain)

            if mbias is not None and self._config['BIAS_CORRECT']:
                if isinstance(mbias, str):
                    mbias = self._convert_fits_to_ccd(mbias, single=True)
                ccd = ccdproc.subtract_bias(ccd, mbias)

            # append the result to a file list for combining
            ldark.append(ccd)
            del ccd

        # combine dark ccds
        combine = ccdproc.combine(ldark, method=method,
                                  mem_limit=_base_conf.MEM_LIMIT_COMBINE,
                                  sigma_clip=True,
                                  sigma_clip_low_thresh=5,
                                  sigma_clip_high_thresh=5,
                                  sigma_clip_func=np.ma.median,
                                  sigma_clip_dev_func=mad_std,
                                  dtype='float32')

        # update fits header
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

        # remove mask and the uncertainty extension
        combine.mask = None
        combine.uncertainty = None

        # change dtype to float32 to keep the file size under control
        combine.data = combine.data.astype('float32')

        # save master bias
        if mdark_fname is not None:
            combine.header['FILENAME'] = os.path.basename(mdark_fname)
            combine.header = ammend_hdr(combine.header)
            combine.write(mdark_fname, overwrite=True)

        # return result
        return combine

    def _create_master_bias(self, files_list, obsparams, mbias_fname=None,
                            trim_section=None, oscan_section=None, namps=1,
                            gain=None, method='average', readnoise=None, error=False,
                            dfilter=None,
                            key_find='find', invert_find=False, sjoin=','):
        """Create master bias file.

        Parameters
        ----------
        files_list: 
            List containing the full fits file paths
        obsparams: dict
            Dictionary with telescope parameter
        mbias_fname: str, path, None, optional
            String containing the absolute path and name of the master bias file.
            Ignored if None. Defaults to None.
        trim_section: str, None, optional
            Sections of the fits image that should be trimmed. Ignored if None. Defaults to None.
        namps: int, optional
            Number of amplifier present in the fits image.
            If namps > 1, the trim is applied to each amplifier and the
             result is re-combined to a ccd object. Defaults to 1.
        gain: float, None, optional
            Detector gain. Usually in el/ADU. Defaults to None.
        method: str, optional
            Method used to combine the single trimmed and corrected bias files.
            Available are ´average´ and ´median´. Defaults to ´average´.
        readnoise: float, None, optional
            Detector readout noise. Typically in electrons.
            If readnoise is None, the readout noise is estimated
            from the standard deviation of a bias image. Defaults to None.
            todo: measure diff between first and last bias and take the
             variation of the difference as measure instead
        error: bool, optional
            If True, an uncertainty map is created.
            Requires a gain and a readout noise. Defaults to False.
        dfilter: dict, None, optional
        key_find: str, optional
        invert_find: bool, optional
        sjoin: str, optional

        Returns
        -------

        """

        if not self.silent:
            self._log.info('  Creating master bias file: %s' % os.path.basename(mbias_fname))

        # check inputs
        if error and (gain is None or readnoise is None):
            self._log.warning('You need to provide "gain" and "readnoise" to compute the error!')
            error = False
        if gain is not None and not isinstance(gain, u.Quantity):
            gain = gain * u.Unit("electron") / u.Unit("adu")
        if readnoise is not None and not isinstance(readnoise, u.Quantity):
            readnoise = readnoise * u.Unit("electron")
        dfilter = {'imagetyp': _base_conf.IMAGETYP_BIAS} if dfilter is None else dfilter

        # get the list with files
        files_list = self._get_file_list(files_list, dfilter,
                                         key_find=key_find, invert_find=invert_find,
                                         abspath=False)
        if not files_list.size > 0:
            return []

        lbias = []
        for filename in files_list:

            # read image to ccd object
            fname = os.path.join(self._tmp_path, filename)
            ccd = CCDData.read(fname, unit=u.Unit("adu"))

            # trim image and update header
            trimmed = True if trim_section is not None else False
            oscan_corrected = True if oscan_section is not None else False
            ccd = self._trim_image(img_ccd=ccd, obsparams=obsparams,
                                   namps=namps, oscansec=oscan_section, trimsec=trim_section,
                                   trimmed=trimmed, oscan_corrected=oscan_corrected)

            # create an uncertainty map
            if error:
                ccd = ccdproc.create_deviation(ccd, gain=gain,
                                               readnoise=readnoise)
            # gain correct
            if gain is not None and self._config['CORRECT_GAIN']:
                ccd = ccdproc.gain_correct(ccd, gain)

            # append the result to a file list for combining
            lbias.append(ccd)
            del ccd

        # combine bias ccds
        combine = ccdproc.combine(lbias, method=method,
                                  mem_limit=_base_conf.MEM_LIMIT_COMBINE,
                                  sigma_clip=True,
                                  sigma_clip_low_thresh=5,
                                  sigma_clip_high_thresh=5,
                                  sigma_clip_func=np.ma.median,
                                  sigma_clip_dev_func=mad_std,
                                  dtype='float32')

        # update fits header
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

        # remove mask and the uncertainty extension
        combine.mask = None
        combine.uncertainty = None

        # change dtype to float32 to keep the file size under control
        combine.data = combine.data.astype('float32')

        # save master bias
        if mbias_fname is not None:
            combine.header['FILENAME'] = os.path.basename(mbias_fname)
            combine.header = ammend_hdr(combine.header)
            combine.write(mbias_fname, overwrite=True)

        # return result
        return combine

    def _trim_image(self, img_ccd, obsparams, namps, oscansec, trimsec, trimmed, oscan_corrected):
        """Trim unwanted areas from image.

        Trim the given image ccd object with fits sections given in ´trim´ and
        depending on the number of amplifier sections.

        Parameters
        ----------
        img_ccd: 
            Image ccd to trim
        obsparams: 
            Observation parameter with reduction information
        namps: 
            Number of amplifiers on the chip.
            If N_amps > 1 trimming is applied to each section, and
            the results are then combined to a new single ccd object
        trimsec:
            Fits section to be trimmed.
        trimmed: 
            Keyword to be added to the header of the trimmed image.
            If trim is applied True, else False

        Returns
        -------
        nccd: 
            New trimmed ccd object for use in other methods

        """
        nccd = img_ccd.copy()
        if namps == 1:
            if oscansec is not None:
                oscan_section = oscansec
                oscan_arr = nccd[oscan_section[0]:oscan_section[1],
                                 oscan_section[2]:oscan_section[3]]
                nccd = ccdproc.subtract_overscan(nccd,
                                                 overscan=oscan_arr,
                                                 add_keyword={'oscan_cor': oscan_corrected},
                                                 median=True, overscan_axis=None)
            if trimsec is not None:
                trim_section = trimsec
                nccd = ccdproc.trim_image(nccd[trim_section[0]:trim_section[1],
                                          trim_section[2]:trim_section[3]],
                                          add_keyword={'trimmed': trimmed})

        else:
            ccd_list = []
            xsize = 0
            ysize = 0
            namps_yx = obsparams['namps_yx'][self._bin_str][namps]

            ni = 0
            for yi in range(namps_yx[0]):
                for xi in range(namps_yx[1]):
                    ampsec_key = obsparams['amplist'][self._bin_str][ni]
                    ampsec = obsparams['ampsec'][self._bin_str][ampsec_key]
                    oscan_section = oscansec[ampsec_key]

                    amp_ccd = ccdproc.trim_image(nccd[ampsec[0]:ampsec[1],
                                                 ampsec[2]:ampsec[3]])

                    oscan_arr = amp_ccd[oscan_section[0]:oscan_section[1],
                                        oscan_section[2]:oscan_section[3]]

                    ncc = ccdproc.subtract_overscan(amp_ccd,
                                                    overscan=oscan_arr,
                                                    median=True, overscan_axis=1)

                    trim_section = trimsec[ampsec_key]
                    ncc = ccdproc.trim_image(ncc[trim_section[0]:trim_section[1],
                                             trim_section[2]:trim_section[3]])

                    xc = np.array([0, ncc.shape[1]], int)
                    yc = np.array([0, ncc.shape[0]], int)
                    if yi == 0:
                        xsize = xsize + ncc.shape[1]
                    else:
                        yc = yc + ncc.shape[0]
                    if xi == 0:
                        ysize = ysize + ncc.shape[0]
                    else:
                        xc = xc + ncc.shape[1]

                    ccd_list.append([(yc, xc), ncc])
                    ni += 1

            # put it all back together
            data = np.zeros((ysize, xsize))
            for i in range(namps):
                y1 = ccd_list[i][0][0][0]
                y2 = ccd_list[i][0][0][1]
                x1 = ccd_list[i][0][1][0]
                x2 = ccd_list[i][0][1][1]
                data[y1:y2, x1:x2] = ccd_list[i][1].data

            # get the unit
            ncc = ccd_list[0][1]
            nccd = ccdproc.CCDData(data, unit=ncc.unit)
            nccd.header = img_ccd.header
            nccd.header['trimmed'] = trimmed
            nccd.header['oscan_cor'] = oscan_corrected

        return nccd

    def _convert_fits_to_ccd(self, lfits: str,
                             key_unit: str = 'BUNIT',
                             key_file: str = 'FILENAME',
                             unit=None, single: bool = False):
        """Convert fits file to ccd object.

        Parameters
        ----------
        lfits: list, dict
            Dictionary or list of dictionaries for conversion to fits-files
        key_unit: str, optional
            Keyword for unit of fits-image stored in the header.
            Defaults to 'BUNIT'.
        key_file: 
            Keyword for file name stored in the header.
        unit: optional
            Fits image unit.
            Defaults to None.
        single: bool, optional
            If True, the input is treated as a single image, else as a list. Defaults to False.

        Returns
        -------
        lccd: list
            List of image ccd objects.
        """
        lccd = []
        if not isinstance(lfits, (tuple, list)):
            lfits = [lfits]
        for fn in lfits:
            fits_unit = unit
            if os.path.exists(fn):
                hdr = fits.getheader(fn)
            else:
                self._log.warning('>>> File "%s" NOT found' % os.path.basename(fn))
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

    @staticmethod
    def _create_dfilter(instrument, obsparams, mode):
        """ Create a filter for ccdproc"""
        key = 'imagetyp' if obsparams['imagetyp'] == 'IMAGETYP' else obsparams['imagetyp']
        def_key = f'imagetyp_{mode}'

        # create data filter for imagetyp and instrument
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

    def _find_calibrations(self, telescope: str, obsparams: dict,
                           date: datetime.date, filters: list,
                           binnings: tuple):
        """Find calibration files and create master calibration file names.

        This method is used to search for the bias, dark and flat files, if they are required,
        copies them to the tmp reduction folder and applies the appropriate reduction method.


        Parameters
        ----------
        telescope: str
            Telescope identifier.
        obsparams: dict
            Dictionary with telescope configuration
        date: 
            Observation date.
        filters:
            List of filters used in the dataset

        Returns
        -------
        _mbias_dict, _mdark_dict, _mflat_dict
        """

        sci_root_path = self._sci_root_path
        tmp_path = self._tmp_path
        suffix = self._suffix
        sci_bin_str = f'{binnings[0]}x{binnings[1]}'

        inst = None
        if self._telescope == 'DDOTI 28-cm f/2.2':
            inst = self._instrument

        # create a range of days before and after the observation night to search for calibrations
        dt_list = list(range(-1 * _base_conf.TIMEDELTA_DAYS, _base_conf.TIMEDELTA_DAYS))
        date_range = np.array([str(date + timedelta(days=i)) for i in dt_list])

        # get base folder for search for calib files
        # fixme: this is not working as intended
        #  The idea is to search for calibration files in the sci folder and if no suitable
        #  calib files are found search for other dates
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

        # create a master calibration folder
        calib_out = Path(f"{search_base_path}/master_calibs/")
        calib_out.mkdir(exist_ok=True)
        self._master_calib_path = calib_out

        # Find all images
        fits_list = Path(search_base_path).rglob(f"*{suffix}")
        fits_list = [f for f in fits_list if '/atmp/' not in str(f)]

        # create an image file collection from the file list
        ic_all = ImageFileCollection(filenames=fits_list)

        # find master calibration files and check if the file exists
        dbias, ddark, dflat, dark_exptimes = self._create_master_fnames(ic_all,
                                                                        obsparams, filters,
                                                                        sci_bin_str)

        # set variables
        self._master_bias_dict = dbias
        self._master_dark_dict = ddark
        self._master_flat_dict = dflat
        self._dark_exptimes = dark_exptimes

        # filter available files according to their image typ
        create_bias = dbias[1]
        if create_bias:
            dfilter, add_filters, regexp = self._create_dfilter(inst, obsparams, 'bias')
            # filter for bias files
            bias_files = self._filter_calib_files(ic_all, obsparams,
                                                  dfilter, add_filters,
                                                  binnings, date, regexp=regexp)

            # copy bias files if no master_bias is available
            if list(bias_files) and self._config['BIAS_CORRECT']:
                new_bias_path = []
                if not self.silent:
                    self._log.info("> Copy raw bias images.")
                for bfile in bias_files:
                    if 'reduced' in bfile or 'master_calibs' in bfile:
                        continue
                    os.system(f"cp -r {bfile} {tmp_path}")
                    file_name = Path(bfile).stem
                    new_bias_path.append(os.path.join(tmp_path, file_name + suffix))

                # prepare bias files
                if not self.silent:
                    self._log.info(f"> Preparing {len(new_bias_path)} ``BIAS`` files for reduction.")
                self._prepare_fits_files(new_bias_path, telescope, obsparams, 'bias')
            else:
                self._config['BIAS_CORRECT'] = False
            del bias_files

        create_dark = np.any(np.array([v[1] for _, v in ddark.items()])) if ddark else False
        if create_dark:
            dfilter, add_filters, regexp = self._create_dfilter(inst, obsparams, 'dark')
            # filter for dark files
            dark_files = self._filter_calib_files(ic_all, obsparams,
                                                  dfilter, add_filters,
                                                  binnings, date, regexp=regexp)

            # copy dark files if no master_dark is available
            if list(dark_files) and self._config['DARK_CORRECT']:
                new_dark_path = []
                if not self.silent:
                    self._log.info("> Copy raw dark images.")
                for dfile in dark_files:
                    if 'reduced' in dfile or 'master_calibs' in dfile:
                        continue
                    os.system(f"cp -r {dfile} {tmp_path}")
                    file_name = Path(dfile).stem
                    new_dark_path.append(os.path.join(tmp_path, file_name + suffix))

                # prepare dark files
                if not self.silent:
                    self._log.info(f"> Preparing {len(new_dark_path)} ``DARK`` files for reduction.")
                self._prepare_fits_files(new_dark_path, telescope, obsparams, 'dark')
            else:
                self._config['DARK_CORRECT'] = False
            del dark_files

        create_flat = np.any(np.array([v[1] for _, v in dflat.items()]))
        if create_flat:

            dfilter, add_filters, regexp = self._create_dfilter(inst, obsparams, 'flat')
            # filter for flat files based on filter
            for filt in filters:
                dfilter[obsparams['filter'].lower()] = filt[1]

                flat_files = self._filter_calib_files(ic_all, obsparams,
                                                      dfilter, add_filters,
                                                      binnings, date, regexp=regexp)

                # copy flat files if no master_flat is available
                if list(flat_files) and self._config['FLAT_CORRECT']:
                    new_flat_path = []
                    if not self.silent:
                        self._log.info("> Copy raw flat images.")
                    for ffile in flat_files:
                        if 'reduced' in ffile or 'master_calibs' in ffile:
                            continue
                        os.system(f"cp -r {ffile} {tmp_path}")
                        file_name = Path(ffile).stem
                        new_flat_path.append(os.path.join(tmp_path, file_name + suffix))

                    # prepare flat files
                    if not self.silent:
                        self._log.info(f"> Preparing {len(new_flat_path)} ``FLAT`` files in"
                                       f" {filt[0]} band for reduction.")
                    self._prepare_fits_files(new_flat_path, telescope, obsparams, 'flat')

                del flat_files

        self._make_mbias = create_bias
        self._make_mdark = create_dark
        self._make_mflat = create_flat

        self._obsparams = obsparams

        del fits_list, ic_all
        gc.collect()

    def _filter_calib_files(self, ic_all, obsparams,
                            dfilter, add_filters,
                            binnings, obs_date=None, regexp=False):
        """ Filter files according to their binning.

        If the first try with a binning keyword fails, use the unbinned image size
        known from a telescope to determine the binning factor
        """

        # Filter with binning keyword
        add_filters_tmp = add_filters.copy()
        if isinstance(obsparams['binning'][0], str) and isinstance(obsparams['binning'][1], str):
            add_filters_tmp[obsparams['binning'][0]] = binnings[0]
            add_filters_tmp[obsparams['binning'][1]] = binnings[1]
        if not add_filters_tmp:
            add_filters = None

        files = self._get_file_list(ic_all, dfilter, key_find='find', regexp=regexp,
                                    invert_find=False, dkeys=add_filters_tmp)
        files = [f for f in files if '/atmp/' not in str(f)]

        if not files:
            if not add_filters_tmp:
                add_filters = None
            files_tmp = self._get_file_list(ic_all, dfilter, key_find='find', regexp=regexp,
                                            invert_find=False, dkeys=add_filters)
            files_tmp = [f for f in files_tmp if '/atmp/' not in str(f)]

            for file_path in files_tmp:
                # load fits file
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
            # load fits file
            hdul = fits.open(file_path, mode='readonly', ignore_missing_end=True)
            hdul.verify('fix')
            prim_hdr = hdul[0].header

            self._log.debug("  Check Obs-Date")
            if 'time-obs'.upper() in prim_hdr:
                time_string = f"{prim_hdr['date-obs'.upper()]}T{prim_hdr['time-obs'.upper()]}"

            else:
                time_string = prim_hdr['date-obs'.upper()]

            # frmt = _base_conf.has_fractional_seconds(time_string)

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

    def _create_master_fnames(self, ic_all, obsparams, filters, binning_ext):
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
        # dbias = []
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
            create_mbias = False if os.path.exists(mbias_path) else True
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
                    create_mdark = False if os.path.exists(mdark_path) else True
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
                create_mflat = False if os.path.exists(mflat_path) else True
                dflat[filt[0]] = (mflat_path, create_mflat)

        del ic_all, obsparams

        return dbias, ddark, dflat, dark_exptimes

    def _get_file_list(self, file_list, dfilter=None, regexp=False, key_find='find',
                       invert_find=False, dkeys=None, abspath=False):
        """Wrapper for _filter_img_file_collection.

        See _filter_img_file_collection() for parameter description.
        """
        if isinstance(file_list, ImageFileCollection):
            file_list = self._filter_img_file_collection(file_list, dfilter, abspath=abspath, regexp=regexp,
                                                         key_find=key_find, invert_find=invert_find, dkeys=dkeys)

        return file_list

    def _prepare_fits_files(self, file_names, telescope, obsparams, imagetyp):
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

            # load fits file
            hdul = fits.open(file_path, mode='readonly', ignore_missing_end=True)
            hdul.verify('fix')
            prim_hdr = hdul[0].header
            new_hdr = prim_hdr.copy()

            # get observation date from header
            match = re.search(r'\d{4}-\d{2}-\d{2}', prim_hdr[obsparams['date_keyword']])
            date = datetime.strptime(match.group(), '%Y-%m-%d').date()
            if date not in dates:
                dates.append(date)

            # setup indices for multi-chip data
            nhdus = len(hdul)
            istart = 0
            if nhdus > 1:
                istart = 1

            # loop from first-data to last HDU
            hduindexes = list(range(nhdus))[istart:]
            for i in hduindexes:
                hdul_subset = hdul[i]
                hdr_subset = hdul_subset.header
                data_subset = hdul_subset.data

                for k, v in hdr_subset.items():
                    if obsparams['keys_to_exclude'] is not None \
                            and k in obsparams['keys_to_exclude']:
                        continue
                    new_hdr[k] = v

                # check the object key in header and set if needed
                if 'OBJECT' not in new_hdr:
                    new_hdr['OBJECT'] = new_hdr[obsparams['object']].strip()

                # check the imagetyp key in header and set
                new_hdr['IMAGETYP'] = imagetyp

                # special case GROND NIR obs
                if i == 0 and telescope == 'GROND_IRIM':
                    filt_arr = np.array(obsparams['cut_order'])

                    for filt in filt_arr:
                        filt_hdr = new_hdr.copy()
                        idx = np.where(filt == filt_arr, True, False)
                        for k in filt_arr[~idx]:
                            [filt_hdr.remove(key) for key, val in filt_hdr.items()
                             if f"{k}_" in key]

                        # update filter keyword
                        filt_key = filt_arr[idx][0]
                        filt_trnsl = obsparams['filter_translations'][filt_key]
                        if filt_trnsl is None:
                            filt_trnsl = 'clear'
                        filt_hdr['FILTER'] = filt_trnsl
                        if filt_trnsl not in filters:
                            filters.append(filt_trnsl)

                        # cut image amplifier regions
                        cut_idx = obsparams['image_cutx'][filt]
                        data_subset2 = data_subset * -1.
                        new_img = data_subset2[:, cut_idx[0]:cut_idx[1]]

                        # update gain and readout noise
                        filt_hdr, file_counter = update_gain_readnoise(new_img, filt_hdr, obsparams,
                                                                       file_counter, filt_trnsl)

                        # update binning
                        filt_hdr = update_binning(filt_hdr, obsparams, binnings)

                        # save result to separate fits file
                        new_fname = f"{fname}_{filt_hdr['FILTER']}{fname_suffix}"
                        new_hdu = fits.PrimaryHDU(data=new_img, header=filt_hdr)
                        new_hdu.writeto(os.path.join(self._tmp_path, new_fname),
                                        output_verify='ignore', overwrite=True)
                    # close header
                    hdul.close()

                    # remove the original file
                    os.system(f"rm -f {file_path}")
                    break

                # update filter
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

                # update binning
                new_hdr = update_binning(new_hdr, obsparams, binnings)

                if i > 0:
                    # update gain and readout noise
                    new_hdr, file_counter = update_gain_readnoise(data_subset, new_hdr,
                                                                  obsparams, file_counter)
                    # remove extension name
                    new_hdr.remove('EXTNAME')

                    # save result to separate fits file
                    new_fname = f"{fname}_{new_hdr['FILTER']}{fname_suffix}"
                    new_hdu = fits.PrimaryHDU(data=data_subset, header=new_hdr)
                    new_hdu.writeto(os.path.join(self._tmp_path, new_fname),
                                    output_verify='ignore', overwrite=True)

                    # remove the original file if done
                    if i == hduindexes[-1]:
                        os.system(f"rm -f {file_path}")
                        hdul.close()
                else:
                    # update gain and readout noise
                    if telescope != 'GROND_OIMG':
                        new_hdr, file_counter = update_gain_readnoise(data_subset, new_hdr,
                                                                      obsparams, file_counter)

                    # save result to separate fits file
                    new_hdu = fits.PrimaryHDU(data=data_subset, header=new_hdr)

                    # os.system(f"ls -la1 {self._tmp_path}")
                    new_hdu.writeto(file_path, output_verify='ignore', overwrite=True)

                    hdul.close()

        return dates[0], filters, binnings

    @staticmethod
    def _filter_img_file_collection(image_file_collection, ldfilter=None,
                                    abspath=False, regexp=False,
                                    key_find='find', invert_find=False,
                                    return_mask=False, key='file',
                                    dkeys=None, copy=True):
        """Filter image file collection object.

        Parameters
        ----------
        image_file_collection: 
        ldfilter: 
        abspath: 
        regexp: 
        key_find: 
        invert_find: 
        return_mask: 
        key: 
        dkeys: 
        copy: 

        Returns
        -------

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

            lfiles = ifc.files_filtered(regex_match=regexp, include_path=abspath, **dfilter)

            lfiles_mask = np.in1d(ifc.summary['file'], lfiles)
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

    @staticmethod
    def find_nearest_dark_exposure(image, dark_exposure_times, tolerance=0.5):
        """
        Find the nearest exposure time of a dark frame to the exposure time of the image,
        raising an error if the difference in exposure time is more than tolerance.

        Parameters
        ----------

        image: astropy.nddata.CCDData
            Image for which a matching dark is needed.

        dark_exposure_times: list
            Exposure times for which there are darks.

        tolerance: float or ``None``, optional
            Maximum difference, in seconds, between the image and the closest dark. Set
            to ``None`` to skip the tolerance test.

        Returns
        -------
        closest_dark_exposure: float
            The Closest dark exposure time to the image.
        """

        dark_exposures = np.array(list(dark_exposure_times))
        idx = np.argmin(np.abs(dark_exposures - image.header['exptime']))
        closest_dark_exposure = dark_exposures[idx]

        if (tolerance is not None and
                np.abs(image.header['exptime'] - closest_dark_exposure) > tolerance):
            raise RuntimeError(f"Closest dark exposure time is {closest_dark_exposure} for "
                               f"flat of exposure time {image.header['exptime']}.")

        return closest_dark_exposure


def compute_2d_background_simple(imgarr: np.ndarray, box_size: int, win_size: int,
                                 bkg_estimator: photutils.background = SExtractorBackground,
                                 rms_estimator: photutils.background = StdBackgroundRMS):
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
        background across the array.  If Background2D fails for any reason, a simpler
        sigma-clipped single-valued array will be computed instead.
    bkg_median:
        The median value (or single sigma-clipped value) of the computed background.
    bkg_rms:
        ND-array the same shape as the input image array which contains the RMS of the
        background across the array.  If Background2D fails for any reason, a simpler
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
        # detect the sources
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
        # create background frame shaped like imgarr populated with sigma-clipped median value
        bkg_background = np.full_like(imgarr, bkg_median)
        # create background frame shaped like imgarr populated with sigma-clipped standard deviation value
        bkg_rms = np.full_like(imgarr, sigcl_std)

    return bkg_background, bkg_median, bkg_rms, bkg_rms_median


def join_path(name: str, directory: str = None):
    """Join path.

    Parameters
    ----------
    name: str
        File name.
    directory: str, optional
        Directory name. Default: None.

    Returns
    -------
    name: str
        Joined path.
    """
    if directory is not None:
        name = os.path.join(directory, os.path.basename(name))
    return name


def update_gain_readnoise(img, hdr, obsparams, file_counter, filt_key=None):
    """ Update gain and readout noise.

    """

    # update gain
    if 'GAIN' not in hdr and obsparams['gain'] is not None:
        if isinstance(obsparams['gain'], str):
            if filt_key is None:
                hdr['GAIN'] = (hdr[obsparams['gain']], 'CCD Gain (e-/ADU)')
            else:
                hdr['GAIN'] = (hdr[obsparams['gain'][filt_key]], 'CCD Gain (e-/ADU)')
        elif isinstance(obsparams['gain'], float):
            hdr['GAIN'] = (obsparams['gain'], 'CCD Gain (e-/ADU)')

    # update readout noise
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
                hdr['IMAGETYP'] == _base_conf.IMAGETYP_BIAS and \
                file_counter == 0:
            bkg_background, bkg_median, bkg_rms, bkg_rms_median = \
                compute_2d_background_simple(img, box_size=11, win_size=5)
            readnoise = bkg_rms_median * hdr['GAIN'] / np.sqrt(2)
            hdr['RON'] = (readnoise, 'CCD Read Out Noise (e-)')
            # file_counter = file_counter + 1
    return hdr, file_counter


def update_binning(hdr, obsparams, binnings):
    """Update header with new binning."""
    binning = get_binning(hdr, obsparams)
    if binning not in binnings:
        binnings.append(binning)
    _bin_str = f'{binning[0]}x{binning[1]}'

    if not (_base_conf.BINNING_DKEYS[0] in hdr and _base_conf.BINNING_DKEYS[1] in hdr):
        hdr[_base_conf.BINNING_DKEYS[0]] = binning[0]
        hdr[_base_conf.BINNING_DKEYS[1]] = binning[1]
    if 'BINNING' not in hdr:
        hdr['binning'] = _bin_str

    return hdr


def get_binning(header, obsparams):
    """ Derive binning from image header.

    Use obsparams['binning'] keywords, unless both keywords are set to 1
    return: tuple (binning_x, binning_y)
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
                # only for RATIR
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
    ccd_file: str, astropy.nddata.CCDData
        String or CCDData object of which the name should be found.
    key: str, optional
        Header keyword to identify file name.
        Only used if the ccd_file is a CCDData object. Default: ``FILENAME``

    Returns
    -------
    name_file: str
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

    hdr: astropy.io.fits.Header
        The FITS header object.
    key: str
        Key to be updated.
    value: object
        Value to be set.

    Returns
    -------
    hdr: astropy.io.fits.Header
        Updated header.
    """
    if value is not None and key is not None:
        hdr[key] = value
    return hdr


def ammend_hdr(header):
    """Remove trailing blank from header.

    Parameters
    ----------
    header: astropy.io.fits.Header
        A FITS header object

    Returns
    -------
    header: astropy.io.fits.Header
        The same fits header object with trailing blanks removed
    """
    if '' in header:
        del header['']
    return header


def add_keys_to_dict(ldict, dkeys, copy=True, force=False):
    """Add dictionary content to the list of dictionaries.

    Parameters
    ----------
    ldict: list, dict
        List of dictionary to which the dict of keys is added.
    dkeys: dict
        Dictionary of keys and values to add to the given list of dictionary
    copy: bool
        Make a deepcopy of input list of dictionary
    force: bool, optional
        Force replacement of key in dictionary if True.
        Default is True.

    Returns
    -------
    ldict: dict
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

    # version check
    _base_conf.check_version(_log)

    ReduceSatObs(input_path=args.input, args=args, silent=args.silent, verbose=args.verbose)


# -----------------------------------------------------------------------------


# standard boilerplate to set 'main' as starting function
if __name__ == '__main__':
    main()
# -----------------------------------------------------------------------------
