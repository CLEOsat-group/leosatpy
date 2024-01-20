#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Name:         dataset.py
# Purpose:      Script to handle given input
#
#
#
#
# Author:       p4adch (cadam)
#
# Created:      04/17/2022
# Copyright:    (c) p4adch 2010-
#
# History:
#
# 17.04.2022
# - file created and basic methods
#
# -----------------------------------------------------------------------------

""" Modules """
import os
import re
import sys
import logging
from inputimeout import (
    inputimeout,
    TimeoutOccurred)

import configparser
import collections

import ccdproc
from datetime import (
    datetime,
    timezone)

import numpy as np
import pandas as pd
from pathlib import Path
from astropy.io import fits
from ccdproc import ImageFileCollection

# Project modules
from . import base_conf as _base_conf
from . import telescope_conf as _tele_conf

# -----------------------------------------------------------------------------

""" Meta-info """
__author__ = "Christian Adam"
__copyright__ = 'Copyright 2021-2023, CLEOSat group'
__credits__ = ["Eduardo Unda-Sanzana, Jeremy Tregloan-Reed, Christian Adam"]
__license__ = "GPL-3.0 license"
__maintainer__ = "Christian Adam"
__email__ = "christian.adam84@gmail.com"
__status__ = "Production"

# -----------------------------------------------------------------------------

# Logging
_log = logging.getLogger(__name__)


class DataSet(object):
    """Initialize and validate given input."""

    def __init__(self, input_args, prefix: str = None,
                 prog_typ: str = "reduceSatObs",
                 prog_sub_typ: list = None,
                 silent: bool = False,
                 log: logging.Logger = _log, log_level: int = 20):
        """ Constructor with default values """

        if prog_sub_typ is None:
            prog_sub_typ = []
        log.setLevel(log_level)

        if not isinstance(input_args, list):
            raise ValueError('Input argument must be a list')

        if len(input_args) > 1:
            if not silent:
                log.info("  Multiple inputs detected")

        # turn prefix and fits suffixes into regular expression
        if prefix is None:
            prefix = ''

        self._datasets = {}
        self._datasets_full = None
        self._obs_data_sorted = []
        self._calib_data_sorted = []
        self._fits_images = []
        self._obsparams = []
        self._instruments = {}
        self._input_args = input_args
        self._config = collections.OrderedDict()
        self._log = log
        self._silent = silent
        self._mode = prog_typ
        self._sub_mode = prog_sub_typ
        self._prefix = prefix
        self._root_dir = _base_conf.ROOT_DIR
        self._s = False

        self._check_input_for_fits()
        self._filter_fits_files()

    def load_config(self):
        """ Load base configuration file """

        config = configparser.ConfigParser()
        config.optionxform = lambda option: option

        # Define the path to the .ini file in the user's home directory
        home = str(Path.home())
        configfile = os.path.join(home, "leosatpy_config.ini")
        try:
            # If the .ini file doesn't already exist in the user's home directory,
            # copy the default one from the package
            if not os.path.exists(configfile):

                import shutil
                default_ini_path = f"{self._root_dir}/leosatpy/leosatpy_config.ini"
                shutil.copy(default_ini_path, configfile)

                self._log.info('> Read configuration')
                self._log.info("  `leosatpy_config.ini` not found in the user's home directory")
                self._log.info(f'  ==> Default config file copied to {configfile}')
                self._log.info('  Please check the config file parameter and values and re-start when ready.')
                sys.exit(0)
            else:
                self._log.info(f'> Read configuration ({configfile})')

        except PermissionError:
            self._log.error(f"  Unable to access or create the .ini "
                            f"file at {configfile} due to insufficient permissions.")
            # handle the error as appropriate for your application
            sys.exit(1)
        except Exception as e:
            self._log.error(f"  An unexpected error occurred: {e}")
            # handle the error as appropriate for your application
            sys.exit(1)

        # read config file
        config.read(configfile)

        conf_groups = ['Reduction', 'Calibration',
                       'Detection', 'Satellite_analysis', 'Plotting']

        if self._mode == 'reduceSatObs':
            conf_groups = ['Reduction']
        if self._mode == 'calibWCS':
            conf_groups = ['Calibration',
                           'Detection', 'Plotting']
        if self._mode == 'satPhotometry':
            conf_groups = ['Detection',
                           'Satellite_analysis', 'Plotting']

        for group in conf_groups:
            items = dict(config.items(group))
            self._config.update(items)
            for key, value in items.items():
                try:
                    val = eval(value)
                except NameError:
                    val = value
                self._config[key] = val

    @property
    def config(self):
        return self._config

    @property
    def instruments(self):
        return self._instruments

    @property
    def instruments_list(self):
        return self._instruments_list

    @property
    def obsparams(self):
        return self._obsparams

    @property
    def datasets(self):
        return self._datasets

    @property
    def datasets_full(self):
        return self._datasets_full

    @property
    def valid_sci_obs(self):
        return self._obs_data_sorted

    @property
    def valid_calib_obs(self):
        return self._calib_data_sorted

    def _exit(self):
        """"""
        self._log.info('Sorry, times up!')
        self._ans = 0
        self._s = True
        raise ValueError

    def get_valid_calib_observations(self, imagetyps_selected: list):
        """"""

        obs_data_sorted = []
        for subset in self._datasets:
            path_arr = []

            for f in subset:
                levels_up = 1
                f_path = Path(f)
                one_up = str(f_path.parents[levels_up - 1])
                root_path = None
                img_sel = None
                for i in imagetyps_selected:
                    r = self.find_parent_folder(one_up, i)
                    if r:
                        root_path = r
                        img_sel = i
                        continue

                path_arr.append([f, f_path.stem, f_path.suffix,
                                 one_up, root_path, img_sel])
            path_arr = np.array(path_arr)

            # make dataframe with [in_path, fname, suffix, parent folder, science_folder]
            obs_df = pd.DataFrame(data=path_arr,
                                  columns=['input', 'file_name', 'suffix',
                                           'file_loc', 'root_path', 'image_typ'])

            # group results by science root path
            grouped_by_root_path = obs_df.groupby('root_path')

            obs_data_sorted.append(grouped_by_root_path)

        self._calib_data_sorted = obs_data_sorted

    def get_valid_sci_observations(self, inst, prog_typ: str = "reduceSatObs"):
        """Search the input argument for valid data"""

        folder_lvl1 = 'reduced'
        folder_lvl2 = 'raw'
        if prog_typ != 'reduceSatObs':
            folder_lvl1 = 'raw'
            folder_lvl2 = 'reduced'

        # exclude files in raw folder in absolute file name and
        # get folder according to assumed folder structure
        path_arr = []
        for f in self._instruments[inst]['dataset']:
            if folder_lvl1 in f:
                continue
            levels_up = 1
            f_path = Path(f)
            one_up = str(f_path.parents[levels_up - 1])
            if folder_lvl2 in one_up:
                levels_up = 2
            path_arr.append([f, f_path.stem, f_path.suffix,
                             one_up, str(f_path.parents[levels_up - 1])])
        path_arr = np.array(path_arr)

        # make dataframe with [in_path, fname, suffix, parent folder, science_folder]
        obs_df = pd.DataFrame(data=path_arr,
                              columns=['input', 'file_name', 'suffix',
                                       'file_loc', 'root_path'])

        # group results by science root path
        grouped_by_root_path = obs_df.groupby('root_path')

        self._obs_data_sorted = grouped_by_root_path

    def _check_input_for_fits(self):
        """ Check the given input and identify fits files. """

        log = self._log
        fits_image_filenames = []

        # set file extension
        regex_ext = re.compile('^' + self._prefix + '.*(fits|FITS|fit|FIT|Fits|fts|FTS)$')

        # loop input arguments and add found fits files to an image list
        for name in self._input_args:
            if os.path.isdir(name):
                name = Path(name).expanduser()
                if not self._silent:
                    log.info("  Directory detected. Searching for .fits file(s).")
                path = os.walk(name)
                for root, dirs, files in path:
                    f = sorted([s for s in files if re.search(regex_ext, s)])
                    if len(f) > 0:

                        # [fits_image_filenames.append(Path(os.path.join(root, s))) for s in f
                        #  if ('master' not in str(s) or 'combined' not in str(s))
                        #  and 'atmp' not in root
                        #  and 'flat' not in str(s) and 'flat' not in root]
                        for s in f:
                            if ('master' not in str(s) or 'combined' not in str(s)) \
                                    and 'atmp' not in root and 'auxiliary' not in root:
                                if self._mode != 'reduceCalibObs':
                                    if 'flat' not in str(s) and 'flat' not in root:
                                        fits_image_filenames.append(Path(os.path.join(root, s)))
                                else:
                                    if 'calibrated' not in root \
                                            and 'reduced' not in root \
                                            and 'master_calibs' not in root:
                                        fits_image_filenames.append(Path(os.path.join(root, s)))

            elif not os.path.isdir(name) \
                    and not os.path.isfile(name) \
                    and not re.search(regex_ext, name):
                if not self._silent:
                    log.info("  Given input does not exist. "
                             "Searching for object(s) in argument.")
                p = Path(name).expanduser()
                _prefix = p.name
                if self._mode == 'reduceCalibObs':
                    _prefix = ''
                regex = re.compile('^' + _prefix + '.*(fits|FITS|fit|FIT|Fits|fts|FTS)$')
                path = os.walk(p.parent)

                for root, dirs, files in path:

                    if self._mode == 'reduceCalibObs':
                        if p.name not in root:
                            continue
                    f = sorted([s for s in files if re.search(regex, s)])

                    if len(f) > 0:
                        for s in f:
                            if ('master' not in str(s) or 'combined' not in str(s)) \
                                    and 'atmp' not in root and 'auxiliary' not in root:
                                if self._mode != 'reduceCalibObs':
                                    if 'flat' not in str(s) and 'flat' not in root:
                                        fits_image_filenames.append(Path(os.path.join(root, s)))
                                else:
                                    if 'calibrated' not in root or 'reduced' not in root:
                                        fits_image_filenames.append(Path(os.path.join(root, s)))

            elif os.path.isfile(name) and re.search(regex_ext, name):
                if self._mode != 'reduceCalibObs':
                    log.debug("  Single FITS-file detected.")
                    fits_image_filenames.append(Path(name).expanduser())
                else:
                    log.error("  Single FITS-file detected. "
                              "Reduction of calibration files not possible. "
                              "Please check the input.")
                    sys.exit()
            else:
                log.error("  NO FITS-file(s) detected. "
                          "Please check the input.")
                sys.exit()

        self._fits_images = fits_image_filenames

    def _filter_fits_files(self):
        """ Filter data for valid science fits files suitable for reduction.

        Returns
        -------
        result: Table
            Filtered dataset(s).
            Contains folder and files suitable for reduction.
        """

        mode = self._mode
        log = self._log
        filenames = self._fits_images
        if len(filenames) == 0:
            log.error('Cannot find any data. This should not happen. '
                      'Please check the input!!!')
            sys.exit()

        # create a file collection with all header data
        data = ImageFileCollection(filenames=filenames)

        # check if instruments in found fits files are supported
        instruments_list = self._get_instruments(filenames)
        instruments_list = sorted(instruments_list)
        inst_dict = collections.OrderedDict({k: {} for k in instruments_list})

        dataset_list = []
        for i in range(len(instruments_list)):
            inst = instruments_list[i]
            tel_id = _tele_conf.INSTRUMENT_IDENTIFIERS[inst]
            obsparams = _tele_conf.TELESCOPE_PARAMETERS[tel_id]
            telescope = obsparams['telescope_keyword']

            inst_dict[inst]['telescope'] = telescope
            inst_dict[inst]['obsparams'] = obsparams
            # filter fits files by science image type
            # todo: implement here a possibility to only use flats,
            #  bias or dark keywords maybe?
            add_filters = {"telescop": telescope}
            if telescope == 'CBNUO-JC':
                add_filters = {'observat': telescope}
            if telescope in ['CTIO 0.9 meter telescope', 'CTIO 4.0-m telescope']:
                add_filters = {'observat': 'CTIO'}
            # if telescope == 'CTIO 4.0-m telescope':
            #     add_filters = {'observat': 'CTIO'}

            add_filters[obsparams['instrume'].lower()] = inst

            if mode == 'reduceCalibObs':
                combos = _base_conf.IMAGETYP_COMBOS
                image_typs = [combos[s] for s in self._sub_mode]
                imagetyp = '|'.join(image_typs)
            else:
                imagetyp = _base_conf.IMAGETYP_COMBOS['light']

            if telescope == 'DDOTI 28-cm f/2.2':
                add_filters['exptype'] = imagetyp
            elif telescope == 'CTIO 4.0-m telescope':
                add_filters['obstype'] = imagetyp
            else:
                add_filters["imagetyp"] = imagetyp

            _files = data.filter(regex_match=True, **add_filters)

            if _files.summary is None or not _files.summary:
                log.error("Given input does NOT contain valid FITS-file(s). "
                          "Please check input.")
                sys.exit()

            # filter again depending on program mode
            if mode == 'reduceSatObs' or mode == 'reduceCalibObs':
                add_filters = {"combined": None, "ast_cal": None}
            elif mode == 'calibWCS':
                add_filters = {"combined": True, "ast_cal": None}
            elif mode == 'satPhotometry':
                if telescope == 'CTIO 4.0-m telescope':
                    add_filters = {"combined": None, "wcscal": 'successful'}
                else:
                    add_filters = {"combined": True, "ast_cal": True}
            else:
                log.error("Mode must be either `reduceSatObs`, `reduceCalibObs`, "
                          "`calibWCS` or `satPhotometry`. "
                          "Please check input.")
                sys.exit()

            _files = _files.filter(regex_match=True, **add_filters)
            if _files.summary is None:
                log.error("NO valid science FITS-file(s) found. "
                          "Please check your input.")
                sys.exit()
            if not self._silent:
                log.info(f'  ==> Found a total of {len(_files.summary)} valid FITS-file(s) '
                         f'for instrument: {inst} @ telescope: {telescope}')

            if mode != 'satPhotometry':
                dataset_list.append(_files.summary['file'].data)
                inst_dict[inst]['dataset'] = _files.summary['file'].data
            else:
                dataset_list.append(self._grouped_by_pointing(_files, inst))
                inst_dict[inst]['dataset'] = self._grouped_by_pointing(_files, inst)

        self._instruments = inst_dict
        self._instruments_list = instruments_list

    def _get_instruments(self, filenames: list):
        """ Identify the instrument and crosscheck with supported instruments.

        """
        log = self._log

        instruments = []
        for idx, filename in enumerate(filenames):
            try:
                hdulist = fits.open(filename, ignore_missing_end=True)
                hdulist.verify('silentfix+ignore')
            except IOError:
                log.error('cannot open file %s' % filename)
                self._fits_images.pop(idx)
                continue

            # open fits header and extract instrument name
            header = hdulist[0].header
            for key in _base_conf.INSTRUMENT_KEYS:
                if key in header:
                    inst = header[key]
                    if inst == 'GROND':
                        det_keywords = _tele_conf.INSTRUMENT_IDENTIFIERS['GROND']
                        [instruments.append(det_keywords[i])
                         for i in det_keywords.keys() if i in header.keys()]
                    else:
                        instruments.append(inst)
                    hdulist.close()
                    break
                else:
                    hdulist.close()

        if len(instruments) == 0:
            log.error('Cannot identify any supported telescope/instrument. Please update'
                      '_base_conf.INSTRUMENT_KEYS accordingly.')
            sys.exit()

        return list(set(instruments))

    @staticmethod
    def _grouped_by_pointing(data: ccdproc.ImageFileCollection, inst):
        """Group files by telescope pointing"""

        # convert to pandas dataframe
        df = data.summary.to_pandas()

        # use only the current instrument
        df = df[df['instrume'] == inst]

        # sort filenames
        df.sort_values(by='file', inplace=True)

        def to_bin(x):
            if inst == 'DECam':
                step = 8.333e-2  # 300 arcsecond
            else:
                step = 3.333e-2  # 60 arcsecond
            return np.floor(x / step) * step

        if inst == 'DECam':

            df['RA_bin'] = df.centra.map(to_bin)
            df['DEC_bin'] = df.centdec.map(to_bin)
        else:

            df['RA_bin'] = df.crval1.map(to_bin)
            df['DEC_bin'] = df.crval2.map(to_bin)

        # group files by selected bins
        by_dist = df.groupby(["RA_bin", "DEC_bin"])

        return by_dist

    @staticmethod
    def find_parent_folder(in_folder: str, imagetyp_sel: str):
        """"""
        parent_folder = in_folder
        folder = None
        removed = -1
        while folder != parent_folder:  # Stop if we hit the file system root
            folder = parent_folder
            removed += 1
            with os.scandir(folder) as ls:
                for f in ls:
                    if f.name == imagetyp_sel and f.path in in_folder:
                        return Path(f.path).parent
            parent_folder = os.path.normpath(os.path.join(folder, os.pardir))

    @staticmethod
    def join_path(name: str, directory: Path = None):
        """Simple method to join paths"""
        if directory is not None:
            name = os.path.join(directory, os.path.basename(name))
        return name

    @staticmethod
    def get_time_stamp() -> str:
        """
        Returns time stamp for now: "2021-10-09 16:18:16"
        """

        now = datetime.now(tz=timezone.utc)
        time_stamp = f"{now:%Y-%m-%d_%H_%M_%S}"

        return time_stamp

    @staticmethod
    def select_file_from_list(data: list) -> tuple:
        """
        Select a value from the data list.

        Parameters
        ----------
        data: list
            List with values from which to select.

        Returns
        -------
         : list
            Selected value
        """

        # Initialize logging for this user-callable function
        _log.setLevel(logging.getLevelName(_log.getEffectiveLevel()))

        x = '0-' + str(len(data) - 1)
        y = zip(data, [str(i) for i in range(len(data))])
        for u, v in y:
            _log.info("    %s" % ' --> '.join((u, v)))

        c = 1
        ans = 0
        s = False

        while not s and c <= 3:
            try:
                title = _base_conf.BCOLORS.OKGREEN \
                        + '[  SELECT] Select folder/file [' + x + '] (default=0): ' \
                        + _base_conf.BCOLORS.ENDC
                # _log.info('  Select folder [' + x + '] (default=0):')
                ans = inputimeout(prompt=title, timeout=_base_conf.DEF_TIMEOUT)
                # ans = int(input(title))
                ans = int(ans)
                _ = data[ans]
                s = True
            except TimeoutOccurred:
                _log.info(" > Time is up.")
                ans = 0
                _log.info(f"   Auto-set to first entry: {data[int(ans)]}")
                s = True
            except KeyboardInterrupt:
                _log.warning("Interrupted by user")
                sys.exit()
            except IndexError:
                _log.error(f"Index Error ... "
                           f"selected index not available.. Try again ({c:d}/3).")
                s = False
                if c == 3:
                    _log.warning("3 times wrong ... Auto-set to first entry.")
                    ans = 0
                    s = True
                c += 1
            except ValueError:
                ans = 0
                _log.info(f" > Auto-set to first entry: {data[int(ans)]}")
                s = True

        idx = int(ans)
        return data[idx], idx
