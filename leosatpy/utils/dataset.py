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
from inputimeout import (inputimeout,
                         TimeoutOccurred)
import time
import threading
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
from . import base_conf as bc
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

    def __init__(self, input_args,
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

        self._datasets = {}
        self._datasets_full = None
        self._obs_data_sorted = []
        self._calib_data_sorted = []
        self._fits_images = []
        self._obsparams = []
        self._instruments = {}
        self._instruments_list = []
        self._input_args = input_args
        self._config = collections.OrderedDict()
        self.log = log
        self.silent = silent
        self.prog_mode = prog_typ
        self._sub_mode = prog_sub_typ
        self._root_dir = bc.ROOT_DIR
        self._s = False

        self.check_input_for_fits_files()
        self.filter_fits_files()

    def load_config(self):
        """ Load base configuration file """

        config = configparser.ConfigParser()
        config.optionxform = lambda option: option

        # Define the path to the .ini file in the user's home directory
        home = str(Path.home())
        configfile = os.path.join(home, "leosatpy_config.ini")
        try:
            # If the .ini file doesn't already exist in the user's home directory,
            # Copy the default one from the package
            if not os.path.exists(configfile):

                import shutil
                default_ini_path = f"{self._root_dir}/leosatpy/leosatpy_config.ini"
                shutil.copy(default_ini_path, configfile)

                self.log.info('> Read configuration')
                self.log.info("  `leosatpy_config.ini` not found in the user's home directory")
                self.log.info(f'  ==> Default config file copied to {configfile}')
                self.log.info('  Please check the config file parameter and values and re-start when ready.')
                sys.exit(0)
            else:
                self.log.info(f'> Read configuration ({configfile})')

        except PermissionError:
            self.log.error(f"  Unable to access or create the .ini "
                           f"file at {configfile} due to insufficient permissions.")
            # Handle the error as appropriate for your application
            sys.exit(1)
        except Exception as e:
            self.log.error(f"  An unexpected error occurred: {e}")
            # Handle the error as appropriate for your application
            sys.exit(1)

        # Read config file
        config.read(configfile)

        conf_groups = ['Reduction', 'Calibration',
                       'Detection', 'Satellite_analysis', 'Plotting']

        if self.prog_mode == 'reduceSatObs':
            conf_groups = ['Reduction']
        if self.prog_mode == 'calibWCS':
            conf_groups = ['Calibration',
                           'Detection', 'Plotting']
        if self.prog_mode == 'satPhotometry':
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

    def get_valid_calib_observations(self, imagetyps_selected: list):
        """

        Parameters
        ----------
        imagetyps_selected

        Returns
        -------

        """
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

            # Make dataframe with [in_path, fname, suffix, parent folder, science_folder]
            obs_df = pd.DataFrame(data=path_arr,
                                  columns=['input', 'file_name', 'suffix',
                                           'file_loc', 'root_path', 'image_typ'])

            # Group results by science root path
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

        # Exclude files in raw folder in absolute file name and
        # Get folder according to assumed folder structure
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

        # Make dataframe with [in_path, fname, suffix, parent folder, science_folder]
        obs_df = pd.DataFrame(data=path_arr,
                              columns=['input', 'file_name', 'suffix',
                                       'file_loc', 'root_path'])

        # Group results by science root path
        grouped_by_root_path = obs_df.groupby('root_path')

        self._obs_data_sorted = grouped_by_root_path

    def check_input_for_fits_files(self):
        """ Check the given input and identify fits files."""

        log = self.log
        fits_image_filenames = []
        regex_ext = re.compile(r'.*\.(fits|fit|fts)$', re.IGNORECASE)

        for name in self._input_args:
            path_arg = Path(name).expanduser()

            if path_arg.is_dir():
                if not self.silent:
                    log.info(f"  Directory detected: {path_arg}. Searching for .fits file(s).")
                fits_image_filenames.extend(self.process_directory_recursive(path_arg, regex_ext))

            elif path_arg.is_file() and re.search(regex_ext, path_arg.name):
                if self.prog_mode != 'reduceCalibObs':
                    log.debug("  Single FITS-file detected.")
                    fits_image_filenames.append(path_arg)
                else:
                    log.error("  Single FITS-file detected. Reduction of calibration files not possible.")
                    sys.exit()
            elif path_arg.parent.is_dir() and not path_arg.is_dir():
                parent_folder = path_arg.parent
                name_part = path_arg.name

                # Check if name_part matches subdirectories in parent_folder
                matching_dirs = [
                    d for d in os.listdir(parent_folder)
                    if name_part in d and (parent_folder / d).is_dir()
                ]

                if matching_dirs:
                    if not self.silent:
                        log.info(f"  Matching directories found for prefix '{name_part}' in {parent_folder}")

                    for match in matching_dirs:
                        target_folder = parent_folder / match
                        fits_image_filenames.extend(
                            self.process_directory_recursive(target_folder, regex_ext))

                # No matching dirs, treat as file prefix
                else:
                    if not self.silent:
                        log.info(
                            f"  No matching subdirectories. "
                            f"Searching for files with prefix '{name_part}' in {parent_folder}")

                    regex_with_prefix = re.compile(r'^' + re.escape(name_part) + r'.*\.(fits|fit|fts)$',
                                                   re.IGNORECASE)
                    fits_image_filenames.extend(
                        self.process_directory_recursive(parent_folder, regex_with_prefix))
            else:
                 if not self.silent:
                    log.error(f"  Path not found: {path_arg}. No matching directories or files.")
                    sys.exit()

        self._fits_images = fits_image_filenames


    def process_directory_recursive(self, path, regex):
        """
        Recursively process a directory to identify FITS files, optionally filtering
        by subdirectory name containing the prefix.

        Parameters
        ----------
        path : Path
            The base directory to start the search from.
        regex : re.Pattern
            Compiled regular expression to match FITS file extensions.

        Returns
        -------
        list of Path
            List of paths to valid FITS files found in the directory.
        """
        fits_files = []
        for root, dirs, files in os.walk(path):
            # Filter and validate FITS files
            for s in sorted(files):
                if re.search(regex, s) and self.is_valid_fits_file(s, root):
                    fits_files.append(Path(os.path.join(root, s)))
        return fits_files


    def is_valid_fits_file(self, filename, root):
        """
        Check if the given FITS file is valid.

        Parameters
        ----------
        filename : str
            Name of the FITS file to validate.
        root : str
            Directory path where the file is located.

        Returns
        -------
        bool
            True if the file is valid, False otherwise.
        """
        if ('master' not in filename or 'combined' not in filename) and \
                'atmp' not in root and 'auxiliary' not in root:
            if self.prog_mode != 'reduceCalibObs':
                # Exclude flats in normal mode
                return 'flat' not in filename and 'flat' not in root
            else:
                # In reduction mode, exclude further directories
                return 'calibrated' not in root and 'reduced' not in root and 'master_calibs' not in root
        return False

    def filter_fits_files(self):
        """ Filter data for valid science fits files suitable for reduction.

        Returns
        -------
        result: Table
            Filtered dataset(s).
            Contains folder and files suitable for reduction.
        """
        mode = self.prog_mode
        log = self.log
        filenames = self._fits_images
        if len(filenames) == 0:
            log.error('Cannot find any data. This should not happen. '
                      'Please check the input!!!')
            sys.exit()

        # Create a file collection with all header data
        data = ImageFileCollection(filenames=filenames)

        # Check if instruments in found fits files are supported
        instruments_list = self.get_instruments(filenames)
        instruments_list = sorted(instruments_list)
        inst_dict = collections.OrderedDict({k: {} for k in instruments_list})

        for i in range(len(instruments_list)):
            inst_val = instruments_list[i]

            tel_id = _tele_conf.INSTRUMENT_IDENTIFIERS[inst_val]
            obsparams = _tele_conf.TELESCOPE_PARAMETERS[tel_id]
            telescope = obsparams['telescope_keyword']

            inst_dict[inst_val]['telescope'] = telescope
            inst_dict[inst_val]['obsparams'] = obsparams

            inst_key = obsparams['instrume'].lower()

            # Filter fits files by science image type
            add_filters = {"telescop": telescope}
            if telescope == 'CBNUO-JC':
                add_filters = {'observat': telescope}
            if telescope in ['CTIO 0.9 meter telescope', 'CTIO 4.0-m telescope']:
                add_filters = {'observat': 'CTIO'}
            if telescope == 'FUT':
                add_filters = {'observat': 'Mt. Kent'}

            add_filters[inst_key] = inst_val

            if mode == 'reduceCalibObs':
                combos = bc.IMAGETYP_COMBOS
                image_typs = [combos[s] for s in self._sub_mode]
                imagetyp = '|'.join(image_typs)
            else:
                imagetyp = bc.IMAGETYP_COMBOS['light']

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

            # Filter again depending on program mode
            if mode == 'reduceSatObs' or mode == 'reduceCalibObs':
                add_filters = {"combined": None, "ast_cal": None}
            elif mode == 'calibWCS':
                if telescope == 'FUT':
                    add_filters = {"combined": None, "wcs-stat": 'ok'}
                else:
                    add_filters = {"combined": True, "ast_cal": None}
            elif mode == 'satPhotometry':
                if telescope == 'CTIO 4.0-m telescope':
                    add_filters = {"combined": None, "wcscal": 'successful'}
                elif telescope == 'FUT':
                    add_filters = {"combined": None, "wcs-stat": 'ok'}
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
            if not self.silent:
                log.info(f'  ==> Found a total of {len(_files.summary)} valid FITS-file(s) '
                         f'for instrument: {inst_val} @ telescope: {telescope}')

            if mode != 'satPhotometry':
                inst_dict[inst_val]['dataset'] = _files.summary['file'].data
            else:
                grouped_by_pointing = self.grouped_by_pointing(_files, inst_val, inst_key)
                inst_dict[inst_val]['dataset'] = grouped_by_pointing

        self._instruments = inst_dict
        self._instruments_list = instruments_list

    def get_instruments(self, filenames: list):
        """ Identify the instrument and crosscheck with supported instruments.

        """
        log = self.log

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
            for key in bc.INSTRUMENT_KEYS:
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
                      'bc.INSTRUMENT_KEYS accordingly.')
            sys.exit()

        return list(set(instruments))

    def _exit(self):
        """"""
        self.log.info('Sorry, time\'s up!')
        self._ans = 0
        self._s = True
        raise ValueError

    @staticmethod
    def grouped_by_pointing(data: ccdproc.ImageFileCollection, inst_val, inst_key):
        """Group files by telescope pointing"""

        # Convert to pandas dataframe
        df = data.summary.to_pandas()

        # use only the current instrument
        df = df[df[inst_key] == inst_val]

        # sort filenames
        df.sort_values(by='file', inplace=True)

        def to_bin(x):
            if inst_val == 'DECam':
                step = 8.333e-2  # 300 arcsecond
            else:
                step = 3.333e-2  # 60 arcsecond
            return np.floor(x / step) * step

        if inst_val == 'DECam':

            df['RA_bin'] = df.centra.map(to_bin)
            df['DEC_bin'] = df.centdec.map(to_bin)
        else:

            df['RA_bin'] = df.crval1.map(to_bin)
            df['DEC_bin'] = df.crval2.map(to_bin)

        # Group files by selected bins
        by_dist = df.groupby(["RA_bin", "DEC_bin"])

        return by_dist

    @staticmethod
    def find_parent_folder(in_folder: str, imagetyp_sel: str):
        """

        Parameters
        ----------
        in_folder
        imagetyp_sel

        Returns
        -------

        """
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
    def select_file_from_list(data: list,
                              timeout: int = 5,
                              max_attempts: int = 3,
                              ans: int = 0,
                              indent: int = 3) -> tuple:
        """
        Prompts user to select an item from a list, with error handling and auto-selection.

        Parameters
        ----------
        data : list
            A list of items to select from.
        timeout : int, optional
            Timeout in seconds for user input. Set to -1 to disable timeout (Default is 5s).
        max_attempts : int, optional
            Maximum number of attempts for user input (Default is 3).
        ans : int, optional
            Default answer index (Default is 0).
        indent : int, optional
            Number of initial whitespaces (Default is 3)

        Returns
        -------
        tuple
            Selected item and its index.
        """
        # Ensure logger is active at current effective level
        _log.setLevel(logging.getLevelName(_log.getEffectiveLevel()))

        # Display selection options
        for idx, item in enumerate(data):
            _log.info(f"{' ':<{indent}}{item} --> {idx}")


        valid_selection = False

        timeout_reached = threading.Event()
        select_str = f"[{bc.BCOLORS.OKGREEN}  SELECT{bc.BCOLORS.ENDC}]"

        def countdown_timer(timeout_val, stop_event, N):
            interval = 0.01
            total_steps = int(timeout_val / interval)
            for step in range(total_steps, 0, -1):
                if stop_event.is_set():
                    break
                remaining = int((step - 1) * interval) + 1
                title_str = (f"{select_str}{bc.BCOLORS.OKGREEN}{' ':<{indent}} "
                             f"Select file/folder [0-{N - 1}] (default=0) "
                             f"[Time left:{remaining:>3}s]: {bc.BCOLORS.ENDC}")
                sys.stdout.write(f"\r{title_str}")
                sys.stdout.flush()
                time.sleep(interval)
            if not stop_event.is_set():
                timeout_reached.set()

        for attempt in range(1, max_attempts + 1):
            stop_timer = threading.Event()
            try:
                title = (
                    f"{select_str}{bc.BCOLORS.OKGREEN}{' ':<{indent}} "
                    f"Select file/folder [0-{len(data) - 1}] (default=0)"
                    f" [Time left: no limit]: {bc.BCOLORS.ENDC}")
                if timeout == -1:
                    sys.stdout.write(f"{title}")
                    sys.stdout.flush()
                    ans = int(input())
                else:
                    timer_thread = threading.Thread(target=countdown_timer,
                                                    args=(timeout, stop_timer,
                                                          len(data)))
                    timer_thread.start()
                    ans = int(inputimeout(prompt="", timeout=timeout))
                    stop_timer.set()
                    timer_thread.join()
                    sys.stdout.write("\r\033[K")  # Clear the line after input
                    if timeout_reached.is_set():
                        raise TimeoutOccurred
                if ans in range(len(data)):
                    valid_selection = True
                    break
                else:
                    _log.error(f"{' ':<{indent}} Invalid selection. "
                               f"Attempt {attempt}/{max_attempts}.")
            except TimeoutOccurred:
                title = f"{' ':<{indent}}Time has expired! Using first entry."
                stop_timer.set()
                sys.stdout.write("\r\033[K")  # Clear the line after error
                _log.warning(title)
                break
            except (ValueError, IndexError):
                stop_timer.set()
                sys.stdout.write("\r\033[K")  # Clear the line after error
                _log.error(f"{' ':<{indent}} Invalid selection. "
                           f"Attempt {attempt}/{max_attempts}.")
            except KeyboardInterrupt:
                stop_timer.set()
                sys.stdout.write("\r\033[K")  # Clear the line after interruption
                _log.warning("Interrupted by user.")
                sys.exit()

        if not valid_selection:
            ans = 0

        return data[ans], ans


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

    return hdr, _bin_str


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
