#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Name:         tables.py
# Purpose:      Helper class to handle tables such as the result table or
#               tle tables
#
#
#
# Author:       p4adch (cadam)
#
# Created:      04/19/2022
# Copyright:    (c) p4adch 2010-
#
# History:
#
# 19.04.2022
# - file created and basic methods
# fixme: add methods for creating and loading a backup of the result
#  table. Include choice for loading the backup if available and no result table
#  is available or it was parsed as argument
# -----------------------------------------------------------------------------

""" Modules """
import csv
import datetime
import gc
import logging
import os
import re
from collections import OrderedDict
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from packaging.version import Version

# Project modules
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

__taskname__ = 'tables'
# -----------------------------------------------------------------------------


# Logging
_log = logging.getLogger(__name__)

frmt = "%Y-%m-%dT%H:%M:%S.%f"


class ObsTables(object):
    """Initialize and validate given input."""

    def __init__(self, config, silent=False, log=_log):
        """ Constructor with default values """
        log.setLevel(logging.getLevelName(log.getEffectiveLevel()))

        self.log = log
        self.silent = silent
        self._obs_info = pd.DataFrame()
        self._obj_info = pd.DataFrame()
        self._sat_info = pd.DataFrame()
        self._roi_info = pd.DataFrame()
        self._roi_data = pd.DataFrame()
        self._ext_oi_info = pd.DataFrame()
        self._ext_oi_data = pd.DataFrame()
        self._glint_mask_info = pd.DataFrame()
        self._glint_mask_data = pd.DataFrame()
        self._faint_trail_info = pd.DataFrame()
        self._faint_trail_data = pd.DataFrame()
        self._vis_info = pd.DataFrame()

        self.def_path = Path(config['WORKING_DIR_PATH']).expanduser().resolve()
        self.def_tbl_name = config['RESULT_TABLE_NAME']
        self.def_roi_name = config['ROI_TABLE_NAME']
        self.def_ext_oi_name = config['EXT_OI_TABLE_NAME']
        self.def_glint_mask_name = config['GLINT_MASK_TABLE_NAME']
        self.def_faint_trail_name = config['FAINT_TRAIL_TABLE_NAME']

        self.res_table_fname = self.def_path / self.def_tbl_name
        self.roi_table_fname = self.def_path / self.def_roi_name
        self.ext_oi_table_fname = self.def_path / self.def_ext_oi_name
        self.glint_mask_table_fname = self.def_path / self.def_glint_mask_name
        self.faint_trail_table_fname = self.def_path / self.def_faint_trail_name

        self.def_cols = bc.DEF_RES_TBL_COL_NAMES
        self.def_col_units = bc.DEF_RES_TBL_COL_UNITS
        self.def_key_transl = bc.DEF_KEY_TRANSLATIONS
        self.res_tbl_data = pd.DataFrame(columns=self.def_cols)
        self.res_tbl_units = OrderedDict(zip(self.def_cols,
                                             self.def_col_units[:len(self.def_cols)]))
        self.res_tbl_unit_df = pd.DataFrame([self.res_tbl_units])

    @property
    def obs_info(self):
        return self._obs_info

    @property
    def faint_trail_data(self):
        return self._faint_trail_data

    @property
    def glint_mask_data(self):
        return self._glint_mask_data

    @property
    def roi_data(self):
        return self._roi_data

    @property
    def ext_oi_data(self):
        return self._ext_oi_data

    @property
    def obj_info(self):
        return self._obj_info

    @property
    def sat_info(self):
        return self._sat_info

    @property
    def vis_data(self):
        return self._vis_info

    def find_tle_file(self, sat_name: str,
                      img_filter: str,
                      src_loc: str):
        """ Check for TLE file location """
        level_up = 0
        datetime_in_folder = ('morning' in src_loc or 'evening' in src_loc)
        filter_in_folder = f"/{img_filter}" in src_loc
        if filter_in_folder:
            level_up = 1
            if datetime_in_folder:
                level_up = 2
        else:
            if datetime_in_folder:
                level_up = 1
        search_path = Path(src_loc).parents[level_up]

        # Get times
        date_obs = self._obj_info['Date-Obs'].values[0]
        obs_date = date_obs.split('-')

        # Get satellite type
        sat_type = self.get_sat_type(sat_name)

        tle_filenames = []
        regex_pattern = rf'(?i)tle_{sat_type}_{obs_date[0]}[-_]{obs_date[1]}[-_]{obs_date[2]}.*\.txt'
        regex = re.compile(regex_pattern, re.IGNORECASE)

        path = os.walk(search_path)
        all_files = []

        for root, dirs, files in path:
            matched_files = [s for s in files if re.search(regex, s)]
            for s in matched_files:
                tle_filenames.append((root, s, Path(os.path.join(root, s))))

            all_files.extend((root, s, Path(os.path.join(root, s))) for s in files)

        def select_tle_file(file_names):
            if len(file_names) == 1:
                return file_names[0]

            idx_arr = np.array([i[1].find('unique') for i in file_names])
            unique_idx = np.where(idx_arr == 0)[0]
            if unique_idx.size > 1:
                self.log.warning("Multiple unique TLE files found. "
                                 "This really should not happen!!! TBW: select")
            elif unique_idx.size == 1:
                return files[unique_idx[0]]

            non_unique_idx = np.where(idx_arr == -1)[0]
            if non_unique_idx.size > 1:
                self.log.warning("Multiple files found. "
                                 "This should not happen!!! TBW: select")
                # TODO: Implement user selection here if required
            else:
                return file_names[non_unique_idx[0]]

        if tle_filenames:
            return select_tle_file(tle_filenames)

        # Fallback to the closest prior TLE file by checking obs_date - 1
        previous_day = datetime.datetime(int(obs_date[0]),
                                         int(obs_date[1]),
                                         int(obs_date[2])) - timedelta(days=1)
        prev_date_pattern = (rf'tle_{sat_type}_'
                             rf'{previous_day.year}[-_]'
                             rf'{previous_day.month:02d}[-_]'
                             rf'{previous_day.day:02d}.*\.txt')

        prev_date_regex = re.compile(prev_date_pattern, re.IGNORECASE)

        fallback_files = [(root, s, path) for root, s, path in all_files if re.search(prev_date_regex, s)]

        if fallback_files:
            return select_tle_file(fallback_files)

        self.log.warning("NO TLE file found!!! "
                         "Check the naming/dates of any TLE if present. Exiting for now!!!")
        return None  # Return None if no suitable file is found

    def find_satellite_in_tle(self, tle_loc, sat_name):
        """

        Parameters
        ----------
        tle_loc
        sat_name

        Returns
        -------

        """
        matches = []
        with open(tle_loc, 'r') as file:
            for line in file:
                if sat_name in line:
                    matches.append(line.strip())  # Add the matching line to the list

        if matches:
            match = matches[0]
            self.log.info(f"    Found match for {sat_name}: {match}")
            return match
        else:
            self.log.warning(f"{sat_name} not found in TLE file!!! "
                             "TBW: Get TLE. "
                             "Exiting for now!!!")
            return None

    # def load_glint_mask_table(self):
    #     """Load table with regions of glints that should be masked and excluded"""
    #     glint_info = self._glint_mask_info
    #
    #     # Files info
    #     fname = self.glint_mask_table_fname
    #     if fname.exists():
    #         if not self.silent:
    #             self.log.info(f'> Read glint mask from: {self.def_glint_mask_name}')
    #         glint_info = pd.read_csv(fname, header=0, delimiter=',')
    #
    #     self._glint_mask_info = glint_info
    #
    # def load_roi_table(self):
    #     """Load the regions of interest info table"""
    #     roi_info = self._roi_info
    #
    #     # Files info
    #     fname = self.roi_table_fname
    #     if fname.exists():
    #         if not self.silent:
    #             self.log.info(f'> Read Region-of-Interest from: {self.def_roi_name}')
    #         roi_info = pd.read_csv(fname, header=0, delimiter=',')
    #
    #     self._roi_info = roi_info
    #
    # def load_ext_oi_table(self):
    #     """Load the extensions of interest info table"""
    #     ext_oi_info = self._ext_oi_info
    #
    #     # Files info
    #     fname = self.ext_oi_table_fname
    #     if fname.exists():
    #         if not self.silent:
    #             self.log.info(f'> Read Extensions-of-Interest from: {self.def_ext_oi_name}')
    #         ext_oi_info = pd.read_csv(fname, header=0, delimiter=',')
    #
    #     self._ext_oi_info = ext_oi_info

    def _load_csv_table(self, data_attribute, file_path, table_name, silent=None):
        """Helper method to load CSV tables with common logic

        Parameters
        ----------
        data_attribute : str
            Name of the attribute to update
        file_path : Path
            The path to the CSV file
        table_name : str
            Display name of the table for logging
        silent : bool, optional
            Override the instance's silent flag

        Returns
        -------
        pd.DataFrame
            The loaded DataFrame
        """
        data = getattr(self, data_attribute)

        if file_path.exists():
            if not (self.silent if silent is None else silent):
                self.log.info(f'> Read {table_name} from: {file_path.name}')
            data = pd.read_csv(file_path, header=0, delimiter=',')

        setattr(self, data_attribute, data)
        return data

    def load_glint_mask_table(self):
        """Load table with regions of glints that should be masked and excluded"""
        return self._load_csv_table(
            '_glint_mask_info',
            self.glint_mask_table_fname,
            'glint mask'
        )

    def load_roi_table(self):
        """Load the regions of interest info table"""
        return self._load_csv_table(
            '_roi_info',
            self.roi_table_fname,
            'Region-of-Interest'
        )

    def load_ext_oi_table(self):
        """Load the extensions of interest info table"""
        return self._load_csv_table(
            '_ext_oi_info',
            self.ext_oi_table_fname,
            'Extensions-of-Interest'
        )

    def load_faint_trail_table(self):
        """Load the faint trail table"""
        return self._load_csv_table(
            '_faint_trail_info',
            self.faint_trail_table_fname,
            'Faint-Trail'
        )

    def load_obs_table(self):
        """Load the observation info table"""
        obs_data = self.res_tbl_data

        fname = self.res_table_fname
        if fname.exists():
            if not self.silent:
                self.log.info(f'> Read observation info from: {self.def_tbl_name}')

            df = pd.read_csv(fname, header=None, delimiter=',')
            columns = df.iloc[0].values

            if len(df) > 1 and self.unit_row_matches(df.iloc[1]):
                obs_data = df.iloc[2:].reset_index(drop=True)
            else:
                obs_data = df.iloc[1:].reset_index(drop=True)

            obs_data.columns = columns

        # else:
        #     self.log.error(f'> {self.def_tbl_name} does not exist!! '
        #                     f'This should not happen.')

        self._obs_info = obs_data

    def unit_row_matches(self, unit_row):
        """Check if the row matches the defined units"""
        return any(cell in self.def_col_units or cell == '' for cell in unit_row)

    # def get_roi(self, fname):
    #     """Get the region of interest"""
    #     roi_info = self._roi_info
    #     if not roi_info.empty:
    #         pos_df = roi_info.copy()
    #
    #         # Convert file name to uppercase
    #         pos_df['File'] = roi_info['File'].str.upper()
    #         fname_upper = fname.upper()
    #         # Check position
    #         pos_mask = pos_df['File'] == fname_upper
    #         pos = np.flatnonzero(pos_mask)
    #
    #         if list(pos):
    #             pos_df = roi_info[pos_mask]
    #             self._roi_data = pos_df
    #
    # def get_ext_oi(self, fname):
    #     """Get extension(s) of interest"""
    #     ext_oi_info = self._ext_oi_info
    #
    #     if not ext_oi_info.empty:
    #         pos_df = ext_oi_info.copy()
    #
    #         # Convert file name to uppercase
    #         pos_df['File'] = ext_oi_info['File'].str.upper()
    #         fname_upper = fname.upper()
    #
    #         # Check position
    #         pos_mask = pos_df['File'] == fname_upper
    #         pos = np.flatnonzero(pos_mask)
    #
    #         if list(pos):
    #             pos_df = ext_oi_info[pos_mask]['HDUs'][pos[0]]
    #             self._ext_oi_data = eval(pos_df)
    #         else:
    #             self._ext_oi_data = []

    def _get_table_data(self, source_attribute, target_attribute, fname, process_func=None):
        """Helper method for retrieving table data based on filename

        Parameters
        ----------
        source_attribute : str
            Name of the attribute containing source data
        target_attribute : str
            Name of the attribute to update with matching data
        fname : str
            Filename to search for
        process_func : callable, optional
            Function to process matched data before assignment

        Returns
        -------
        Any
            The matched and processed data
        """
        source_data = getattr(self, source_attribute)

        # Default empty result
        result = pd.DataFrame() if process_func is None else []

        if not source_data.empty:
            # Convert file name to uppercase for case-insensitive matching
            fname_upper = fname.upper()

            # Filter by filename
            mask = source_data['File'].str.upper() == fname_upper
            matches = source_data[mask]

            if not matches.empty:
                result = matches if process_func is None else process_func(matches)

        # Store result in target attribute
        setattr(self, target_attribute, result)
        return result

    def get_roi(self, fname):
        """Get the region of interest"""
        return self._get_table_data('_roi_info', '_roi_data', fname)

    def get_ext_oi(self, fname):
        """Get extension(s) of interest"""

        def process_ext_oi(matches):
            try:
                return eval(matches['HDUs'].iloc[0])
            except (IndexError, KeyError):
                return []

        return self._get_table_data('_ext_oi_info', '_ext_oi_data', fname, process_ext_oi)

    def _get_hdu_table_data(self, source_attribute, target_attribute, fname, hdu_idx):
        """Helper method for retrieving table data based on filename and HDU index

        Parameters
        ----------
        source_attribute : str
            Name of the attribute containing source data
        target_attribute : str
            Name of the attribute to update with matching data
        fname : str
            Filename to search for
        hdu_idx : int
            HDU index to match

        Returns
        -------
        pd.DataFrame
            The matched data
        """
        source_data = getattr(self, source_attribute)

        # Default empty result
        result = pd.DataFrame()

        if not source_data.empty:
            # Convert file name to uppercase for case-insensitive matching
            fname_upper = fname.upper()

            # Filter by filename and HDU index
            mask = (source_data['File'].str.upper() == fname_upper) & (source_data['HDU'] == hdu_idx)
            matches = source_data[mask]

            if not matches.empty:
                result = matches

        # Store result in target attribute
        setattr(self, target_attribute, result)
        return result

    def get_glint_mask(self, fname, hdu_idx):
        """Get Glint masks

        Parameters
        ----------
        fname : str
            Filename to search for in the glint mask table
        hdu_idx : int
            HDU index to match

        Returns
        -------
        pd.DataFrame
            The matched glint mask data
        """
        return self._get_hdu_table_data(
            '_glint_mask_info',
            '_glint_mask_data',
            fname,
            hdu_idx
        )

    def get_faint_trail(self, fname, hdu_idx):
        """Get faint trail data

        Parameters
        ----------
        fname : str
            Filename to search for in the faint trail table
        hdu_idx : int
            HDU index to match

        Returns
        -------
        pd.DataFrame
            The matched faint trail data
        """
        return self._get_hdu_table_data(
            '_faint_trail_info',
            '_faint_trail_data',
            fname,
            hdu_idx
        )

    # def get_glint_mask(self, fname, hdu_idx):
    #     """Get Glint masks
    #
    #     Parameters
    #     ----------
    #     fname : str
    #         Filename to search for in the glint mask table
    #     hdu_idx : int
    #         HDU index to match
    #
    #     Returns
    #     -------
    #     pd.DataFrame
    #         The matched glint mask data
    #     """
    #     glint_info = self._glint_mask_info
    #
    #     if glint_info.empty:
    #         self._glint_mask_data = pd.DataFrame()
    #         return
    #
    #     # Convert file name to uppercase for case-insensitive matching
    #     fname_upper = fname.upper()
    #
    #     # Filter by filename then by HDU index (more efficient than making a copy)
    #     mask = (glint_info['File'].str.upper() == fname_upper) & (glint_info['HDU'] == hdu_idx)
    #     matches = glint_info[mask]
    #
    #     # Store matched data
    #     if not matches.empty:
    #         self._glint_mask_data = matches
    #     else:
    #         self._glint_mask_data = pd.DataFrame()
    #
    # def get_faint_trail(self, fname, hdu_idx):
    #     """Get faint trail data
    #
    #     Parameters
    #     ----------
    #     fname : str
    #         Filename to search for in the faint trail table
    #     hdu_idx : int
    #         HDU index to match
    #
    #     Returns
    #     -------
    #     pd.DataFrame
    #         The matched faint trail data
    #     """
    #     faint_info = self._faint_trail_info
    #
    #     if faint_info.empty:
    #         self._faint_trail_data = pd.DataFrame()
    #         return
    #
    #     # Convert file name to uppercase for case-insensitive matching
    #     fname_upper = fname.upper()
    #
    #     # Filter by filename then by HDU index (more efficient than making a copy)
    #     mask = (faint_info['File'].str.upper() == fname_upper) & (faint_info['HDU'] == hdu_idx)
    #     matches = faint_info[mask]
    #
    #     # Store matched data
    #     if not matches.empty:
    #         self._faint_trail_data = matches
    #     else:
    #         self._faint_trail_data = pd.DataFrame()

    # def get_glint_mask(self, fname, hdu_idx):
    #     """Get Glint masks"""
    #
    #     glint_info = self._glint_mask_info
    #
    #     if not glint_info.empty:
    #         pos_df = glint_info.copy()
    #
    #         # Convert file name to uppercase
    #         pos_df['File'] = glint_info['File'].str.upper()
    #         fname_upper = fname.upper()
    #
    #         # Check position
    #         pos_mask = (pos_df['File'] == fname_upper)
    #         pos = np.flatnonzero(pos_mask)
    #
    #         if list(pos):
    #             df = glint_info[pos_mask]
    #             hdu_match = (df['HDU'] == hdu_idx)
    #             if np.any(hdu_match):
    #
    #                 if len(np.array(hdu_match)) == 1:
    #                     idx = pos[hdu_match]
    #                     self._glint_mask_data = glint_info.iloc[idx]
    #                 else:
    #                     idx = pos[np.where(hdu_match)[0]]
    #                     self._glint_mask_data = glint_info.iloc[idx]

    def get_object_data(self, fname, kwargs, obsparams, hdu_idx):
        """
        Extract a row from the observation info table.

        Parameters
        ----------
        fname : str
            The name of the file to extract data from.
        kwargs : dict
            Additional keyword arguments containing observation parameters.
        obsparams : dict
            Dictionary containing observation parameters.
        hdu_idx : int
            Index of the HDU (Header Data Unit).

        Returns
        -------
        None
        """
        obs_info = self._obs_info
        n_ext = obsparams['n_ext']

        # Make a copy
        obs_info_tmp = obs_info.copy()

        # Convert file name to uppercase
        obs_info_tmp['File'] = obs_info['File'].str.upper()
        fname_upper = fname.upper()

        # Check position
        mask = obs_info_tmp['File'] == fname_upper
        pos = np.flatnonzero(mask)

        ra_key, ra_val = self.get_position_data(kwargs, obsparams, 'ra', 'RA')
        de_key, de_val = self.get_position_data(kwargs, obsparams, 'dec', 'DEC')
        inst_key, inst_val = self.get_position_data(kwargs, obsparams, 'instrume', 'Instrument')
        obj_key, obj_val = self.get_position_data(kwargs, obsparams, 'object', 'Object')

        key_list = ['RA', 'DEC', 'Instrument', 'Object']
        key_dict = {'RA': [ra_val], 'DEC': [de_val], 'Instrument': [inst_val],
                    'Object': [obj_val]}

        if n_ext > 1:
            key_list = ['RA', 'DEC', 'Instrument', 'Object', 'HDU_idx']
            key_dict['HDU_idx'] = [f"{hdu_idx:.1f}"]

        if list(pos):
            self.log.debug("  Possible match with previous entry found. Check RA and DEC.")

            df = obs_info[mask][key_list].astype(str)

            coord_match = df.isin(key_dict).all(axis='columns')

            if coord_match.any():
                if len(coord_match) == 1:
                    self.log.debug("  Coordinate/Pointing match found. Updating.")
                    idx = pos[coord_match]
                    self._obj_info = obs_info.iloc[idx]
                else:
                    self.log.debug("  Multiple Coordinate/Pointing matches found."
                                   " This should not happen.")
                    idx = pos[np.where(coord_match)[0]]
                    self._obj_info = obs_info.iloc[idx]
            else:
                self.log.error("  NO Coordinate/Pointing match found."
                               " This should not happen!!!")
        else:
            self.log.error("  File not found. This should not happen!!!")

    def get_satellite_id(self, obj_in):
        """
        Get satellite id from object name.

        This function takes an object name, normalizes it, and determines the satellite type and unique ID if present.

        Parameters
        ----------
        obj_in : str
            The input object name.

        Returns
        -------
        tuple
            A tuple containing the satellite identifier and a unique ID (if present).
        """
        self.log.debug("  Identify Satellite/Object name")

        # Standardize object name: uppercase and replace spaces with hyphens
        obj_name = obj_in.upper().replace(' ', '-')

        # Extract unique ID if present (e.g., "-IDxxxx")
        unique_id = None
        if '-ID' in obj_name:
            idx = obj_name.index('-ID')
            unique_id = obj_name[idx + 1:idx + 8]
            obj_name = obj_name[:idx]

        # Remove extensions like "-N" if present
        if '-N' in obj_name:
            idx = obj_name.index('-N')
            obj_name = obj_name[:idx]

        # Remove binning identifiers (e.g., "_1X1", "-2X2")
        for suffix in ['_1X1', '-1X1', '_2X2', '-2X2', '_4X4', '-4X4']:
            if suffix in obj_name:
                obj_name = obj_name.replace(suffix, '')

        # Replace remaining underscores with hyphens
        obj_name = obj_name.replace('_', '-')

        # Identify satellite type
        sat_name = obj_name
        found = False
        for key, props in bc.SATELLITE_PROPERTIES.items():
            if any(identifier in obj_name for identifier in props['identifiers']):
                sat_name = key
                found = True
                break

        # Extract satellite number if present and format accordingly
        parts = obj_name.split('-')
        nb_str = None
        for part in parts:
            if part.isdigit():
                nb_str = part
                break

        if found:
            if nb_str:
                sat_id = bc.SATELLITE_PROPERTIES[sat_name]['format_number'](nb_str)
            else:
                sat_id = sat_name
        else:
            # If not a known satellite observation, return the original object name without modifications
            sat_id = obj_name

        return sat_id, unique_id

    def get_position_data(self, kwargs, obsparams, obskey, dkey):
        """

        Parameters
        ----------
        kwargs
        obsparams
        obskey
        dkey

        Returns
        -------

        """
        key_transl = self.def_key_transl
        for i in range(len(key_transl[dkey])):
            key = key_transl[dkey][i]
            if key in kwargs and key == obsparams[obskey]:
                val = str(kwargs[key])
                return key, val

    def update_obs_table(self, file, kwargs, obsparams):
        """
        Load the observation obs info table.

        Load the table with observation information.
        If the file does not exist, it is created.
        """
        obs_info = self._obs_info
        fname = self.res_table_fname
        n_ext = obsparams['n_ext']

        # exposure time
        expt = kwargs['EXPTIME']

        # Update the columns in case the table was changed
        for col in self.def_cols:
            if col not in obs_info:
                # insert column at the end
                obs_info.insert(len(obs_info.columns), col, '')

        # Reorder columns
        obs_info = obs_info[self.def_cols]

        # Replace all None values with NAN
        obs_info = obs_info.replace(to_replace=[None], value=np.nan, inplace=False)

        # Make a copy
        obs_info_tmp = obs_info.copy()

        # Convert file name to uppercase
        obs_info_tmp['File'] = obs_info['File'].str.upper()
        file_upper = file.upper()

        # Check the position of data
        mask = obs_info_tmp['File'] == file_upper
        pos = np.flatnonzero(mask)

        ra_key, ra_val = self.get_position_data(kwargs, obsparams, 'ra', 'RA')
        de_key, de_val = self.get_position_data(kwargs, obsparams, 'dec', 'DEC')
        inst_key, inst_val = self.get_position_data(kwargs, obsparams, 'instrume', 'Instrument')
        obj_key, obj_val = self.get_position_data(kwargs, obsparams, 'object', 'Object')

        key_list = ['RA', 'DEC', 'Instrument', 'Object']
        key_dict = {'RA': [ra_val], 'DEC': [de_val], 'Instrument': [inst_val],
                    'Object': [obj_val]}

        if n_ext > 1:
            hdu_idx = str(float(kwargs['HDU_IDX']))
            key_list = ['RA', 'DEC', 'Instrument', 'Object', 'HDU_idx']
            key_dict['HDU_idx'] = [hdu_idx]

        if not list(pos):
            self.log.debug("  File name not found. Adding new entry.")
            obs_info = self.add_row(obs_info, file, kwargs, self.def_cols,
                                    expt, obj_key, ra_val, de_val)
        else:
            self.log.debug("  Possible match with previous entry found. "
                           "Check RA, DEC, object and instrument.")

            df = obs_info[mask][key_list].astype(str)

            coord_match = df.isin(key_dict).all(axis=1)
            if coord_match.any():

                self.log.debug("  Coordinate/Pointing, object and instrument match found. Updating.")
                idx = pos[np.where(coord_match)[0]]
                obs_info = self.update_row(obs_info, idx, kwargs, self.def_cols,
                                           expt, obj_key, ra_val, de_val)

            else:
                self.log.debug("  NO Coordinate/Pointing, object and instrument match found. Adding entry.")
                obs_info = self.add_row(obs_info, file, kwargs,
                                        self.def_cols, expt, obj_key, ra_val, de_val)

        # Check for duplicates
        bool_series = obs_info.duplicated(keep='first')
        obs_info = obs_info[~bool_series]

        obs_info = obs_info.sort_values(by=['Date-Obs', 'Obs-Start', 'File'],
                                        ascending=[True, True, True]).reset_index(drop=True)

        self._obs_info = obs_info
        self.save_table(fname, obs_info)

        del obs_info, obs_info_tmp
        gc.collect()

    def update_row(self, obs_info, idx, kwargs, def_cols, expt, obj_key, ra, dec):
        """
        Update row in result table.

        This function updates a specific row in the result table with the provided
        observation information and parameters.
        """

        for i in range(1, len(def_cols)):
            dcol_name = def_cols[i]
            for key, value in kwargs.items():
                if dcol_name in self.def_key_transl and \
                        key in self.def_key_transl[dcol_name]:
                    if dcol_name == 'Object':
                        # Create uniform satellite id from object name
                        sat_name, alt_id = self.get_satellite_id(kwargs[obj_key])
                        obs_info.loc[idx, dcol_name] = value
                        obs_info.loc[idx, 'Sat-Name'] = sat_name
                        obs_info.loc[idx, 'UniqueID'] = alt_id
                    elif dcol_name == 'Date-Obs':
                        t_short, t_start, t_mid, t_stop = self.update_times(value, expt, kwargs)
                        obs_info.loc[idx, dcol_name] = t_short
                        obs_info.loc[idx, 'Obs-Start'] = t_start
                        obs_info.loc[idx, 'Obs-Mid'] = t_mid.strftime('%H:%M:%S.%f')[:-3]
                        obs_info.loc[idx, 'Obs-Stop'] = t_stop.strftime('%H:%M:%S.%f')[:-3]
                    elif dcol_name == 'WCS_cal':
                        obs_info.loc[idx, dcol_name] = 'T' if value or value == 'successful' else 'F'
                    elif dcol_name == 'RA':
                        obs_info.loc[idx, dcol_name] = ra
                    elif dcol_name == 'DEC':
                        obs_info.loc[idx, dcol_name] = dec
                    else:
                        obs_info.loc[idx, dcol_name] = value
                else:
                    if key == dcol_name and dcol_name not in ['Obs-Start', 'Obs-Mid', 'Obs-Stop', 'Sat-Name']:
                        obs_info.loc[idx, key] = value

        return obs_info

    def add_row(self, obs_info, file, kwargs, def_cols, expt, obj_key, ra, dec):
        """ Add new row to result table"""
        new_dict = {def_cols[0]: [file]}
        for i in range(len(def_cols)):
            dcol_name = def_cols[i]
            for key, value in kwargs.items():
                if dcol_name in self.def_key_transl and \
                        key in self.def_key_transl[dcol_name]:
                    if dcol_name == 'Object':
                        # Create uniform satellite id from object name
                        sat_name, alt_id = self.get_satellite_id(kwargs[obj_key])
                        new_dict[dcol_name] = value
                        new_dict['Sat-Name'] = sat_name
                        new_dict['UniqueID'] = alt_id
                    elif dcol_name == 'Date-Obs':
                        t_short, t_start, t_mid, t_stop = self.update_times(value, expt, kwargs)
                        new_dict[dcol_name] = t_short
                        new_dict['Obs-Start'] = t_start
                        new_dict['Obs-Mid'] = t_mid.strftime('%H:%M:%S.%f')[:-3]
                        new_dict['Obs-Stop'] = t_stop.strftime('%H:%M:%S.%f')[:-3]
                    elif dcol_name == 'WCS_cal':
                        new_dict[dcol_name] = 'T' if value or value == 'successful' else 'F'
                    elif dcol_name == 'RA':
                        new_dict[dcol_name] = ra
                    elif dcol_name == 'DEC':
                        new_dict[dcol_name] = dec
                    else:
                        new_dict[dcol_name] = value
                else:
                    if key == dcol_name and dcol_name not in ['Obs-Start', 'Obs-Mid', 'Obs-Stop', 'Sat-Name']:
                        new_dict[dcol_name] = value

        new_df = pd.DataFrame(new_dict)

        # Pandas version < 2.0
        if Version(pd.__version__) < Version('2.0'):
            obs_info = obs_info.append(new_df, ignore_index=False)
        else:
            obs_info = pd.concat([obs_info, new_df], ignore_index=False)

        del new_df, new_dict
        gc.collect()

        return obs_info

    def read_vis_file(self, vis_file_list: list):
        """Read the visibility file"""

        col_names = bc.DEF_VIS_TBL_COL_NAMES
        df_list = []
        for i in range(len(vis_file_list)):
            vis_loc = vis_file_list[i]
            file = vis_loc[1]

            # Create 3 empty lists to hold row data
            data, data2, data3 = ([] for _ in range(3))

            # Load the .csv into a list
            with open(file, 'r') as f:
                reader = csv.reader(f)
                lines = list(reader)

            # Check if the first line has multiple words
            has_multiple_words = self.contains_multiple_words(lines[1][0])
            two_lines = False if has_multiple_words else True

            for line in lines[1:]:

                # Split line using regex
                row = re.split(pattern=r'\s\s|\s|\t', string=line[0])
                l_row = len(row)

                if two_lines:

                    if l_row in [1, 2]:
                        test_str = row[0] if l_row == 1 else '-'.join(row)
                        sat_id, alt_id, unique_id = self.identify_satellite(test_str)
                        data2.append([sat_id, alt_id, unique_id])
                    if l_row == 13:
                        data3.append(row)
                else:
                    test_str, rest = (row[0], row[1:]) if l_row == 14 else ('-'.join(row[:2]), row[2:])
                    sat_id, alt_id, unique_id = self.identify_satellite(test_str)
                    tmp = [sat_id, alt_id, unique_id]
                    tmp += rest
                    data.append(tmp)
            if two_lines:
                for j in range(len(data2)):
                    tmp = data2[j]
                    tmp += data3[j]
                    data.append(tmp)

            df = pd.DataFrame(data=data, columns=col_names)
            df.replace(to_replace=[None], value=np.nan, inplace=True)
            df.reset_index()
            df_list.append(df)

            del data, df
            gc.collect()

        df = pd.concat(df_list)
        df.reset_index()

        self._vis_info = df

        del df, df_list
        gc.collect()

    def save_table(self, fname, obs_info):
        """Method to save the result table to a csv file."""
        df_units = self.res_tbl_unit_df
        if obs_info.empty:
            df_data = pd.concat([pd.DataFrame([self.def_cols], columns=self.def_cols),
                                 df_units], ignore_index=True)
        else:
            df_data = pd.concat([pd.DataFrame([self.def_cols], columns=self.def_cols),
                                 df_units, obs_info], axis=0, ignore_index=True)

        df_data.to_csv(fname, index=False, header=False)

    def update_times(self, value, expt, kwargs):
        """ Update Observation date, start, mid, and end times."""
        self.log.debug("  Check Obs-Date")

        tel_key = 'TELESCOP' if 'TELESCOP' in kwargs else 'OBSERVAT'
        if 'TIME-OBS' in kwargs and kwargs[tel_key] != 'CTIO 4.0-m telescope':
            t = pd.to_datetime(f"{kwargs['DATE-OBS']}T{kwargs['TIME-OBS']}",
                               format=frmt, utc=False)
        else:
            t = pd.to_datetime(value, format=frmt, utc=False)

        t_short = t.strftime('%Y-%m-%d')
        t_start = t.strftime('%H:%M:%S.%f')[:-3]
        t_mid = t + timedelta(seconds=float(expt) / 2)
        t_stop = t + timedelta(seconds=float(expt))

        return t_short, t_start, t_mid, t_stop

    @staticmethod
    def contains_multiple_words(s):
        return len(re.compile(r' ').split(s)) > 1

    @staticmethod
    def identify_satellite(test_str):

        unique_id = re.findall(r'(ID-\d+)', test_str)
        unique_id = unique_id[0] if unique_id else None

        if unique_id is not None:
            test_str = test_str.replace(f'-{unique_id}', '')
        alt_id = re.findall(r'-\((\w+)\)', test_str)
        alt_id = alt_id[0] if alt_id else None

        if alt_id is not None:
            test_str = test_str.replace(f'-({alt_id})', '')

        return test_str, alt_id, unique_id

    @staticmethod
    def get_sat_type(sat_name):
        """
        Determine the satellite type based on the satellite name.

        Parameters
        ----------
        sat_name : str
            The name of the satellite.

        Returns
        -------
        str
            The satellite type corresponding to the given satellite name.
            Returns 'oneweb' if no specific match is found.
        """
        sat_type_mapping = {
            'STARLINK': 'starlink',
            'BLUEWALKER': 'bluewalker',
            'KUIPER': 'kuiper',
            'SPACEMOBILE': '[spacemobile|all]'
        }

        # Iterate over the mapping to find a matching satellite type
        for key, value in sat_type_mapping.items():
            if key in sat_name.upper():
                return value

        # Default to 'oneweb' if no match is found
        return 'oneweb'
