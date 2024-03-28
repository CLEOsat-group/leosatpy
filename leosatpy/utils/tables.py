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
import datetime
import gc
import os
import logging
import re
import csv
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import timedelta
from datetime import datetime as dt
from astropy import units as u
from astropy.coordinates import SkyCoord
from packaging.version import Version

# Project modules
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

        self._log = log
        self._silent = silent
        self._obs_info = pd.DataFrame()
        self._obj_info = pd.DataFrame()
        self._sat_info = pd.DataFrame()
        self._roi_info = pd.DataFrame()
        self._roi_data = pd.DataFrame()
        self._ext_oi_info = pd.DataFrame()
        self._ext_oi_data = pd.DataFrame()
        self._glint_mask_info = pd.DataFrame()
        self._glint_mask_data = pd.DataFrame()

        self._vis_info = pd.DataFrame()
        self._def_path = Path(config['RESULT_TABLE_PATH']).expanduser().resolve()
        self._def_tbl_name = config['RESULT_TABLE_NAME']
        self._def_roi_name = config['ROI_TABLE_NAME']
        self._def_ext_oi_name = config['EXT_OI_TABLE_NAME']
        self._def_glint_mask_name = config['GLINT_MASK_TABLE_NAME']

        self._def_cols = _base_conf.DEF_RES_TBL_COL_NAMES
        self._def_col_units = _base_conf.DEF_RES_TBL_COL_UNITS
        self._fname_res_table = self._def_path / self._def_tbl_name
        self._fname_roi_table = self._def_path / self._def_roi_name
        self._fname_ext_oi_table = self._def_path / self._def_ext_oi_name
        self._fname_glint_mask_table = self._def_path / self._def_glint_mask_name
        self._def_key_transl = _base_conf.DEF_KEY_TRANSLATIONS
        self._create_obs_table()

    @property
    def obs_info(self):
        return self._obs_info

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

    def find_satellite_in_tle(self, tle_loc, sat_name):
        matches = []
        with open(tle_loc, 'r') as file:
            for line in file:
                if sat_name in line:
                    matches.append(line.strip())  # Add the matching line to the list

        if matches:
            match = matches[0]
            self._log.info(f"    Found match for {sat_name}: {match}")
            return match
        else:
            self._log.warning(f"{sat_name} not found in TLE file!!! "
                              "TBW: Get TLE. "
                              "Exiting for now!!!")
            return None

    def find_tle_file(self, sat_name: str,
                      img_filter: str,
                      src_loc: str):
        """ Check for TLE file location """
        tle_filenames = []
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

        # get times
        date_obs = self._obj_info['Date-Obs'].values[0]
        obs_date = date_obs.split('-')

        sat_type = 'oneweb'
        if 'STARLINK' in sat_name:
            sat_type = 'starlink'
        if 'BLUEWALKER' in sat_name:
            sat_type = 'bluewalker'

        regex_pattern = rf'(?i)tle_{sat_type}_{obs_date[0]}[-_]{obs_date[1]}[-_]{obs_date[2]}.*\.txt'
        regex = re.compile(regex_pattern,
                           re.IGNORECASE)

        path = os.walk(search_path)
        for root, dirs, files in path:
            f = sorted([s for s in files if re.search(regex, s)])

            if len(f) > 0:
                [tle_filenames.append((root, s, Path(os.path.join(root, s)))) for s in f]

        if not tle_filenames:
            self._log.warning("NO TLE file found!!! "
                              "Check the naming/dates of any TLE if present. "
                              "Exiting for now!!!")
            return None
        
        elif len(tle_filenames) > 1:
            # first check if one file has unique in name
            idx_arr = np.array([i[0].find('unique') for i in tle_filenames])
            unique_idx = np.where(idx_arr == 0)[0]
            if unique_idx and len(unique_idx) > 1:
                self._log.warning("Multiple unique TLE files found. "
                                  "This really should not happen!!! TBW: select")
            elif unique_idx and len(unique_idx) == 1:
                return tle_filenames[unique_idx[0]]
            # if there is still more than one file, let the user select
            non_unique_idx = np.where(idx_arr == -1)[0]
            if len(non_unique_idx) > 1:
                self._log.warning("Multiple files found. This should not happen!!! TBW: select")
                # fixme: add possibility to select correct file
            else:
                return tle_filenames[non_unique_idx[0]]
        else:
            return tle_filenames[0]

    def load_visibility_file(self,
                             img_filter: str,
                             src_loc: str):
        """Method to find and load visibility data into table."""

        vis_filenames = []
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

        vis_search_path = Path(src_loc).parents[level_up]
        path = os.walk(vis_search_path)
        for root, dirs, files in path:
            f = sorted([s for s in files if s == 'visible.txt' or 'visible' in s])

            if len(f) > 0:
                [vis_filenames.append((s, Path(os.path.join(root, s)))) for s in f]

        if not vis_filenames:
            self._log.warning("NO visibility file found!!! "
                              "TBW: calculate visibility from TLE file for obs-mid time. "
                              "Exiting for now!!!")
            return False

        # read visibility file found or calculated
        self._read_vis_file(vis_file_list=vis_filenames)

        return True

    def load_glint_mask_table(self):
        """Load table with regions of glints that should be masked and excluded"""
        glint_info = self._glint_mask_info

        # files info
        fname = self._fname_glint_mask_table
        if fname.exists():
            if not self._silent:
                self._log.info(f'> Read glint mask from: {self._def_glint_mask_name}')
            glint_info = pd.read_csv(fname, header=0, delimiter=',')

        self._glint_mask_info = glint_info

    def load_roi_table(self):
        """Load the regions of interest info table"""
        roi_info = self._roi_info

        # files info
        fname = self._fname_roi_table
        if fname.exists():
            if not self._silent:
                self._log.info(f'> Read Region-of-Interest from: {self._def_roi_name}')
            roi_info = pd.read_csv(fname, header=0, delimiter=',')

        self._roi_info = roi_info

    def load_ext_oi_table(self):
        """Load the extensions of interest info table"""
        ext_oi_info = self._ext_oi_info

        # files info
        fname = self._fname_ext_oi_table
        if fname.exists():
            if not self._silent:
                self._log.info(f'> Read Extensions-of-Interest from: {self._def_ext_oi_name}')
            ext_oi_info = pd.read_csv(fname, header=0, delimiter=',')

        self._ext_oi_info = ext_oi_info

    def load_obs_table(self):
        """Load the observation info table"""
        obs_info = self._obs_info

        # files info
        fname = self._fname_res_table
        if fname.exists():
            if not self._silent:
                self._log.info(f'> Read observation info from: {self._def_tbl_name}')
            obs_info = pd.read_csv(fname, header=0, delimiter=',')
            # obs_info.drop([0], axis=0, inplace=True)
            # obs_info.reset_index(drop=True)
            # obs_info = obs_info.apply(pd.to_numeric, errors='ignore')
        else:
            self._log.error(f'> {self._def_tbl_name} does not exist!! '
                            f'This should not happen.')
        self._obs_info = obs_info

    def get_roi(self, fname):
        """Get the region of interest"""
        roi_info = self._roi_info
        if not roi_info.empty:
            pos_df = roi_info.copy()

            # convert file name to uppercase
            pos_df['File'] = roi_info['File'].str.upper()
            fname_upper = fname.upper()
            # check position
            pos_mask = pos_df['File'] == fname_upper
            pos = np.flatnonzero(pos_mask)

            if list(pos):
                pos_df = roi_info[pos_mask]
                self._roi_data = pos_df

    def get_ext_oi(self, fname):
        """Get extension(s) of interest"""
        ext_oi_info = self._ext_oi_info

        if not ext_oi_info.empty:
            pos_df = ext_oi_info.copy()

            # convert file name to uppercase
            pos_df['File'] = ext_oi_info['File'].str.upper()
            fname_upper = fname.upper()

            # check position
            pos_mask = pos_df['File'] == fname_upper
            pos = np.flatnonzero(pos_mask)

            if list(pos):

                pos_df = ext_oi_info[pos_mask]['Array_N'][pos[0]]
                self._ext_oi_data = eval(pos_df)
            else:
                self._ext_oi_data = []

    def get_glint_mask(self, fname, hdu_idx):
        """Get Glint masks"""

        glint_info = self._glint_mask_info

        if not glint_info.empty:
            pos_df = glint_info.copy()

            # convert file name to uppercase
            pos_df['File'] = glint_info['File'].str.upper()
            fname_upper = fname.upper()

            # check position
            pos_mask = (pos_df['File'] == fname_upper)
            pos = np.flatnonzero(pos_mask)

            if list(pos):
                df = glint_info[pos_mask]
                hdu_match = (df['HDU'] == hdu_idx)
                if np.any(hdu_match):

                    if len(np.array(hdu_match)) == 1:
                        idx = pos[hdu_match]
                        self._glint_mask_data = glint_info.iloc[idx]
                    else:
                        idx = pos[np.where(hdu_match)[0]]
                        self._glint_mask_data = glint_info.iloc[idx]

    def get_object_data(self, fname, kwargs, obsparams, hdu_idx):
        """Extract a row from the obs info table"""
        obs_info = self._obs_info
        n_ext = obsparams['n_ext']

        # make a copy
        obs_info_tmp = obs_info.copy()

        # convert file name to uppercase
        obs_info_tmp['File'] = obs_info['File'].str.upper()
        fname_upper = fname.upper()

        # check position
        mask = obs_info_tmp['File'] == fname_upper
        pos = np.flatnonzero(mask)

        ra_key, ra_val = self._get_position_data(kwargs, obsparams, 'ra', 'RA')
        de_key, de_val = self._get_position_data(kwargs, obsparams, 'dec', 'DEC')
        inst_key, inst_val = self._get_position_data(kwargs, obsparams, 'instrume', 'Instrument')
        obj_key, obj_val = self._get_position_data(kwargs, obsparams, 'object', 'Object')

        key_list = ['RA', 'DEC', 'Instrument', 'Object']
        key_dict = {'RA': [ra_val], 'DEC': [de_val], 'Instrument': [inst_val],
                    'Object': [obj_val]}

        if n_ext > 1:
            key_list = ['RA', 'DEC', 'Instrument', 'Object', 'HDU_idx']
            key_dict['HDU_idx'] = [f"{hdu_idx:.1f}"]

        if list(pos):
            self._log.debug("  Possible match with previous entry found. Check RA and DEC.")

            df = obs_info[mask][key_list].astype(str)

            coord_match = df.isin(key_dict).all(axis='columns')

            if coord_match.any():
                if len(coord_match) == 1:
                    self._log.debug("  Coordinate/Pointing match found. Updating.")
                    idx = pos[coord_match]
                    self._obj_info = obs_info.iloc[idx]
                else:
                    self._log.debug("  Multiple Coordinate/Pointing matches found."
                                    " This should not happen.")
                    idx = pos[np.where(coord_match)[0]]
                    self._obj_info = obs_info.iloc[idx]
            else:
                self._log.error("  NO Coordinate/Pointing match found."
                                " This should not happen!!!")
        else:
            self._log.error("  File not found. This should not happen!!!")

    def get_sat_data_from_visibility(self, sat_name: str, obs_date, pointing: tuple):
        """Extract a row from the visibility info table"""
        vis_info = self._vis_info

        ra_unit = u.deg
        matches = [":", " "]
        if any(x in str(pointing[0]) for x in matches):
            ra_unit = u.hourangle
        c0 = SkyCoord(ra=pointing[0], dec=pointing[1], unit=(ra_unit, u.deg))

        mask = vis_info['ID'] == sat_name
        pos = np.flatnonzero(mask)

        if list(pos):

            if len(pos) == 1:
                self._sat_info = vis_info[mask]
                return

            vis_info_masked = vis_info[mask]

            date_info = vis_info_masked[['UT Date', 'UT time']].copy(deep=True)
            date_info['Time'] = date_info.apply(lambda row:
                                                datetime.datetime.strptime(f"{row['UT Date']}T{row['UT time']}",
                                                                           "%Y-%m-%dT%H:%M:%S"), axis=1)

            date_arr = date_info['Time'].to_numpy(dtype=object)
            dt_secs = np.array([abs((date_arr[i] - obs_date).total_seconds())
                                for i in range(date_arr.shape[0])])
            pos_info = vis_info_masked[['SatRA', 'SatDEC']].to_numpy(dtype=object)
            c1 = SkyCoord(ra=pos_info[:, 0], dec=pos_info[:, 1], unit=(u.hourangle, u.deg))
            sep = c1.separation(c0).arcsecond

            dt_idx = np.argmin(dt_secs)
            sep_idx = np.argmin(sep)

            mask = dt_idx if not dt_idx == sep_idx else sep_idx

            self._sat_info = vis_info_masked.iloc[mask].to_frame(0).T
        else:
            self._log.error("  Satellite ID not found in visibility file. "
                            "This should not happen!!!")

    def get_satellite_id(self, obj_in):
        """Get satellite id from object name"""

        self._log.debug("  Identify Satellite/Object name")

        # convert to uppercase
        obj_name = obj_in.upper()
        # convert _ to -
        if ' ' in obj_name:
            obj_name = str(obj_name).replace(' ', '-')

        # identify unique ID if present
        unique_id = None
        if '-ID' in obj_name:
            idx = obj_name.index('-ID')
            unique_id = obj_name[idx + 1:idx + 8]
            obj_name = obj_name[:idx]

        # remove extensions to name
        if '-N' in obj_name:
            idx = obj_name.index('-N')
            obj_name = obj_name[:idx]

        # remove binning from name
        for i in [1, 2, 4]:
            bin_str1 = f'_{i}X{i}'
            bin_str2 = f'-{i}X{i}'
            if bin_str1 in obj_name:
                obj_name = str(obj_name).replace(bin_str1, '')
            if bin_str2 in obj_name:
                obj_name = str(obj_name).replace(bin_str2, '')

        # convert _ to -
        if '_' in obj_name:
            obj_name = str(obj_name).replace('_', '-')

        # OneWeb
        sat_name = 'ONEWEB' if ('OW' in obj_name or 'ONEWEB' in obj_name) else obj_name

        # StarLink
        sat_name = 'STARLINK' if 'STARLINK' in obj_name else sat_name

        # Bluewalker
        sat_name = 'BLUEWALKER' if ('BW' in obj_name or 'BLUEWALKER' in obj_name) else sat_name

        # get the satellite id number and convert to 4 digit string
        nb_str = ''.join(n for n in obj_name if n.isdigit())
        sat_id = f"{sat_name}"
        if nb_str != '':
            sat_id = f"{sat_name}-{int(nb_str):d}" if 'BLUEWALKER' in sat_name else f"{sat_name}-{int(nb_str):04d}"

        return sat_id, unique_id

    def _get_position_data(self, kwargs, obsparams, obskey, dkey):
        key_transl = self._def_key_transl
        for i in range(len(key_transl[dkey])):
            key = key_transl[dkey][i]
            if key in kwargs and key == obsparams[obskey]:
                val = str(kwargs[key])
                return key, val

    def update_obs_table(self, file, kwargs, obsparams):
        """Load the observation obs info table.

        Load the table with observation information.
        If the file does not exist, it is created.
        """

        obs_info = self._obs_info
        fname = self._fname_res_table
        n_ext = obsparams['n_ext']

        # exposure time
        expt = kwargs['EXPTIME']

        # update the columns in case the table was changed
        for col in self._def_cols:
            if col not in obs_info:
                # insert column at the end
                obs_info.insert(len(obs_info.columns), col, '')

        # reorder columns
        obs_info = obs_info[self._def_cols]

        # replace all None values with NAN
        obs_info = obs_info.replace(to_replace=[None], value=np.nan, inplace=False)

        # make a copy
        obs_info_tmp = obs_info.copy()

        # convert file name to uppercase
        obs_info_tmp['File'] = obs_info['File'].str.upper()
        file_upper = file.upper()

        # check the position of data
        mask = obs_info_tmp['File'] == file_upper
        pos = np.flatnonzero(mask)

        ra_key, ra_val = self._get_position_data(kwargs, obsparams, 'ra', 'RA')
        de_key, de_val = self._get_position_data(kwargs, obsparams, 'dec', 'DEC')
        inst_key, inst_val = self._get_position_data(kwargs, obsparams, 'instrume', 'Instrument')
        obj_key, obj_val = self._get_position_data(kwargs, obsparams, 'object', 'Object')

        key_list = ['RA', 'DEC', 'Instrument', 'Object']
        key_dict = {'RA': [ra_val], 'DEC': [de_val], 'Instrument': [inst_val],
                    'Object': [obj_val]}

        if n_ext > 1:
            # hdu index
            hdu_idx = str(float(kwargs['HDU_IDX']))
            key_list = ['RA', 'DEC', 'Instrument', 'Object', 'HDU_idx']
            key_dict['HDU_idx'] = [hdu_idx]

        if not list(pos):
            self._log.debug("  File name not found. Adding new entry.")
            obs_info = self._add_row(obs_info, file, kwargs, self._def_cols,
                                     expt, obj_key, ra_val, de_val)
        else:
            self._log.debug("  Possible match with previous entry found. "
                            "Check RA, DEC, object and instrument.")

            df = obs_info[mask][key_list].astype(str)

            coord_match = df.isin(key_dict).all(axis=1)
            if coord_match.any():

                self._log.debug("  Coordinate/Pointing, object and instrument match found. Updating.")
                idx = pos[np.where(coord_match)[0]]
                obs_info = self._update_row(obs_info, idx, kwargs, self._def_cols,
                                            expt, obj_key, ra_val, de_val)

            else:
                self._log.debug("  NO Coordinate/Pointing, object and instrument match found. Adding entry.")
                obs_info = self._add_row(obs_info, file, kwargs,
                                         self._def_cols, expt, obj_key, ra_val, de_val)

        # check for duplicates
        bool_series = obs_info.duplicated(keep='first')
        obs_info = obs_info[~bool_series]

        obs_info = obs_info.sort_values(by=['Date-Obs', 'Obs-Start', 'File'],
                                        ascending=[True, True, True]).reset_index(drop=True)

        # save to csv
        obs_info.to_csv(fname,
                        header=self._def_cols,
                        index=False)
        self._obs_info = obs_info

        del obs_info, obs_info_tmp
        gc.collect()

    def _update_row(self, obs_info, idx, kwargs, def_cols, expt, obj_key, ra, dec):
        """ Update row in result table"""

        for i in range(1, len(def_cols)):
            dcol_name = def_cols[i]
            for key, value in kwargs.items():
                if dcol_name in self._def_key_transl and \
                        key in self._def_key_transl[dcol_name]:
                    if dcol_name == 'Object':
                        # create uniform satellite id from object name
                        sat_name, alt_id = self.get_satellite_id(kwargs[obj_key])
                        obs_info.loc[idx, dcol_name] = value
                        obs_info.loc[idx, 'Sat-Name'] = sat_name
                        obs_info.loc[idx, 'UniqueID'] = alt_id
                    elif dcol_name == 'Date-Obs':
                        t_short, t_start, t_mid, t_stop = self._update_times(value, expt, kwargs)
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

    def _add_row(self, obs_info, file, kwargs, def_cols, expt, obj_key, ra, dec):
        """ Add new row to result table"""
        new_dict = {def_cols[0]: [file]}
        for i in range(len(def_cols)):
            dcol_name = def_cols[i]
            for key, value in kwargs.items():
                if dcol_name in self._def_key_transl and \
                        key in self._def_key_transl[dcol_name]:
                    if dcol_name == 'Object':
                        # create uniform satellite id from object name
                        sat_name, alt_id = self.get_satellite_id(kwargs[obj_key])
                        new_dict[dcol_name] = value
                        new_dict['Sat-Name'] = sat_name
                        new_dict['UniqueID'] = alt_id
                    elif dcol_name == 'Date-Obs':
                        t_short, t_start, t_mid, t_stop = self._update_times(value, expt, kwargs)
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

    def _get_daytime(self):
        """ Extract day time from observation date """
        date_obs = self._obj_info['Date-Obs'].values[0]
        obs_date = date_obs.replace('-', '_').split('_')
        obs_mid = self._obj_info['Obs-Mid'].values[0]
        datem = dt.strptime(obs_mid, '%H:%M:%S.%f')
        daytime = 'morning' if (datem.hour % 24) // 12 == 0 else 'evening'

        return obs_date, daytime

    def _read_vis_file(self,
                       vis_file_list: list):
        """Read the visibility file"""

        col_names = _base_conf.DEF_VIS_TBL_COL_NAMES
        df_list = []
        for i in range(len(vis_file_list)):
            vis_loc = vis_file_list[i]
            file = vis_loc[1]

            # create 3 empty lists to hold row data
            data, data2, data3 = ([] for _ in range(3))

            # load the .csv into a list
            with open(file, 'r') as f:
                reader = csv.reader(f)
                lines = list(reader)

            # check if the first line has multiple words
            has_multiple_words = self.contains_multiple_words(lines[1][0])
            two_lines = False if has_multiple_words else True

            for line in lines[1:]:

                # split line using regex
                row = re.split(pattern=r'\s\s|\s|\t', string=line[0])
                l_row = len(row)

                if two_lines:

                    if l_row in [1, 2]:
                        test_str = row[0] if l_row == 1 else '-'.join(row)
                        sat_id, alt_id, unique_id = self._identify_satellite(test_str)
                        data2.append([sat_id, alt_id, unique_id])
                    if l_row == 13:
                        data3.append(row)
                else:
                    test_str, rest = (row[0], row[1:]) if l_row == 14 else ('-'.join(row[:2]), row[2:])
                    sat_id, alt_id, unique_id = self._identify_satellite(test_str)
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

    def _create_obs_table(self):
        """Method to create csv file that contains basic model info"""

        fname = self._fname_res_table
        df = pd.DataFrame(columns=self._def_cols)
        if not fname.exists():
            if not self._silent:
                self._log.info('> Create obs_info data frame')
                df.to_csv(fname, header=self._def_cols, index=False)
        self._obs_info = df.copy()

    def _update_times(self, value, expt, kwargs):
        """ Update Observation date, start, mid, and end times."""
        self._log.debug("  Check Obs-Date")
        if ('time-obs'.upper() in kwargs and 'telescop'.upper() in kwargs and
                kwargs['telescop'.upper()] != 'CTIO 4.0-m telescope'):
            t = pd.to_datetime(f"{kwargs['date-obs'.upper()]}T{kwargs['time-obs'.upper()]}",
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
    def _identify_satellite(test_str):

        unique_id = re.findall(r'(ID-\d+)', test_str)
        unique_id = unique_id[0] if unique_id else None

        if unique_id is not None:
            test_str = test_str.replace(f'-{unique_id}', '')
        alt_id = re.findall(r'-\((\w+)\)', test_str)
        alt_id = alt_id[0] if alt_id else None

        if alt_id is not None:
            test_str = test_str.replace(f'-({alt_id})', '')

        return test_str, alt_id, unique_id
