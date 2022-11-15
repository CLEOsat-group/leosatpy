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
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import timedelta
from datetime import datetime as dt
from astropy import units as u
from astropy.coordinates import SkyCoord

# Project modules
import config.base_conf as _base_conf

# -----------------------------------------------------------------------------

""" Meta-info """
__author__ = "Christian Adam"
__copyright__ = 'Copyright 2021, UA, LEOSat observations'
__credits__ = ["Christian Adam, Eduardo Unda-Sanzana, Jeremy Tregloan-Reed"]
__license__ = "Free"
__version__ = "0.3.1"
__maintainer__ = "Christian Adam"
__email__ = "christian.adam84@gmail.com"
__status__ = "Production"
# -----------------------------------------------------------------------------

# changelog
# version 0.1.0 alpha
# version 0.2.0 improved detection of duplicates and similar data
#               added methods for adding and updating rows to table
# version 0.2.1 added satellite name identification from object name in header
# version 0.3.0 added support for tle data
# version 0.3.1 added sorting of table, and minor fixes


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
        self._vis_info = pd.DataFrame()
        self._def_path = Path(config['RESULT_TABLE_PATH']).expanduser().resolve()
        self._def_name = config['RESULT_TABLE_NAME']
        self._def_cols = _base_conf.DEF_RES_TBL_COL_NAMES
        self._def_col_units = _base_conf.DEF_RES_TBL_COL_UNITS
        self._fname_table = self._def_path / self._def_name

        self._create_obs_table()

    @property
    def obs_info(self):
        return self._obs_info

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
        """ Check for tle file location """
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

        # regex = re.compile(r'tle_{0}_{1}_.+.txt'.format(sat_type,
        #                                                 obs_date))

        regex = re.compile(rf'tle_{sat_type}_{obs_date[0]}[-_]{obs_date[1]}[-_]{obs_date[2]}.+.txt')
        # print(regex)
        path = os.walk(search_path)
        for root, dirs, files in path:
            f = sorted([s for s in files if re.search(regex, s)])
            # f = sorted([s for s in files if f'tle_{sat_type}' in s])
            if len(f) > 0:
                [tle_filenames.append((s, Path(os.path.join(root, s)))) for s in f]

        if not tle_filenames:
            self._log.warning("NO tle file found!!! "
                              "TBW: Get tle. "
                              "Exiting for now!!!")
            return None
            # sys.exit(1)
        elif len(tle_filenames) > 1:
            # first check if one file has unique in name
            idx_arr = np.array([i[0].find('unique') for i in tle_filenames])
            unique_idx = np.where(idx_arr == 0)[0]
            if unique_idx and len(unique_idx) > 1:
                self._log.warning("Multiple unique tle files found. "
                                  "This really should not happen!!! TBW: select")
            elif unique_idx and len(unique_idx) == 1:
                return tle_filenames[unique_idx[0]]
            # if there is still more than one file, user select
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
                              "TBW: calculate visibility from tle file for obs-mid time. "
                              "Exiting for now!!!")
            return False

        # read visibility file found or calculated
        self._read_vis_file(vis_file_list=vis_filenames)

        return True

    def load_obs_table(self):
        """Load the observation info table"""
        obs_info = self._obs_info

        # files info
        fname = self._fname_table
        if fname.exists():
            if not self._silent:
                self._log.info(f'> Read {self._def_name}')
            obs_info = pd.read_csv(fname, header=0, delimiter=',')
            # obs_info.drop([0], axis=0, inplace=True)
            # obs_info.reset_index(drop=True)
            # obs_info = obs_info.apply(pd.to_numeric, errors='ignore')
        else:
            self._log.error(f'> {self._def_name} does not exist!! '
                            f'This should not happen.')
        self._obs_info = obs_info

    def get_object_data(self, fname, kwargs, radec_separator=':'):
        """Extract a row from the obs info table"""
        obs_info = self._obs_info

        # make a copy
        obs_info_tmp = obs_info.copy()

        # convert file name to uppercase
        obs_info_tmp['File'] = obs_info['File'].str.upper()
        fname_upper = fname.upper()

        # check position
        mask = obs_info_tmp['File'] == fname_upper
        pos = np.flatnonzero(mask)

        if list(pos):
            self._log.debug("  Possible match with previous entry found. Check RA and DEC.")
            # check pointing in RA and DEC to exclude duplicates
            ra_file = [kwargs[k] for k in _base_conf.DEF_KEY_TRANSLATIONS['RA']
                       if k in kwargs][0]
            dec_file = [kwargs[k] for k in _base_conf.DEF_KEY_TRANSLATIONS['DEC']
                        if k in kwargs][0]

            df = obs_info[mask][['RA', 'DEC']].astype(str)

            mask = (df.select_dtypes(object)
                    .apply(lambda x: x.str.contains(radec_separator, case=False))
                    .any().any())
            if not mask:
                df = df.astype(float)

            coord_match = df.isin({'RA': [ra_file], 'DEC': [dec_file]}).all(axis=1)
            if coord_match.any():
                self._log.debug("  Coordinate/Pointing match found. Updating.")
                idx = pos[coord_match]
                self._obj_info = obs_info.iloc[idx]
            else:
                self._log.error("  NO Coordinate/Pointing match found. This should not happen!!!")
        else:
            self._log.error("  File not found. This should not happen!!!")

    def get_sat_data_from_visibility(self, sat_name: str, obs_date, pointing: tuple):
        """Extract a row from the visibility info table"""
        vis_info = self._vis_info
        c0 = SkyCoord(ra=pointing[0], dec=pointing[1], unit=(u.hourangle, u.deg))

        mask = vis_info['ID'] == sat_name
        pos = np.flatnonzero(mask)

        if list(pos):

            if len(pos) == 1:
                self._sat_info = vis_info[mask]
                return

            vis_info_masked = vis_info[mask]

            date_info = vis_info_masked[['UT Date', 'UT time']].copy(deep=True)
            # print(date_info)
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

            mask = sep_idx if not dt_idx == sep_idx else dt_idx

            self._sat_info = vis_info_masked.iloc[mask].to_frame(0).T
        else:
            self._log.error("  Satellite ID not found in visibility file. "
                            "This should not happen!!!")

    def get_satellite_id(self, obj_in):
        """Get satellite id from object name"""
        # todo: implement new addition of unique ID in sat name
        self._log.debug("  Identify Satellite name")

        # convert to uppercase
        obj_name = obj_in.upper()

        # remove extensions to name
        if '-N' in obj_name:
            obj_name = str(obj_name).replace('-N', '')

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
        sat_name = 'ONEWEB' if ('OW' in obj_name or 'ONEWEB' in obj_name) else 'N/A'

        # StarLink
        sat_name = 'STARLINK' if 'STARLINK' in obj_name else sat_name

        # Bluewalker
        sat_name = 'BLUEWALKER' if 'BLUEWALKER' in obj_name else sat_name

        # get the satellite id number and convert to 4 digit string
        nb_str = ''.join(n for n in obj_name if n.isdigit())
        sat_id = f"{sat_name}-{int(nb_str):d}" if 'BLUEWALKER' in sat_name else f"{sat_name}-{int(nb_str):04d}"

        return sat_id

    def update_obs_table(self, file, kwargs, radec_separator=':'):
        """Load the observation obs info table.

        Load table with observation information.
        If the file does not exist, it is created.
        """

        obs_info = self._obs_info
        fname = self._fname_table

        # update the columns in case the table was changed
        # if not self._silent:
        #     self._log.info('  Updating column names')
        for col in self._def_cols:
            if col not in obs_info:
                # insert column at the end
                obs_info.insert(len(obs_info.columns), col, '')

        # reorder columns
        obs_info = obs_info[self._def_cols]

        # replace all None values with NAN
        obs_info.replace(to_replace=[None], value=np.nan, inplace=True)

        # exposure time
        expt = kwargs['EXPTIME']

        # update rows
        # if not self._silent:
        #     self._log.info('  Updating table values')

        # make a copy
        obs_info_tmp = obs_info.copy()

        # convert file name to uppercase
        obs_info_tmp['File'] = obs_info['File'].str.upper()
        file_upper = file.upper()

        # check position
        mask = obs_info_tmp['File'] == file_upper
        pos = np.flatnonzero(mask)

        if not list(pos):
            self._log.debug("  File not found. Adding new entry.")
            obs_info = self._add_row(obs_info, file, kwargs, self._def_cols, expt)
        else:
            self._log.debug("  Possible match with previous entry found. Check RA and DEC.")

            # check pointing in RA and DEC to exclude duplicates
            ra_file = [kwargs[k] for k in _base_conf.DEF_KEY_TRANSLATIONS['RA'] if k in kwargs][0]
            dec_file = [kwargs[k] for k in _base_conf.DEF_KEY_TRANSLATIONS['DEC'] if k in kwargs][0]

            df = obs_info[mask][['RA', 'DEC']].astype(str)

            mask = (df.select_dtypes(object)
                    .apply(lambda x: x.str.contains(radec_separator, case=False))
                    .any().any())

            if not mask:
                df = df.astype(float)
            coord_match = df.isin({'RA': [ra_file], 'DEC': [dec_file]}).all(axis=1)
            if coord_match.any():
                self._log.debug("  Coordinate/Pointing match found. Updating.")
                idx = pos[coord_match]
                obs_info = self._update_row(obs_info, idx, kwargs, self._def_cols, expt)
            else:
                self._log.debug("  NO Coordinate/Pointing match found. Adding entry.")
                obs_info = self._add_row(obs_info, file, kwargs, self._def_cols, expt)

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
        # if not self._silent:
        #     self._log.info('  Table updated')
        del obs_info, obs_info_tmp
        gc.collect()

    def _update_row(self, obs_info, idx, kwargs, def_cols, expt):
        """ Update row in result table"""

        for i in range(1, len(def_cols)):
            dcol_name = def_cols[i]
            for key, value in kwargs.items():
                if dcol_name in _base_conf.DEF_KEY_TRANSLATIONS and \
                        key in _base_conf.DEF_KEY_TRANSLATIONS[dcol_name]:
                    if dcol_name == 'Object':
                        # create uniform satellite id from object name
                        sat_name = self.get_satellite_id(kwargs['OBJECT'])
                        obs_info.loc[idx, dcol_name] = value
                        obs_info.loc[idx, 'Sat-Name'] = sat_name
                    elif dcol_name == 'Date-Obs':
                        t_short, t_start, t_mid, t_stop = self._update_times(value, expt, kwargs)

                        obs_info.loc[idx, dcol_name] = t_short
                        obs_info.loc[idx, 'Obs-Start'] = t_start
                        obs_info.loc[idx, 'Obs-Mid'] = t_mid.strftime('%H:%M:%S.%f')[:-3]
                        obs_info.loc[idx, 'Obs-Stop'] = t_stop.strftime('%H:%M:%S.%f')[:-3]
                    elif dcol_name == 'WCS_cal':
                        obs_info.loc[idx, dcol_name] = 'T' if value else 'F'
                    else:
                        obs_info.loc[idx, dcol_name] = value
                else:
                    if key == dcol_name and dcol_name not in ['Obs-Start', 'Obs-Mid', 'Obs-Stop', 'Sat-Name']:
                        obs_info.loc[idx, key] = value

        return obs_info

    def _add_row(self, obs_info, file, kwargs, def_cols, expt):
        """ Add new row to result table"""
        new_dict = {def_cols[0]: [file]}
        for i in range(len(def_cols)):
            dcol_name = def_cols[i]
            for key, value in kwargs.items():
                if dcol_name in _base_conf.DEF_KEY_TRANSLATIONS and \
                        key in _base_conf.DEF_KEY_TRANSLATIONS[dcol_name]:
                    if dcol_name == 'Object':
                        # create uniform satellite id from object name
                        sat_name = self.get_satellite_id(kwargs['OBJECT'])
                        new_dict[dcol_name] = value
                        new_dict['Sat-Name'] = sat_name
                    elif dcol_name == 'Date-Obs':
                        t_short, t_start, t_mid, t_stop = self._update_times(value, expt, kwargs)

                        new_dict[dcol_name] = t_short
                        new_dict['Obs-Start'] = t_start
                        new_dict['Obs-Mid'] = t_mid.strftime('%H:%M:%S.%f')[:-3]
                        new_dict['Obs-Stop'] = t_stop.strftime('%H:%M:%S.%f')[:-3]
                    elif dcol_name == 'WCS_cal':
                        new_dict[dcol_name] = 'T' if value else 'F'
                    else:
                        new_dict[dcol_name] = value
                else:
                    if key == dcol_name and dcol_name not in ['Obs-Start', 'Obs-Mid', 'Obs-Stop', 'Sat-Name']:
                        new_dict[dcol_name] = value

        new_df = pd.DataFrame(new_dict)
        obs_info = obs_info.append(new_df, ignore_index=False)

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

            # load into list
            with open(file, 'r') as f:
                reader = csv.reader(f)
                lines = list(reader)

            # check if the first line has multiple words
            has_multiple_words = self.contains_multiple_words(lines[1][0])
            two_lines = False if has_multiple_words else True

            for line in lines[1:]:

                # split line using regex
                row = re.split(r'\s\s|\s|\t', line[0])
                l_row = len(row)

                if two_lines:
                    # print(two_lines, l_row)
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

        fname = self._fname_table
        df = pd.DataFrame(columns=self._def_cols)
        if not fname.exists():
            if not self._silent:
                self._log.info('> Create obs_info data frame')
                df.to_csv(fname, header=self._def_cols, index=False)
        self._obs_info = df.copy()

    def _update_times(self, value, expt, kwargs):
        """ Update Observation date, start, mid, and end times."""
        self._log.debug("  Check Obs-Date")
        if 'time-obs'.upper() in kwargs:
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


def main():
    """ Main procedure """


# -----------------------------------------------------------------------------


# standard boilerplate to set 'main' as starting function
if __name__ == '__main__':
    main()
# -----------------------------------------------------------------------------
