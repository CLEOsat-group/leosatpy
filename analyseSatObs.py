#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Name:         analyseSatObs.py
# Purpose:      Detect satellite trails and measure integrated flux and
#               differential magnitudes
#
#
# Author:       p4adch (cadam)
#
# Created:      05/05/2022
# Copyright:    (c) p4adch 2010-
#
# Changelog
# version 0.1.0 added first method's
# version 0.2.0 added final calculations and stored the result
# version 0.3.0 added plots for trail parameter estimation
# version 0.3.1 bug fixes and optimization
# version 0.4.0 changed detection method to include subtraction of images if available to remove stars
#               and help with detection
# version 0.4.1 bug fixes and optimization of documentation
# version 0.4.2 bug fixes in reversed search for sat trails
# version 0.4.3 added observer satellite range to results
# version 0.5.0 remake of standard stars selection and photometry
# -----------------------------------------------------------------------------

""" Modules """
# STDLIB
import argparse
import gc
import itertools
import operator
import os
import logging
import sys
import warnings
import time
import configparser
import collections
from datetime import (timedelta, datetime, timezone)

from pathlib import Path

# THIRD PARTY
import pandas as pd
import numpy as np
import astroalign

# astropy
import astropy.table
from astropy.io import fits
from astropy import units as u
from astropy.stats import sigma_clip
from astropy.utils.exceptions import (AstropyUserWarning, AstropyWarning)
from astropy.wcs import WCS
from astropy.visualization import (LinearStretch, LogStretch, SqrtStretch)
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.coordinates import (Angle, SkyCoord, EarthLocation)
from astropy.stats import (sigma_clipped_stats, mad_std)
from skimage.draw import disk
from skimage.measure import label

from photutils.aperture import (RectangularAperture, RectangularAnnulus, CircularAperture)
from photutils.centroids import (centroid_sources, centroid_2dg, centroid_com)
from photutils.segmentation import (SegmentationImage, detect_sources, detect_threshold)

import ephem
import pyorbital
from pyorbital.orbital import Orbital

from scipy.stats import norm
from scipy import ndimage as nd

# plotting; optional
try:
    import matplotlib
except ImportError:
    plt = None
    mpl = None
    warnings.warn('matplotlib not found, plotting is disabled', AstropyUserWarning)
else:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec  # GRIDSPEC !
    from matplotlib.ticker import AutoMinorLocator
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.patches as mpl_patches
    import matplotlib.lines as mlines
    import matplotlib.colors as mcolors
    import matplotlib.cm as cm
    import matplotlib.ticker as ticker

    # matplotlib parameter
    mpl.use('Qt5Agg')
    mpl.rc("lines", linewidth=1.2)
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42

    mpl.rc('figure', dpi=300, facecolor='w', edgecolor='k')
    mpl.rc('text.latex', preamble=r'\usepackage{sfmath}')

from utils.arguments import ParseArguments
from utils.dataset import DataSet
from utils.tables import ObsTables
import utils.sources as sext
import utils.satellites as sats
import utils.transformations as imtrans
import utils.photometry as phot
import config.base_conf as _base_conf

# -----------------------------------------------------------------------------

""" Meta-info """
__author__ = "Christian Adam"
__copyright__ = 'Copyright 2021, UA, LEOSat observations'
__credits__ = ["Christian Adam, Eduardo Unda-Sanzana, Jeremy Tregloan-Reed"]
__license__ = "Free"
__version__ = "0.4.2"
__maintainer__ = "Christian Adam"
__email__ = "christian.adam84@gmail.com"
__status__ = "Production"

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

# Warnings
# np.seterr(divide='ignore', invalid='ignore')
# np.errstate(invalid='ignore')
# warnings.simplefilter('ignore', category=AstropyWarning)

__taskname__ = 'analyseSatObs'

# figure size
figsize = (10, 6)

pass_str = _base_conf.BCOLORS.PASS + "SUCCESSFUL" + _base_conf.BCOLORS.ENDC
fail_str = _base_conf.BCOLORS.FAIL + "FAILED" + _base_conf.BCOLORS.ENDC

frmt = "%Y-%m-%dT%H:%M:%S.%f"


# todo: rethink the config structure!!!
# -----------------------------------------------------------------------------


# noinspection PyAugmentAssignment
class AnalyseSatObs(object):
    """Perform photometric, i.e. integrated flux and differential magnitude,
     analysis on satellite trails.
     """

    def __init__(self,
                 input_path: str, args: argparse.Namespace = None,
                 silent: bool = False, verbose: bool = False,
                 log: logging.Logger = _log, log_level: int = _log_level):
        """ Constructor with default values """

        if args is None:
            raise ValueError('Args argument must be a dict or argparse.Namespace.')

        plot_images = args.plot_images
        ignore_warnings = args.ignore_warnings
        if silent:
            verbose = False
            ignore_warnings = True
            plot_images = False

        if verbose:
            log = logging.getLogger()
            log.setLevel("debug".upper())
            log_level = log.level

        if plt is None or silent or not plot_images:
            plt.ioff()

        if ignore_warnings:
            _base_conf.load_warnings()

        # set variables
        self._root_path = Path(os.getcwd())
        self._config = collections.OrderedDict()
        self._root_dir = _base_conf.ROOT_DIR
        self._input_path = input_path
        self._dataset_object = None
        self._dataset_all = None

        self._band = args.band
        self._catalog = args.catalog
        self._photo_ref_cat_fname = args.ref_cat_phot_fname
        self._cutouts = args.cutout
        self._vignette = args.vignette
        self._vignette_rectangular = args.vignette_rectangular

        self._log = log
        self._log_level = log_level
        self._hdu_idx = args.hdu_idx
        self._force_extract = args.force_detection
        self._force_download = args.force_download
        self._plot_images = plot_images
        self._work_dir = None
        self._plt_path = None
        self._aux_path = None
        self._cat_path = None
        self._sat_id = None
        self._silent = silent
        self._verbose = verbose

        self._instrument = None
        self._telescope = None
        self._obsparams = None

        self._obsTable = None
        self._obs_info = None
        self._obj_info = None
        self._roi_info = None

        self._observer = None

        # run trail analysis
        self._run_trail_analysis()

    def _run_trail_analysis(self):
        """Prepare and run satellite trail photometry"""

        StartTime = time.perf_counter()
        self._log.info('====> Analyse satellite trail init <====')
        self._load_config()
        self._obsTable = ObsTables(config=self._config)
        self._obsTable.load_obs_table()
        self._obs_info = ObsTables.obs_info
        self._obj_info = ObsTables.obj_info

        # Load Region-of-Interest
        self._obsTable.load_roi_table()

        if not self._silent:
            self._log.info("> Search input and prepare data")
        self._log.debug("  > Check input argument(s)")

        # prepare dataset from input argument
        ds = DataSet(input_args=self._input_path,
                     prog_typ='satPhotometry',
                     log=self._log, log_level=self._log_level)
        self._dataset_all = ds

        # set variables for use
        inst_list = ds.instruments_list
        inst_data = ds.instruments
        N_inst = len(inst_list)

        fails = []
        pass_counter = 0
        time_stamp = self.get_time_stamp()

        # loop telescopes
        for i in range(N_inst):
            inst = inst_list[i]
            self._instrument = inst
            self._telescope = inst_data[inst]['telescope']
            self._obsparams = inst_data[inst]['obsparams']
            self._dataset_object = inst_data[inst]['dataset']

            # loop over groups and run reduction for each group
            for obj_pointing, file_list in self._dataset_object:
                files = file_list['file'].values
                N_files = len(file_list)

                unique_obj = np.unique(np.array([self._obsTable.get_satellite_id(f)[0]
                                                 for f in file_list['object'].values]))
                if len(unique_obj) > 1:
                    self._log.warning("More than one object name identified. "
                                      "This should not happen!!! Please check the file header. "
                                      f"Skip pointing RA={obj_pointing[0]}, "
                                      f"DEC={obj_pointing[1]}.")
                    fails.append([self._telescope, ';'.join(unique_obj),
                                  'IDError', 'Multiple ID for same pointing.',
                                  'Check/Compare FITS header and visibility files for naming issues.'])
                    continue

                sat_name, _ = self._obsTable.get_satellite_id(unique_obj[0])
                if not self._silent:
                    self._log.info("====> Analyse satellite trail run <====")
                    self._log.info(
                        f"> Analysing {N_files} fits files for {sat_name} with pointing "
                        f"RA={obj_pointing[0]}, DEC={obj_pointing[1]} "
                        f"observed with the {self._telescope} telescope")

                # result, error = self._analyse_sat_trails(files=files, sat_name=sat_name)
                try:
                    result, error = self._analyse_sat_trails(files=files, sat_name=sat_name)
                except Exception as e:
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                    self._log.critical(f"Unexpected behaviour: {e} in file {fname}, "
                                       f"line {exc_tb.tb_lineno}")
                    error = [str(e), f'file: {fname}, line: {exc_tb.tb_lineno}',
                             'Please report to christian.adam84@gmail.com']
                    result = False

                if result:
                    self._log.info(f">> Report: Satellite detection and analysis was {pass_str}")
                    pass_counter += 1
                else:
                    self._log.info(f">> Report: Satellite detection and analysis has {fail_str}")
                    location = os.path.dirname(files[0])
                    tmp_fail = [location, self._telescope, unique_obj[0]]
                    tmp_fail += error
                    fails.append(tmp_fail)
                if not self._silent:
                    self._log.info(f'====>  Analysis of {sat_name} finished <====')

                del result
                gc.collect()

        # save fails
        if fails:
            fname = Path(self._config['RESULT_TABLE_PATH']) / f'fails_analyseSatObs_{time_stamp}.log'
            with open(fname, "w", encoding="utf8") as file:
                file.write('\t'.join(['Location', 'Telescope',
                                      'SatID', 'ErrorMsg', 'Cause', 'Hint']) + '\n')
                [file.write('\t'.join(f) + '\n') for f in fails]

        EndTime = time.perf_counter()
        dt = EndTime - StartTime
        td = timedelta(seconds=dt)

        if not self._silent:
            self._log.info(f"Program execution time in hh:mm:ss: {td}")
        self._log.info('====>  Satellite analysis finished <====')
        sys.exit(0)

    def _load_config(self):
        """ Load base configuration file """

        # configuration file name
        configfile = f"{self._root_dir}/leosatpy_config.ini"
        config = configparser.ConfigParser()
        config.optionxform = lambda option: option

        self._log.info('> Read configuration')
        config.read(configfile)

        for group in ['Satellite_analysis', 'Detection']:
            items = dict(config.items(group))
            self._config.update(items)
            for key, value in items.items():
                try:
                    val = eval(value)
                except NameError:
                    val = value
                self._config[key] = val

    def _analyse_sat_trails(self, files: list, sat_name: str):
        """ Run satellite trail detection and photometry.

        Detect which image has a trail, determine parameters,
        perform relative photometry with standard stars,
        and calculate angles and corrections from tle data.

        Parameters
        ----------
        files :
            Input file list
        sat_name:
            Satellite designation.

        """

        obsparams = self._obsparams

        # identify images with satellite trails and get parameters
        if self._silent:
            self._log.info("> Identify image with satellite trail(s)")
        data_dict, fail_msg = self._identify_trails(files)
        if data_dict is None:
            return False, fail_msg

        # collect information on the satellite image
        trail_img_idx = data_dict['trail_img_idx'][0]
        trail_img_dict = self._get_img_dict(data_dict, trail_img_idx, obsparams)

        # get catalogs excluding the trail area and including it
        trail_img_dict, state, fail_msg = self._filter_catalog(data=trail_img_dict)

        if not state:
            self._log.critical("No matching sources detected!!! Skipping further steps. ")
            del obsparams, data_dict, trail_img_dict
            gc.collect()
            return False, fail_msg

        # select std stars and perform aperture photometry
        std_photometry_results = self._run_std_photometry(data=trail_img_dict)
        (std_apphot, std_filter_keys, optimum_aprad,
         mag_conv, mag_corr, qlf_aprad, state, fail_msg) = std_photometry_results
        if not state:
            self._log.critical("Standard star selection and photometry has FAILED!!! "
                               "Skipping further steps. ")
            del obsparams, std_photometry_results, std_apphot, data_dict, trail_img_dict
            gc.collect()
            return False, fail_msg

        # prepare trail image and perform aperture photometry with optimum aperture
        sat_apphot = self._run_sat_photometry(data_dict=data_dict,
                                              img_dict=trail_img_dict,
                                              qlf_aprad=qlf_aprad)

        # load tle data, extract info for the satellite and calc mags, angles etc.
        state, fail_msg = self._get_final_results(img_dict=trail_img_dict,
                                                  sat_name=sat_name,
                                                  sat_apphot=sat_apphot,
                                                  std_apphot=std_apphot,
                                                  std_filter_keys=std_filter_keys,
                                                  mag_conv=mag_conv, mag_corr=mag_corr)

        del obsparams, data_dict, trail_img_dict, std_photometry_results, sat_apphot
        gc.collect()

        if not state:
            return False, fail_msg

        return True, fail_msg

    def _get_final_results(self,
                           img_dict: dict,
                           sat_name: str,
                           sat_apphot: dict,
                           std_apphot: astropy.table.Table,
                           std_filter_keys: dict,
                           mag_conv: bool, mag_corr: tuple):
        """Get final results

        Use all collected data to estimate all parameters required for publication.
        Calculate observed and estimated lengths, magnitudes and angles employing the available
        telemetry and observation datasets.

        Parameters
        ----------
        img_dict:
            Dictionary with trail image data
        sat_name:
            Satellite identifier
        sat_apphot:
            Catalog with satellite photometry
        std_apphot:
            Catalog with standard star photometry
        std_filter_keys:
            Standard star filter band and error names
        """

        obj_info = self._obj_info

        hdr = img_dict['hdr']
        obsparams = self._obsparams
        file_name = img_dict['fname']
        n_trails_max = self._config['N_TRAILS_MAX']

        # observation and location data
        geo_lon = obsparams['longitude']
        geo_lat = obsparams['latitude']
        geo_alt = obsparams['altitude'] / 1.e3  # in km

        # get additional info
        exptime = hdr['exptime']
        pixelscale = hdr['pixscale']  # arcsec/pix
        pixelscale_err = 0.05 * pixelscale

        if not self._silent:
            self._log.info("> Perform final analysis of satellite trail data")

        # get nominal orbit altitude
        sat_h_orb_ref = _base_conf.SAT_HORB_REF['ONEWEB']
        if 'STARLINK' in sat_name:
            sat_h_orb_ref = _base_conf.SAT_HORB_REF['STARLINK']
        if 'BLUEWALKER' in sat_name:
            sat_h_orb_ref = _base_conf.SAT_HORB_REF['BLUEWALKER']

        # get time delta of obs mid time
        date_obs = obj_info['Date-Obs'].to_numpy(dtype=object)[0]
        obs_mid = obj_info['Obs-Mid'].to_numpy(dtype=object)[0]
        obs_start = obj_info['Obs-Start'].to_numpy(dtype=object)[0]
        obs_end = obj_info['Obs-Stop'].to_numpy(dtype=object)[0]
        dtobj_obs = datetime.strptime(f"{date_obs}T{obs_mid}", "%Y-%m-%dT%H:%M:%S.%f")

        # standard star data
        _aper_sum = np.array(list(zip(std_apphot['flux_counts_aper'],
                                      std_apphot['flux_counts_err_aper'])))
        _mags = np.array(list(zip(std_apphot[std_filter_keys[0]],
                                  std_apphot[std_filter_keys[1]])))
        std_pos = np.array(list(zip(std_apphot['xcentroid'],
                                    std_apphot['ycentroid'])))

        if not self._silent:
            self._log.info("  > Find tle file location")
        # get tle file and calculate angular velocity
        tle_location = self._obsTable.find_tle_file(sat_name=sat_name,
                                                    img_filter=hdr['FILTER'],
                                                    src_loc=img_dict['loc'])

        if tle_location is None:
            return False, ['LoadTLEError',
                           'TLE file not found',
                           'Check that naming of the tle file']

        # if not pd.isnull(sat_info['AltID']).any():
        #     sat_name = f'{sat_name} ({sat_info["AltID"].values[0]})'
        # if not pd.isnull(sat_info['UniqueID']).any() and 'BLUEWALKER' not in sat_name:
        #     sat_name = f'{sat_name}-{sat_info["UniqueID"].values[0]}'
        if not self._silent:
            self._log.info("  > Calculate satellite information from tle")
        sat_vel, sat_vel_err, tle_pos_df = self._get_data_from_tle(hdr=hdr, sat_name=sat_name,
                                                                   tle_location=str(tle_location[2]),
                                                                   date_obs=date_obs,
                                                                   obs_times=[obs_start, obs_mid, obs_end],
                                                                   exptime=exptime,
                                                                   dt=self._config['DT_STEP_SIZE'])

        sat_vel = float('%.3f' % sat_vel)
        sat_vel_err = float('%.3f' % sat_vel_err)

        sat_info = tle_pos_df[tle_pos_df['Obs_Time'] == dtobj_obs.strftime(frmt)[:-3]]

        # required satellite data
        sat_info.reset_index(drop=True)
        result = sat_info.iloc[0].to_dict()
        sat_lon = sat_info['SatLon'].to_numpy(dtype=float)[0]
        sat_lat = sat_info['SatLat'].to_numpy(dtype=float)[0]
        sat_alt = sat_info['SatAlt'].to_numpy(dtype=float)[0]
        sat_az = sat_info['SatAz'].to_numpy(dtype=float)[0]
        sat_elev = sat_info['SatElev'].to_numpy(dtype=float)[0]

        for i in range(n_trails_max):
            reg_info = img_dict["trail_data"]['reg_info_lst'][i]
            if not self._silent:
                self._log.info("  > Combine data to calculate angles and magnitudes")

            # observed trail length
            obs_trail_len = reg_info['width'] * pixelscale
            obs_trail_len_err = np.sqrt((pixelscale * reg_info['e_width']) ** 2
                                        + (reg_info['width'] * pixelscale_err) ** 2)

            # estimated trail length and count
            est_trail_len = exptime * sat_vel
            est_trail_len_err = np.sqrt(est_trail_len)

            # time on detector
            time_on_det = obs_trail_len / sat_vel
            time_on_det_err = np.sqrt((obs_trail_len_err / sat_vel) ** 2
                                      + (obs_trail_len * sat_vel_err / sat_vel ** 2) ** 2)

            # calculate solar incidence angle theta_1
            sun_inc_ang = sats.sun_inc_elv_angle(dtobj_obs,
                                                 (sat_lat, sat_lon))

            # range, distance observer satellite
            dist_obs_sat = sats.get_obs_range(sat_elev, sat_alt, geo_alt, geo_lat)

            # calculate observer phase angle theta_2
            phi = sats.get_observer_angle(sat_lat, geo_lat, sat_alt, geo_alt, dist_obs_sat)

            # eta, distance observer to satellite on the surface of the earth
            # eta = sats.get_distance((geo_lat, geo_lon), (sat_lat, sat_lon))
            # phi_rad = np.arcsin((eta / sat_alt) * np.sin(np.deg2rad(sat_elev)))
            # phi = np.rad2deg(phi_rad)

            sun_sat_ang, sun_phase_angle, sun_az, sun_alt = sats.get_solar_phase_angle(sat_az=sat_az,
                                                                                       sat_alt=sat_elev,
                                                                                       geo_loc=(geo_lat, geo_lon),
                                                                                       obs_range=dist_obs_sat,
                                                                                       obsDate=dtobj_obs)
            # get scaled estimated magnitude with scale factor -5log(range/h_orb_ref)
            mag_scale = -5. * np.log10((dist_obs_sat / sat_h_orb_ref))

            # observed trail count
            obs_trail_count = sat_apphot[i]['aperture_sum_bkgsub'].values[0]
            obs_trail_count_err = sat_apphot[i]['aperture_sum_bkgsub_err'].values[0]

            # estimated trail count
            est_trail_count = obs_trail_count * (est_trail_len / obs_trail_len)
            est_trail_count_err = ((obs_trail_count + obs_trail_count_err)
                                   * ((est_trail_len + est_trail_len_err)
                                      / (obs_trail_len - obs_trail_len_err))) - est_trail_count

            obs_mag_avg_w = None
            obs_mag_avg_err = None
            est_mag_avg_w = None
            est_mag_avg_err = None
            est_mag_avg_w_scaled = None
            est_mag_avg_err_scaled = None

            if obs_trail_count > 0.:
                # observed satellite magnitude relative to std stars
                obs_mag_avg = sats.get_average_magnitude(flux=obs_trail_count,
                                                         flux_err=obs_trail_count_err,
                                                         std_fluxes=_aper_sum,
                                                         mag_corr=mag_corr,
                                                         std_mags=_mags)
                obs_mag_avg_w, obs_mag_avg_err = obs_mag_avg

                # estimated magnitude difference
                est_mag_avg = sats.get_average_magnitude(flux=est_trail_count,
                                                         flux_err=est_trail_count_err,
                                                         std_fluxes=_aper_sum,
                                                         mag_corr=mag_corr,
                                                         std_mags=_mags)
                est_mag_avg_w, est_mag_avg_err = est_mag_avg

                # estimated magnitude difference scaled with mag scale
                est_mag_avg_scaled = sats.get_average_magnitude(flux=est_trail_count,
                                                                flux_err=est_trail_count_err,
                                                                std_fluxes=_aper_sum,
                                                                std_mags=_mags,
                                                                mag_corr=mag_corr,
                                                                mag_scale=mag_scale)

                est_mag_avg_w_scaled, est_mag_avg_err_scaled = est_mag_avg_scaled

            tbl_names = [obsparams['ra'], obsparams['dec'],
                         obsparams['instrume'], obsparams['object'],
                         obsparams['exptime'],
                         'SatAngularSpeed', 'e_SatAngularSpeed',
                         'ObsSatRange',
                         'ToD', 'e_ToD',
                         'ObsTrailLength', 'e_ObsTrailLength',
                         'EstTrailLength', 'e_EstTrailLength',
                         'SunSatAng', 'SunPhaseAng', 'SunIncAng', 'ObsAng',
                         'ObsMag', 'e_ObsMag',
                         'EstMag', 'e_EstMag',
                         'EstScaleMag', 'e_EstScaleMag',
                         'SunAzAng', 'SunElevAng',
                         'FluxScale', 'MagScale', 'MagCorrect', 'e_MagCorrect',
                         'dt_tle-obs', 'mag_conv']

            ra = hdr[obsparams['ra']]
            dec = hdr[obsparams['dec']]
            if obsparams['radec_separator'] == 'XXX':
                ra = round(hdr[obsparams['ra']], _base_conf.ROUND_DECIMAL)
                dec = round(hdr[obsparams['dec']], _base_conf.ROUND_DECIMAL)

            tbl_vals = [ra, dec,
                        hdr[obsparams['instrume']],
                        hdr[obsparams['object']],
                        hdr[obsparams['exptime']],
                        sat_vel, sat_vel_err, dist_obs_sat,
                        time_on_det, time_on_det_err,
                        obs_trail_len, obs_trail_len_err,
                        est_trail_len, est_trail_len_err,
                        sun_sat_ang, sun_phase_angle, sun_inc_ang, phi,
                        obs_mag_avg_w, obs_mag_avg_err,
                        est_mag_avg_w, est_mag_avg_err,
                        est_mag_avg_w_scaled, est_mag_avg_err_scaled,
                        sun_az, sun_alt, (est_trail_len / obs_trail_len), mag_scale,
                        mag_corr[0], mag_corr[1], 0,
                        'T' if mag_conv else 'F']

            for k, v in zip(tbl_names, tbl_vals):
                result[k] = v

            result['RADEC_Separator'] = obsparams['radec_separator']
            file = img_dict['fname']
            self._obsTable.update_obs_table(file=file,
                                            kwargs=result, obsparams=obsparams)

            if not self._silent:
                self._log.info("  > Plot trail detection result (including tle positions)")

            # final plot
            self._plot_final_result(img_dict=img_dict,
                                    reg_data=reg_info,
                                    obs_date=(date_obs, obs_mid),
                                    expt=exptime,
                                    std_pos=std_pos,
                                    file_base=file_name,
                                    tle_pos=tle_pos_df)

        # if not self._silent:
        #     if dt_sec > exptime / 2.:
        #         self._log.warning('  Time difference lager than half of exposure time!!! '
        #                           'Please consider using more accurate visibility data.')

        del _mags, _aper_sum, sat_info, obj_info, hdr, obsparams, img_dict, sat_apphot, std_apphot
        gc.collect()

        return True, []

    def _run_sat_photometry(self,
                            data_dict: dict,
                            img_dict: dict, qlf_aprad: bool):
        """Perform aperture photometry on satellite trails"""

        config = self._config
        obspar = self._obsparams
        imgarr = img_dict['imgarr']
        img_bkg = img_dict['bkg_data']['bkg']
        img_bkg_rms = img_dict['bkg_data']['bkg_rms_med']
        hdr = img_dict['hdr']
        file_name = img_dict['fname']
        fwhm = img_dict['kernel_fwhm'][0]
        n_trails = self._config['N_TRAILS_MAX']

        if not self._silent:
            self._log.info("> Perform photometry on satellite trails")

        # choose reference image if available (!!! only first at the moment !!!)
        ref_img_idx = None
        has_ref = False
        N_ref = None
        if list(data_dict['ref_img_idx']):
            idx_list = data_dict['ref_img_idx']
            ref_img_idx = data_dict['ref_img_idx'][0]
            if len(data_dict['ref_img_idx']) > 1:
                self._log.info(f"  Found {len(idx_list)} reference images to choose from.")
                _, idx = self._dataset_all.select_file_from_list(data=np.array(data_dict['file_names'])[idx_list])
                ref_img_idx = data_dict['ref_img_idx'][np.array(idx)]
            has_ref = True
            N_ref = len(data_dict['ref_img_idx'])

        src_mask = None
        df = img_dict['cats_cleaned_rem']['ref_cat_cleaned']
        if not has_ref and not df.empty:
            src_mask = np.zeros(imgarr.shape)

            # filter apertures outside of image
            trail_stars = df.drop((df.loc[(df['xcentroid'] < 0) |
                                          (df['ycentroid'] < 0) |
                                          (df['xcentroid'] > imgarr.shape[1]) |
                                          (df['ycentroid'] > imgarr.shape[0])]).index)

            # positions of sources detected in the trail image
            _pos = np.array(list(zip(trail_stars['xcentroid'],
                                     trail_stars['ycentroid'])))

            for i in range(_pos.shape[0]):
                rr, cc = disk((_pos[:, 1][i], _pos[:, 0][i]), 2. * fwhm)
                src_mask[rr, cc] = 1

            src_mask = np.ma.make_mask(src_mask)

        ref_img_warped = None
        if has_ref:
            img_mask = data_dict['image_mask'][ref_img_idx]
            ref_imgarr = data_dict['images'][ref_img_idx]
            ref_bkg = data_dict['bkg_data'][ref_img_idx]['bkg']
            ref_bkg_rms = data_dict['bkg_data'][ref_img_idx]['bkg_rms_med']

            img_bkg_sub = imgarr - img_bkg
            if img_mask is not None:
                img_bkg_sub[img_mask] = img_bkg_rms
            img_bkg_sub[img_bkg_sub < 0.] = img_bkg_rms

            ref_img_bkg_sub = ref_imgarr - ref_bkg
            if img_mask is not None:
                ref_img_bkg_sub[img_mask] = ref_bkg_rms
            ref_img_bkg_sub[ref_img_bkg_sub < 0.] = ref_bkg_rms

            try:
                T, _ = astroalign.find_transform(img_bkg_sub, ref_img_bkg_sub,
                                                 detection_sigma=3.,
                                                 max_control_points=100,
                                                 min_area=9)
            except (TypeError, astroalign.MaxIterError):
                T, _ = astroalign.find_transform(nd.gaussian_filter(img_bkg_sub, 2.),
                                                 nd.gaussian_filter(ref_img_bkg_sub, 2.),
                                                 detection_sigma=5.,
                                                 max_control_points=100,
                                                 min_area=9)

            ref_img_warped, footprint = astroalign.apply_transform(T,
                                                                   ref_imgarr,
                                                                   imgarr,
                                                                   propagate_mask=True)
            ref_bkg_warped, footprint = astroalign.apply_transform(T,
                                                                   ref_bkg,
                                                                   img_bkg,
                                                                   propagate_mask=True)
            ref_img_warped = ref_img_warped - ref_bkg_warped

        # loop over each detected trail to get optimum aperture and photometry
        sat_phot_dict = {}
        for i in range(n_trails):
            reg_info = img_dict["trail_data"]['reg_info_lst'][i]

            src_pos = [reg_info['coords']]
            src_pos_err = [reg_info['coords_err']]
            ang = Angle(reg_info['orient_deg'], 'deg')

            if has_ref:
                alpha = 30,
                sigma_blurr = 3
                blurred_f = nd.gaussian_filter(ref_img_warped, 2.)
                filter_blurred_f = nd.gaussian_filter(blurred_f, sigma_blurr)
                sharpened = blurred_f + alpha * (blurred_f - filter_blurred_f)

                mean, _, std = sigma_clipped_stats(sharpened, grow=False)
                sharpened -= mean

                threshold = detect_threshold(sharpened, 1.5, mean, std)
                sources = detect_sources(sharpened,
                                         threshold=threshold,
                                         npixels=5, connectivity=8)
                if sources is not None:
                    segm = SegmentationImage(sources.data)

                    footprint = None  # np.ones((2, 2))
                    src_mask = segm.make_source_mask(footprint=footprint)

            width = reg_info['width']
            optimum_apheight = config['APER_RAD'] * fwhm * 2.
            w_in = reg_info['width'] + config['RSKYIN'] * fwhm
            w_out = reg_info['width'] + config['RSKYOUT'] * fwhm

            # this offset can be used in case two parallel trails are close together and affect the
            # background aperture of the other
            h_offset = 15 if 'BLUEWALKER-3-016' in file_name else 0
            h_in = config['RSKYIN'] * fwhm * 2. + h_offset
            h_out = config['RSKYOUT'] * fwhm * 2. + h_offset

            if not self._silent:
                self._log.info(f'  > Measure photometry for satellite trail.')

            _, _, sat_phot, _, _ = phot.get_aper_photometry(image=imgarr,
                                                            src_pos=src_pos[0],
                                                            mask=src_mask,
                                                            aper_mode='rect',
                                                            width=width, height=optimum_apheight,
                                                            w_in=w_in, w_out=w_out,
                                                            h_in=h_in, h_out=h_out,
                                                            theta=ang)

            # this can be helpful
            # annulus_aperture = RectangularAnnulus(src_pos[0], w_in=w_in, w_out=w_out,
            #                                       h_in=h_in, h_out=h_out,
            #                                       theta=ang
            #                                       )
            # aperture = RectangularAperture(src_pos[0],
            #                                w=width,
            #                                h=optimum_apheight,
            #                                theta=ang
            #                                )
            # plt.imshow(imgarr*~src_mask, interpolation='nearest')
            #
            # ap_patches = aperture.plot(**{'color': 'white',
            #                               'lw': 2, 'label': 'Photometry aperture'})
            # ann_patches = annulus_aperture.plot(**{'color': 'red',
            #                                        'lw': 2, 'label': 'Background annulus'})
            # handles = (ap_patches[0], ann_patches[0])
            # plt.legend(loc='best', facecolor='#458989', labelcolor='white',
            #            handles=handles, prop={'weight': 'bold', 'size': 11})
            # plt.show()

            sat_phot_dict[i] = sat_phot

            # convert pixel to world coordinates with uncertainties
            w = WCS(hdr)
            px = np.array([src_pos[0][0],
                           src_pos[0][0] + src_pos_err[0][0],
                           src_pos[0][0] - src_pos_err[0][0]])
            py = np.array([src_pos[0][1],
                           src_pos[0][1] + src_pos_err[0][1],
                           src_pos[0][1] - src_pos_err[0][1]])

            wx, wy = w.wcs_pix2world(px, py, 0)
            wx_err = np.mean(abs(np.subtract(wx, wx[0])[1:])) * 3600.  # in arcseconds
            wy_err = np.mean(abs(np.subtract(wy, wy[0])[1:])) * 3600.  # in arcseconds

            file = img_dict['fname']

            ra = hdr[obspar['ra']]
            dec = hdr[obspar['dec']]
            if obspar['radec_separator'] == 'XXX':
                ra = round(hdr[obspar['ra']], _base_conf.ROUND_DECIMAL)
                dec = round(hdr[obspar['dec']], _base_conf.ROUND_DECIMAL)
            hdr[obspar['ra']] = ra
            hdr[obspar['dec']] = dec

            kwargs = {obspar['ra']: ra,
                      obspar['dec']: dec,
                      obspar['object']: hdr[obspar['object']],
                      obspar['instrume']: hdr[obspar['instrume']],
                      obspar['exptime']: hdr[obspar['exptime']],
                      'TrailCX': src_pos[0][1],  # in px
                      'e_TrailCX': src_pos_err[0][1],  # in px
                      'TrailCY': src_pos[0][0],  # in px
                      'e_TrailCY': src_pos_err[0][0],  # in px
                      'TrailCRA': wx[0],  # in deg
                      'e_TrailCRA': wx_err,  # in arcseconds
                      'TrailCDEC': wy[0],  # in deg
                      'e_TrailCDEC': wy_err,  # in arcseconds
                      'TrailANG': reg_info['orient_deg'],  # in deg
                      'e_TrailANG': reg_info['e_orient_deg'],  # in deg
                      'OptAperHeight': '%3.2f' % optimum_apheight,
                      'HasRef': 'T' if has_ref else 'F', 'NRef': N_ref,
                      'QlfAperRad': 'T' if qlf_aprad else 'F'}  # in px

            self._obsTable.update_obs_table(file=file, kwargs=kwargs,
                                            obsparams=obspar)
            self._obsTable.get_object_data(fname=file_name, kwargs=hdr,
                                           obsparams=obspar)
            self._obj_info = self._obsTable.obj_info

        del imgarr, ref_img_warped, df, config, obspar, img_bkg, hdr, data_dict, img_dict
        gc.collect()

        return sat_phot_dict

    def _save_fits(self, data: np.ndarray, fname: str):
        """Save Fits file

        Parameters
        ----------
        data:
            Array with fits data
        fname:
            File name
        """
        folder = self._aux_path
        fname = folder / fname
        fits.writeto(filename=f'{fname}.fits',
                     data=data,
                     output_verify='ignore',
                     overwrite=True)

        del data
        gc.collect()

    def _run_std_photometry(self, data: dict):
        """Perform aperture photometry on standard stars.

        Select standard stars and perform aperture photometry
        with bkg annulus sky estimation.
        Use the growth curve to determine the best aperture radius;
        recenter pixel coordinates before execution using centroid poly_func from photutils

        Parameters
        ----------
        data:
            Dictionary with all image data
        """

        config = self._config
        obspar = self._obsparams
        imgarr = data['imgarr']
        hdr = data['hdr']
        std_cat = data['cats_cleaned']['ref_cat_cleaned']
        std_fkeys = data['std_fkeys']
        mag_conv = data['mag_conv']
        file_name = data['fname']
        kernel_fwhm = data['kernel_fwhm'][0]
        exp_time = hdr[obspar['exptime']]
        gain = 1  # hdr['gain']
        rdnoise = 0  # hdr['ron']

        if not self._silent:
            self._log.info("> Perform photometry on standard stars")
            self._log.info("  > Select standard stars")

        # select only good sources if possible
        if 'include_fwhm' in std_cat.columns:
            idx = std_cat['include_fwhm']
        else:
            idx = [True] * len(std_cat)
        std_cat = std_cat[idx]
        # std_cat = std_cat.head(100)

        # get reference positions
        src_pos = np.array(list(zip(std_cat['xcentroid'], std_cat['ycentroid'])))

        if not self._silent:
            self._log.info(f'  > Get optimum aperture from signal-to-noise ratio (S/N) of '
                           f'standard stars')
        result = phot.get_optimum_aper_rad(image=imgarr,
                                           std_cat=std_cat,
                                           fwhm=kernel_fwhm,
                                           exp_time=exp_time,
                                           config=config,
                                           r_in=config['RSKYIN'],
                                           r_out=config['RSKYOUT'],
                                           gain=gain,
                                           rdnoise=rdnoise,
                                           silent=self._silent)

        fluxes, rapers, max_snr_aprad, opt_aprad, qlf_aprad = result

        # plot result of aperture photometry
        self._plot_aperture_photometry(fluxes=fluxes,
                                       rapers=rapers,
                                       opt_aprad=opt_aprad,
                                       file_base=file_name)

        # update aperture values
        config['APER_RAD'] = opt_aprad
        config['INF_APER_RAD'] = opt_aprad + 1.5
        config['RSKYIN'] = opt_aprad + 1.
        config['RSKYOUT'] = opt_aprad + 2.
        self._config = config

        if not self._silent:
            self._log.info(f'  > Measure photometry for {len(std_cat):d} standard stars.')
        std_cat = phot.get_std_photometry(image=imgarr, std_cat=std_cat,
                                          src_pos=src_pos,
                                          fwhm=kernel_fwhm, config=config)

        # estimate magnitude correction due to the finite aperture used
        if not self._silent:
            self._log.info(f'  > Get magnitude correction for finite aperture')
        mag_corr = self._get_magnitude_correction(df=std_cat,
                                                  file_base=file_name)
        if not mag_corr:
            del data, imgarr, catalog, std_cat, result, fluxes
            gc.collect()

            return None, None, None, None, None, None, False, ['MagCorrectError',
                                                               'Error during interpolation.',
                                                               'Check number and '
                                                               'quality of standard stars and SNR']

        if not self._silent:
            self._log.info('    ==> estimated magnitude correction: '
                           '{:.3f} +/- {:.3f} (mag)'.format(mag_corr[0], mag_corr[1]))

        del imgarr, result, src_pos, fluxes, rapers, data
        gc.collect()

        return std_cat, std_fkeys, opt_aprad, mag_conv, mag_corr, qlf_aprad, True, ['', '', '']

    def _get_magnitude_correction(self, df: pd.DataFrame, file_base: str):
        """Estimate magnitude correction"""
        flux_ratios = df['flux_counts_inf_aper'] / df['flux_counts_aper']

        corrections = np.array([-2.5 * np.log10(i) if i > 0.0 else np.nan for i in flux_ratios])
        corrections = corrections[~np.isnan(corrections)]
        mask = np.array(~sigma_clip(corrections, sigma=3.,
                                    maxiters=None, cenfunc=np.nanmedian, stdfunc=mad_std).mask)
        corrections_cleaned = corrections[mask]

        aper_correction = np.nanmedian(corrections_cleaned, axis=None)
        aper_correction_err = np.nanstd(corrections_cleaned, axis=None)

        fig = plt.figure(figsize=(10, 6))
        gs = gridspec.GridSpec(1, 1)
        ax = fig.add_subplot(gs[0, 0])

        # the histogram of the data
        n, bins, patches = ax.hist(corrections_cleaned, bins='auto',
                                   density=True, facecolor='b', alpha=0.75,
                                   label='Aperture Correction', rwidth=0.9)

        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 500)

        mu, sigma = norm.fit(bins)
        best_fit_line = norm.pdf(x, mu, sigma)

        ax.plot(x, best_fit_line, 'r-', label='PDF')
        ax.legend(numpoints=2, loc='upper right', frameon=False)

        ax.set_xlabel("Correction (mag)")
        ax.set_ylabel("Probability Density")

        # Save the plot
        fname = f"plot.photometry.std_mag_corr_{file_base}"
        self._save_plot(fname=fname)

        if self._plot_images:
            plt.show()

        plt.close(fig=fig)

        return aper_correction, aper_correction_err

    def _filter_catalog(self, data: dict):
        """"""
        if not self._silent:
            self._log.info("> Remove sources close to trails from catalog")

        imgarr = data['imgarr']
        fwhm = data['kernel_fwhm'][0]
        # src_tbl = data['src_tbl']
        ref_tbl_photo = data['src_tbl']
        ref_cat_str = 'ref_cat_cleaned'

        # create a masked image for detected trail
        mask, mask_list = self._create_mask(image=imgarr,
                                            reg_data=data['trail_data']['reg_info_lst'],
                                            fac=5.)
        mask = mask * 1.

        data['trail_data']['mask'] = mask
        data['trail_data']['mask_list'] = mask_list

        if ref_tbl_photo.empty or not mask[mask > 0.].any():
            del imgarr, ref_tbl_photo, mask_list
            return data, False, ['IntegrityError',
                                 'Empty data frame or empty mask',
                                 'Check input data, trail/sources detection']
        # split catalog in stars outside and inside the trail mask
        ref_cat_cln, ref_cat_removed = sext.clean_catalog_trail(imgarr=imgarr, mask=mask,
                                                                catalog=ref_tbl_photo,
                                                                fwhm=fwhm)
        if ref_cat_cln is None:
            del imgarr, ref_tbl_photo, ref_cat_cln, ref_cat_removed, mask_list
            return data, False, ['SrcMatchError',
                                 'No standard stars left '
                                 'after removing stars close to trail',
                                 'To be solved']

        del imgarr, mask_list

        data['cats_cleaned'] = {ref_cat_str: ref_cat_cln}
        data['cats_cleaned_rem'] = {ref_cat_str: ref_cat_removed}

        return data, True, ['', '', '']

    def _get_img_dict(self, data: dict, idx: int,
                      obsparam: dict,
                      mode: str = 'trail',
                      mask: np.ndarray = None):
        """Get the data for either the reference image or the trail image

        Parameters
        ----------
        data: dict
            Dictionary with basic info for all object files
        idx: int
            Index of image selection, either for the trail or reference image

        Returns
        -------
        result : dict
            Dictionary with all relevant data for image
        """
        config = obsparam

        imgarr = data['images'][idx]

        # noinspection PyAugmentAssignment
        if mode == 'ref':
            imgarr = imgarr * mask

        hdr = data['header'][idx]
        file_name = data['file_names'][idx]
        bkg_data = data['bkg_data'][idx]
        bkg_file_name = bkg_data['bkg_fname']
        location = data['src_loc'][idx]
        catalog = data['catalog'][idx]
        band = data['band'][idx]
        trail_data = data['trail_data'][idx]
        img_mask = data['image_mask'][idx]

        # get wcs
        wcsprm = WCS(hdr).wcs

        # extract sources on detector
        params_to_add = dict(_fov_radius=None,
                             _filter_val=band,
                             _trail_data=trail_data,
                             _vignette=self._vignette,
                             _vignette_rectangular=self._vignette_rectangular,
                             _cutouts=self._cutouts,
                             _src_cat_fname=None, _ref_cat_fname=None,
                             _photo_ref_cat_fname=self._photo_ref_cat_fname,
                             estimate_bkg=False, bkg_fname=bkg_file_name,
                             _force_extract=self._force_extract,
                             _force_download=self._force_download,
                             _plot_images=self._plot_images,
                             image_mask=img_mask)

        # add to configuration
        config.update(params_to_add)
        for key, value in self._config.items():
            config[key] = value
        src_tbl, std_fkeys, mag_conv, kernel_fwhm, state = sext.get_photometric_catalog(file_name, self._cat_path,
                                                                                        imgarr, hdr, wcsprm, catalog,
                                                                                        silent=self._silent,
                                                                                        **config)

        res_keys = ['imgarr', 'hdr', 'fname', 'loc', 'cat', 'band', 'src_tbl', 'kernel_fwhm',
                    'trail_data', 'bkg_data', 'std_fkeys', 'mag_conv']

        res_vals = [imgarr, hdr, file_name, location, catalog, band,
                    src_tbl, kernel_fwhm, trail_data,
                    bkg_data, std_fkeys, mag_conv]

        result = dict(zip(res_keys, res_vals))

        if not self._silent:
            self._log.info("> Load object information")

        ra = hdr[obsparam['ra']]
        dec = hdr[obsparam['dec']]
        if obsparam['radec_separator'] == 'XXX':
            ra = round(hdr[obsparam['ra']], _base_conf.ROUND_DECIMAL)
            dec = round(hdr[obsparam['dec']], _base_conf.ROUND_DECIMAL)
        hdr[obsparam['ra']] = ra
        hdr[obsparam['dec']] = dec

        self._obsTable.get_object_data(fname=file_name, kwargs=hdr,
                                       obsparams=obsparam)
        self._obj_info = self._obsTable.obj_info

        return result

    def _identify_trails(self, files: list):
        """Identify image with the satellite trail(s) and collect image info and trail parameters."""

        hdu_idx = self._hdu_idx
        obspar = self._obsparams
        data_dict = {'images': [], 'header': [],
                     'file_names': [], 'loc': [], 'src_loc': [],
                     'telescope': [], 'tel_key': [],
                     'has_trail': [], 'trail_data': [],
                     'trail_img_idx': None, 'ref_img_idx': None,
                     'catalog': [], 'band': [],
                     'src_cat': [], 'astro_ref_cat': [], 'photo_ref_cat': [],
                     'bkg_data': [], 'image_mask': []}

        n_imgs = len(files)
        for f in range(n_imgs):
            file_loc = files[f]

            # split file string into a path and filename
            file_name = Path(file_loc).stem
            file_name_clean = file_name.replace('_cal', '')
            location = os.path.dirname(file_loc)

            # load fits file
            img_mask = None
            with fits.open(file_loc) as hdul:
                hdul.verify('fix')
                hdr = hdul[hdu_idx].header
                imgarr = hdul[hdu_idx].data.astype('float32')
                if 'mask' in hdul and obspar['apply_mask']:
                    img_mask = hdul['MASK'].data.astype(bool)

            # check the used filter band
            filter_val = hdr['FILTER']
            if self._band is not None:
                filter_val = self._band
            catalog = sext.select_reference_catalog(band=filter_val, source=self._catalog)

            # check if the background file exists
            levels_up = 2
            f_path = Path(file_loc)
            one_up = str(f_path.parents[levels_up - 1])
            aux_path = Path(one_up, 'auxiliary')
            if not aux_path.exists():
                aux_path.mkdir(exist_ok=True)

            cat_path = Path(one_up, 'catalogs')
            if not cat_path.exists():
                cat_path.mkdir(exist_ok=True)

            plt_path = Path(one_up, 'figures')
            if not plt_path.exists():
                plt_path.mkdir(exist_ok=True)
            sat_id, _ = self._obsTable.get_satellite_id(hdr[obspar['object']])
            plt_path_final = plt_path / sat_id
            if not plt_path_final.exists():
                plt_path_final.mkdir(exist_ok=True)

            self._sat_id = sat_id
            self._work_dir = Path(one_up)
            self._aux_path = aux_path
            self._cat_path = cat_path
            self._plt_path = plt_path_final

            # set background file name and create folder
            bkg_fname_short = file_name.replace('_cal', '_bkg')
            bkg_fname = os.path.join(aux_path, bkg_fname_short)

            estimate_bkg = True
            if os.path.isfile(f'{bkg_fname}.fits'):
                estimate_bkg = False

            box_size = self._config['BKG_BOX_SIZE']
            win_size = self._config['BKG_MED_WIN_SIZE']
            bkg_data = sext.compute_2d_background(imgarr,
                                                  estimate_bkg=estimate_bkg,
                                                  bkg_fname=(bkg_fname,
                                                             bkg_fname_short),
                                                  box_size=box_size,
                                                  win_size=win_size, mask=img_mask)

            bkg, bkg_med, bkg_rms, bkg_rms_med = bkg_data

            # subtract background
            img = imgarr - bkg

            if n_imgs >= 1:
                _config = self._config
                _config['sat_id'] = sat_id
                _config['pixscale'] = hdr['PIXSCALE']
                _config['fwhm'] = hdr['FWHM']

                img[img < 0.] = 0
                if img_mask is not None:
                    img[img_mask] = 0

                ccd_mask_dict = {'CTIO': [[1405, 1791, 1920, 1957]],
                                 'CBNUO-JC': [[225, 227, 1336, 2048]]}
                detect_mask = None
                if self._telescope in ['CTIO', 'CBNUO-JC']:
                    detect_mask = np.zeros(img.shape)
                    ccd_mask_list = ccd_mask_dict[self._telescope]
                    for yx in ccd_mask_list:
                        detect_mask[yx[0]:yx[1], yx[2]:yx[3]] = 1
                    detect_mask = np.where(detect_mask == 1, True, False)
                roi_img = img.copy()

                # extract region of interest for detection
                self._obsTable.get_roi(file_name_clean)
                roi_info = self._obsTable.roi_data
                _config['roi_offset'] = None
                if not roi_info.empty:
                    yx = roi_info.to_numpy()[0][1:]

                    roi_img = img[yx[2]:yx[3], yx[0]:yx[1]]
                    _config['roi_offset'] = (yx[0], yx[2])

                if not self._silent:
                    self._log.info(f"> Find satellite trail(s) in image: {file_name_clean}")

                # extract satellite trails from the image and create mask
                masks_dict, has_trail = sats.detect_sat_trails(image=roi_img,
                                                               config=_config,
                                                               alpha=30, mask=detect_mask,
                                                               silent=self._silent)

                kwargs = hdr.copy()
                kwargs['HasTrail'] = 'T' if has_trail else 'F'

                data_dict['has_trail'].append(has_trail)
                data_dict['trail_data'].append(masks_dict)
                if has_trail:
                    if not self._silent:
                        self._log.info("  > Plot trail detection results")
                    self._plot_fit_results(masks_dict, file_name_clean)

                if obspar['radec_separator'] == 'XXX':
                    kwargs[obspar['ra']] = round(hdr[obspar['ra']],
                                                 _base_conf.ROUND_DECIMAL)
                    kwargs[obspar['dec']] = round(hdr[obspar['dec']],
                                                  _base_conf.ROUND_DECIMAL)

                self._obsTable.update_obs_table(file=file_name_clean,
                                                kwargs=kwargs, obsparams=obspar)
                self._obj_info = self._obsTable.obj_info

                del masks_dict

            data_dict['telescope'].append(self._telescope)
            data_dict['images'].append(imgarr)
            data_dict['header'].append(hdr)
            data_dict['catalog'].append(catalog)
            data_dict['band'].append(filter_val)
            data_dict['image_mask'].append(img_mask)
            data_dict['file_names'].append(file_name_clean)
            data_dict['loc'].append(location)
            data_dict['src_loc'].append(one_up)

            # add bkg data to dict
            bkg_dict = {'bkg_fname': (bkg_fname, bkg_fname_short),
                        'bkg': bkg, 'bkg_med': bkg_med,
                        'bkg_rms': bkg_rms,
                        'bkg_rms_med': bkg_rms_med}
            data_dict['bkg_data'].append(bkg_dict)

            del imgarr, img, bkg_data, bkg, bkg_rms, bkg_dict
            gc.collect()

        # group images
        has_trail_check = np.array(data_dict['has_trail'])
        if (n_imgs == 1 and not np.all(has_trail_check)) or \
                (n_imgs > 1 and not np.any(has_trail_check)):
            self._log.critical("None of the input images have satellite trail(s) detected!!! "
                               "Skipping further steps.")

            return None, ['DetectError',
                          'Satellite trail(s) detection fail.',
                          'Difficult to solve.']
        else:
            data_dict['trail_img_idx'] = np.asarray(has_trail_check).nonzero()[0]
            data_dict['ref_img_idx'] = np.asarray(~has_trail_check).nonzero()[0]

        return data_dict, None

    def _get_data_from_tle(self, hdr, sat_name: str, tle_location: str,
                           date_obs, obs_times,
                           exptime, dt=None):
        """ Calculate angular velocity from tle, satellite and observer location """

        wcs = WCS(hdr)
        column_names = ['Obs_Time', 'RA', 'DEC', 'x', 'y',
                        'UT Date', 'UT time', 'SatLon', 'SatLat', 'SatAlt', 'SatAz',
                        'SatElev', 'SatRA', 'SatDEC',
                        'SunRA', 'SunDEC', 'SunZenithAngle']

        tle_pos_path = Path(self._work_dir, 'tle_predictions')
        if not tle_pos_path.exists():
            tle_pos_path.mkdir(exist_ok=True)

        pos_times = [datetime.strptime(f"{date_obs}T{i}",
                                       frmt) for i in obs_times]

        fname = f"tle_predicted_positions_{sat_name.upper()}_{pos_times[0].strftime(frmt)[:-3]}.csv"
        fname = os.path.join(tle_pos_path, fname)

        # get ephemeris
        self._set_observer()

        obs_lat = self._obsparams["latitude"]
        obs_lon = self._obsparams["longitude"]
        obs_ele = self._obsparams["altitude"]  # in meters
        loc = EarthLocation(lat=obs_lat * u.deg,
                            lon=obs_lon * u.deg,
                            height=obs_ele * u.m)

        try:
            satellite = Orbital(sat_name, tle_file=tle_location)
        except KeyError:
            sat_name = sat_name.replace('-', ' ')
            satellite = Orbital(sat_name, tle_file=tle_location)

        if dt is not None:
            numseconds = int(exptime / dt)
            delta_time = dt
            base = pos_times[0]
            pos_times = [pos_times[0] - timedelta(seconds=dt), base]
            c = 0
            while c < numseconds:
                base += timedelta(seconds=dt)
                pos_times.append(base)
                c += 1
        else:
            delta_time = exptime / 2
            pos_times.insert(0, pos_times[0] - timedelta(seconds=delta_time))
        # print(pos_times)
        positions = np.empty((len(pos_times), 17), object)

        for t, val in enumerate(pos_times):
            sun_coordinates = pyorbital.astronomy.sun_ra_dec(val)

            sat_az, sat_elev = satellite.get_observer_look(utc_time=val,
                                                           lon=obs_lon,
                                                           lat=obs_lat,
                                                           alt=obs_ele / 1000.0)

            radec = SkyCoord(alt=sat_elev * u.deg, az=sat_az * u.deg,
                             obstime=val, frame='altaz', location=loc)
            pix_coords = radec.to_pixel(wcs)
            sun_zenith = pyorbital.astronomy.sun_zenith_angle(val, obs_lon, obs_lat)
            sun_ra_hms = Angle(sun_coordinates[0], u.rad).hour
            sun_dec_dms = Angle(sun_coordinates[1], u.rad).degree
            sat_lon, sat_lat, sat_alt = satellite.get_lonlatalt(val)
            sat_ra = Angle(radec.icrs.ra.value, u.degree).to_string(unit=u.hour)
            sat_dec = Angle(radec.icrs.dec.value, u.degree).to_string(unit=u.degree, sep=':')

            to_add = [val.strftime(frmt)[:-3],
                      radec.icrs.ra.value, radec.icrs.dec.value,
                      pix_coords[0], pix_coords[1], val.strftime("%Y-%m-%d"), val.strftime('%H:%M:%S.%f')[:-3],
                      sat_lon, sat_lat, sat_alt, sat_az, sat_elev, sat_ra, sat_dec,
                      sun_ra_hms, sun_dec_dms, sun_zenith]
            # print(to_add)
            positions[t, :] = to_add

        velocity_list = [np.nan]
        for i in range(1, positions.shape[0]):
            ra = np.deg2rad(positions[i, 1])
            dec = np.deg2rad(positions[i, 2])
            prev_ra = np.deg2rad(positions[i - 1, 1])
            prev_dec = np.deg2rad(positions[i - 1, 2])

            dtheta = 2. * np.arcsin(np.sqrt(np.sin(0.5 * (dec - prev_dec)) ** 2.
                                            + np.cos(dec) * np.cos(prev_dec)
                                            * np.sin(0.5 * (ra - prev_ra)) ** 2.))

            dtheta *= 206264.806

            velocity_list += [dtheta / delta_time]

        pos_df = pd.DataFrame(data=positions,
                              columns=column_names)
        pos_df['pos_mask'] = pos_df.apply(lambda row: (0 < row['x']) and
                                                      (hdr['NAXIS1'] > row['x']) and
                                                      (0 < row['y']) and
                                                      (hdr['NAXIS2'] > row['y']), axis=1)
        pos_df['Obs_Time'] = pd.to_datetime(pos_df['Obs_Time'],
                                            format=frmt,
                                            utc=False)

        pos_df['angular_velocity'] = velocity_list

        angular_velocity = np.nanmean(velocity_list)
        e_angular_velocity = np.nanstd(velocity_list)

        # save results
        pos_df[1:].to_csv(fname, index=False)

        return angular_velocity, e_angular_velocity, pos_df

    def _set_observer(self):
        """"""
        observer = ephem.Observer()
        observer.epoch = "2000"
        observer.pressure = 1010
        observer.temp = 15

        observatory_latitude = self._obsparams["latitude"]  # degrees
        observer.lat = np.radians(observatory_latitude)

        observatory_longitude = self._obsparams["longitude"]  # degrees
        observer.lon = np.radians(observatory_longitude)

        observer.elevation = self._obsparams["altitude"]  # in meters
        self._observer = observer

    def _plot_fit_results(self, data: dict, file_base: str):
        """Plot trail detection and parameter fit results"""
        n_trails_max = self._config['N_TRAILS_MAX']

        for i in range(n_trails_max):
            reg_data = data['reg_info_lst'][i]
            param_fit_dict = reg_data['param_fit_dict']
            trail_imgs_dict = reg_data['detection_imgs']

            # plot detection process
            fname_trail_det = f"plot.satellite.detection_{file_base}_T{i + 1:02d}"
            self._plot_trail_detection(param_dict=trail_imgs_dict, fname=fname_trail_det)

            # plot hough space
            fname_houg_space = f"plot.satellite.hough.space_{file_base}_T{i + 1:02d}"
            self._plot_hough_space(param_dict=param_fit_dict, fname=fname_houg_space)

            # plot quadratic fit result
            fname_houg_quad_fit = f"plot.satellite.hough.parameter_{file_base}_T{i + 1:02d}"
            self._plot_voting_variance(fit_results=param_fit_dict,
                                       reg_data=reg_data,
                                       fname=fname_houg_quad_fit)

            # plot linear fit result
            fname_houg_lin_fit = f"plot.satellite.hough.centroidxy_{file_base}_T{i + 1:02d}"
            self._plot_centroid_fit(fit_results=param_fit_dict,
                                    reg_data=reg_data,
                                    fname=fname_houg_lin_fit)

            del trail_imgs_dict, param_fit_dict, reg_data
            gc.collect()

        del data
        gc.collect()

    def _plot_trail_detection(self, param_dict: dict, fname: str,
                              img_norm: str = 'lin', cmap: str = None):
        """Plot trail detection"""

        img = param_dict['img_sharp']
        segm = param_dict['segm_map']
        label_image = param_dict['img_labeled']
        label_image.relabel_consecutive(start_label=1)
        trail_mask = param_dict['trail_mask']

        # Define normalization and the colormap
        vmin = np.percentile(img, 5)
        vmax = np.percentile(img, 99.)
        if img_norm == 'lin':
            nm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=LinearStretch())
        elif img_norm == 'log':
            nm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=LogStretch())
        else:
            nm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=SqrtStretch())
        if cmap is None:
            cmap = 'Greys'

        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(2, 2)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])

        ax1.imshow(img, origin='lower',
                   cmap=cmap,
                   interpolation='none',
                   norm=nm,
                   aspect='auto'
                   )
        ax2.imshow(segm, origin='lower', cmap=segm.cmap, interpolation='nearest',
                   aspect='auto')

        ax3.imshow(label_image, origin='lower', interpolation='nearest',
                   aspect='auto')

        ax4.imshow(trail_mask, origin='lower', interpolation='nearest',
                   aspect='auto')

        # format axes
        ax_list = [ax1, ax2, ax3, ax4]
        for i in range(len(ax_list)):
            ax = ax_list[i]

            ax.set_xlim(0, img.shape[1])
            ax.set_ylim(0, img.shape[0])

            minorLocatorx = AutoMinorLocator(5)
            minorLocatory = AutoMinorLocator(5)

            ax.xaxis.set_minor_locator(minorLocatorx)
            ax.yaxis.set_minor_locator(minorLocatory)

            ax.tick_params(
                axis='both',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                direction='in',  # inward or outward ticks
                left=True,  # ticks along the left edge are off
                right=True,  # ticks along the right edge are off
                top=True,  # ticks along the top edge are off
                bottom=True,  # ticks along the bottom edge are off
                width=1.,
                color='k'  # tick color
            )
            if i > 0:
                ax.tick_params(
                    which='both',  # both major and minor ticks are affected
                    axis='both',  # changes apply to the x-axis
                    color='w'  # tick color
                )
            if i == 1 or i == 3:
                ax.set_yticklabels([])
            else:
                ax.set_ylabel(r'Pixel y direction')

            if i == 0 or i == 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel(r'Pixel x direction')
            #
            # ax.set_xlabel(r'Pixel x direction')

        fig.tight_layout(h_pad=0.4, w_pad=0.5, pad=1.)

        # Save the plot
        self._save_plot(fname=fname)

        if self._plot_images:
            plt.show()

        plt.close(fig=fig)

        del img, segm, label_image, trail_mask
        gc.collect()

    def _plot_centroid_fit(self, fit_results: dict, fname: str, reg_data: dict):
        """"""

        result = fit_results['lin_fit']
        x = fit_results['lin_x']
        y = fit_results['lin_y']

        fig = plt.figure(figsize=figsize)

        gs = gridspec.GridSpec(1, 1)
        ax = fig.add_subplot(gs[0, 0])
        ax.plot(x, y, 'bo')
        ax.plot(x, result.best_fit, 'r-')

        # format axis
        ax.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            direction='in',  # inward or outward ticks
            left=True,  # ticks along the left edge are off
            right=True,  # ticks along the right edge are off
            top=True,  # ticks along the top edge are off
            bottom=True,  # ticks along the bottom edge are off
            width=1.,
            color='k'  # tick color
        )

        minorLocatorx = AutoMinorLocator(5)
        minorLocatory = AutoMinorLocator(5)

        ax.xaxis.set_minor_locator(minorLocatorx)
        ax.yaxis.set_minor_locator(minorLocatory)

        x_str = r'$\tan(\theta)$'
        if 'sin' in fit_results['lin_y_label']:
            x_str = r'$\tan(\theta)^{-1}$'

        ax.set_xlabel(x_str)
        ax.set_ylabel(fit_results['lin_y_label'])

        # create a list with two empty handles (or more if needed)
        handles = [mpl_patches.Rectangle((0, 0), 1, 1, fc="white", ec="white",
                                         lw=0, alpha=0)] * 3

        # create the corresponding number of labels (= the text you want to display)
        eq_str = r'$= \mathrm{y}_{c}\tan(\theta) + \mathrm{x}_{c}$'
        if 'sin' in fit_results['lin_y_label']:
            eq_str = r'$= \mathrm{y}_{c}\tan(\theta)^{-1} + \mathrm{x}_{c}$'
        labels = [r"{0}".format(fit_results['lin_y_label'])
                  + eq_str,
                  r"$\mathrm{x}_{c}$="
                  + r"{0:.2f}$\pm${1:.2f} (pixel)".format(reg_data['coords'][0],
                                                          reg_data['coords_err'][0]),
                  r"$\mathrm{y}_{c}$="
                  + r"{0:.2f}$\pm${1:.2f} (pixel)".format(reg_data['coords'][1],
                                                          reg_data['coords_err'][1])]

        # create the legend, suppressing the blank space of the empty line symbol and the
        # padding between symbol and label by setting handlelenght and handletextpad
        legend = ax.legend(handles, labels, loc='best', fontsize='medium',
                           fancybox=False, framealpha=0.25,
                           handlelength=0, handletextpad=0)
        legend.set_draggable(state=1)

        # plt.tight_layout()

        # Save the plot
        self._save_plot(fname=fname)

        if self._plot_images:
            plt.show()

        plt.close(fig=fig)

        del fit_results, reg_data
        gc.collect()

    def _plot_voting_variance(self, fit_results: dict, fname: str, reg_data: dict):
        """Plot results from quadratic fit.

        Plot of voting variance fit used to obtain length, width, and orientation of the trail.
        """

        result = fit_results['quad_fit']

        x = fit_results['quad_x']
        y = fit_results['quad_y']

        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(1, 1)
        ax = fig.add_subplot(gs[0, 0])
        ax.plot(np.rad2deg(x), y, 'bo')
        ax.plot(np.rad2deg(x), result.best_fit, 'r-')

        # format axis
        ax.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            direction='in',  # inward or outward ticks
            left=True,  # ticks along the left edge are off
            right=True,  # ticks along the right edge are off
            top=True,  # ticks along the top edge are off
            bottom=True,  # ticks along the bottom edge are off
            width=1.,
            color='k'  # tick color
        )

        minorLocatorx = AutoMinorLocator(5)
        minorLocatory = AutoMinorLocator(5)

        ax.xaxis.set_minor_locator(minorLocatorx)
        ax.yaxis.set_minor_locator(minorLocatory)

        ax.set_xlabel(r'$\theta$ (deg)')
        ax.set_ylabel(r'$\sigma^{2}$ (pixel$^{2}$)')

        # create a list with two empty handles (or more if needed)
        handles = [mpl_patches.Rectangle((0, 0), 1, 1, fc="white", ec="white",
                                         lw=0, alpha=0)] * 4

        # create the corresponding number of labels (= the text you want to display)
        labels = [r"$\sigma^{2} = \mathrm{a}_{2}\theta^{2} "
                  r"+ \mathrm{a}_{1}\theta "
                  r"+ \mathrm{a}_{0}$",
                  r"$\theta_{0} = $" + r"{0:.2f} (deg)".format(reg_data['orient_deg'] - 90.),
                  r"$L = $" + r"{0:.2f}$\pm${1:.2f} (pixel)".format(reg_data['width'],
                                                                    reg_data['e_width']),
                  r"$T = $" + r"{0:.2f}$\pm${1:.2f} (pixel)".format(reg_data['height'],
                                                                    reg_data['e_height'])]

        # create the legend, suppressing the blank space of the empty line symbol and the
        # padding between symbol and label by setting handlelenght and handletextpad
        legend = ax.legend(handles, labels, loc='best', fontsize='medium',
                           fancybox=False, framealpha=0.25,
                           handlelength=0, handletextpad=0)
        legend.set_draggable(state=1)

        # Save the plot
        self._save_plot(fname=fname)

        if self._plot_images:
            plt.show()

        plt.close(fig=fig)

        del fit_results, reg_data
        gc.collect()

    def _plot_hough_space(self, param_dict: dict, fname: str,
                          img_norm: str = 'lin', cmap: str = None):
        """Plot Hough space"""

        img = param_dict['Hij_sub']
        rows = param_dict['Hij_rho']
        cols = param_dict['Hij_theta']

        # Define normalization and the colormap
        vmin = 0.  # np.percentile(img, 5)
        vmax = np.percentile(img, 99.995)
        if img_norm == 'lin':
            nm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=LinearStretch())
        elif img_norm == 'log':
            nm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=LogStretch())
        else:
            nm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=SqrtStretch())
        if cmap is None:
            cmap = 'Greys'

        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(1, 1)
        ax = fig.add_subplot(gs[0, 0])

        ax.imshow(img, origin='lower',
                  extent=[cols[0], cols[-1],
                          rows[0], rows[-1]],
                  cmap=cmap,
                  interpolation='none',
                  norm=nm,
                  aspect='auto')

        # format axis
        ax.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            direction='in',  # inward or outward ticks
            left=True,  # ticks along the left edge are off
            right=True,  # ticks along the right edge are off
            top=True,  # ticks along the top edge are off
            bottom=True,  # ticks along the bottom edge are off
            width=1.,
            color='k'  # tick color
        )

        minorLocatorx = AutoMinorLocator(5)
        minorLocatory = AutoMinorLocator(5)

        ax.xaxis.set_minor_locator(minorLocatorx)
        ax.yaxis.set_minor_locator(minorLocatory)

        ax.set_xlabel(r'$\theta$ (deg)')
        ax.set_ylabel(r'$\rho$ (pixel)')

        plt.tight_layout()

        # Save the plot
        self._save_plot(fname=fname)

        if self._plot_images:
            plt.show()

        plt.close(fig=fig)

        del img, param_dict
        gc.collect()

    def _plot_photometry_snr(self, src_pos: list, fluxes: np.ndarray,
                             apers: list, optimum_aper: float,
                             file_base: str, mode: str = 'sat'):
        """Plot the Signal-to-Noise ratio results from curve-of-growth estimation"""

        flux_med = np.nanmedian(fluxes[:, :, 2], axis=1) / np.nanmax(np.nanmedian(fluxes[:, :, 2], axis=1))
        fig = plt.figure(figsize=figsize)

        gs = gridspec.GridSpec(1, 2)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])

        # plot fluxes
        for r in range(len(src_pos)):
            ax1.plot(apers, fluxes[:, r, 0] / fluxes[0, r, 0], c='k')

        # add legend
        if mode != 'sat':
            black_line = mlines.Line2D([], [], color='k', marker='',
                                       markersize=15, label='STD star flux')
            ax1.legend([black_line], ['STD star flux'], loc=1, numpoints=1)

        # plot fluxes
        ax2.plot(apers, flux_med, 'r-', lw=2)

        for r in range(len(src_pos)):
            ax2.plot(apers, fluxes[:, r, 2] / np.nanmax(fluxes[:, r, 2]), 'k-', alpha=0.2)

        # indicate optimum aperture
        ax2.axvline(x=optimum_aper, c='k', ls='--')

        # add legend
        black_line = mlines.Line2D([], [], color='k', marker='',
                                   markersize=15, label='STD star S/N')
        best_line = mlines.Line2D([], [], color='r', marker='',
                                  markersize=15, label='median S/N')
        if mode == 'sat':
            ax2.legend([best_line, black_line], ['S/N', 'Trail flux'],
                       loc=1, numpoints=1)
        else:
            ax2.legend([best_line, black_line], ['median S/N', 'STD star S/N'],
                       loc=1, numpoints=1)

        # format axes
        ax_list = [ax1, ax2]
        for i in range(len(ax_list)):
            ax = ax_list[i]

            ax.tick_params(
                axis='both',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                direction='in',  # inward or outward ticks
                left=True,  # ticks along the left edge are off
                right=True,  # ticks along the right edge are off
                top=True,  # ticks along the top edge are off
                bottom=True,  # ticks along the bottom edge are off
                width=1.,
                color='k'  # tick color
            )

            minorLocatorx = AutoMinorLocator(5)
            minorLocatory = AutoMinorLocator(5)

            ax.xaxis.set_minor_locator(minorLocatorx)
            ax.yaxis.set_minor_locator(minorLocatory)

            if i == 0:
                ax.set_ylabel(r'Normalized Flux$_{\mathrm{aper}}$')
            else:
                ax.set_ylabel(r'Normalized S/N')

            ax.set_xlabel(r'Aperture radius (pixel)')

        fig.tight_layout(h_pad=0.4, w_pad=0.5, pad=1.)

        # Save the plot
        if mode == 'sat':
            # fname = f"{file_name}_T{i + 1:2d}_ref_sub.fits"
            fname = f"plot.photometry.snr_sat_{file_base}"
        else:
            fname = f"plot.photometry.snr_std_{file_base}"

        # Save the plot
        self._save_plot(fname=fname)

        if self._plot_images:
            plt.show()

        plt.close(fig=fig)

        del src_pos, fluxes
        gc.collect()

    def _plot_aperture_photometry(self, fluxes: np.ndarray, rapers: list,
                                  opt_aprad: float, file_base: str):
        """"""
        snr_arr = fluxes[:, :, 2]
        snr_med = np.nanmedian(snr_arr, axis=1) / np.nanmax(np.nanmedian(snr_arr, axis=1))
        snr_max = np.array([np.nanmax(snr_arr[:, r]) for r in range(fluxes.shape[1])])

        fig = plt.figure(figsize=(10, 6), constrained_layout=True, layout='compressed')
        # plt.set_loglevel('WARNING')
        widths = [50, 1]
        gs = gridspec.GridSpec(2, 2, figure=fig, width_ratios=widths)

        # set up the normalization and the colormap
        normalize = mcolors.Normalize(vmin=np.min(snr_max), vmax=np.max(snr_max))
        colormap = cm.jet

        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)

        for r in range(fluxes.shape[1]):
            color = colormap(normalize(snr_max[r]))
            ax1.plot(rapers, fluxes[:, r, 2] / np.nanmax(fluxes[:, r, 2]),
                     color=color, ls='--', alpha=0.25)
            ax2.plot(rapers, fluxes[:, r, 0] / np.nanmax(fluxes[:, r, 0]),
                     color=color, ls='--', alpha=0.25)

        ax1.plot(rapers, snr_med, 'r-', lw=2, label='Median COG')
        # indicate optimum aperture
        ax1.axvline(x=opt_aprad, c='b', ls='--')
        ax2.axvline(x=opt_aprad, c='b', ls='--',
                    label='Optimum Radius\n'
                          + r'$\left(r_{\mathrm{max, S/N}}\times 1.25\right)$')

        ax2.axhline(y=1., c='k', ls=':', label=r'100 %')
        ax2.axhline(y=0.95, c='k', ls='-.', label=r'95 %')

        ax1.set_xlim(xmin=0, xmax=5)
        ax2.set_ylim(ymin=0.8)

        ax3 = fig.add_subplot(gs[:, 1])
        # set up the color-bar
        scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=colormap)
        scalarmappaple.set_array(snr_max)
        cbar = fig.colorbar(scalarmappaple, cax=ax3)
        tick_locator = ticker.MaxNLocator(nbins=5)

        cbar.locator = tick_locator
        cbar.ax.minorticks_on()
        cbar.ax.get_yaxis().labelpad = 10
        cbar.ax.set_ylabel('Max S/N', rotation=270)
        cbar.update_ticks()

        cbar.ax.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            direction='in',
            left=True,  # ticks along the bottom edge are off
            right=True,  # ticks along the top edge are off
            top=True,
            bottom=True,
            width=1,
            color='k')
        cbar.update_ticks()
        # format axes
        ax_list = [ax1, ax2]
        for i in range(len(ax_list)):
            ax = ax_list[i]

            ax.tick_params(
                axis='both',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                direction='in',  # inward or outward ticks
                left=True,  # ticks along the left edge are off
                right=True,  # ticks along the right edge are off
                top=True,  # ticks along the top edge are off
                bottom=True,  # ticks along the bottom edge are off
                width=1.,
                color='k'  # tick color
            )

            minorLocatorx = AutoMinorLocator(4)
            minorLocatory = AutoMinorLocator(4)

            ax.xaxis.set_minor_locator(minorLocatorx)
            ax.yaxis.set_minor_locator(minorLocatory)

            if i == 0:
                ax.set_ylabel(r'SNR (normalized)')
            else:
                ax.set_ylabel(r'Flux (normalized)')

            ax.set_xlabel(r'Aperture radius (1/FWHM)')

        plt.setp(ax1.get_xticklabels(), visible=False)
        ax1.set(xlabel=None)
        ax1.legend(loc='lower right')
        ax2.legend(loc='lower right')

        # fig.tight_layout(h_pad=0.4, w_pad=0.5, pad=1.)
        fname = f"plot.photometry.snr_std_{file_base}"

        # Save the plot
        self._save_plot(fname=fname)

        if self._plot_images:
            plt.show()

        plt.close(fig=fig)

        del fluxes
        gc.collect()

    def _plot_final_result(self, img_dict: dict, reg_data: dict,
                           obs_date: tuple, expt: float, std_pos: np.ndarray,
                           file_base: str, tle_pos: pd.DataFrame = None,
                           img_norm: str = 'lin', cmap: str = 'Greys'):
        """Plot final result of satellite photometry"""

        img = img_dict['imgarr']
        band = img_dict['band']
        bkg = img_dict['bkg_data']['bkg']
        hdr = img_dict['hdr']
        wcsprm = WCS(hdr).wcs
        obsparams = self._obsparams

        # label_img = np.where(mask, 1, np.nan)
        # label_img = SegmentationImage(label(src_mask))
        # label_img.relabel_consecutive(start_label=1)

        image = img - bkg  # subtract background
        image[image < 0.] = 0.

        # get the scale and angle
        scale = wcsprm.cdelt[0]
        angle = np.arccos(wcsprm.pc[0][0])

        theta = angle / 2. / np.pi * 360.
        if wcsprm.pc[0][0] < 0.:
            theta *= -1.

        # get y-axis length as limit
        plim = img.shape[0]
        x0 = plim * 0.955
        y0 = plim * 0.955
        if self._telescope == 'DK-1.54':
            x0 = plim * 0.955
            y0 = plim * 0.025
            if theta < 0.:
                x0 = plim * 0.025
                y0 = plim * 0.955
        if self._telescope == 'Takahashi FSQ 85':
            x0 = plim * 1.25
            y0 = plim * 0.025
            if theta < 0.:
                x0 = plim * 0.025
                y0 = plim * 0.955

        larr = self._config['ARROW_LENGTH'] * plim  # 15% of the image size
        length = self._config['LINE_LENGTH']
        xlemark = length / (scale * 60.)
        xstmark = 0.025 * plim
        xenmark = xstmark + xlemark
        ystmark = plim * (1. - 0.975)

        sat_pos = reg_data['coords']
        sat_ang = Angle(reg_data['orient_deg'], 'deg')
        trail_w = reg_data['width']
        trail_h = float(self._obj_info['OptAperHeight'])

        sat_aper = RectangularAperture(sat_pos, w=trail_w, h=trail_h, theta=sat_ang)
        std_apers = CircularAperture(std_pos, r=12.)

        # Define normalization and the colormap
        vmin = np.percentile(image, 5)
        vmax = np.percentile(image, 99)
        if img_norm == 'lin':
            nm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=LinearStretch())
        elif img_norm == 'log':
            nm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=LogStretch())
        else:
            nm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=SqrtStretch())
        if cmap is None:
            cmap = 'cividis'

        fig = plt.figure(figsize=figsize)

        gs = gridspec.GridSpec(1, 1)
        ax = fig.add_subplot(gs[0, 0])

        # plot the image
        ax.imshow(image, origin='lower',
                  cmap=cmap,
                  interpolation='bicubic',
                  norm=nm,
                  aspect='equal')

        # add apertures
        std_apers.plot(axes=ax, **{'color': 'red', 'lw': 1.25, 'alpha': 0.75})
        sat_aper.plot(axes=ax, **{'color': 'green', 'lw': 1.25, 'alpha': 0.75})

        if tle_pos is not None:
            pos_data = tle_pos[tle_pos['pos_mask']]
            if not pos_data.empty and pos_data.shape[0] > 3:
                mid_idx = pos_data.shape[0] // 2
                pos_data_cut = pos_data.iloc[[mid_idx - 1, mid_idx, mid_idx + 1]]

                ax.arrow(pos_data_cut['x'].values[0], pos_data_cut['y'].values[0],
                         pos_data_cut['x'].values[2] - pos_data_cut['x'].values[0],
                         pos_data_cut['y'].values[2] - pos_data_cut['y'].values[0],
                         length_includes_head=True,
                         **{'head_width': 20, 'color': 'blue',
                            'lw': 1.5,
                            'alpha': 0.85})

                ax.plot(pos_data['x'], pos_data['y'], **{'color': 'blue',
                                                         'lw': 1.5, 'ls': '--',
                                                         'alpha': 0.75})
        ra = hdr[obsparams['ra']]
        dec = hdr[obsparams['dec']]
        unit = (u.hourangle, u.deg)
        if obsparams['radec_separator'] == 'XXX':
            unit = (u.deg, u.deg)
            ra = round(hdr[obsparams['ra']], _base_conf.ROUND_DECIMAL)
            dec = round(hdr[obsparams['dec']], _base_conf.ROUND_DECIMAL)

        c = SkyCoord(ra=ra, dec=dec, unit=unit,
                     obstime=hdr[obsparams['date_keyword']])

        xy = c.to_pixel(WCS(hdr))
        ax.scatter(xy[0], xy[1], **{'color': 'blue', 'marker': '+', 's': 100,
                                    'linewidths': 1.5, 'alpha': 0.75})

        # Add compass
        self._add_compass(ax=ax, x0=x0, y0=y0, larr=larr, theta=theta, color='k')
        ax.plot([xstmark, xenmark], [ystmark, ystmark], color='k', lw=1.25)
        ax.annotate(text=r"{0:.0f}'".format(length),
                    xy=(((xstmark + xenmark) / 2.), ystmark + 5), xycoords='data',
                    textcoords='data', ha='center', c='k')

        # format axis
        ax.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            direction='in',  # inward or outward ticks
            left=True,  # ticks along the left edge are off
            right=True,  # ticks along the right edge are off
            top=True,  # ticks along the top edge are off
            bottom=True,  # ticks along the bottom edge are off
            width=1.,
            color='k'  # tick color
        )

        minorLocatorx = AutoMinorLocator(5)
        minorLocatory = AutoMinorLocator(5)

        ax.xaxis.set_minor_locator(minorLocatorx)
        ax.yaxis.set_minor_locator(minorLocatory)

        ax.set_ylabel(r'Pixel y direction')
        ax.set_xlabel(r'Pixel x direction')

        ax.set_xlim(xmin=0, xmax=img.shape[1])
        ax.set_ylim(ymin=0, ymax=img.shape[0])

        # create a list with two empty handles (or more if needed)
        handles = [mpl_patches.Rectangle((0, 0), 1, 1, fc="white", ec="white",
                                         lw=0, alpha=0)] * 2

        # create the corresponding number of labels (= the text you want to display)
        labels = [f"{self._sat_id} {self._telescope} / {band} band",
                  f"{obs_date[0]} {obs_date[1]} / {expt:.1f}s"]

        # create the legend, suppressing the blank space of the empty line symbol and the
        # padding between symbol and label by setting handlelenght and handletextpad
        legend = ax.legend(handles, labels, loc=2, fontsize='small',
                           fancybox=False, framealpha=0.25,
                           handlelength=0, handletextpad=0)
        legend.set_draggable(state=1)
        legend.set_zorder(2.5)

        plt.tight_layout()

        # Save the plot
        fname = f"plot.satellite.final_{file_base}"
        self._save_plot(fname=fname)

        if self._plot_images:
            plt.show()

        plt.close(fig=fig)

        del img, image, hdr, wcsprm, sat_aper, std_apers, img_dict, reg_data, std_pos
        gc.collect()

    def _save_plot(self, fname: str):
        """Save plots"""

        fname = os.path.join(self._plt_path, fname)

        fig_type = self._config['FIG_TYPE']
        fig_dpi = self._config['FIG_DPI']
        if fig_type == 'png':
            plt.savefig(fname + '.png', format='png', dpi=fig_dpi)
            os.system(f'mogrify -trim {fname}.png ')
        elif fig_type == 'pdf':
            self._log.setLevel("warning".upper())
            plt.savefig(fname + '.pdf', format='pdf', dpi=fig_dpi)
            self._log.setLevel("info".upper())
            os.system('pdfcrop ' + fname + '.pdf ' + fname + '.pdf')
        else:
            self._log.info(f"Figure will be saved as .png and.pdf")
            plt.savefig(fname + '.png', format='png', dpi=fig_dpi)
            os.system(f'mogrify -trim {fname}.png ')
            self._log.setLevel("warning".upper())
            plt.savefig(fname + '.pdf', format='pdf', dpi=fig_dpi)
            self._log.setLevel("info".upper())
            os.system('pdfcrop ' + fname + '.pdf ' + fname + '.pdf')

    @staticmethod
    def get_time_stamp() -> str:
        """
        Returns time stamp for now: "2021-10-09 16:18:16"
        """

        now = datetime.now(tz=timezone.utc)
        time_stamp = f"{now:%Y-%m-%d_%H_%M_%S}"

        return time_stamp

    @staticmethod
    def _get_mask(src_pos: tuple,
                  width: float,
                  height: float,
                  theta: float,
                  img_size: tuple):
        """Extract single rectangular mask from an image"""

        aperture = RectangularAperture(src_pos,
                                       w=width,
                                       h=height,
                                       theta=theta)

        rect_mask = aperture.to_mask(method='center')
        mask_img = rect_mask.to_image(img_size)
        mask_img = np.where(mask_img == 1., True, False)
        mask = mask_img * 1.

        return mask, aperture

    @staticmethod
    def _create_mask(image: np.ndarray, reg_data: dict, fac: float = 3.):
        """Create a boolean mask from detected trails"""

        mask = np.zeros(image.shape)
        mask = np.where(mask == 0., False, True)
        mask_list = []
        for r in range(len(reg_data)):
            positions = reg_data[r]['coords']
            ang = Angle(reg_data[r]['orient_deg'], 'deg')
            aperture = RectangularAperture(positions,
                                           w=reg_data[r]['width'],
                                           h=fac * reg_data[r]['height'],
                                           theta=ang)

            rect_mask = aperture.to_mask(method='exact')
            mask_img = rect_mask.to_image(image.shape)
            mask_img = np.where(mask_img == 1., True, False)
            mask |= mask_img
            mask_list.append(mask_img * 1.)
            del mask_img, rect_mask, aperture, positions
            gc.collect()

        del image, reg_data
        gc.collect()

        return mask, mask_list

    @staticmethod
    def _add_compass(ax: plt.Axes, x0: float, y0: float,
                     larr: float, theta: float, color: str = 'black'):
        """Make a Ds9 like compass for image"""

        # East
        theta = theta * np.pi / 180.
        x1, y1 = larr * np.cos(theta), larr * np.sin(theta)
        ax.arrow(x0, y0, x1, y1, head_width=10, color=color, zorder=2)
        ax.text(x0 + 1.25 * x1, y0 + 1.25 * y1, 'E', color=color)

        # North
        theta = theta + 90. * np.pi / 180.
        x1, y1 = larr * np.cos(theta), larr * np.sin(theta)
        ax.arrow(x0, y0, x1, y1, head_width=10, color=color, zorder=2)
        ax.text(x0 + 1.25 * x1, y0 + 1.25 * y1, 'N', color=color)
        return ax


# -----------------------------------------------------------------------------


def main():
    """ Main procedure """
    pargs = ParseArguments(prog_typ='satPhotometry')
    args = pargs.args_parsed
    main.__doc__ = pargs.args_doc

    AnalyseSatObs(input_path=args.input, args=args,
                  silent=args.silent, verbose=args.verbose)


# -----------------------------------------------------------------------------


# standard boilerplate to set 'main' as starting function
if __name__ == '__main__':
    main()
# -----------------------------------------------------------------------------
