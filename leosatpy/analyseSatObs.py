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
import os
import logging
import sys
import warnings
import time
import collections
from datetime import (timedelta, datetime, timezone)
from pathlib import Path

# THIRD PARTY
import pandas as pd
import numpy as np
import astroalign
from skimage.draw import disk

# astropy
import astropy.table
from astropy.io import fits
from astropy import units as u
from astropy.stats import (sigma_clip, sigma_clipped_stats)
from astropy.utils.exceptions import AstropyUserWarning
from astropy import wcs
from astropy.visualization import (LinearStretch, LogStretch, SqrtStretch, ZScaleInterval)
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.coordinates import (Angle, SkyCoord, EarthLocation)

# photutils
from photutils.aperture import (RectangularAperture, RectangularAnnulus, CircularAperture)
from photutils.segmentation import (SourceCatalog, detect_sources, detect_threshold)

# pyorbital
from pyorbital.orbital import Orbital
from pyorbital.astronomy import (sun_ra_dec, sun_zenith_angle)

# scipy
from scipy.stats import norm

# multiprocessing
from concurrent.futures import ProcessPoolExecutor
from functools import partial

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
    from matplotlib.ticker import AutoMinorLocator, ScalarFormatter
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

# Project modules
try:
    import leosatpy
except ModuleNotFoundError:
    from utils.version import __version__
    from utils.arguments import ParseArguments
    from utils.dataset import DataSet, update_binning
    from utils.tables import ObsTables
    from utils.select_trail import confirm_trail_and_select_aperture_gui
    import utils.sources as sext
    import utils.satellites as sats
    import utils.transformations as imtrans
    import utils.photometry as phot
    import utils.base_conf as bc
else:
    from leosatpy.utils.version import __version__
    from leosatpy.utils.arguments import ParseArguments
    from leosatpy.utils.dataset import DataSet, update_binning
    from leosatpy.utils.tables import ObsTables
    from leosatpy.utils.select_trail import confirm_trail_and_select_aperture_gui
    import leosatpy.utils.sources as sext
    import leosatpy.utils.satellites as sats
    import leosatpy.utils.transformations as imtrans
    import leosatpy.utils.photometry as phot
    import leosatpy.utils.base_conf as bc

# -----------------------------------------------------------------------------

""" Meta-info """
__author__ = "Christian Adam"
__copyright__ = 'Copyright 2021-2023, CLEOSat group'
__credits__ = ["Eduardo Unda-Sanzana, Jeremy Tregloan-Reed, Christian Adam"]
__license__ = "GPL-3.0 license"
__version__ = __version__
__maintainer__ = "Christian Adam"
__email__ = "christian.adam84@gmail.com"
__status__ = "Production"

__taskname__ = 'analyseSatObs'

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

pass_str = bc.BCOLORS.PASS + "SUCCESSFUL" + bc.BCOLORS.ENDC
fail_str = bc.BCOLORS.FAIL + "FAILED" + bc.BCOLORS.ENDC

frmt = "%Y-%m-%dT%H:%M:%S.%f"


# -----------------------------------------------------------------------------


# noinspection PyAugmentAssignment
class AnalyseSatObs(object):
    """Perform photometric, i.e., integrated flux and differential magnitude,
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
            bc.load_warnings()

        # Set variables
        self._root_path = Path(os.getcwd())
        self._config = collections.OrderedDict()
        self._root_dir = bc.ROOT_DIR
        self._input_path = input_path
        self._dataset_object = None
        self._dataset_all = None

        self.band = args.band
        self._catalog = args.catalog
        self._photo_ref_cat_fname = args.ref_cat_phot_fname

        self._log = log
        self._log_level = log_level
        self._hdu_idx = args.hdu_idx
        self.force_extract = args.force_detection
        self.force_download = args.force_download
        self.plot_images = plot_images
        self._figsize = (10, 6)
        self._work_dir = None
        self._plt_path = None
        self._aux_path = None
        self._cat_path = None
        self._sat_id = None
        self.sat_vel = None
        self.mag_scale = None
        self._silent = silent
        self._verbose = verbose

        self._instrument = None
        self._telescope = None
        self._obsparams = None

        self._obsTable = None
        self._obj_info = None

        self._glint_mask_data = None
        self._image_mask_collection = {}

        self._faint_trail_data = None
        self._manual_trail_selection = args.select_faint_trail

        self.fails = []
        self.current_location = ''
        self.current_object = ''
        self.current_file_name = ''
        self.current_hdu_idx = ''
        self.current_hdu_exec_state = []

        # Run the trail analysis
        self.run_trail_analysis()

    def run_trail_analysis(self):
        """Prepare and run satellite trail photometry"""

        starttime = time.perf_counter()
        self._log.info('====> Analyse satellite trail init <====')

        if not self._silent:
            self._log.info("> Search input and prepare data")
        self._log.debug("  > Check input argument(s)")

        # Prepare dataset from input argument
        ds = DataSet(input_args=self._input_path,
                     prog_typ='satPhotometry',
                     log=self._log, log_level=self._log_level)

        # Load configuration
        ds.load_config()
        self._config = ds.config
        self._figsize = self._config['FIG_SIZE']

        # Load observation data table
        self._obsTable = ObsTables(config=self._config)
        self._obsTable.load_obs_table()
        self._obj_info = ObsTables.obj_info

        # Load Region-of-Interest
        self._obsTable.load_roi_table()

        # Load Extensions-of-Interest
        self._obsTable.load_ext_oi_table()

        # Load Glint mask
        self._obsTable.load_glint_mask_table()

        # Load faint trail data
        self._obsTable.load_faint_trail_table()

        self._dataset_all = ds

        # Set variables for use
        inst_list = ds.instruments_list
        inst_data = ds.instruments
        n_inst = len(inst_list)

        pass_counter = 0
        time_stamp = self.get_time_stamp()

        # Loop over telescopes
        for i in range(n_inst):
            inst = inst_list[i]
            self._instrument = inst
            self._telescope = inst_data[inst]['telescope']
            self._obsparams = inst_data[inst]['obsparams']
            self._dataset_object = inst_data[inst]['dataset']

            # Loop over groups and run analysis for each group
            for obj_pointing, file_list in self._dataset_object:
                files = file_list['file'].values
                n_files = len(file_list)

                unique_obj = np.unique(np.array([self._obsTable.get_satellite_id(f)[0]
                                                 for f in file_list['object'].values]))
                self.current_location = os.path.dirname(files[0])
                self.current_object = unique_obj[0]
                if len(unique_obj) > 1:
                    self._log.warning("More than one object name identified. "
                                      "This should not happen!!! Please check the file header. "
                                      f"Skip pointing RA={obj_pointing[0]:.4f}, "
                                      f"DEC={obj_pointing[1]:.4f}.")

                    self.update_failed_processes(['SatIDError',
                                                  f'Multiple ID for same pointing. {";".join(unique_obj)}',
                                                  'Check/Compare FITS header and visibility files for naming issues.'])
                    continue

                sat_name, _ = self._obsTable.get_satellite_id(unique_obj[0])
                if not self._silent:
                    self._log.info("====> Analyse satellite trail run <====")
                    self._log.info(
                        f"> Analysing {n_files} FITS-file(s) for {sat_name} with pointing "
                        f"RA={obj_pointing[0]:.4f}, DEC={obj_pointing[1]:.4f} "
                        f"observed with the {self._telescope} telescope")
                self._sat_id = sat_name

                # In case of an error, comment the following line to see where the error occurs;
                # This needs some fixing and automation
                # exec_state = self.analyse_sat_trails(files=files, sat_name=sat_name)
                try:
                    exec_state = self.analyse_sat_trails(files=files, sat_name=sat_name)
                except Exception as e:
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                    self._log.critical(f"Unexpected behaviour: {e} in file {fname}, "
                                       f"line {exc_tb.tb_lineno}")
                    exec_state = 0
                    self.update_failed_processes([str(e), 'Unexpected behaviour',
                                                  'Please create a ticket url'])

                if exec_state == 1:
                    self._log.info(f">> Report: Satellite detection and analysis was {pass_str}")
                    pass_counter += 1

                else:
                    state_str = ''
                    if exec_state == -1:
                        state_str = 'partially '
                    self._log.info(f">> Report: Satellite detection and analysis has {state_str}{fail_str}")

                if not self._silent:
                    self._log.info(f'====>  Analysis of {sat_name} finished <====')

        # Save failed processes
        self.save_failed_processes(time_stamp)

        endtime = time.perf_counter()
        dt = endtime - starttime
        td = timedelta(seconds=dt)

        if not self._silent:
            self._log.info(f"Program execution time in hh:mm:ss: {td}")
        self._log.info('====>  Satellite analysis finished <====')
        sys.exit(0)

    def analyse_sat_trails(self, files, sat_name):
        """ Run satellite trail detection and photometry.

        Detect which image has a trail, determine parameters,
        perform relative photometry with standard stars,
        and calculate angles and corrections from TLE data.

        Parameters
        ----------
        files : list
            Input file list
        sat_name : str
            Satellite designation.

        """
        obsparams = self._obsparams
        self.current_file_name = ''
        self.current_hdu_idx = ''

        # Identify images with satellite trails and get parameters
        if self._silent:
            self._log.info("> Identify image(s) with satellite trail(s)")
        data_dict, fail_msg = self.identify_trails(files)
        # data_dict = None
        if data_dict is None:
            tmp = ['', '']
            tmp.extend(fail_msg)
            return False, None, [tmp]

        # Collect information on the satellite image and download the standard star catalog
        trail_img_idx_list = data_dict['trail_img_idx']
        final_status = 1
        fail_messages = [[] for _ in range(len(trail_img_idx_list))]
        for img_idx in range(len(trail_img_idx_list)):

            if not self._silent:
                self._log.info(f"==> Process image {img_idx + 1}/{len(trail_img_idx_list)} <==")

            trail_img_idx = data_dict['trail_img_idx'][img_idx]
            file_name = data_dict['file_names'][trail_img_idx]

            self.current_file_name = file_name
            self.current_hdu_idx = ''
            self.current_hdu_exec_state = []

            trail_img_dict, state = self.get_img_dict(data_dict, trail_img_idx, obsparams)
            if not np.all(self.current_hdu_exec_state):
                self._log.critical("> None of the image HDUs have matching standard stars!!! "
                                   "Skipping further steps. ")

                final_status = 0
                continue

            # Get catalogs excluding the trail area and including it
            trail_img_dict = self.filter_catalog(data=trail_img_dict)

            if not np.all(self.current_hdu_exec_state):
                self._log.critical("> None of the image HDUs have matching standard stars!!! "
                                   "Skipping further steps. ")

                final_status = 0
                continue

            trail_hdus = trail_img_dict['hdus']
            for h, hdu_idx in enumerate(trail_hdus):
                self.current_hdu_idx = f"HDU #{hdu_idx:02d}"
                if not self._silent:
                    self._log.info(f"==> Run photometry/analysis on HDU #{hdu_idx:02d} [{h + 1}/{len(trail_hdus)}] <==")

                if not self.current_hdu_exec_state[h]:
                    self._log.critical(f"  Not enough data to proceed!!! "
                                       "Skipping further steps.")
                    final_status = -1
                    continue

                # Mask the glint
                self._obsTable.get_glint_mask(file_name, hdu_idx)
                glint_mask_info = self._obsTable.glint_mask_data

                self._glint_mask_data = None
                self._image_mask_collection = {}

                analyse_glint = False
                if not glint_mask_info.empty and glint_mask_info['HDU'].values[0] == hdu_idx:
                    image = trail_img_dict['images'][h]

                    x0 = glint_mask_info['xc'].values[0]
                    y0 = glint_mask_info['yc'].values[0]
                    length = glint_mask_info['length'].values[0]
                    width = glint_mask_info['width'].values[0]
                    ang = np.deg2rad(glint_mask_info['ang'].values[0])

                    self._glint_mask_data = self.get_mask((x0, y0),
                                                          length, width, ang, image.shape,
                                                          scale=1.)
                    analyse_glint = True

                # Select std stars and perform aperture photometry
                std_photometry_results = self.run_std_photometry(data=trail_img_dict, hidx=h, hdu_idx=hdu_idx)
                (std_apphot, std_filter_keys, optimum_aprad,
                 mag_conv, mag_corr, qlf_aprad, state) = std_photometry_results

                if not state:
                    self._log.critical("> Standard star selection and/or photometry has FAILED!!! "
                                       "Skipping further steps. ")
                    self.current_hdu_exec_state[h] = False
                    final_status = -1
                    continue

                # Prepare trail image and perform aperture photometry with optimum aperture
                sat_apphot = self.run_satellite_photometry(data_dict=data_dict,
                                                           img_dict=trail_img_dict,
                                                           img_idx=trail_img_idx,
                                                           hidx=h, hdu_idx=hdu_idx,
                                                           qlf_aprad=qlf_aprad)

                # Load TLE data, extract information for the satellite and calc mags, and angles, etc.
                state = self.get_final_results(img_dict=trail_img_dict,
                                               sat_name=sat_name,
                                               sat_apphot=sat_apphot,
                                               std_apphot=std_apphot,
                                               std_filter_keys=std_filter_keys,
                                               mag_conv=mag_conv, mag_corr=mag_corr,
                                               hidx=h, hdu_idx=hdu_idx)

                if not state:
                    self.current_hdu_exec_state[h] = False
                    final_status = -1
                    continue

                # todo: fix problems with indices at some angles
                try:
                    self.analyse_trail_profile(img_dict=trail_img_dict,
                                               sat_name=sat_name,
                                               std_apphot=std_apphot,
                                               std_filter_keys=std_filter_keys,
                                               mag_corr=mag_corr,
                                               hidx=h, hdu_idx=hdu_idx,
                                               analyse_glint=analyse_glint)
                except Exception as e:
                    fail_msg = ['ProfileFitError',
                                'Error during profile analysis',
                                str(e)]
                    self.update_failed_processes(fail_msg)
                    self.current_hdu_exec_state[h] = False
                    continue

                del std_photometry_results, sat_apphot

        del obsparams, data_dict, trail_img_dict

        return final_status

    def analyse_trail_profile(self,
                              img_dict: dict,
                              sat_name: str,
                              std_apphot: astropy.table.Table,
                              std_filter_keys: dict,
                              mag_corr: tuple,
                              hidx: int, hdu_idx: int, analyse_glint: bool = False):
        """ Analyse trail profile and glint (if available)"""

        # Setup stuff
        config = self._config
        obspar = self._obsparams
        n_ext = obspar['n_ext']

        file_name = img_dict['fname']
        file_name_plot = file_name if n_ext == 0 else f'{file_name}_HDU{hdu_idx:02d}'

        image = img_dict['images'][hidx]
        image_hdr = img_dict['hdr'][hidx]
        image_fwhm = img_dict['kernel_fwhm'][hidx][0]
        # image_fwhm_err = img_dict['kernel_fwhm'][hidx][1]
        image_detector_mask = img_dict['img_mask'][hidx]
        reg_data_all = img_dict["trail_data"][hidx]['reg_info_lst']

        # n_trails = self._config['N_TRAILS_MAX']

        # Standard star data
        _aper_sum = np.array(list(zip(std_apphot['flux_counts_aper'],
                                      std_apphot['flux_counts_err_aper'])))
        _mags = np.array(list(zip(std_apphot[std_filter_keys[0]],
                                  std_apphot[std_filter_keys[1]])))
        # std_pos = np.array(list(zip(std_apphot['xcentroid'],
        #                             std_apphot['ycentroid'])))

        # Get satellite velocity
        sat_vel, sat_vel_err = self.sat_vel

        fac = 2.  # scale factor by which the height is increased
        r = 0  # trail index

        # Get additional info
        exptime = image_hdr['exptime']
        pixelscale = image_hdr['pixscale']  # arcsec/pix
        # pixelscale_err = 0.05 * pixelscale

        # Create the combined image mask
        image_mask_combined = np.zeros(image.shape)
        image_mask_combined = np.ma.make_mask(image_mask_combined, dtype=bool, shrink=False)

        image_mask_sources = np.zeros(image.shape)
        image_mask_sources = np.ma.make_mask(image_mask_sources, dtype=bool, shrink=False)

        src_mask = self._image_mask_collection['sources_mask']
        if src_mask is not None:
            image_mask_sources |= src_mask

        image_mask_exclude = np.ones(image.shape)
        image_mask_exclude = np.ma.make_mask(image_mask_exclude, dtype=bool, shrink=False)

        # Combine with detector mask
        if image_detector_mask is not None:
            image_mask_combined |= image_detector_mask

        # Mask borders
        image_mask_combined = sext.make_border_mask(image_mask_combined, borderLen=20)

        if not self._silent:
            self._log.info("> Perform analysis of satellite trail profile")

        # Get the image background subtracted
        image_bkg_fnme = img_dict['bkg_data'][hidx]['bkg_fname']
        img_processed = self.process_image(image,
                                           self._telescope,
                                           image_bkg_fnme)
        image_bkg_sub, image_bkg, image_bkg_rms, image_mask_negative = img_processed

        # Add mask with pixel < 0
        image_mask_combined |= image_mask_negative

        image_mask_trail, _ = self.create_mask(image=image_bkg_sub,
                                               reg_data=reg_data_all,
                                               fac=fac)

        center_trail = reg_data_all[r]['coords']
        height_trail = int(round(fac * reg_data_all[r]['height']))
        L_trail = int(round(reg_data_all[r]['width']))

        ang_trail = reg_data_all[r]['orient_deg']

        # print(center_trail, height_trail, L_trail, ang_trail)

        # Exclude the glint if present
        if self._glint_mask_data is not None:
            image_mask_glint, _ = self._glint_mask_data
            image_mask_exclude *= ~image_mask_glint

        image_mask_combined |= image_mask_sources

        #
        image_mask_exclude *= ~image_mask_combined

        # Get a copy of the image
        image_bkg_sub_masked = image_bkg_sub.copy()
        image_bkg_sub_masked[~image_mask_exclude] = np.nan

        # Get profiles perpendicular to the center line
        profiles_trail = sats.perpendicular_line_profile(center_trail,
                                                         L=L_trail,
                                                         theta=ang_trail,
                                                         perp_length=height_trail,
                                                         num_samples=L_trail,
                                                         use_pixel_sampling=True)
        # Get pixel values along each profile
        profiles_data_trail = sats.get_pixel_values(image_bkg_sub_masked, profiles_trail, height_trail)

        # print(profiles)
        # print(profiles_data_trail)
        # print(profiles_data_trail.shape)

        # Number of profiles and length of each profile
        num_profiles_minor_trail = profiles_data_trail.shape[0]
        num_profiles_major_trail = profiles_data_trail.shape[1]

        x_minor_trail = np.arange(0, num_profiles_minor_trail, 1)
        x_major_trail = np.arange(0, num_profiles_major_trail, 1)

        # Get stats along the major axis
        mean_minor_trail, median_minor_trail, std_minor_trail = sigma_clipped_stats(profiles_data_trail,
                                                                                    axis=1,
                                                                                    cenfunc=np.nanmedian,
                                                                                    stdfunc=np.nanstd,
                                                                                    maxiters=None
                                                                                    )
        # print(median)
        # print(std)
        median_minor_trail = np.ma.masked_invalid(median_minor_trail)
        std_minor_trail = np.ma.masked_invalid(std_minor_trail)

        # Select which type of model to fit; gaussian and lognormal are currently available
        model_type_1comp = 'gaussian'
        model_type_2comp = 'gaussian'
        # model_type_1comp = 'lognormal'
        # model_type_2comp = 'lognormal'

        # Fit line profile along minor axis using 1 model + bkg
        fit_result_trail_minor = sats.fit_single_model(x=x_minor_trail, y=median_minor_trail,
                                                       fwhm=image_fwhm, yerr=None,
                                                       model_type=model_type_1comp)

        # Fit line profile along minor axis using 2 models + bkg
        fit_result_trail_2comp_minor = sats.fit_line_profile_two_comps(x=x_minor_trail, y=median_minor_trail,
                                                                       fwhm=image_fwhm, yerr=None,
                                                                       model_type=model_type_2comp)
        if not self._silent:
            self._log.info("  Plot trail profile along minor axis")
        # Plot the line profile + fit result along minor axis
        # the fit components can be commented and will not be plotted
        self.plot_profile_along_axis(x=x_minor_trail,
                                     y=median_minor_trail,
                                     yerr=std_minor_trail,
                                     fit_result_1comp=fit_result_trail_minor,
                                     model_type_1comp=model_type_1comp,
                                     fit_result_2comp=fit_result_trail_2comp_minor,
                                     model_type_2comp=model_type_2comp,
                                     axis='minor',
                                     fname=file_name_plot, is_glint=False)
        # Plot 3D surface plot
        # self.plot_profile_3D(x_major_trail, x_minor_trail, profiles_data_trail)

        ####################################################
        # this is the part where the glint comes into play
        ####################################################

        # Load glint data and apply analysis
        glint_mask_info = self._obsTable.glint_mask_data

        if not analyse_glint or glint_mask_info.empty or glint_mask_info['HDU'].values[0] != hdu_idx:
            return None

        # Reset mask
        image_bkg_sub_masked = image_bkg_sub.copy()

        # Get mask data
        x0 = glint_mask_info['xc'].values[0]
        y0 = glint_mask_info['yc'].values[0]
        center_glint = (x0, y0)
        L_glint = glint_mask_info['length'].values[0]

        # Get profiles perpendicular to the center line
        profiles_glint = sats.perpendicular_line_profile(center_glint, L_glint, ang_trail, height_trail,
                                                         num_samples=L_glint,
                                                         use_pixel_sampling=True)
        # Get pixel values along each profile
        profiles_data_glint = sats.get_pixel_values(image_bkg_sub_masked, profiles_glint, height_trail)

        # Subtract mean along minor axis along major axis
        profiles_data_glint_corrected = (profiles_data_glint.T - mean_minor_trail).T

        # Mask invalid pixel
        profiles_data_glint = np.ma.masked_invalid(profiles_data_glint_corrected)
        # print(profiles_data_glint)

        # Number of profiles and length of each profile
        num_profiles_minor_glint = profiles_data_glint.shape[0]
        num_profiles_major_glint = profiles_data_glint.shape[1]

        # Get x values along each axis
        x_minor_glint = np.arange(0, num_profiles_minor_glint, 1)
        x_major_glint = np.arange(0, num_profiles_major_glint, 1)

        # Get position of global maximum
        max_pos = np.unravel_index(np.nanargmax(profiles_data_glint), profiles_data_glint.shape)
        yo, xo = max_pos
        # print(yo, xo)

        # sats.fit_3D_profile_two_gaussian(x_major_glint, x_minor_glint,
        #                                  profiles_data_glint, image_fwhm)

        # Plot 3D
        # self.plot_profile_3D(x_major_glint, x_minor_glint,
        #                       profiles_data_glint,
        #                       np.nanmax(median_minor_trail),
        #                       split_cmap=True)

        profile_major_glint = profiles_data_glint[int(yo), :]
        profile_minor_glint = profiles_data_glint[:, int(xo)]

        model_type_1comp = 'gaussian'
        model_type_2comp = 'gaussian'

        # Fit line profile along minor axis using 1 model + bkg
        fit_result_glint_major = sats.fit_single_model(x=x_major_glint, y=profile_major_glint,
                                                       fwhm=image_fwhm, yerr=None,
                                                       model_type=model_type_1comp)

        # Fit line profile along minor axis using 2 models + bkg
        fit_result_glint_2comp_major = sats.fit_line_profile_two_comps(x=x_major_glint, y=profile_major_glint,
                                                                       fwhm=image_fwhm, yerr=None,
                                                                       model_type=model_type_2comp)

        if not self._silent:
            self._log.info("    Plot glint profile along major axis")

        # Plot the line profile + fit result along major axis
        # the fit components can be commented and will not be plotted
        self.plot_profile_along_axis(x=x_major_glint,
                                     y=profile_major_glint,
                                     yerr=None,
                                     fit_result_1comp=fit_result_glint_major,
                                     model_type_1comp=model_type_1comp,
                                     fit_result_2comp=fit_result_glint_2comp_major,
                                     model_type_2comp=model_type_2comp,
                                     axis='major',
                                     fname=file_name_plot, is_glint=True)

        # Example use of the function
        model_params_major = {
            'Gaussian1': {'A': fit_result_glint_2comp_major.params['m1_amplitude'].value,
                          'mu': fit_result_glint_2comp_major.params['m1_center'].value,
                          'sigma': fit_result_glint_2comp_major.params['m1_sigma'].value},
            'Gaussian2': {'A': fit_result_glint_2comp_major.params['m2_amplitude'].value,
                          'mu': fit_result_glint_2comp_major.params['m2_center'].value,
                          'sigma': fit_result_glint_2comp_major.params['m2_sigma'].value}
        }

        # Select the model type
        model_type_1comp = 'gaussian'
        model_type_2comp = 'gaussian'

        # Fit line profile along minor axis using 1 model + bkg
        fit_result_glint_minor = sats.fit_single_model(x=x_minor_glint, y=profile_minor_glint,
                                                       fwhm=image_fwhm, yerr=None,
                                                       model_type=model_type_1comp)

        # Fit line profile along minor axis using 2 models + bkg
        fit_result_glint_2comp_minor = sats.fit_line_profile_two_comps(x=x_minor_glint, y=profile_minor_glint,
                                                                       fwhm=image_fwhm, yerr=None,
                                                                       model_type=model_type_2comp)
        if not self._silent:
            self._log.info("    Plot glint profile along minor axis")

        # Plot the line profile + fit result along minor axis
        # The fit components can be commented and will not be plotted
        self.plot_profile_along_axis(x=x_minor_glint,
                                     y=profile_minor_glint,
                                     yerr=None,
                                     fit_result_1comp=fit_result_glint_minor,
                                     model_type_1comp=model_type_1comp,
                                     fit_result_2comp=fit_result_glint_2comp_minor,
                                     model_type_2comp=model_type_2comp,
                                     axis='minor',
                                     fname=file_name_plot, is_glint=True)

        # Calculate fit bounds
        model_params_minor = {
            'Gaussian1': {'A': fit_result_glint_2comp_minor.params['m1_amplitude'].value,
                          'mu': fit_result_glint_2comp_minor.params['m1_center'].value,
                          'sigma': fit_result_glint_2comp_minor.params['m1_sigma'].value},
            'Gaussian2': {'A': fit_result_glint_2comp_minor.params['m2_amplitude'].value,
                          'mu': fit_result_glint_2comp_minor.params['m2_center'].value,
                          'sigma': fit_result_glint_2comp_minor.params['m2_sigma'].value}
        }

        # Find 3-sigma bounds
        bounds_major = sats.find_gaussian_bounds(model_params_major, sigma=3)
        bounds_minor = sats.find_gaussian_bounds(model_params_minor, sigma=3)

        # Get inputs for aperture creation
        aper_height_glint = bounds_minor[1] - bounds_minor[0]
        aper_width_glint = bounds_major[1] - bounds_major[0]
        aper_pos_glint = (aper_width_glint / 2 + bounds_major[0], aper_height_glint / 2 + bounds_minor[0])

        # Get fluxes via aperture photometry
        phot_glint = phot.get_glint_photometry(profiles_data_glint, profiles_data_glint.mask,
                                               aper_pos_glint,
                                               aper_width_glint, aper_height_glint, 0., config)

        phot_glint_df = pd.DataFrame.from_dict(phot_glint, orient='index')
        obs_len = aper_width_glint * pixelscale

        # Get time on detector
        glint_duration = obs_len / sat_vel
        glint_duration_err = (obs_len * sat_vel_err / sat_vel ** 2)

        phot_glint_df['area_arc'] = phot_glint_df.apply(lambda row: row.area_pix * pixelscale ** 2, axis=1)

        # Apply the function and create a new DataFrame
        mag_df = phot_glint_df.apply(self.get_glint_magnitudes, axis=1,
                                     args=(obs_len, exptime, pixelscale, (sat_vel, sat_vel_err),
                                           _aper_sum, _mags, mag_corr))
        combined_df = pd.concat([phot_glint_df, mag_df], axis=1)

        col_names = combined_df.columns

        add_columns = {
            'File': file_name,
            'SatName': sat_name,
            'GlintLength_pix': aper_width_glint,
            'GlintLength_arc': obs_len,
            'EventDuration': glint_duration,
            'e_EventDuration': glint_duration_err,
            # Add more columns and values as needed
        }

        new_cols = list(add_columns.keys())
        new_cols.extend(col_names)

        # Add new columns from the list
        for column, value in add_columns.items():
            combined_df[column] = None  # Initialize with None
            combined_df.at['with_sat_src', column] = value  # Set value for the first row

        result = combined_df[new_cols]

        if not self._silent:
            self._log.info("    Save glint photometry to .csv")

        fname = f"result.phot.glint_{file_name_plot}.csv"
        fname = os.path.join(self._work_dir, fname)

        # Save the result to a CSV file
        result.to_csv(fname,
                      header=result.columns,
                      index=False)
        if not self._silent:
            self._log.info("    Plot 2D cutout + profiles along major/minor axes")

        self.plot_glint_results(profiles_data_glint,
                                (aper_pos_glint, aper_width_glint, aper_height_glint, 0.),
                                (bounds_major, bounds_minor),
                                [(xo, yo), fit_result_glint_2comp_major, fit_result_glint_2comp_minor],
                                file_name_plot)
        # plt.show()

    def get_glint_magnitudes(self, row, obs_len, exptime, pixelscale, sat_vel, std_fluxes, std_mags, mag_corr):
        """ Calculate the magnitudes for the glint """

        mag_scale = self.mag_scale

        # Estimated trail length
        est_len = exptime * sat_vel[0]
        est_len_err = exptime * sat_vel[1]

        # Observed trail flux count and area
        flux_obs = row['flux_counts']
        flux_obs_err = row['flux_counts_err']
        area = row['area_pix'] * pixelscale ** 2

        # Calculating estimated trail count
        flux_est = flux_obs * (est_len / obs_len)

        # Calculating relative errors
        rel_error = ((flux_obs_err / flux_obs) ** 2 +
                     (est_len_err / est_len) ** 2) ** 0.5

        # Calculating absolute error in estimated trail count
        flux_est_err = flux_est * rel_error

        param_list = [[['ObsMagGlint', 'e_ObsMagGlint'],
                       flux_obs, flux_obs_err, None, None],
                      [['EstMagGlint', 'e_EstMagGlint'],
                       flux_est, flux_est_err, None, None],
                      [['EstScaleMagGlint', 'e_EstScaleMagGlint'],
                       flux_est, flux_est_err, mag_scale, None],
                      [['ObsMagGlintSurf', 'e_ObsMagGlintSurf'],
                       flux_obs, flux_obs_err, None, area],
                      [['EstMagGlintSurf', 'e_EstMagGlintSurf'],
                       flux_est, flux_est_err, None, area],
                      [['EstScaleMagGlintSurf', 'e_EstScaleMagGlintSurf'],
                       flux_est, flux_est_err, mag_scale, area],
                      [['ObsMagGlintSurfSec', 'e_ObsMagGlintSurfSec'],
                       flux_obs, flux_obs_err, None, area * exptime],
                      [['EstMagGlintSurfSec', 'e_EstMagGlintSurfSec'],
                       flux_est, flux_est_err, None, area * exptime],
                      [['EstScaleMagGlintSurfSec', 'e_EstScaleMagGlintSurfSec'],
                       flux_est, flux_est_err, mag_scale, area * exptime],
                      ]

        col_names = []
        data_list = []
        for p in param_list:
            mag_avg = sats.get_average_magnitude(flux=p[1],
                                                 flux_err=p[2],
                                                 std_fluxes=std_fluxes,
                                                 mag_corr=mag_corr,
                                                 std_mags=std_mags, mag_scale=p[3], area=p[4])

            col_names.extend(p[0])
            data_list.extend(mag_avg)
            del p, mag_avg

        return pd.Series(data_list, index=col_names)

    def get_final_results(self,
                          img_dict: dict,
                          sat_name: str,
                          sat_apphot: dict,
                          std_apphot: astropy.table.Table,
                          std_filter_keys: dict,
                          mag_conv: bool, mag_corr: tuple,
                          hidx: int, hdu_idx: int):

        """ Combine data and get final results for the found trails.

        Use collected data to estimate all parameters such as observed and
        estimated lengths, magnitudes, and angles employing the available telemetry and observation datasets.

        :param img_dict: Dictionary containing trail image data.
        :param sat_name: String representing the satellite identifier.
        :param sat_apphot: Catalog containing satellite photometry data.
        :param std_apphot: Catalog containing standard star photometry data.
        :param std_filter_keys: List or array of standard star filter band and error keyword names.
        """

        obj_info = self._obj_info

        hdr = img_dict['hdr'][hidx]
        obsparams = self._obsparams
        n_ext = obsparams['n_ext']

        file_name = img_dict['fname']
        file_name_plot = file_name if n_ext == 0 else f'{file_name}_HDU{hdu_idx:02d}'

        n_trails_max = self._config['N_TRAILS_MAX']

        has_mag_zp = False
        mag_zp = None

        # Observation and location data
        geo_lon = obsparams['longitude']
        geo_lat = obsparams['latitude']
        geo_alt = obsparams['altitude'] / 1.e3  # in km

        # Get additional info
        exptime = hdr['exptime']
        pixelscale = float(hdr['pixscale'])  # arcsec/pix
        pixelscale_err = 0.05 * pixelscale

        if not self._silent:
            self._log.info("> Perform final analysis of satellite trail data")

        # Get nominal orbit altitude
        sat_h_orb_ref = bc.get_nominal_orbit_altitude(sat_name)
        # sat_h_orb_ref = bc.SAT_HORB_REF['ONEWEB']
        # if 'STARLINK' in sat_name:
        #     sat_h_orb_ref = bc.SAT_HORB_REF['STARLINK']
        # if 'BLUEWALKER' in sat_name:
        #     sat_h_orb_ref = bc.SAT_HORB_REF['BLUEWALKER']

        # Get time delta of obs mid-time
        date_obs = obj_info['Date-Obs'].to_numpy(dtype=object)[0]
        obs_mid = obj_info['Obs-Mid'].to_numpy(dtype=object)[0]
        obs_start = obj_info['Obs-Start'].to_numpy(dtype=object)[0]
        obs_end = obj_info['Obs-Stop'].to_numpy(dtype=object)[0]
        dtobj_obs = datetime.strptime(f"{date_obs}T{obs_mid}", "%Y-%m-%dT%H:%M:%S.%f")

        # Standard star data
        _aper_sum = np.array(list(zip(std_apphot['flux_counts_aper'],
                                      std_apphot['flux_counts_err_aper'])))
        _mags = np.array(list(zip(std_apphot[std_filter_keys[0]],
                                  std_apphot[std_filter_keys[1]])))
        std_pos = np.array(list(zip(std_apphot['xcentroid'],
                                    std_apphot['ycentroid'])))

        # Get TLE file and calculate angular velocity
        if not self._silent:
            self._log.info("  > Find TLE file location")
        tle_location = self._obsTable.find_tle_file(sat_name=sat_name,
                                                    img_filter=hdr['FILTER'],
                                                    src_loc=img_dict['loc'])
        # print(img_dict['loc'])
        if tle_location is None:
            self.update_failed_processes(['LoadTLEError',
                                          'TLE file not found',
                                          'Check that name of the TLE file'])
            return False

        if not self._silent:
            self._log.info("  > Check if satellite is in the TLE file")
        tle_sat_name = self._obsTable.find_satellite_in_tle(tle_loc=tle_location[2],
                                                            sat_name=sat_name)

        if tle_sat_name is None:
            self.update_failed_processes(['LoadSatError',
                                          'Satellite file not found',
                                          'Check that name of the satellite'])
            return False

        # if not pd.isnull(sat_info['AltID']).any():
        #     sat_name = f'{sat_name} ({sat_info["AltID"].values[0]})'
        # if not pd.isnull(sat_info['UniqueID']).any() and 'BLUEWALKER' not in sat_name:
        #     sat_name = f'{sat_name}-{sat_info["UniqueID"].values[0]}'
        if not self._silent:
            self._log.info("  > Calculate satellite information from TLE")

        sat_vel, sat_vel_err, tle_pos_df = self.get_data_from_tle(hdr=hdr, sat_name=(tle_sat_name, sat_name),
                                                                  tle_location=str(tle_location[2]),
                                                                  date_obs=date_obs,
                                                                  obs_times=[obs_start, obs_mid, obs_end],
                                                                  exptime=exptime,
                                                                  dt=self._config['DT_STEP_SIZE'])

        self.sat_vel = (sat_vel, sat_vel_err)

        sat_vel = float('%.3f' % sat_vel)
        sat_vel_err = float('%.3f' % sat_vel_err)
        sat_info = tle_pos_df[tle_pos_df['Obs_Time'] == dtobj_obs.strftime(frmt)[:-3]]

        # Required satellite data
        sat_info.reset_index(drop=True)
        result = sat_info.iloc[0].to_dict()
        sat_lon = sat_info['SatLon'].to_numpy(dtype=float)[0]
        sat_lat = sat_info['SatLat'].to_numpy(dtype=float)[0]
        sat_alt = sat_info['SatAlt'].to_numpy(dtype=float)[0]
        sat_az = sat_info['SatAz'].to_numpy(dtype=float)[0]
        sat_elev = sat_info['SatElev'].to_numpy(dtype=float)[0]

        # Calculate solar incidence angle theta_1
        sun_inc_ang = sats.sun_inc_elv_angle(dtobj_obs,
                                             (sat_lat, sat_lon))

        # Range, distance observer satellite
        dist_obs_sat = sats.get_obs_range(sat_elev, sat_alt, geo_alt, geo_lat)

        # Calculate observer phase angle theta_2
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
        # Get scaled estimated magnitude with scale factor -5log(range/h_orb_ref)
        mag_scale = -5. * np.log10((dist_obs_sat / sat_h_orb_ref))
        self.mag_scale = mag_scale

        for i in range(n_trails_max):
            reg_info = img_dict["trail_data"][hidx]['reg_info_lst'][i]
            if not self._silent:
                self._log.info("  > Combine data to calculate angles and magnitudes")

            # Observed trail length
            obs_trail_len = reg_info['width'] * pixelscale
            obs_trail_len_err = np.sqrt((pixelscale * reg_info['e_width']) ** 2
                                        + (reg_info['width'] * pixelscale_err) ** 2)

            # Estimated trail length and count
            est_trail_len = exptime * sat_vel
            est_trail_len_err = np.sqrt(est_trail_len)

            # Time on detector
            time_on_det = obs_trail_len / sat_vel
            time_on_det_err = np.sqrt((obs_trail_len_err / sat_vel) ** 2
                                      + (obs_trail_len * sat_vel_err / sat_vel ** 2) ** 2)

            # Observed trail count
            obs_trail_count = sat_apphot[i]['aperture_sum_bkgsub'].values[0]
            obs_trail_count_err = sat_apphot[i]['aperture_sum_bkgsub_err'].values[0]

            # Calculating estimated trail count
            est_trail_count = obs_trail_count * (est_trail_len / obs_trail_len)

            # Calculating relative errors
            rel_error = ((obs_trail_count_err / obs_trail_count) ** 2 +
                         (est_trail_len_err / est_trail_len) ** 2 +
                         (obs_trail_len_err / obs_trail_len) ** 2) ** 0.5

            # Calculating absolute error in estimated trail count
            est_trail_count_err = est_trail_count * rel_error

            mag_corr_sat = sat_apphot[i]['mag_corr'].values[0]

            obs_mag_avg_w = None
            obs_mag_avg_err = None
            est_mag_avg_w = None
            est_mag_avg_err = None
            est_mag_avg_w_scaled = None
            est_mag_avg_err_scaled = None

            obs_mag_zp = None
            obs_mag_zp_err = None
            est_mag_zp = None
            est_mag_zp_err = None
            est_mag_zp_scaled = None
            est_mag_zp_err_scaled = None

            if obs_trail_count > 0.:
                # Observed satellite magnitude relative to std stars
                obs_mag_avg = sats.get_average_magnitude(flux=obs_trail_count,
                                                         flux_err=obs_trail_count_err,
                                                         std_fluxes=_aper_sum,
                                                         mag_corr=mag_corr,
                                                         mag_corr_sat=mag_corr_sat,
                                                         std_mags=_mags)
                obs_mag_avg_w, obs_mag_avg_err = obs_mag_avg

                # Estimated magnitude difference
                est_mag_avg = sats.get_average_magnitude(flux=est_trail_count,
                                                         flux_err=est_trail_count_err,
                                                         std_fluxes=_aper_sum,
                                                         mag_corr=mag_corr,
                                                         mag_corr_sat=mag_corr_sat,
                                                         std_mags=_mags)
                est_mag_avg_w, est_mag_avg_err = est_mag_avg

                # Estimated magnitude difference scaled with mag scale
                est_mag_avg_scaled = sats.get_average_magnitude(flux=est_trail_count,
                                                                flux_err=est_trail_count_err,
                                                                std_fluxes=_aper_sum,
                                                                std_mags=_mags,
                                                                mag_corr=mag_corr,
                                                                mag_corr_sat=mag_corr_sat,
                                                                mag_scale=mag_scale)
                est_mag_avg_w_scaled, est_mag_avg_err_scaled = est_mag_avg_scaled

                # If there is a magnitude zero point, also calculate the result
                # use https://www.gnu.org/software/gnuastro/manual/html_node/Brightness-flux-magnitude.html
                if 'MAGZPT' in hdr:
                    has_mag_zp = True
                    mag_zp = hdr['MAGZPT']
                    obs_mag_zp, obs_mag_zp_err = sats.get_magnitude_zpmag(obs_trail_count,
                                                                          obs_trail_count_err,
                                                                          mag_zp,
                                                                          mag_corr_sat=mag_corr_sat)

                    est_mag_zp, est_mag_zp_err = sats.get_magnitude_zpmag(est_trail_count,
                                                                          est_trail_count_err,
                                                                          mag_zp, mag_corr_sat=mag_corr_sat)

                    est_mag_zp_scaled, est_mag_zp_err_scaled = sats.get_magnitude_zpmag(est_trail_count,
                                                                                        est_trail_count_err,
                                                                                        mag_zp,
                                                                                        mag_corr_sat=mag_corr_sat,
                                                                                        mag_scale=mag_scale)

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
                         'HasZp', 'MagZp',
                         'ObsMag_zp', 'e_ObsMag_zp',
                         'EstMag_zp', 'e_EstMag_zp',
                         'EstScaleMag_zp', 'e_EstScaleMag_zp',
                         'SunAzAng', 'SunElevAng',
                         'FluxScale', 'MagScale',
                         'MagCorrect', 'e_MagCorrect',
                         'MagCorrectSat',
                         'dt_tle-obs', 'mag_conv']

            # Make sure that the RA and DEC are consistent and rounded to the same decimal
            ra = hdr[obsparams['ra']]
            dec = hdr[obsparams['dec']]
            if obsparams['radec_separator'] == 'XXX':
                ra = round(hdr[obsparams['ra']], bc.ROUND_DECIMAL)
                dec = round(hdr[obsparams['dec']], bc.ROUND_DECIMAL)

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
                        'T' if has_mag_zp else 'F', mag_zp,
                        obs_mag_zp, obs_mag_zp_err,
                        est_mag_zp, est_mag_zp_err,
                        est_mag_zp_scaled, est_mag_zp_err_scaled,
                        sun_az, sun_alt, (est_trail_len / obs_trail_len), mag_scale,
                        mag_corr[0], mag_corr[1], mag_corr_sat, 0,
                        'T' if mag_conv else 'F']

            for k, v in zip(tbl_names, tbl_vals):
                result[k] = v

            result['RADEC_Separator'] = obsparams['radec_separator']
            result['HDU_IDX'] = hdu_idx

            self._obsTable.update_obs_table(file=file_name,
                                            kwargs=result, obsparams=obsparams)

            if not self._silent:
                self._log.info("  > Plot trail detection result (including TLE positions)")

            # final plot
            self.plot_final_result(img_dict=img_dict,
                                   reg_data=reg_info,
                                   obs_date=(date_obs, obs_mid),
                                   expt=exptime,
                                   std_pos=std_pos,
                                   file_base=file_name_plot,
                                   tle_pos=tle_pos_df,
                                   hidx=hidx)
        bc.clean_up(_mags, _aper_sum, sat_info,
                    obj_info, hdr, obsparams, img_dict, sat_apphot, std_apphot)

        return True

    def run_satellite_photometry(self,
                                 data_dict: dict,
                                 img_dict: dict, img_idx: int, hidx: int, hdu_idx: int, qlf_aprad: bool):
        """Perform aperture photometry on the satellite trails"""

        config = self._config
        obspar = self._obsparams
        n_ext = obspar['n_ext']

        file_name = img_dict['fname']
        file_name_plot = file_name if n_ext == 0 else f'{file_name}_HDU{hdu_idx:02d}'

        trail_img = img_dict['images'][hidx]
        trail_img_hdr = img_dict['hdr'][hidx]
        trail_img_bkg_fnme = img_dict['bkg_data'][hidx]['bkg_fname']

        image_fwhm = img_dict['kernel_fwhm'][hidx][0]
        image_fwhm_err = img_dict['kernel_fwhm'][hidx][1]
        image_detector_mask = img_dict['img_mask'][hidx]
        reg_data_all = img_dict["trail_data"][hidx]['reg_info_lst']

        n_trails = config['N_TRAILS_MAX']
        n_refs = None
        ref_image_idx = None
        ref_image_hdu_idx = None
        has_ref = False
        sat_phot_dict = {}

        # Get image parameter
        exp_time = trail_img_hdr[obspar['exptime']]
        gain = 1.
        rdnoise = 0.

        # if isinstance(obspar['gain'], str) and obspar['gain'] in trail_img_hdr:
        #     gain = trail_img_hdr[obspar['gain']]
        # elif isinstance(obspar['gain'], float):
        #     gain = obspar['gain']
        # if isinstance(obspar['readnoise'], str) and obspar['readnoise'] in trail_img_hdr:
        #     rdnoise = trail_img_hdr[obspar['readnoise']]
        # elif isinstance(obspar['readnoise'], float):
        #     rdnoise = obspar['readnoise']

        if not self._silent:
            self._log.info("> Perform photometry on satellite trail(s)")

        # Choose reference image if available (only first at the moment)
        idx_list = list(data_dict['ref_img_idx'][img_idx])
        if idx_list:
            ref_image_idx = idx_list[0]
            if len(idx_list) > 1:
                self._log.info(f"  Found {len(idx_list)} reference images to choose from.")
                data = np.array(data_dict['file_names'])[idx_list]
                _, idx = self._dataset_all.select_file_from_list(data=data,
                                                                 config=config['DEF_SELECT_TIMEOUT'])
                ref_image_idx = data_dict['ref_img_idx'][img_idx][np.array(idx)]

            # Select reference image HDU
            ref_image_hdu_idx = np.where(hdu_idx == np.array(data_dict['hdus'][ref_image_idx]))[0]

            has_ref = True
            n_refs = len(idx_list)

        else:
            self._log.info(f"  No reference image found. Proceeding without ...")

        # Create the combined image mask
        image_mask_combined = np.zeros(trail_img.shape)
        image_mask_combined = np.ma.make_mask(image_mask_combined, dtype=bool, shrink=False)

        # Create the source mask for sources detected along the trail
        src_mask = np.zeros(trail_img.shape)
        src_mask = np.ma.make_mask(src_mask, dtype=bool, shrink=False)

        # Combine with detector mask
        if image_detector_mask is not None:
            image_mask_combined |= image_detector_mask

        # Mask saturated objects
        image_mask_saturation = (trail_img > 0.95 * config['sat_lim'])
        image_mask_combined |= image_mask_saturation

        # Subtract background if necessary
        trail_img_processed = self.process_image(trail_img,
                                                 self._telescope,
                                                 trail_img_bkg_fnme)
        trail_img_bkg_sub, trail_img_bkg, trail_img_bkg_rms, _ = trail_img_processed

        df = img_dict['cats_cleaned_rem'][hidx]['ref_cat_cleaned']
        if not has_ref and not df.empty:

            # Filter apertures outside image
            trail_stars = df.drop((df.loc[(df['xcentroid'] < 0) |
                                          (df['ycentroid'] < 0) |
                                          (df['xcentroid'] > trail_img.shape[1]) |
                                          (df['ycentroid'] > trail_img.shape[0])]).index)

            # Positions of sources detected in the trail image
            _pos = np.array(list(zip(trail_stars['xcentroid'],
                                     trail_stars['ycentroid'])))

            # Create the mask for each position
            for i in range(_pos.shape[0]):
                rr, cc = disk((_pos[:, 1][i], _pos[:, 0][i]), 2.5 * image_fwhm)
                src_mask[rr, cc] = True

        if has_ref:
            if not self._silent:
                self._log.info("  > Align reference image and create source mask")

            # Load reference image and mask
            ref_imgarr = data_dict['images'][ref_image_idx][ref_image_hdu_idx[0]]
            ref_img_mask = data_dict['image_mask'][ref_image_idx][ref_image_hdu_idx[0]]
            ref_bkg_fname = data_dict['bkg_data'][ref_image_idx][ref_image_hdu_idx[0]]['bkg_fname']

            # Subtract background if necessary
            trail_img_processed = self.process_image(ref_imgarr,
                                                     self._telescope,
                                                     ref_bkg_fname)
            ref_img_bkg_sub, ref_bkg, ref_bkg_rms, _ = trail_img_processed

            # Create a saturation mask
            ref_img_mask_sat = (ref_img_bkg_sub > 0.95 * config['sat_lim'])

            # Update the image mask
            ref_img_mask = ref_img_mask | ref_img_mask_sat if ref_img_mask is not None else ref_img_mask_sat

            if not self._silent:
                self._log.info("    Find transformation")

            # New version: find transformation between reference and trail image
            src_mask = self.align_ref_img_and_create_mask(ref_img_bkg_sub, trail_img_bkg_sub,
                                                          ref_img_mask, image_mask_combined,
                                                          ref_bkg, image_fwhm, config)

            # Clean up
            bc.clean_up(ref_bkg, ref_bkg_rms, trail_img_bkg, trail_img_bkg_rms,
                        ref_img_mask, ref_imgarr, ref_img_bkg_sub)

        # Create a mask for the detected trail
        trail_mask, _ = self.create_mask(image=trail_img_bkg_sub,
                                         reg_data=reg_data_all,
                                         fac=3.)
        trail_mask |= image_mask_combined

        # Detect sources in the trail image with the trail masked
        trail_sources, mask = self.detect_sources_in_image(image=trail_img_bkg_sub,
                                                           detect_background=0.,
                                                           detect_error=None,
                                                           mask=trail_mask, npixels=5, nsigma=3.)

        # Combine the source mask with the trail mask
        if trail_sources is not None:
            src_mask |= mask
            del mask

        del trail_sources

        self._image_mask_collection['sources_mask'] = src_mask

        # Mask the glint, if available and add to the source mask
        if self._glint_mask_data is not None:
            mask = self._glint_mask_data[0]
            src_mask |= mask
            del mask

        # Loop over each detected trail and get optimum aperture and photometry
        for i in range(n_trails):

            # This offset can be used in case two parallel trails are close together and affect the
            # background aperture of the other
            h_offset = 15 if 'BLUEWALKER-3-016' in file_name else 0

            reg_info = img_dict["trail_data"][hidx]['reg_info_lst'][i]

            # Get trail parameters
            src_pos = reg_info['coords']
            src_pos_err = reg_info['coords_err']
            width = reg_info['width']  # i.e. length of the trail
            ang = Angle(reg_info['orient_deg'], 'deg')

            # Get positional error from astrometric calibration if available
            std_x = float(trail_img_hdr['POSERRX']) if 'POSERRX' in trail_img_hdr else 0.
            std_y = float(trail_img_hdr['POSERRY']) if 'POSERRY' in trail_img_hdr else 0.

            # Combine with the positional error of the trail detection
            src_pos_err_x = np.sqrt(src_pos_err[0] ** 2 + std_x ** 2)
            src_pos_err_y = np.sqrt(src_pos_err[1] ** 2 + std_y ** 2)

            if not self._silent:
                self._log.info(f'  > Estimate optimum aperture from signal-to-noise ratio (S/N) of '
                               f'satellite trail')

            # Combine estimated masks
            image_mask_combined |= src_mask

            # Get optimum trail aperture size
            result = phot.get_opt_aper_trail(image=trail_img,
                                             src_pos=[src_pos],
                                             img_mask=image_mask_combined,
                                             fwhm=image_fwhm,
                                             exp_time=exp_time,
                                             config=config,
                                             width=width,
                                             theta=ang,
                                             gain=gain,
                                             rdnoise=rdnoise)

            # Unpack the results
            fluxes, rapers, max_snr_aprad, opt_aprad, qlf_aptrail = result

            # Plot the result of the aperture photometry
            self.plot_aperture_photometry(fluxes=fluxes,
                                          rapers=rapers,
                                          opt_aprad=opt_aprad,
                                          file_base=file_name_plot,
                                          mode='trail')

            # Update aperture values with the optimum aperture
            config['APER_RAD_RECT'] = opt_aprad
            config['INF_APER_RAD_RECT'] = opt_aprad + 5

            config['RSKYIN_OPT_RECT'] = opt_aprad + 4.
            config['RSKYOUT_OPT_RECT'] = opt_aprad + 6.

            # Update the configuration
            self._config = config

            phot_df = phot.get_sat_photometry(trail_img, image_mask_combined, [src_pos],
                                              image_fwhm, width, ang,
                                              config)

            flux_ratios = phot_df['flux_counts_inf_aper'] / phot_df['flux_counts_aper']
            mag_corr = -2.5 * np.log10(flux_ratios)

            if not self._silent:
                self._log.info(f'  > Get magnitude correction for finite aperture')
                self._log.info(f'    ==> estimated magnitude correction: {mag_corr[0]:.4f} (mag)')

            # Set optimum aperture parameters
            optimum_apheight = opt_aprad * image_fwhm * 2.
            r_in = opt_aprad + 4.
            r_out = opt_aprad + 6.
            w_in = width + r_in * image_fwhm
            w_out = width + r_out * image_fwhm

            h_in = r_in * image_fwhm * 2. + h_offset
            h_out = r_out * image_fwhm * 2. + h_offset

            if not self._silent:
                self._log.info(f'  > Measure photometry for satellite trail')

            # Perform aperture photometry on satellite trail
            _, _, sat_phot, _, _, _ = phot.get_aper_photometry(image=trail_img,
                                                               src_pos=src_pos,
                                                               mask=image_mask_combined,
                                                               aper_mode='rect',
                                                               width=width, height=optimum_apheight,
                                                               w_in=w_in, w_out=w_out,
                                                               h_in=h_in, h_out=h_out,
                                                               theta=ang)
            # Add the estimated magnitude correction to the photometry table
            sat_phot['mag_corr'] = mag_corr

            # Update the photometry dictionary
            sat_phot_dict[i] = sat_phot

            # Convert pixel to world coordinates with uncertainties
            w = wcs.WCS(trail_img_hdr)
            px = np.array([src_pos[0],
                           src_pos[0] + src_pos_err_x,
                           src_pos[0] - src_pos_err_x])
            py = np.array([src_pos[1],
                           src_pos[1] + src_pos_err_y,
                           src_pos[1] - src_pos_err_y])

            # Convert to world coordinates and calculate the error in arcseconds
            wx, wy = w.wcs_pix2world(px, py, 0)
            wx_err = np.mean(abs(np.subtract(wx, wx[0])[1:])) * 3600.  # in arcseconds
            wy_err = np.mean(abs(np.subtract(wy, wy[0])[1:])) * 3600.  # in arcseconds

            # Make sure that the RA and DEC are consistent and rounded to the same decimal
            ra = trail_img_hdr[obspar['ra']]
            dec = trail_img_hdr[obspar['dec']]
            if obspar['radec_separator'] == 'XXX':
                ra = round(trail_img_hdr[obspar['ra']], bc.ROUND_DECIMAL)
                dec = round(trail_img_hdr[obspar['dec']], bc.ROUND_DECIMAL)
            trail_img_hdr[obspar['ra']] = ra
            trail_img_hdr[obspar['dec']] = dec

            kwargs = {obspar['ra']: ra,
                      obspar['dec']: dec,
                      obspar['object']: trail_img_hdr[obspar['object']],
                      obspar['instrume']: trail_img_hdr[obspar['instrume']],
                      obspar['exptime']: trail_img_hdr[obspar['exptime']],
                      'HDU_IDX': hdu_idx,
                      'FWHM': image_fwhm,  # in px
                      'e_FWHM': image_fwhm_err,  # in px
                      'TrailCX': src_pos[0],  # in px
                      'e_TrailCX': src_pos_err_x,  # in px
                      'TrailCY': src_pos[1],  # in px
                      'e_TrailCY': src_pos_err_y,  # in px
                      'TrailCRA': wx[0],  # in deg
                      'e_TrailCRA': wx_err,  # in arcseconds
                      'TrailCDEC': wy[0],  # in deg
                      'e_TrailCDEC': wy_err,  # in arcseconds
                      'TrailANG': reg_info['orient_deg'],  # in deg
                      'e_TrailANG': reg_info['e_orient_deg'],  # in deg
                      'OptAperHeight': '%3.2f' % optimum_apheight,
                      'HasRef': 'T' if has_ref else 'F', 'NRef': n_refs,
                      'QlfAperRad': 'T' if qlf_aprad else 'F'}  # in px

            # Update the observation result table
            self._obsTable.update_obs_table(file=img_dict['fname'], kwargs=kwargs,
                                            obsparams=obspar)
            # Get the updated object information
            self._obsTable.get_object_data(fname=file_name, kwargs=trail_img_hdr,
                                           obsparams=obspar, hdu_idx=hdu_idx)
            self._obj_info = self._obsTable.obj_info

        # Clean up
        bc.clean_up(trail_img, trail_img_bkg_sub, df,
                    config, obspar, trail_img_hdr, data_dict, img_dict)

        return sat_phot_dict

    def align_ref_img_and_create_mask(self, ref_img, trail_img,
                                      ref_img_mask, trail_img_mask,
                                      ref_img_bkg,
                                      image_fwhm, config):
        """
        Align the reference image with the trail image and create a mask for the reference image.

        Notes
        -----
            - Assumes that the reference image and the trail image are already background subtracted.

        """
        # Set the box size
        fwhm_multiplier = config['ISOLATE_SOURCES_FWHM_SEP']
        box_size = int(np.ceil(fwhm_multiplier * image_fwhm)) + 0.5

        # Exclude pixels near the border
        trail_img_mask = sext.make_border_mask(trail_img_mask, borderLen=2 * box_size)
        ref_img_mask = sext.make_border_mask(ref_img_mask, borderLen=2 * box_size)

        # Detect sources in the trail image
        img_sources, _ = self.detect_sources_in_image(image=trail_img,
                                                      detect_background=0.,
                                                      detect_error=None,
                                                      nsigma=3.0, npixels=5, connectivity=8,
                                                      mask=trail_img_mask)

        # Create a new index column for merging
        source_df = img_sources.reset_index(drop=True).reset_index(drop=False, names='obs_idx')

        # Detect sources in the reference image
        ref_sources, _ = self.detect_sources_in_image(image=ref_img,
                                                      detect_background=0.,
                                                      detect_error=None,
                                                      nsigma=3.0, npixels=5, connectivity=8,
                                                      mask=ref_img_mask)
        ref_sources.reset_index(drop=True, inplace=True)

        # Match observed data to reference data
        matches_df = imtrans.get_source_matches(obs_df=source_df, ref_df=ref_sources,
                                                match_radius=image_fwhm)

        # Merge matches with observed sources and create a mask of sources matched within the match radius
        matched_src_df = pd.merge(source_df, matches_df, on='obs_idx', how='left')
        has_m = matched_src_df["has_match"]
        dist_mask = pd.Series(False, index=matched_src_df.index)
        dist_mask.loc[has_m.index[has_m]] = matched_src_df.loc[has_m, 'closest_distance'] < image_fwhm

        # Compare the catalogs and get sources matched within the match radius
        src_subset_df, ref_subset_df = imtrans.extract_sources_subset(matched_src_df, ref_sources,
                                                                      has_m, dist_mask=dist_mask)

        # Extract the positions of the matched sources
        obs_xy = src_subset_df[["xcentroid", "ycentroid"]].values
        ref_xy = ref_subset_df[["xcentroid", "ycentroid"]].values

        # Find the transformation between the two sets of positions
        tform, _ = astroalign.find_transform(ref_xy, obs_xy, max_control_points=50)

        # Apply the transformation to the reference image
        ref_img_warped, ref_img_mask_warped = astroalign.apply_transform(tform,
                                                                         ref_img,
                                                                         trail_img,
                                                                         propagate_mask=True)

        # Warp and subtract background from the reference image, if needed
        if self._telescope != 'CTIO 4.0-m telescope':
            ref_bkg_warped, _ = astroalign.apply_transform(tform,
                                                           ref_img_bkg,
                                                           trail_img,
                                                           propagate_mask=True)
            ref_img_warped -= ref_bkg_warped
            del ref_bkg_warped

        # Create a new mask for the warped reference image
        ref_img_mask = np.zeros(ref_img_warped.shape)
        ref_img_mask = np.ma.make_mask(ref_img_mask, dtype=bool, shrink=False)
        ref_img_mask |= ref_img_mask_warped

        # Mask saturated pixel in the warped reference image
        ref_img_warped_mask_sat = (ref_img_warped > 0.95 * config['sat_lim'])
        ref_img_mask |= ref_img_warped_mask_sat

        ref_img_warped_negative_mask = (ref_img_warped < 0.)
        ref_img_mask |= ref_img_warped_negative_mask

        # Mask pixels near the border
        ref_img_mask = sext.make_border_mask(ref_img_mask, borderLen=2. * box_size)

        # Detect sources in the warped reference image
        warped_ref_src, warped_ref_src_mask = self.detect_sources_in_image(image=ref_img_warped,
                                                                           detect_background=0.,
                                                                           detect_error=None,
                                                                           nsigma=2.5, npixels=9,
                                                                           connectivity=8,
                                                                           mask=ref_img_mask)

        # Clean up
        bc.clean_up(ref_img, trail_img, ref_img_mask, trail_img_mask,
                    ref_img_bkg, ref_img_warped, ref_img_mask, obs_xy, ref_xy,
                    src_subset_df, ref_subset_df, tform, img_sources, ref_sources,
                    matches_df, matched_src_df)

        return warped_ref_src_mask

    def process_image(self, image, telescope, bkg_fname):
        """
        Process the image based on the telescope type and apply background subtraction.
        """
        if telescope == 'CTIO 4.0-m telescope':
            negative_mask = (image < 0.)
            return image, 0., self._config['sky_noise'], negative_mask
        else:
            background, _, background_rms, _ = self.load_background_data(bkg_fname)
            img_bkg_sub, negative_mask = self.subtract_background(image, background)
            return img_bkg_sub, background, background_rms, negative_mask

    def update_failed_processes(self, fail_msg):
        """

        Parameters
        ----------
        fail_msg

        Returns
        -------

        """
        tmp_fail = [self.current_location, self._telescope, self.current_object,
                    self.current_file_name, self.current_hdu_idx]
        tmp_fail.extend(fail_msg)

        self.fails.append(tmp_fail)

    def save_failed_processes(self, time_stamp):
        """

        Parameters
        ----------
        time_stamp

        Returns
        -------

        """
        # Save failed processes
        if self.fails:
            fail_path = Path(self._config['WORKING_DIR_PATH']).expanduser().resolve()
            fail_fname = fail_path / f'fails_analyseSatObs_{time_stamp}.log'
            self._log.info(f">> FAILED analyses are stored here: {fail_fname}")
            with open(fail_fname, "w", encoding="utf8") as file:
                file.write('\t'.join(['Location', 'Telescope',
                                      'SatID', 'File', 'HDU#', 'ErrorType', 'ErrorMsg', 'Note']) + '\n')
                [file.write('\t'.join(f) + '\n') for f in self.fails]

    def run_std_photometry(self, data, hidx, hdu_idx):
        """Perform aperture photometry on standard stars.

        Select standard stars and perform aperture photometry with bkg annulus sky estimation.
        Use the growth curve to determine the best aperture radius;
        re-center pixel coordinates before execution using centroid poly_func from photutils

        Parameters
        ----------
        data : dict
            Dictionary with all image data
        hidx : int

        hdu_idx : int

        """
        config = self._config
        obspar = self._obsparams
        n_ext = obspar['n_ext']

        file_name = data['fname']
        file_name_plot = file_name if n_ext == 0 else f'{file_name}_HDU{hdu_idx:02d}'

        img_mask = data['img_mask'][hidx]
        config['telescope_keyword'] = self._telescope
        imgarr = data['images'][hidx]
        hdr = data['hdr'][hidx]
        bkg_data = data['bkg_data'][hidx]
        bkg_file_name = bkg_data['bkg_fname']
        std_cat = data['cats_cleaned'][hidx]['ref_cat_cleaned']
        std_fkeys = data['std_fkeys'][hidx]
        mag_conv = data['mag_conv'][hidx]
        kernel_fwhm = data['kernel_fwhm'][hidx][0]
        exp_time = hdr[obspar['exptime']]
        gain = 1.
        rdnoise = 0.

        # Get wcs
        wcsprm = wcs.WCS(hdr)

        if not self._silent:
            self._log.info(f"> Perform photometry on standard stars")
            self._log.info("  > Select standard stars")

        # Select only good sources if possible
        if 'include_fwhm' in std_cat.columns:
            idx = std_cat['include_fwhm']
        else:
            idx = [True] * len(std_cat)
        std_cat = std_cat[idx]

        # Get reference positions
        src_pos = np.array(list(zip(std_cat['xcentroid'], std_cat['ycentroid'])))

        if not self._silent:
            self._log.info(f'  > Get optimum aperture from signal-to-noise ratio (S/N) of '
                           f'standard stars')

        mask = (imgarr > config['sat_lim'])
        if img_mask is not None:
            mask |= img_mask

        trail_img_processed = self.process_image(image=imgarr,
                                                 telescope=self._telescope,
                                                 bkg_fname=bkg_file_name)

        image_bkg_sub, image_bkg, image_bkg_rms, negative_mask = trail_img_processed

        mask |= negative_mask
        config['img_mask'] = mask

        result = phot.get_optimum_aper_rad(image=image_bkg_sub,
                                           std_cat=std_cat,
                                           fwhm=kernel_fwhm,
                                           exp_time=exp_time,
                                           config=config,
                                           aper_rad=config['APER_RAD'],
                                           r_in=config['RSKYIN'],
                                           r_out=config['RSKYOUT'],
                                           gain=gain,
                                           rdnoise=rdnoise,
                                           silent=self._silent)

        fluxes, rapers, max_snr_aprad, opt_aprad, qlf_aprad = result

        # Plot result of aperture photometry
        self.plot_aperture_photometry(fluxes=fluxes,
                                      rapers=rapers,
                                      opt_aprad=opt_aprad,
                                      file_base=file_name_plot)

        # Update aperture values
        config['APER_RAD_OPT'] = opt_aprad
        config['INF_APER_RAD_OPT'] = opt_aprad + 1.5
        config['RSKYIN_OPT'] = opt_aprad + 1.
        config['RSKYOUT_OPT'] = opt_aprad + 2.
        self._config = config

        if not self._silent:
            self._log.info(f'  > Measure photometry for standard stars. N={len(std_cat):d}')
        std_cat = phot.get_std_photometry(image=image_bkg_sub, std_cat=std_cat,
                                          src_pos=src_pos,
                                          fwhm=kernel_fwhm, config=config)

        # Estimate magnitude correction due to the finite aperture used
        if not self._silent:
            self._log.info(f'  > Get magnitude correction for finite aperture')
        mag_corr = self.get_magnitude_correction(df=std_cat,
                                                 file_base=file_name_plot)
        if not mag_corr:
            bc.clean_up(data, imgarr, image_bkg_sub, std_cat, result, fluxes)
            self.update_failed_processes(['MagCorrectError',
                                          'Error during interpolation.',
                                          'Check number and '
                                          'quality of standard stars and SNR'])
            return None, None, None, None, None, None, False

        if not self._silent:
            self._log.info('    ==> estimated magnitude correction: '
                           '{:.3f} +/- {:.3f} (mag)'.format(mag_corr[0], mag_corr[1]))

        bc.clean_up(imgarr, image_bkg_sub, result, src_pos, fluxes, rapers, data)

        if not self._silent:
            self._log.info("  > Save standard star photometry")
        std_photometry_cat_fname = f'{self._cat_path}/{file_name_plot}_std_star_photometry_cat'

        sext.save_catalog(cat=std_cat, wcsprm=wcsprm,
                          out_name=std_photometry_cat_fname,
                          kernel_fwhm=data['kernel_fwhm'][hidx],
                          mag_corr=mag_corr,
                          std_fkeys=std_fkeys,
                          mode='ref_photo')

        return std_cat, std_fkeys, opt_aprad, mag_conv, mag_corr, qlf_aprad, True

    def get_magnitude_correction(self, df: pd.DataFrame, file_base: str):
        """Estimate magnitude correction"""
        flux_ratios = df['flux_counts_inf_aper'] / df['flux_counts_aper']

        corrections = np.array([-2.5 * np.log10(i) if i > 0.0 else np.nan for i in flux_ratios])
        corrections = corrections[~np.isnan(corrections)]
        mask = np.array(~sigma_clip(corrections, sigma=3.,
                                    cenfunc=np.nanmedian).mask)
        corrections_cleaned = corrections[mask]

        aper_correction = np.nanmedian(corrections_cleaned, axis=None)
        aper_correction_err = np.nanstd(corrections_cleaned, axis=None)

        fig = plt.figure(figsize=self._figsize)
        gs = gridspec.GridSpec(1, 1)
        ax = fig.add_subplot(gs[0, 0])

        # Create a histogram of the data
        ax.hist(corrections_cleaned, bins='auto',
                density=True, facecolor='b',
                alpha=0.75,
                label='Aperture Correction')

        xmin, xmax = ax.get_xlim()
        x = np.linspace(xmin, xmax, 500)

        mu, sigma = norm.fit(corrections_cleaned)
        best_fit_line = norm.pdf(x, mu, sigma)

        ax.plot(x, best_fit_line, 'r-', label='PDF')

        ax.axvline(x=aper_correction, color='k')

        ax.legend(numpoints=2, loc='upper right', frameon=False)

        ax.set_xlabel("Correction (mag)")
        ax.set_ylabel("Probability Density")

        # Save the plot
        fname = f"plot.photometry.std_mag_corr_{file_base}"
        self.save_plot(fname=fname)

        if self.plot_images:
            plt.show()

        plt.close(fig=fig)

        return aper_correction, aper_correction_err

    def filter_catalog(self, data: dict):
        """

        Parameters
        ----------
        data

        Returns
        -------

        """
        if not self._silent:
            self._log.info("> Remove sources close to trail(s) from catalog")

        hdus = data['hdus']
        ref_cat_str = 'ref_cat_cleaned'
        cats_cleaned = []
        cats_cleaned_rem = []
        for i, hdu_idx in enumerate(hdus):
            if not self._silent:
                self._log.info(f"  Processing: HDU #{hdu_idx:02d} [{i + 1}/{len(hdus)}]")

            if not self.current_hdu_exec_state[i]:
                self._log.critical("  No matching standard stars found!!! Skipping further steps. ")

                self.update_failed_processes(['IntegrityError',
                                              'No reference star data',
                                              'Check reference catalog input data'])
                self.current_hdu_exec_state[i] = False
                continue

            imgarr = data['images'][i]
            ref_tbl_photo = data['src_tbl'][i]

            # Create a masked image for detected trail
            mask, mask_list = self.create_mask(image=imgarr,
                                               reg_data=data['trail_data'][i]['reg_info_lst'],
                                               fac=5.)

            if ref_tbl_photo is None or ref_tbl_photo.empty or not mask.any():
                del imgarr, ref_tbl_photo, mask_list, mask
                self.update_failed_processes(['IntegrityError',
                                              'Empty reference star data frame or empty mask',
                                              'Check input data, trail/sources detection'])
                self.current_hdu_exec_state[i] = False
                continue

            # Split the catalog in stars outside and inside the trail mask
            fwhm = data['kernel_fwhm'][i][0]
            ref_cat_cln, ref_cat_removed = sext.clean_catalog_trail(imgarr=imgarr, mask=mask,
                                                                    catalog=ref_tbl_photo,
                                                                    fwhm=fwhm)
            if ref_cat_cln is None:
                self.update_failed_processes(['SrcMatchError',
                                              'No standard stars left after removing stars close to trail',
                                              'To be solved'])
                self.current_hdu_exec_state[i] = False
                del imgarr, ref_tbl_photo, ref_cat_cln, ref_cat_removed, mask_list, mask
                continue

            del imgarr, mask_list, mask

            cats_cleaned.append({ref_cat_str: ref_cat_cln})
            cats_cleaned_rem.append({ref_cat_str: ref_cat_removed})

        data['cats_cleaned'] = cats_cleaned
        data['cats_cleaned_rem'] = cats_cleaned_rem

        del cats_cleaned, cats_cleaned_rem, hdus

        return data

    def get_img_dict(self, data: dict, idx: int, obsparam: dict):
        """Get the data for either the reference image or the trail image.

        Parameters
        ----------
        data : dict
            A Dictionary with basic info for all object files
        idx : int
            Index of image selection, either for the trail or reference image
        obsparam : dict

        Returns
        -------
        result : dict
            A Dictionary with all relevant data for image analysis
        """
        res_keys = ['images', 'hdus', 'has_trail', 'img_mask', 'hdr',
                    'fname', 'loc', 'cat', 'band', 'src_tbl',
                    'kernel_fwhm',
                    'trail_data', 'bkg_data', 'std_fkeys', 'mag_conv']

        config = obsparam
        n_ext = obsparam['n_ext']
        file_name = data['file_names'][idx]
        location = data['src_loc'][idx]
        catalog = data['catalog'][idx]
        band = data['band'][idx]
        has_trails = data['has_trail'][idx]
        hdus = data['hdus'][idx]

        status_list = []
        image_list = []
        image_mask_list = []
        hdu_list = []
        hdr_list = []
        has_trails_list = []
        bkg_data_list = []
        src_tbl_list = []
        fwhm_list = []
        trail_data_list = []
        std_filter_key_list = []
        mag_conv_list = []

        for h, hdu_idx in enumerate(hdus):

            if not has_trails[h]:
                if not self._silent:
                    self._log.info(f"> No trail found. Skipping HDU #{hdu_idx:02d} [{h + 1}/{len(hdus)}]")
                continue

            if not self._silent:
                self._log.info(f"> Extract source information: HDU #{hdu_idx:02d} [{h + 1}/{len(hdus)}]")

            imgarr = data['images'][idx][h]
            hdr = data['header'][idx][h]
            bkg_data = data['bkg_data'][idx][h]
            bkg_file_name = bkg_data['bkg_fname']
            trail_data = data['trail_data'][idx][h]
            img_mask = data['image_mask'][idx][h]

            # Get the WCS from the header
            wcsprm = wcs.WCS(hdr)

            if not self._silent:
                self._log.info("> Load object information")

            # Make sure that the RA and DEC are consistent and rounded to the same decimal
            ra = hdr[obsparam['ra']]
            dec = hdr[obsparam['dec']]
            if obsparam['radec_separator'] == 'XXX':
                ra = round(hdr[obsparam['ra']], bc.ROUND_DECIMAL)
                dec = round(hdr[obsparam['dec']], bc.ROUND_DECIMAL)
            hdr[obsparam['ra']] = ra
            hdr[obsparam['dec']] = dec

            self._obsTable.get_object_data(fname=file_name, kwargs=hdr,
                                           obsparams=obsparam, hdu_idx=hdu_idx)
            self._obj_info = self._obsTable.obj_info

            if self._telescope in ['CTIO 0.9 meter telescope']:
                img_fwhm = 2 * np.sqrt(2 * np.log(2)) * hdr['SEEING']
            else:
                img_fwhm = hdr['FWHM']

            estimate_bkg = False
            if self.force_extract:
                estimate_bkg = True

            # Check binning keyword
            if isinstance(obsparam['binning'][0], str) and isinstance(obsparam['binning'][1], str):
                bin_x = int(hdr[obsparam['binning'][0]])
                bin_y = int(hdr[obsparam['binning'][1]])
            else:
                bin_x = int(obsparam['binning'][0])
                bin_y = int(obsparam['binning'][1])

            bin_str = f'{bin_x}x{bin_y}'

            # Extract sources on detector
            params_to_add = dict(fov_radius=None,
                                 _filter_val=band,
                                 _trail_data=trail_data,
                                 src_cat_fname=None, ref_cat_fname=None,
                                 photo_ref_cat_fname=self._photo_ref_cat_fname,
                                 bin_str=bin_str,
                                 estimate_bkg=estimate_bkg, bkg_fname=bkg_file_name,
                                 force_extract=self.force_extract,
                                 force_download=self.force_download,
                                 image_mask=img_mask, img_fwhm=img_fwhm)

            # Add to configuration
            config.update(params_to_add)
            for key, value in self._config.items():
                config[key] = value

            sat_lim = config['saturation_limit']
            if isinstance(config['saturation_limit'], str) and config['saturation_limit'] in hdr:
                sat_lim = hdr[obsparam['saturation_limit']]
            config['sat_lim'] = sat_lim
            if self._telescope == 'CTIO 4.0-m telescope':
                config['sky_noise'] = hdr['SKYNOISE']

            # Get FWHM and minimum separation between sources
            fwhm_guess = float(self._obj_info['FWHM'].values[0]) if 'FWHM' in self._obj_info else None
            if fwhm_guess is not None:
                config['ISOLATE_SOURCES_INIT_SEP'] = int(np.ceil(config['ISOLATE_SOURCES_FWHM_SEP'] * fwhm_guess))

            file_name_cat = file_name if n_ext == 0 else f'{file_name}_HDU{hdu_idx:02d}'
            phot_result = sext.get_photometric_catalog(file_name_cat, self._cat_path,
                                                       imgarr, hdr, wcsprm, catalog,
                                                       silent=self._silent,
                                                       **config)
            src_tbl, std_fkeys, mag_conv, fwhm, state, fail_msg = phot_result

            image_list.append(imgarr)
            hdu_list.append(hdu_idx)
            hdr_list.append(hdr)
            bkg_data_list.append(bkg_data)
            has_trails_list.append(has_trails[h])
            trail_data_list.append(trail_data)
            image_mask_list.append(img_mask)

            src_tbl_list.append(src_tbl)
            std_filter_key_list.append(std_fkeys)
            mag_conv_list.append(mag_conv)
            fwhm_list.append(fwhm)
            status_list.append(state)
            self.current_hdu_exec_state.append(state)
            if not state:
                self.update_failed_processes(fail_msg)

            del imgarr, trail_data, hdr

        res_vals = [image_list, hdu_list, has_trails_list,
                    image_mask_list, hdr_list, file_name, location, catalog, band,
                    src_tbl_list, fwhm_list, trail_data_list,
                    bkg_data_list, std_filter_key_list, mag_conv_list]

        result = dict(zip(res_keys, res_vals))

        bc.clean_up(res_vals, image_list, hdu_list, has_trails_list,
                    image_mask_list, hdr_list, file_name, location, catalog, band,
                    src_tbl_list, fwhm_list, trail_data_list,
                    bkg_data_list, std_filter_key_list, mag_conv_list)

        return result, status_list

    def identify_trails(self, files: list):
        """Identify image with the satellite trail(s) and collect image information and trail parameters."""

        # hdu_idx = self._hdu_idx
        obspar = self._obsparams
        n_imgs = len(files)
        n_ext = obspar['n_ext']

        hdu_idx_list = list(range(1, n_ext + 1)) if n_ext > 0 else [0]

        # Prepare data dict
        data_dict_col_names = ['images', 'hdus', 'header', 'file_names', 'loc', 'src_loc',
                               'telescope', 'tel_key',
                               'has_trail', 'trail_data',
                               'catalog', 'band',
                               'src_cat', 'astro_ref_cat', 'photo_ref_cat',
                               'bkg_data', 'image_mask']
        data_dict = {c: [] for c in data_dict_col_names}
        data_dict['trail_img_idx'] = []
        data_dict['ref_img_idx'] = []

        ccd_mask_dict = {'CTIO 0.9 meter telescope': [[1405, 1791, 1850, 1957],
                                                      [1520, 1736, 1850, 1957],
                                                      [1022, 1830, 273, 275],
                                                      [1022, 1791, 1845, 1957]],
                         'CBNUO-JC': [[225, 227, 1336, 2048]]}

        # Check the extension table for entries
        has_trail_list = [[] for _ in range(n_imgs)]
        trail_data_list = [[] for _ in range(n_imgs)]
        images = [[] for _ in range(n_imgs)]
        hdus = [[] for _ in range(n_imgs)]
        headers = [[] for _ in range(n_imgs)]
        image_masks = [[] for _ in range(n_imgs)]
        bkg_data_dict_list = [[] for _ in range(n_imgs)]
        # apply_trail_detect = [False for _ in range(n_imgs)]
        has_ext_oi_data = [False for _ in range(n_imgs)]
        for f in range(n_imgs):
            file_loc = files[f]

            # Split file string into a path and filename
            file_name = Path(file_loc).stem
            file_name_clean = file_name.replace('_cal', '')

            # Load extensions of interest if available
            self._obsTable.get_ext_oi(file_name_clean)
            ext_oi_data = self._obsTable.ext_oi_data
            if list(ext_oi_data):
                hdu_idx_list = [int(i + 1) for i in ext_oi_data] if n_ext > 1 else ext_oi_data
                has_ext_oi_data[f] = True

        for f in range(n_imgs):
            file_loc = files[f]
            file_dir = os.path.dirname(file_loc)

            # Check if the required folders exist
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
            sat_id = self._sat_id

            plt_path_final = plt_path / sat_id
            if not plt_path_final.exists():
                plt_path_final.mkdir(exist_ok=True)
            self._plt_path = plt_path_final

            self._work_dir = Path(one_up)
            self._aux_path = aux_path
            self._cat_path = cat_path

            # Split file string into a path and filename
            file_name = Path(file_loc).stem
            file_name_clean = file_name.replace('_cal', '')

            if not self._silent:
                self._log.info(f"> Find satellite trail(s) in image: {file_name_clean}")

            for i, hdu_idx in enumerate(hdu_idx_list):

                apply_detect = True
                img_mask = None

                # Load fits file
                with fits.open(file_loc) as hdul:
                    hdul.verify('fix')
                    hdr = hdul[hdu_idx].header
                    imgarr = hdul[hdu_idx].data.astype('float32')
                    if n_ext > 0:
                        hdr = hdul[0].header
                        ext_hdr = hdul[hdu_idx].header

                        # Combine headers
                        [hdr.set(k, value=v) for (k, v) in ext_hdr.items()]

                    if 'mask' in hdul and obspar['apply_mask']:
                        img_mask = hdul['MASK'].data.astype(bool)

                # Set the background file name and check if the file already exists
                bkg_str = '_bkg' if n_ext == 0 else f'HDU{hdu_idx:02d}_bkg'
                bkg_fname_short = f'{file_name_clean}_{bkg_str}'
                if '_cal' in file_name:
                    bkg_fname_short = file_name.replace('_cal', bkg_str)

                estimate_bkg = True
                bkg_fname = os.path.join(aux_path, bkg_fname_short)
                if os.path.isfile(f'{bkg_fname}.fits') and not self.force_extract:
                    estimate_bkg = False

                # Add bkg data to dict
                bkg_dict = {'bkg_fname': (bkg_fname, bkg_fname_short)}
                bkg_data_dict_list[f].append(bkg_dict)

                detect_mask = None
                if self._telescope in ['CTIO 0.9 meter telescope', 'CBNUO-JC']:
                    detect_mask = np.zeros(imgarr.shape)
                    ccd_mask_list = ccd_mask_dict[self._telescope]
                    for yx in ccd_mask_list:
                        detect_mask[yx[0]:yx[1], yx[2]:yx[3]] = 1
                    detect_mask = np.where(detect_mask == 1, True, False)

                result_dict = {}
                has_trail = False
                automated_selection = False
                if hdu_idx == 31 and self._telescope == 'CTIO 4.0-m telescope':
                    apply_detect = False

                if np.any(has_ext_oi_data) and not has_ext_oi_data[f]:
                    apply_detect = False

                if apply_detect:
                    if detect_mask is not None:
                        if img_mask is None:
                            img_mask = detect_mask
                        else:
                            img_mask |= detect_mask

                    _config = self._config
                    if self._telescope == 'CTIO 4.0-m telescope':
                        img = imgarr
                    else:
                        box_size = self._config['BKG_BOX_SIZE']
                        win_size = self._config['BKG_MED_WIN_SIZE']
                        bkg, _, _, _ = sext.compute_2d_background(imgarr,
                                                                  estimate_bkg=estimate_bkg,
                                                                  bkg_fname=(bkg_fname,
                                                                             bkg_fname_short),
                                                                  box_size=box_size,
                                                                  win_size=win_size, mask=img_mask)

                        # Subtract background
                        img = imgarr - bkg
                        del bkg

                    # Add satellite id to config
                    _config['sat_id'] = sat_id

                    # The FUT data can already be processed, but we have to verify the header to ensure
                    # the required header keywords are present
                    if self._telescope == 'FUT':
                        self.verify_fut_header(img, hdr, obspar, _config, img_mask)

                    # Update the config
                    _config['pixscale'] = hdr['PIXSCALE']
                    if self._telescope in ['CTIO 0.9 meter telescope']:
                        _config['fwhm'] = 2 * np.sqrt(2 * np.log(2)) * hdr['SEEING']
                    else:
                        _config['fwhm'] = hdr['FWHM']

                    # Create a copy of the original image
                    img_orig = img.copy()

                    # Set negative/masked values to zero
                    img[img < 0.] = 0
                    if img_mask is not None:
                        img[img_mask] = 0

                    # Extract the region of interest for detection
                    roi_img = img.copy()
                    self._obsTable.get_roi(file_name_clean)
                    roi_info = self._obsTable.roi_data
                    _config['roi_offset'] = None
                    if not roi_info.empty:
                        yx = roi_info.to_numpy()[0][1:]
                        roi_img = img[yx[2]:yx[3], yx[0]:yx[1]]
                        _config['roi_offset'] = (yx[0], yx[2])

                    # # Mask the glint mask
                    # print(file_name_clean)
                    # self._obsTable.get_glint_mask(file_name_clean, hdu_idx)
                    # glint_mask_info = self._obsTable.glint_mask_data
                    # _config['glint_mask'] = None
                    # _config['has_glint_mask'] = False
                    # if not glint_mask_info.empty and glint_mask_info['HDU'].values[0] == hdu_idx:
                    #     print('hdu_idx', hdu_idx)
                    #     print('glint_mask_info', glint_mask_info)
                    #     x0 = glint_mask_info['xc'].values[0]
                    #     y0 = glint_mask_info['yc'].values[0]
                    #     length = glint_mask_info['length'].values[0]
                    #     width = glint_mask_info['width'].values[0]
                    #     ang = np.deg2rad(glint_mask_info['ang'].values[0])
                    #     print(x0, y0, length, width, ang)
                    #     glint_mask, _ = self.get_mask((x0, y0), length, width, ang, img.shape, scale=1.)
                    #     plt.figure()
                    #     plt.imshow(glint_mask*img, origin='lower', vmax=100)
                    #     plt.show()
                    #     _config['has_glint_mask'] = True

                    # This is a special case from the CTIO 90cm telescope
                    _config['use_sat_mask'] = False
                    if file_name_clean == 'n2_028':
                        _config['use_sat_mask'] = True

                    sat_lim = obspar['saturation_limit']
                    if isinstance(obspar['saturation_limit'], str) and obspar['saturation_limit'] in hdr:
                        sat_lim = hdr[obspar['saturation_limit']]
                    _config['sat_lim'] = sat_lim
                    if self._telescope == 'CTIO 4.0-m telescope':
                        _config['sky_noise'] = hdr['SKYNOISE']

                    # Get faint trail data from file
                    self._obsTable.get_faint_trail(file_name_clean, hdu_idx)
                    faint_trail_data = self._obsTable.faint_trail_data
                    if self._manual_trail_selection or not faint_trail_data.empty:
                        manual_trail_data_dict = {}
                        # If faint trail data is available, use it and skip the selection via GUI, else select via GUI
                        if not faint_trail_data.empty:
                            if not self._silent:
                                if hdu_idx == 0:
                                    self._log.info(f"  > Using faint trail data from file")
                                else:
                                    self._log.info(f"  > Using faint trail data from file for: "
                                                   f"HDU #{hdu_idx:02d} [{i + 1}/{len(hdu_idx_list)}] ")

                            # Use the faint trail data
                            src_pos = (faint_trail_data['xc'].values[0], faint_trail_data['yc'].values[0])

                            # Get the width, height and angle of the faint trail
                            width, height, theta = self.get_corrected_dimensions(
                                faint_trail_data['width'].values[0],
                                faint_trail_data['height'].values[0],
                                faint_trail_data['angle'].values[0]
                            )
                            manual_trail_data_dict = {
                                'coords': src_pos,
                                'coords_err': (0.5, 0.5),
                                'width': width,
                                'e_width': 0.5,
                                'height': height,
                                'e_height': 0.5,
                                'theta': theta,
                                'e_theta': 0.5,
                                'param_fit_dict': {},
                                'detection_imgs': {'roi_img': roi_img}
                            }

                            # Set the has_trail flag
                            has_trail = True
                        else:
                            if not self._silent:
                                if hdu_idx == 0:
                                    self._log.info(f"  > Manually select satellite trail via GUI")
                                else:
                                    self._log.info(f"  > Manually select satellite trail via GUI for: HDU #{hdu_idx:02d} "
                                                   f"[{i + 1}/{len(hdu_idx_list)}] ")

                            # Let user interactively select the aperture
                            aperture_params, has_trail = confirm_trail_and_select_aperture_gui(roi_img)
                            if aperture_params is not None:
                                src_pos = aperture_params['position']

                                # Extract width, height and angle from the aperture parameters
                                width, height, theta = self.get_corrected_dimensions(aperture_params['width'],
                                                                                     aperture_params['height'],
                                                                                     aperture_params['theta'])
                                manual_trail_data_dict = {
                                    'coords': src_pos,
                                    'coords_err': (0.5, 0.5),
                                    'width': width,
                                    'e_width': 0.5,
                                    'height': height,
                                    'e_height': 0.5,
                                    'orient_deg': theta,
                                    'e_orient_deg': 0.5,
                                    'param_fit_dict': {},
                                    'detection_imgs': {'roi_img': roi_img}
                                }
                        # Create a dictionary with the manual trail data
                        result_dict = {
                            'reg_info_lst': [manual_trail_data_dict],
                            'N_trails': 1,
                        } if has_trail else None

                    else:

                        if not self._silent:
                            if hdu_idx == 0:
                                self._log.info(f"  > Running automatic satellite trail detection")
                            else:
                                self._log.info(f"  > Running automatic satellite trail detection for: HDU #{hdu_idx:02d}"
                                               f" [{i + 1}/{len(hdu_idx_list)}] ")

                        # Extract satellite trails from the image and create mask
                        result_dict, has_trail = sats.detect_sat_trails(image=roi_img,
                                                                       config=_config,
                                                                       image_orig=img_orig,
                                                                       amount=_config['SHARPEN_AMOUNT'],
                                                                       mask=detect_mask,
                                                                       silent=self._silent)

                        automated_selection = True
                    # sys.exit()
                    del img, roi_img, _config
                else:
                    if not self._silent:
                        if hdu_idx == 0:
                            self._log.info(f"  >> Image was marked as reference (no trail) by the User. "
                                           f"Skipping satellite trail detection")
                        else:
                            self._log.info(f"  >> HDU was marked as reference (no trail) by the User. "
                                           f"Skipping satellite trail detection for: "
                                           f"HDU #{hdu_idx:02d} [{i + 1}/{len(hdu_idx_list)}] ")
                    pass

                kwargs = hdr.copy()
                kwargs['HDU_IDX'] = hdu_idx
                kwargs['DetPosID'] = None if n_ext == 0 else hdr[obspar['detector']]
                kwargs['HasTrail'] = 'T' if has_trail else 'F'
                kwargs['NTrail'] = result_dict['N_trails'] if has_trail else None
                kwargs['TrailDetMethod'] = 'Auto' if automated_selection else 'Manual'
                if has_trail:

                    # Plot results
                    file_name_plot = file_name_clean if n_ext == 0 else f'{file_name_clean}_HDU{hdu_idx:02d}'
                    if automated_selection:

                        # Plot the fit results
                        self.plot_fit_results(result_dict, file_name_plot)

                        # Cleanup images that are not needed any more
                        for k in range(len(result_dict['reg_info_lst'])):
                            result_dict['reg_info_lst'][k].pop('detection_imgs')
                            [result_dict['reg_info_lst'][k]['param_fit_dict'].pop(j)
                             for j in ['Hij_sub', 'Hij_rho', 'Hij_theta']]
                    else:
                        # Create a plot of the selected aperture
                        self.plot_manual_trail_selection(result_dict, file_name_plot)

                # Update the result table
                if obspar['radec_separator'] == 'XXX':
                    kwargs[obspar['ra']] = round(hdr[obspar['ra']],
                                                 bc.ROUND_DECIMAL)
                    kwargs[obspar['dec']] = round(hdr[obspar['dec']],
                                                  bc.ROUND_DECIMAL)

                self._obsTable.update_obs_table(file=file_name_clean,
                                                kwargs=kwargs, obsparams=obspar)
                self._obj_info = self._obsTable.obj_info

                has_trail_list[f].append(has_trail)
                trail_data_list[f].append(result_dict)

                del result_dict

                # Check the used filter band
                filter_val = hdr['FILTER'] if self._telescope != 'CTIO 4.0-m telescope' else hdr['BAND']
                if self.band is not None:
                    filter_val = self.band
                catalog = sext.select_reference_catalog(band=filter_val, source=self._catalog)
                if i == 0:
                    data_dict['catalog'].append(catalog)
                    data_dict['band'].append(filter_val)
                    data_dict['telescope'].append(self._telescope)
                    data_dict['file_names'].append(file_name_clean)
                    data_dict['loc'].append(file_dir)
                    data_dict['src_loc'].append(one_up)

                images[f].append(imgarr)
                hdus[f].append(hdu_idx)
                headers[f].append(hdr)
                image_masks[f].append(img_mask)

                del imgarr, bkg_dict

        # Group images
        data_dict['has_trail'] = has_trail_list
        data_dict['trail_data'] = trail_data_list
        data_dict['bkg_data'] = bkg_data_dict_list
        data_dict['images'] = images
        data_dict['hdus'] = hdus
        data_dict['header'] = headers
        data_dict['image_mask'] = image_masks

        has_trail_check = np.array(data_dict['has_trail'])
        test_for_any = np.any(has_trail_check, axis=1)

        if not np.any(test_for_any):
            self._log.critical("None of the input images have satellite trail(s) detected!!! "
                               "Skipping further steps.")
            self.update_failed_processes(['DetectError',
                                          'Satellite trail(s) detection has failed.',
                                          'Difficult to solve.'])
            return None, ['DetectError',
                          'Satellite trail(s) detection fail.',
                          'Difficult to solve.']
        else:
            data_dict['trail_img_idx'] = list(np.asarray(test_for_any).nonzero()[0])

            # Indices of all rows
            all_rows = np.arange(has_trail_check.shape[0])

            # Use broadcasting to find indices of other rows
            other_rows = all_rows[:, np.newaxis] != all_rows
            other_row_indices = np.where(other_rows)

            # Reshape to get the desired format
            other_rows_formatted = other_row_indices[1].reshape(has_trail_check.shape[0], -1)
            other_rows_formatted = [list(i) for i in other_rows_formatted]
            # print(other_rows_formatted)
            data_dict['ref_img_idx'] = other_rows_formatted

        return data_dict, None

    @staticmethod
    def get_corrected_dimensions(width, height, theta):
        """
        Get the corrected dimensions of a rectangle based on its width, height, and angle.

        Parameters
        ----------
        width : float
            Width of the rectangle.
        height : float
            Height of the rectangle.
        theta : float
            Angle in degrees.

        Returns
        -------
        tuple
            Corrected width and height.
        """
        if height > width:
            width, height = height, width
            theta = (theta + 90) % 360

        return width, height, theta

    @staticmethod
    def verify_fut_header(img, hdr, obsparams, config, img_mask=None):
        """
        Verify and update essential keywords in the header for FUT telescope.

        Parameters
        ----------

        img : numpy.ndarray, optional
            Image data to calculate FWHM if FWHM is missing
        hdr : astropy.io.fits.Header
            The header to verify and update
        obsparams : dict
            Telescope configuration containing default values
        config : dict
            A dictionary with configuration parameters
        img_mask : numpy.ndarray, optional
            A boolean mask to apply to the image

        Returns
        -------
        astropy.io.fits.Header
            Updated header with required keywords
        """
        # Extract the WCS from the header
        wcsprm = wcs.WCS(hdr).wcs

        # Check for BINNING
        hdr, bin_str = update_binning(hdr, obsparams, [])

        # Update configuration
        config_updated = obsparams.copy()
        params_to_add = dict(image_mask=img_mask,
                             bin_str=bin_str)

        # Add the parameters to the configuration
        config_updated.update(params_to_add)
        for key, value in config.items():
            config_updated[key] = value

        # Get detector saturation limit
        sat_lim = config_updated['saturation_limit']
        if isinstance(config_updated['saturation_limit'], str) and config_updated['saturation_limit'] in hdr:
            sat_lim = hdr[obsparams['saturation_limit']]
        config_updated['sat_lim'] = sat_lim
        config_updated['image_shape'] = img.shape

        # Check for PIXSCALE
        if 'PIXSCALE' not in hdr:
            # Calculate pixel scale from WCS
            wcs_pixscale = imtrans.get_wcs_pixelscale(wcsprm)
            pix_scale = np.mean(abs(wcs_pixscale)) * 3600. if wcs_pixscale is not np.nan else np.nan
            scale_x = abs(wcs_pixscale[0]) * 3600.
            scale_y = abs(wcs_pixscale[1]) * 3600.
            hdr['PIXSCALE'] = (pix_scale, 'Average pixel scale [arcsec/pixel]')
            hdr['SCALEX'] = (f'{scale_x:.5f}', 'Pixel scale in X [arcsec/pixel]')
            hdr['SCALEY'] = (f'{scale_y:.5f}', 'Pixel scale in Y [arcsec/pixel]')

        # Check for FWHM
        if 'FWHM' not in hdr:
            # Calculate FWHM using source detection
            kernel_fwhm, source_cat, state, fail_msg = sext.build_source_catalog(
                img, img_mask, None, None, **config_updated)

            fwhm = kernel_fwhm[0]
            e_fwhm = kernel_fwhm[1]
            hdr['FWHM'] = (f'{fwhm:.3f}', 'FWHM [pixel]')
            hdr['FWHMERR'] = (f'{e_fwhm:.3f}', 'FWHM error [pixel]')

        del img, img_mask, obsparams, config

        return hdr

    def get_data_from_tle(self, hdr, sat_name: tuple, tle_location: str,
                          date_obs, obs_times,
                          exptime, dt=None):
        """ Calculate angular velocity from TLE, satellite and observer location """

        image_wcs = wcs.WCS(hdr)
        column_names = ['Obs_Time', 'RA', 'DEC', 'x', 'y',
                        'UT Date', 'UT time', 'SatLon', 'SatLat', 'SatAlt', 'SatAz',
                        'SatElev', 'SatRA', 'SatDEC',
                        'SunRA', 'SunDEC', 'SunZenithAngle']

        sat_name_tle = sat_name[0]

        try:
            satellite = Orbital(sat_name_tle, tle_file=tle_location)
        except KeyError:
            sat_name_tle = sat_name_tle.replace('-', ' ')
            satellite = Orbital(sat_name_tle, tle_file=tle_location)

        tle_pos_path = Path(self._work_dir, 'tle_predictions')
        if not tle_pos_path.exists():
            tle_pos_path.mkdir(exist_ok=True)

        pos_times = [datetime.strptime(f"{date_obs}T{i}",
                                       frmt) for i in obs_times]

        fname = f"tle_predicted_positions_{sat_name[1].upper()}_{pos_times[0].strftime(frmt)[:-3]}.csv"
        fname = os.path.join(tle_pos_path, fname)

        # Set observer location
        obs_lat = self._obsparams["latitude"]
        obs_lon = self._obsparams["longitude"]
        obs_ele = self._obsparams["altitude"]  # in meters
        loc = EarthLocation(lat=obs_lat * u.deg,
                            lon=obs_lon * u.deg,
                            height=obs_ele * u.m)

        if dt is not None:
            delta_time = dt
            pos_times = self.calculate_pos_times(pos_times[0], exptime, dt)
        else:
            delta_time = exptime / 2
            pos_times.insert(0, pos_times[0] - timedelta(seconds=delta_time))

        total_pos_times = len(pos_times)
        try:
            # Use functools.partial to create a new function that has some of the parameters pre-filled
            func = partial(self.compute_for_single_time, obs_lon=obs_lon, obs_lat=obs_lat, obs_ele=obs_ele,
                           image_wcs=image_wcs, satellite=satellite, loc=loc, total=total_pos_times, use_lock=True)
            # Map the function over the range of rapers and get the results
            with ProcessPoolExecutor() as executor:
                results = list(executor.map(func, pos_times, range(1, total_pos_times + 1)))

            bc.print_progress_bar(total_pos_times, total_pos_times, length=80,
                                  color=bc.BCOLORS.OKGREEN, use_lock=True)

        finally:
            executor.shutdown(wait=True)
            sys.stdout.write('\n')
        del executor

        positions = np.array(results, dtype=object)

        velocity_list = [np.nan]
        for i in range(1, positions.shape[0]):
            ra = np.deg2rad(positions[i, 1])
            dec = np.deg2rad(positions[i, 2])
            prev_ra = np.deg2rad(positions[i - 1, 1])
            prev_dec = np.deg2rad(positions[i - 1, 2])

            d_theta = 2. * np.arcsin(np.sqrt(np.sin(0.5 * (dec - prev_dec)) ** 2.
                                             + np.cos(dec) * np.cos(prev_dec)
                                             * np.sin(0.5 * (ra - prev_ra)) ** 2.))

            d_theta *= 206264.806

            velocity_list += [d_theta / delta_time]

        pos_df = pd.DataFrame(data=positions,
                              columns=column_names)
        pos_df['pos_mask'] = pos_df.apply(lambda row: (0 < float(row['x'])) and
                                                      (hdr['NAXIS1'] > float(row['x'])) and
                                                      (0 < float(row['y'])) and
                                                      (hdr['NAXIS2'] > float(row['y'])), axis=1)
        pos_df['Obs_Time'] = pd.to_datetime(pos_df['Obs_Time'],
                                            format=frmt,
                                            utc=False)

        pos_df['angular_velocity'] = velocity_list

        angular_velocity = np.nanmean(velocity_list)
        e_angular_velocity = np.nanstd(velocity_list)

        # Save results
        pos_df[1:].to_csv(fname, index=False)

        return angular_velocity, e_angular_velocity, pos_df

    def plot_fit_results(self, data: dict, file_base: str):
        """Plot trail detection and parameter fit results"""
        n_trails_max = self._config['N_TRAILS_MAX']
        img_list = ['img_sharp', 'segm_map', 'img_labeled', 'trail_mask']
        fname_ext = ['sharpened', 'segments', 'segments_cleaned', 'HT_input']

        for i in range(n_trails_max):
            if not self._silent:
                self._log.info(f"  > Save trail detection plots: Trail #{i + 1:02d}")

            reg_data = data['reg_info_lst'][i]
            param_fit_dict = reg_data['param_fit_dict']
            trail_imgs_dict = reg_data['detection_imgs']

            # Plot detection process
            # fname_trail_det = f"plot.satellite.detection_combined_{file_base}_T{i + 1:02d}"
            # self.plot_trail_detection_combined(param_dict=trail_imgs_dict, fname=fname_trail_det)
            if not self._silent:
                self._log.info("    Segmentation plots")
            for k, v in enumerate(img_list):
                fname_trail_det = f"plot.satellite.detection_{fname_ext[k]}_{file_base}_T{i + 1:02d}"
                self.plot_trail_detection_single(param_dict=trail_imgs_dict, img_key=v,
                                                 fname=fname_trail_det)
            if not self._silent:
                self._log.info("    Hough Transform accumulator")
            # Plot hough space
            fname_hough_space = f"plot.satellite.hough.space_{file_base}_T{i + 1:02d}"
            self.plot_hough_space(param_dict=param_fit_dict, fname=fname_hough_space)
            if not self._silent:
                self._log.info("    Derived trail properties")

            # Plot quadratic fit result
            fname_hough_quad_fit = f"plot.satellite.hough.parameter_{file_base}_T{i + 1:02d}"
            self.plot_voting_variance(fit_results=param_fit_dict,
                                      reg_data=reg_data,
                                      fname=fname_hough_quad_fit)

            # Plot linear fit result
            fname_hough_lin_fit = f"plot.satellite.hough.centroidxy_{file_base}_T{i + 1:02d}"
            self.plot_centroid_fit(fit_results=param_fit_dict,
                                   reg_data=reg_data,
                                   fname=fname_hough_lin_fit)
            bc.clean_up(trail_imgs_dict, param_fit_dict, reg_data)

        del data

    def plot_manual_trail_selection(self, data: dict, file_base: str,
                                    img_norm: str = 'lin', cmap: str = None):
        """

        Parameters
        ----------
        data : dict
            A dictionary containing trail detection data
        file_base : str
            The filename to save the plot
        img_norm : str
            Normalization type for the image, can be 'lin', 'log' or 'sqrt'
        cmap : str
            Colormap to use for the image, default is 'gray_r'

        Returns
        -------

        """
        n_trails_max = self._config['N_TRAILS_MAX']
        for i in range(n_trails_max):
            if not self._silent:
                self._log.info(f"  > Save manual trail selection plot: Trail #{i + 1:02d}")

            reg_data = data['reg_info_lst'][i]
            fname = f"plot.satellite.detection_manual_{file_base}_T{i + 1:02d}"
            img = reg_data['detection_imgs']['roi_img']
            img[img < 0.] = 0.

            # Define normalization and the colormap
            vmin = np.percentile(img, 1.)
            vmax = np.percentile(img, 99.5)

            if img_norm == 'lin':
                nm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=LinearStretch())
            elif img_norm == 'log':
                nm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=LogStretch())
            else:
                nm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=SqrtStretch())

            if cmap is None:
                cmap = 'gray_r'

            # Prepare the data for plotting
            src_pos = reg_data['coords']
            width = reg_data['width']
            height = reg_data['height']
            theta = Angle(reg_data['orient_deg'], unit='deg')

            # Calculate zoom region (0.25 * min image size) centered on src_pos
            ny, nx = img.shape
            zoom_size = int(min(nx, ny) * 0.1)
            x0, y0 = src_pos
            x1 = int(max(0, x0 - zoom_size // 2))
            x2 = int(min(nx, x0 + zoom_size // 2))
            y1 = int(max(0, y0 - zoom_size // 2))
            y2 = int(min(ny, y0 + zoom_size // 2))

            zoom_img = img[y1:y2, x1:x2]

            aperture = RectangularAperture(src_pos, w=width, h=height, theta=theta)
            aperture_zoom = RectangularAperture(positions=(x0 - x1, y0 - y1), w=width, h=height, theta=theta)
            rect = mpl_patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                         fill=False, color='blue', lw=1.5)

            fig, ax1 = plt.subplots(figsize=self._figsize)

            ax1.imshow(img, origin='lower',
                  cmap=cmap,
                  interpolation='nearest',
                  norm=nm)

            ap_patches = aperture.plot(ax=ax1, **{'color': 'lime', 'lw': 1.5,
                                                  'label': 'Selected aperture'})

            ax1.add_patch(rect)

            # Create inset axes for zoom
            ax2 = ax1.inset_axes([0.7, 0.7, 0.3, 0.3])
            ax2.imshow(zoom_img,
                       origin='lower',
                       interpolation='nearest', norm=nm, cmap=cmap)

            # Add white edge to inset
            for spine in ax2.spines.values():
                spine.set_edgecolor('blue')
                spine.set_linewidth(1.5)

            aperture_zoom.plot(ax=ax2, **{'color': 'lime', 'lw': 1.5})

            ax2.set_xticks([])
            ax2.set_yticks([])

            # Add lines connecting the corners of the cyan box to the inset
            ax1.annotate('', xy=(x1, y2), xycoords='data', xytext=(0.7, 1.0),
                         textcoords='axes fraction', arrowprops=dict(arrowstyle='-', color='blue'))
            ax1.annotate('', xy=(x2, y1), xycoords='data', xytext=(1.0, 0.7),
                         textcoords='axes fraction', arrowprops=dict(arrowstyle='-', color='blue'))

            # Add legend to main plot
            handles = (ap_patches[0], )
            ax1.legend(loc='best', facecolor='#458989', labelcolor='white',
                       handles=handles, prop={'weight': 'normal', 'size': 11})

            # Format axis
            ax1.tick_params(
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

            minorlocatorx = AutoMinorLocator(5)
            minorlocatory = AutoMinorLocator(5)

            ax1.xaxis.set_minor_locator(minorlocatorx)
            ax1.yaxis.set_minor_locator(minorlocatory)

            ax1.set_ylabel(r'Pixel y direction')
            ax1.set_xlabel(r'Pixel x direction')

            ax1.set_xlim(xmin=0, xmax=img.shape[1])
            ax1.set_ylim(ymin=0, ymax=img.shape[0])

            plt.tight_layout()

            # Save the plot
            self.save_plot(fname=fname)

            if self.plot_images:
                plt.show()

            plt.close(fig=fig)

        del img

    def plot_trail_detection_single(self, param_dict: dict, img_key: str,
                                    fname: str,
                                    img_norm: str = 'lin', cmap: str = None):
        """Plot each trail detection result in a single figure"""

        img = param_dict[img_key] if img_key in ['trail_mask', 'img_sharp'] else param_dict[img_key][0]

        if cmap is None:
            cmap = 'Greys'

        if img_key == 'img_sharp':
            img[img < 0.] = 0.

            # Define normalization and the colormap
            vmin = np.percentile(img, 1.)
            vmax = np.percentile(img, 98)

            if img_norm == 'lin':
                nm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=LinearStretch())
            elif img_norm == 'log':
                nm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=LogStretch())
            else:
                nm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=SqrtStretch())
        else:
            nm = None
            cmap = param_dict[img_key][1] if img_key != 'trail_mask' else 'binary'

        fig = plt.figure(figsize=self._figsize)
        gs = gridspec.GridSpec(1, 1)
        ax = fig.add_subplot(gs[0, 0])

        ax.imshow(img, origin='lower',
                  cmap=cmap,
                  interpolation='nearest',
                  norm=nm,
                  aspect='auto')

        tick_color = "k"
        if img_key not in ['img_sharp', 'trail_mask']:
            tick_color = "w"

        # Format axis
        ax.tick_params(
            axis='both',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            direction='in',  # inward or outward ticks
            left=True,  # ticks along the left edge are off
            right=True,  # ticks along the right edge are off
            top=True,  # ticks along the top edge are off
            bottom=True,  # ticks along the bottom edge are off
            width=1.,
            color=tick_color  # tick color
        )

        minorlocatorx = AutoMinorLocator(5)
        minorlocatory = AutoMinorLocator(5)

        ax.xaxis.set_minor_locator(minorlocatorx)
        ax.yaxis.set_minor_locator(minorlocatory)

        ax.set_xlabel(r'Pixel x direction')
        ax.set_ylabel(r'Pixel y direction')

        plt.tight_layout()

        # Save the plot
        self.save_plot(fname=fname)

        if self.plot_images:
            plt.show()

        plt.close(fig=fig)

        del img, param_dict

    def plot_trail_detection_combined(self, param_dict: dict, fname: str,
                                      img_norm: str = 'lin', cmap: str = None):
        """Plot trail detection results combined in a single figure"""

        img = param_dict['img_sharp']
        segm = param_dict['segm_map']
        label_image = param_dict['img_labeled'][0]
        # label_image.relabel_consecutive(start_label=1)
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

        fig = plt.figure(figsize=self._figsize)
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
        ax2.imshow(segm[0], origin='lower', cmap=segm[1], interpolation='nearest',
                   aspect='auto')

        ax3.imshow(label_image, origin='lower', interpolation='nearest',
                   aspect='auto')

        ax4.imshow(trail_mask, origin='lower', interpolation='nearest',
                   aspect='auto')

        # Format axes
        ax_list = [ax1, ax2, ax3, ax4]
        for i in range(len(ax_list)):
            ax = ax_list[i]

            ax.set_xlim(0, img.shape[1])
            ax.set_ylim(0, img.shape[0])

            minorlocatorx = AutoMinorLocator(5)
            minorlocatory = AutoMinorLocator(5)

            ax.xaxis.set_minor_locator(minorlocatorx)
            ax.yaxis.set_minor_locator(minorlocatory)

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
        self.save_plot(fname=fname)

        if self.plot_images:
            plt.show()

        plt.close(fig=fig)

        bc.clean_up(img, segm, label_image, trail_mask)

    def plot_centroid_fit(self, fit_results: dict, fname: str, reg_data: dict):
        """"""

        result = fit_results['lin_fit']
        x = fit_results['lin_x']
        y = fit_results['lin_y']

        fig = plt.figure(figsize=self._figsize)

        gs = gridspec.GridSpec(1, 1)
        ax = fig.add_subplot(gs[0, 0])
        ax.plot(x, y, 'bo')
        ax.plot(x, result.best_fit, 'r-')

        # Format axis
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

        minorlocatorx = AutoMinorLocator(5)
        minorlocatory = AutoMinorLocator(5)

        ax.xaxis.set_minor_locator(minorlocatorx)
        ax.yaxis.set_minor_locator(minorlocatory)

        x_str = r'$\tan(\theta)$'
        if 'sin' in fit_results['lin_y_label']:
            x_str = r'$\tan(\theta)^{-1}$'

        ax.set_xlabel(x_str)
        ax.set_ylabel(fit_results['lin_y_label'])

        # Create a list with two empty handles (or more if needed)
        handles = [mpl_patches.Rectangle((0, 0), 1, 1, fc="white", ec="white",
                                         lw=0, alpha=0)] * 3

        # Create the corresponding number of labels
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

        # Create the legend, suppressing the blank space of the empty line symbol and the
        # padding between symbol and label by setting handlelenght and handletextpad
        legend = ax.legend(handles, labels, loc='best', fontsize='medium',
                           fancybox=False, framealpha=0.25,
                           handlelength=0, handletextpad=0)
        legend.set_draggable(state=True)

        # plt.tight_layout()

        # Save the plot
        self.save_plot(fname=fname)

        if self.plot_images:
            plt.show()

        plt.close(fig=fig)

        del fit_results, reg_data

    def plot_voting_variance(self, fit_results: dict, fname: str, reg_data: dict):
        """Plot results from quadratic fit.

        Plot of voting variance fit used to obtain length, width, and orientation of the trail.
        """

        result = fit_results['quad_fit']

        x = fit_results['quad_x']
        y = fit_results['quad_y']

        fig = plt.figure(figsize=self._figsize)
        gs = gridspec.GridSpec(1, 1)
        ax = fig.add_subplot(gs[0, 0])
        ax.plot(np.rad2deg(x), y, 'bo')
        ax.plot(np.rad2deg(x), result.best_fit, 'r-')

        # Format axis
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

        minorlocatorx = AutoMinorLocator(5)
        minorlocatory = AutoMinorLocator(5)

        ax.xaxis.set_minor_locator(minorlocatorx)
        ax.yaxis.set_minor_locator(minorlocatory)

        ax.set_xlabel(r'$\theta$ (deg)')
        ax.set_ylabel(r'$\sigma^{2}$ (pixel$^{2}$)')

        # Create a list with two empty handles (or more if needed)
        handles = [mpl_patches.Rectangle((0, 0), 1, 1, fc="white", ec="white",
                                         lw=0, alpha=0)] * 4

        # Create the corresponding number of labels
        labels = [r"$\sigma^{2} = \mathrm{a}_{2}\theta^{2} "
                  r"+ \mathrm{a}_{1}\theta "
                  r"+ \mathrm{a}_{0}$",
                  r"$\theta_{0} = $" + r"{0:.2f} (deg)".format(reg_data['orient_deg'] - 90.),
                  r"$L = $" + r"{0:.2f}$\pm${1:.2f} (pixel)".format(reg_data['width'],
                                                                    reg_data['e_width']),
                  r"$T = $" + r"{0:.2f}$\pm${1:.2f} (pixel)".format(reg_data['height'],
                                                                    reg_data['e_height'])]

        # Create the legend, suppressing the blank space of the empty line symbol and the
        # padding between symbol and label by setting handlelenght and handletextpad
        legend = ax.legend(handles, labels, loc='best', fontsize='medium',
                           fancybox=False, framealpha=0.25,
                           handlelength=0, handletextpad=0)
        legend.set_draggable(state=True)

        # Save the plot
        self.save_plot(fname=fname)

        if self.plot_images:
            plt.show()

        plt.close(fig=fig)

        del fit_results, reg_data

    def plot_hough_space(self, param_dict: dict, fname: str,
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

        fig = plt.figure(figsize=self._figsize)
        gs = gridspec.GridSpec(1, 1)
        ax = fig.add_subplot(gs[0, 0])

        ax.imshow(img, origin='lower',
                  extent=(cols[0], cols[-1], rows[0], rows[-1]),
                  cmap=cmap,
                  interpolation='none',
                  norm=nm,
                  aspect='auto')

        # Format axis
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

        minorlocatorx = AutoMinorLocator(5)
        minorlocatory = AutoMinorLocator(5)

        ax.xaxis.set_minor_locator(minorlocatorx)
        ax.yaxis.set_minor_locator(minorlocatory)

        ax.set_xlabel(r'$\theta$ (deg)')
        ax.set_ylabel(r'$\rho$ (pixel)')

        plt.tight_layout()

        # Save the plot
        self.save_plot(fname=fname)

        if self.plot_images:
            plt.show()

        plt.close(fig=fig)

        del img, rows, cols, param_dict

    def plot_photometry_snr(self, src_pos: list, fluxes: np.ndarray,
                            apers: list, optimum_aper: float,
                            file_base: str, mode: str = 'sat'):
        """Plot the Signal-to-Noise ratio results from curve-of-growth estimation"""

        flux_med = np.nanmedian(fluxes[:, :, 2], axis=1) / np.nanmax(np.nanmedian(fluxes[:, :, 2], axis=1))
        fig = plt.figure(figsize=self._figsize)

        gs = gridspec.GridSpec(nrows=1, ncols=2)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])

        # Plot fluxes
        for r in range(len(src_pos)):
            ax1.plot(apers, fluxes[:, r, 0] / fluxes[0, r, 0], c='k')

        # Add legend
        if mode != 'sat':
            black_line = mlines.Line2D(xdata=[], ydata=[], color='k', marker='',
                                       markersize=15, label='STD star flux')
            ax1.legend([black_line], ['STD star flux'], loc=1, numpoints=1)

        # Plot fluxes
        ax2.plot(apers, flux_med, 'r-', lw=2)

        for r in range(len(src_pos)):
            ax2.plot(apers, fluxes[:, r, 2] / np.nanmax(fluxes[:, r, 2]), 'k-', alpha=0.2)

        # indicate optimum aperture
        ax2.axvline(x=optimum_aper, c='k', ls='--')

        # Add legend
        black_line = mlines.Line2D(xdata=[], ydata=[], color='k', marker='',
                                   markersize=15, label='STD star S/N')
        best_line = mlines.Line2D(xdata=[], ydata=[], color='r', marker='',
                                  markersize=15, label='median S/N')
        if mode == 'sat':
            ax2.legend([best_line, black_line], ['S/N', 'Trail flux'],
                       loc=1, numpoints=1)
        else:
            ax2.legend([best_line, black_line], ['median S/N', 'STD star S/N'],
                       loc=1, numpoints=1)

        # Format axes
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

            minorlocatorx = AutoMinorLocator(5)
            minorlocatory = AutoMinorLocator(5)

            ax.xaxis.set_minor_locator(minorlocatorx)
            ax.yaxis.set_minor_locator(minorlocatory)

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
        self.save_plot(fname=fname)

        if self.plot_images:
            plt.show()

        plt.close(fig=fig)

        del src_pos, fluxes

    def plot_aperture_photometry(self, fluxes: np.ndarray, rapers: list,
                                 opt_aprad: float, file_base: str, mode='std'):
        """"""
        snr_arr = fluxes[:, :, 2]
        snr_med = np.nanmedian(snr_arr, axis=1) / np.nanmax(np.nanmedian(snr_arr, axis=1))
        snr_max = np.array([np.nanmax(snr_arr[:, r]) for r in range(fluxes.shape[1])])

        if mode == 'std':
            fname = f"plot.photometry.snr_std_{file_base}"
            snr_label = 'Mean COG'
            x_label = r'Aperture radius (1/FWHM)'
            vline_label = ('Optimum Radius\n'
                           + r'$\left(r_{\mathrm{max, S/N}}\times 1.25\right)$')

        else:
            fname = f"plot.photometry.snr_sat_{file_base}"
            rapers = [2. * r for r in rapers]
            opt_aprad = 2. * opt_aprad
            # xmax *= 2.
            snr_label = 'COG'
            x_label = r'Aperture height (1/FWHM)'
            vline_label = ('Optimum Height\n'
                           + r'$\left(h_{\mathrm{max, S/N}}\times 1.25\right)$')
        xmax = np.max(rapers)

        fig = plt.figure(figsize=self._figsize, constrained_layout=True, layout='compressed')

        widths = [50, 1]
        gs = gridspec.GridSpec(2, 2, figure=fig, width_ratios=widths)

        # Set up the normalization and the colormap
        normalize = mcolors.Normalize(vmin=np.min(snr_max), vmax=np.max(snr_max))
        colormap = mpl.colormaps['jet']

        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)

        for r in range(fluxes.shape[1]):
            color = colormap(normalize(snr_max[r]))
            ax1.plot(rapers, fluxes[:, r, 2] / np.nanmax(fluxes[:, r, 2]),
                     color=color, ls='--', alpha=0.25)
            ax2.plot(rapers, fluxes[:, r, 0] / np.nanmax(fluxes[:, r, 0]),
                     color=color, ls='--', alpha=0.25)

        ax1.plot(rapers, snr_med, 'r-', lw=2, label=snr_label)

        # indicate optimum aperture
        ax1.axvline(x=opt_aprad, c='b', ls='--')
        ax2.axvline(x=opt_aprad, c='b', ls='--',
                    label=vline_label)

        ax2.axhline(y=1., c='k', ls=':', label=r'100 %')
        ax2.axhline(y=0.95, c='k', ls='-.', label=r'95 %')

        ax1.set_xlim(xmin=0, xmax=xmax)
        ax2.set_ylim(ymin=0.8)

        ax3 = fig.add_subplot(gs[:, 1])

        # Set up the color-bar
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

        # Format axes
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

            minorlocatorx = AutoMinorLocator(4)
            minorlocatory = AutoMinorLocator(4)

            ax.xaxis.set_minor_locator(minorlocatorx)
            ax.yaxis.set_minor_locator(minorlocatory)

            if i == 0:
                ax.set_ylabel(r'SNR (normalized)')
            else:
                ax.set_ylabel(r'Flux (normalized)')

            ax.set_xlabel(x_label)

        plt.setp(ax1.get_xticklabels(), visible=False)
        ax1.set(xlabel=None)
        ax1.legend(loc='lower right')
        ax2.legend(loc='lower right')

        # fig.tight_layout(h_pad=0.4, w_pad=0.5, pad=1.)

        # Save the plot
        self.save_plot(fname=fname)

        if self.plot_images:
            plt.show()

        plt.close(fig=fig)

        del fluxes

    def plot_final_result(self, img_dict: dict, reg_data: dict,
                          obs_date: tuple, expt: float, std_pos: np.ndarray,
                          file_base: str, hidx: int, tle_pos: pd.DataFrame = None,
                          img_norm: str = 'lin', cmap: str = 'Greys'):
        """Plot final result of satellite photometry"""

        img = img_dict['images'][hidx]
        img_mask = img_dict['img_mask'][hidx]
        band = img_dict['band']

        hdr = img_dict['hdr'][hidx]

        wcsprm = wcs.WCS(wcs.WCS(hdr).to_header()).wcs
        obsparams = self._obsparams

        if self._telescope == 'CTIO 4.0-m telescope':
            image = img
        else:
            bkg_fname = img_dict['bkg_data'][hidx]['bkg_fname']
            if Path(f'{bkg_fname[0]}.fits').exists():
                bkg, _, _, _ = sext.load_background_file(bkg_fname, silent=True)
                image = img - bkg  # Subtract background
            else:
                image = img

        image[image < 0.] = 0.
        # plt.figure()
        # plt.imshow(image, origin='lower')
        if img_mask is not None:
            image[img_mask] = 0.
        # plt.figure()
        # plt.imshow(image, origin='lower')

        matrix = wcsprm.pc * wcsprm.cdelt[:, np.newaxis]

        # Scale
        scale_x = np.sqrt(matrix[0, 0] ** 2 + matrix[0, 1] ** 2)

        # Rotation angle
        theta_rad = -np.arctan2(matrix[1, 0], matrix[1, 1])

        scale_length = self._config['LINE_LENGTH']  # in arcmin
        scale_arrow = self._config['ARROW_LENGTH']  # in percent

        sat_pos = reg_data['coords']
        sat_ang = Angle(reg_data['orient_deg'], 'deg')
        trail_w = reg_data['width']
        trail_h = float(self._obj_info['OptAperHeight'])

        # adjust position offset
        px_offset = -0.
        adjusted_sat_pos = [(x + px_offset, y + px_offset) for x, y in [sat_pos]]
        adjusted_std_pos = [(x + px_offset, y + px_offset) for x, y in std_pos]

        sat_aper = RectangularAperture(adjusted_sat_pos, w=trail_w, h=trail_h, theta=sat_ang)
        std_apers = CircularAperture(adjusted_std_pos, r=12.)

        # Define normalization and the colormap
        vmin = np.percentile(image, 1.)
        vmax = np.percentile(image, 98.)
        if img_norm == 'lin':
            nm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=LinearStretch())
        elif img_norm == 'log':
            nm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=LogStretch())
        else:
            nm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=SqrtStretch())
        if cmap is None:
            cmap = 'cividis'

        fig = plt.figure(figsize=self._figsize)

        gs = gridspec.GridSpec(1, 1)
        ax = fig.add_subplot(gs[0, 0])

        # Plot the image
        ax.imshow(image, origin='lower',
                  cmap=cmap,
                  interpolation='bicubic',
                  norm=nm,
                  aspect='equal')

        # Add apertures
        std_apers.plot(axes=ax, **{'color': 'darkorange', 'lw': 1.25, 'alpha': 0.95})
        sat_aper.plot(axes=ax, **{'color': 'green', 'lw': 1.25, 'alpha': 0.75})
        if self._glint_mask_data is not None:
            glint_aper = self._glint_mask_data[1]
            glint_aper.plot(axes=ax, **{'color': 'red', 'lw': 1.5, 'alpha': 0.95})

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

        # Make sure that the RA and DEC are consistent and rounded to the same decimal
        ra = hdr[obsparams['ra']]
        dec = hdr[obsparams['dec']]
        unit = (u.hourangle, u.deg)
        if obsparams['radec_separator'] == 'XXX':
            unit = (u.deg, u.deg)
            ra = round(hdr[obsparams['ra']], bc.ROUND_DECIMAL)
            dec = round(hdr[obsparams['dec']], bc.ROUND_DECIMAL)

        c = SkyCoord(ra=ra, dec=dec, unit=unit,
                     obstime=hdr[obsparams['date_keyword']])

        xy = c.to_pixel(wcs.WCS(hdr))
        ax.scatter(xy[0], xy[1], **{'color': 'blue', 'marker': '+', 's': 100,
                                    'linewidths': 1.5, 'alpha': 0.75})

        # Add compass
        self.add_compass(ax=ax, image_shape=img.shape,
                         scale_arrow=scale_arrow, theta_rad=theta_rad,
                         color='k')

        # Add scale
        self.add_scale(ax=ax, image_shape=img.shape, scale_x=scale_x, length=scale_length, color='k')

        # Format axis
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

        minorlocatorx = AutoMinorLocator(5)
        minorlocatory = AutoMinorLocator(5)

        ax.xaxis.set_minor_locator(minorlocatorx)
        ax.yaxis.set_minor_locator(minorlocatory)

        ax.set_ylabel(r'Pixel y direction')
        ax.set_xlabel(r'Pixel x direction')

        ax.set_xlim(xmin=0, xmax=img.shape[1])
        ax.set_ylim(ymin=0, ymax=img.shape[0])

        # Create a list with two empty handles (or more if needed)
        handles = [mpl_patches.Rectangle((0, 0), 1, 1, fc="white", ec="white",
                                         lw=0, alpha=0)] * 2

        # Create the corresponding number of labels
        labels = [f"{self._sat_id} {self._telescope} / {band} band",
                  f"{obs_date[0]} {obs_date[1]} / {expt:.1f}s"]

        # Create the legend, suppressing the blank space of the empty line symbol and the
        # padding between symbol and label by setting handlelenght and handletextpad
        legend = ax.legend(handles, labels, loc=2, fontsize='small',
                           fancybox=False, framealpha=0.25,
                           handlelength=0, handletextpad=0)
        legend.set_draggable(state=False)
        legend.set_zorder(2.5)

        plt.tight_layout()

        # Save the plot
        fname = f"plot.satellite.final_{file_base}"
        self.save_plot(fname=fname)

        if self.plot_images:
            plt.show()

        plt.close(fig=fig)

        bc.clean_up(img, image, hdr, wcsprm, sat_aper, std_apers, img_dict, reg_data, std_pos)

    def plot_profile_along_axis(self, x, y, yerr=None,
                                fit_result_1comp=None, model_type_1comp='gaussian',
                                fit_result_2comp=None, model_type_2comp='gaussian',
                                axis='minor', fname=None, is_glint=False):
        """Plot profile along the specified axis of the cutout."""

        label_data = 'Median along major axis' if axis == 'minor' else 'Data along major axis'

        fig = plt.figure(figsize=self._figsize)
        gs = gridspec.GridSpec(1, 1)
        ax = fig.add_subplot(gs[0, 0])

        ax.errorbar(x=x, y=y, yerr=yerr, fmt='.', color='k', label=label_data)
        x_new = np.linspace(x.min(), x.max(), 1000)

        # Plot the one-component fit
        if fit_result_1comp is not None:
            y_eval = fit_result_1comp.eval(fit_result_1comp.params, x=x_new)
            label = f'Best fit single {model_type_1comp.capitalize()} model'
            ax.plot(x_new, y_eval, 'g-', lw=1.5, label=label)

        # Plot the two-component fit
        if fit_result_2comp is not None:
            y_eval = fit_result_2comp.eval(fit_result_2comp.params, x=x_new)
            label = f'Best fit 2 {model_type_2comp.capitalize()} model'
            ax.plot(x_new, y_eval, 'r-', lw=1.5, label=label)

            comps = fit_result_2comp.eval_components(x=x_new)
            for comp_name, comp_data in comps.items():
                style = '--' if 'const' not in comp_name else ':'
                label = f'{comp_name.capitalize()} component'
                ax.plot(x_new, comp_data, style, alpha=0.9, label=label)

        # Format axis
        ax.tick_params(axis='both', which='both', direction='in',
                       left=True, right=True, top=True, bottom=True,
                       width=1., color='k')

        minorlocatorx = AutoMinorLocator(5)
        minorlocatory = AutoMinorLocator(5)

        ax.xaxis.set_minor_locator(minorlocatorx)
        ax.yaxis.set_minor_locator(minorlocatory)

        label_x = 'Major' if axis == 'major' else 'Minor'
        ax.set_xlabel(f'{label_x} axis position (pixel)')
        ax.set_ylabel('Flux (counts)')
        ax.legend(loc=1, numpoints=1)

        plt.tight_layout()

        # Save the plot
        profile_type = 'glint' if is_glint else 'trail'
        fname_plot = f"plot.{profile_type}.profile.{axis}_axis_{fname}"
        self.save_plot(fname=fname_plot)

        if self.plot_images:
            plt.show()

        plt.close(fig=fig)

    def plot_glint_results(self, image, aper, bounds, fit_data, file_base):
        """Plot cutout of glint with projections and best fit """
        aper_pos, aper_width, aper_height, theta = aper
        bounds_x, bounds_y = bounds
        xc, yc = aper_pos

        xo, yo = fit_data[0]
        fit_major = fit_data[1]
        fit_minor = fit_data[2]

        # Number of profiles and length of each profile
        h, w = image.shape

        # Get x values along each axis
        x = np.arange(0, w, 1)
        y = np.arange(0, h, 1)

        x_fit_major = np.linspace(x.min(), x.max(), 1000)
        x_fit_minor = np.linspace(y.min(), y.max(), 1000)

        X, Y = np.meshgrid(x, y)

        scale = phot.order_shift(abs(image))

        image_plt = image / scale

        aperture = RectangularAperture(aper_pos,
                                       w=aper_width,
                                       h=aper_height,
                                       theta=theta)

        y_eval_major = fit_major.eval(fit_major.params, x=x_fit_major)
        y_eval_minor = fit_minor.eval(fit_minor.params, x=x_fit_minor)

        vmin, vmax = (ZScaleInterval(n_samples=1500, contrast=1)).get_limits(image_plt)

        plt.ioff()

        fig = plt.figure(figsize=self._figsize)

        ncols = 2
        nrows = 2

        heights = [1, 0.3]
        widths = [1, 0.3]

        gs = gridspec.GridSpec(nrows, ncols,
                               wspace=0.05,
                               hspace=0.05,
                               height_ratios=heights,
                               width_ratios=widths
                               )

        ax1 = fig.add_subplot(gs[0:1, 0:1])
        ax1_B = fig.add_subplot(gs[1, 0:1])
        ax1_R = fig.add_subplot(gs[0:1, 1])

        image_plt[image_plt.mask] = 0.

        ax1.imshow(image_plt,
                   interpolation='nearest',
                   origin='lower',
                   aspect='auto',
                   vmin=vmin,
                   vmax=vmax
                   )

        aperture.plot(ax=ax1, **{'color': 'red', 'ls': '-', 'lw': 1.25, 'alpha': 0.5})

        ax1.axvline(xo, ls='--', color='black', alpha=0.5)
        ax1.axhline(yo, ls='--', color='black', alpha=0.5)

        ax1_B.axvline(xo, ls='--', color='black', alpha=0.5)
        ax1_R.axhline(yo, ls='--', color='black', alpha=0.5)

        ax1_B.axvline(bounds_x[0], ls='-', color='red')
        ax1_B.axvline(bounds_x[1], ls='-', color='red')

        ax1_R.axhline(bounds_y[0], ls='-', color='red')
        ax1_R.axhline(bounds_y[1], ls='-', color='red')

        ax1_R.step(image_plt[:, int(xo)], Y[:, int(xo)], color='blue', where='mid')
        ax1_B.step(X[int(yo), :], image_plt[int(yo), :], color='blue', where='mid')

        ax1_B.plot(x_fit_major, y_eval_major / scale, 'g-', lw=1.25)
        ax1_R.plot(y_eval_minor / scale, x_fit_minor, 'g-', lw=1.25)

        # Format axes
        ax_list = [ax1, ax1_B, ax1_R]
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

            minorlocatorx = AutoMinorLocator(5)
            minorlocatory = AutoMinorLocator(5)

            ax.xaxis.set_minor_locator(minorlocatorx)
            ax.yaxis.set_minor_locator(minorlocatory)

        ax1.tick_params(labelbottom=False)

        ax1_R.yaxis.set_label_position("right")
        ax1_R.yaxis.tick_right()

        # ax1.axis('off')

        ax1_B.set_ylabel(r'Flux (counts $\times 10^{%d}$)' % np.log10(scale))
        ax1_R.set_xlabel(r'Flux (counts $\times 10^{%d}$)' % np.log10(scale))

        ax1_R.set_ylabel('Y pixel')
        ax1_B.set_xlabel('X pixel')

        ax1_R.set_ylim(ax1.get_ylim()[0] - 0.5, ax1.get_ylim()[1] + 0.5)
        ax1_B.set_xlim(ax1.get_xlim()[0] - 0.5, ax1.get_xlim()[1] + 0.5)

        ax1_R.tick_params(axis='x', rotation=-90)

        # Save the plot
        fname = f"plot.glint.final_{file_base}"
        self.save_plot(fname=fname)

        if self.plot_images:
            plt.show()

        plt.close(fig=fig)

    def plot_profile_3D(self, x, y, z, split_z_lim=None, split_cmap=False, fname=None, is_glint=False):
        """Plot 3D profile."""

        X, Y = np.meshgrid(x, y)

        Z = np.array(z)

        fig = plt.figure(figsize=self._figsize)

        gs = gridspec.GridSpec(1, 1)
        ax = fig.add_subplot(gs[0, 0], projection='3d')

        if split_cmap:
            criterion = 1.5 * split_z_lim
            Z1 = np.ma.masked_less(Z, criterion)  # larger values
            Z2 = np.ma.masked_greater_equal(Z, criterion)

            ax.plot_surface(X, Y, Z2, cmap=mpl.colormaps['coolwarm'],
                            antialiased=False, edgecolor=None, zorder=-1,
                            rstride=5, cstride=1,
                            # alpha=0.5
                            )

            ax.plot_surface(X, Y, Z1, cmap=mpl.colormaps['cividis'],
                            antialiased=False, edgecolor=None, zorder=2,
                            rstride=5, cstride=1,
                            # alpha=0.5
                            )

            ax.view_init(elev=45, azim=-50)
        else:

            ax.plot_surface(X, Y, Z, cmap=mpl.colormaps['cividis'],
                            antialiased=False,
                            rstride=5, cstride=1,
                            # alpha=0.5
                            )

        # Format axis
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

        minorlocatorx = AutoMinorLocator(5)
        minorlocatory = AutoMinorLocator(5)

        ax.xaxis.set_minor_locator(minorlocatorx)
        ax.yaxis.set_minor_locator(minorlocatory)

        # Set scientific notation for z-axis
        ax.zaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='z', scilimits=(0, 0))

        ax.set_xlabel('Major axis position (pixel)')
        ax.set_ylabel('Minor axis position (pixel)')
        ax.set_zlabel('Flux (counts)')

        ax.set_xlim(xmin=0, xmax=len(x))
        ax.set_ylim(ymin=0, ymax=len(y))
        # ax.legend(loc=1, numpoints=1)

        # plt.tight_layout()

        # Save the plot
        profile_type = 'glint' if is_glint else 'trail'
        fname_plot = f"plot.{profile_type}.profile.2D_{fname}"
        self.save_plot(fname=fname_plot)

        if self.plot_images:
            plt.show()

        plt.close(fig=fig)

    @staticmethod
    def plot_catalog_positions(source_catalog, reference_catalog, image=None):
        """Plot image with source and reference catalog positions marked"""

        ref_cat = reference_catalog

        fig, ax = plt.subplots(figsize=(10, 10))

        # Plot the positions of the source catalog
        ax.scatter(source_catalog['xcentroid'], source_catalog['ycentroid'], c='blue', marker='o',
                   label='Source Catalog', alpha=0.7)

        # Plot the positions of the reference catalog
        ax.scatter(ref_cat['xcentroid'], ref_cat['ycentroid'], c='red', marker='x',
                   label='Reference Catalog',
                   alpha=0.7)

        if image is not None:
            plt.imshow(image, vmin=0, vmax=2000)

        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.legend()
        plt.title('Source Catalog and Reference Catalog Positions')
        # plt.show()

    @staticmethod
    def load_background_data(bkg_fname):
        """
        Load background data for the given image index and HDU index.
        """
        return sext.load_background_file(bkg_fname, silent=True)

    @staticmethod
    def subtract_background(image, background):
        """
        Subtract the background from the image and handle negative values and masking.
        """
        image_bkg_sub = image - background

        negative_mask = (image_bkg_sub < 0.)

        return image_bkg_sub, negative_mask

    @staticmethod
    def process_warped_background(transform, ref_bkg, trail_image_bkg, ref_bkg_rms, trail_image_bkg_rms):
        """
        Apply the given transformation to the reference background and its RMS,
        returning the warped background and RMS for detection.
        """
        ref_bkg_warped, _ = astroalign.apply_transform(transform, ref_bkg, trail_image_bkg,
                                                       propagate_mask=True)
        ref_bkg_rms_warped, _ = astroalign.apply_transform(transform, ref_bkg_rms, trail_image_bkg_rms,
                                                           propagate_mask=True)

        return ref_bkg_warped, ref_bkg_rms_warped

    @staticmethod
    def detect_sources_in_image(image, detect_background, detect_error, mask,
                                nsigma=2.0, npixels=5, connectivity=8):
        """
        Detect sources in the image.
        """
        threshold = detect_threshold(image, nsigma=nsigma, background=detect_background,
                                     error=detect_error,
                                     mask=mask)
        segm = detect_sources(image, threshold=threshold,
                              npixels=npixels, connectivity=connectivity, mask=mask)

        # Create a catalog of sources
        catalog = SourceCatalog(image, segm, mask=mask)

        # Select columns
        sources = catalog.to_table(
            columns=['label', 'xcentroid', 'ycentroid',
                     'max_value', 'segment_flux',
                     'eccentricity', 'fwhm', 'gini']).to_pandas()

        # Rename the flux column
        sources.rename(columns={"segment_flux": "flux"}, inplace=True)
        sources['mag'] = -2.5 * np.log10(sources['flux'].values)

        # Sort by maximum value
        sources.sort_values(by='flux', ascending=False, inplace=True)
        mask = segm.make_source_mask(footprint=None)
        return sources, mask

    @staticmethod
    def compute_for_single_time(val, iteration, obs_lon, obs_lat, obs_ele, image_wcs,
                                satellite, loc,
                                total, use_lock=False):
        """Calculate satellite position for a single time point.

        Computes the position of a satellite at a specific time and converts it to
        image pixel coordinates using the provided WCS information.

        Parameters
        ----------
        val : datetime
            The UTC time for the satellite position calculation.
        iteration : int
            Index or counter for the current iteration, used for progress tracking.
        obs_lon : float
            Observer's longitude in degrees.
        obs_lat : float
            Observer's latitude in degrees.
        obs_ele : float
            Observer's elevation in meters.
        image_wcs : astropy.wcs.WCS
            World Coordinate System object for the image.
        satellite : pyorbital.orbital.Orbital
            Satellite object with orbital parameters.
        loc : astropy.coordinates.EarthLocation
            Observer's location on Earth.
        total : int
            Total number of positions to calculate, used for progress reporting.
        use_lock : bool, optional
            Whether to use thread locking for console output, default is False.

        Returns
        -------
        list
            A List containing calculated satellite parameters:
            [time, RA, DEC, x, y, altitude, longitude, latitude, azimuth, elevation]
        """

        sat_az, sat_elev = satellite.get_observer_look(utc_time=val,
                                                       lon=obs_lon,
                                                       lat=obs_lat,
                                                       alt=obs_ele / 1000.0)

        radec = SkyCoord(alt=sat_elev * u.deg, az=sat_az * u.deg,
                         obstime=val, frame='altaz', location=loc)
        pix_coords = radec.to_pixel(image_wcs)

        sun_coordinates = sun_ra_dec(val)
        sun_zenith = sun_zenith_angle(val, obs_lon, obs_lat)
        sun_ra_hms = Angle(sun_coordinates[0], u.rad).hour
        sun_dec_dms = Angle(sun_coordinates[1], u.rad).degree

        sat_lon, sat_lat, sat_alt = satellite.get_lonlatalt(val)
        sat_ra = Angle(radec.icrs.ra.value, u.degree).to_string(unit=u.hour)
        sat_dec = Angle(radec.icrs.dec.value, u.degree).to_string(unit=u.degree, sep=':')

        bc.print_progress_bar(iteration, total, length=80,
                              color=bc.BCOLORS.OKGREEN, use_lock=use_lock)

        return [val.strftime(frmt)[:-3],
                radec.icrs.ra.value, radec.icrs.dec.value,
                pix_coords[0], pix_coords[1], val.strftime("%Y-%m-%d"), val.strftime('%H:%M:%S.%f')[:-3],
                sat_lon, sat_lat, sat_alt, sat_az, sat_elev, sat_ra, sat_dec,
                sun_ra_hms, sun_dec_dms, sun_zenith]

    @staticmethod
    def calculate_pos_times(start_time, exptime, dt):
        """Calculate the time intervals for the satellite position calculation.

        Creates a set of evenly spaced time points for determining satellite positions
        during an observation exposure.

        Parameters
        ----------
        start_time : datetime
            The starting time of the observation (UTC).
        exptime : float
            Exposure time of the observation in seconds.
        dt : float or None
            Time step between points in seconds. If None, a default value will be used.

        Returns
        -------
        list
            A list of datetime objects representing the time points for satellite
            position calculations.
        """
        # Calculate total number of intervals
        num_intervals = int(exptime / dt) if dt else 1

        # Generate all pos_times, starting from the time step before the initial time
        pos_times = [start_time - timedelta(seconds=dt)]
        pos_times += [start_time + timedelta(seconds=i * dt) for i in range(num_intervals + 1)]

        return pos_times

    @staticmethod
    def add_compass(ax: plt.Axes, image_shape, scale_arrow, theta_rad: float, color: str = 'black'):
        """Add a compass indicator to the given matplotlib axes.

        Creates a directional compass (similar to DS9 style) on the image to indicate
        North and East directions based on the image's rotation angle.

        Parameters
        ----------
        ax : matplotlib.pyplot.Axes
            The matplotlib axes object where the compass will be added.
        image_shape : tuple
            The shape of the image (height, width) in pixels.
        scale_arrow : float
            Scale factor for the compass arrow length, relative to the image size.
        theta_rad : float
            Rotation angle of the image in radians, used to determine compass orientation.
        color : str, optional
            Color of the compass elements (arrows and labels), default is 'black'.
        """

        text_off = 0.1

        theta_deg = np.rad2deg(theta_rad)

        length_arrow = scale_arrow * min(image_shape)
        length_box = (scale_arrow + 0.05) * image_shape[1]

        theta_north_rad = theta_rad + np.pi / 2.
        theta_east_rad = theta_rad + np.pi

        dx = length_box * np.cos(theta_rad)
        dy = length_box * np.sin(theta_rad)

        north_dx = length_arrow * np.cos(theta_north_rad)
        north_dy = length_arrow * np.sin(theta_north_rad)

        east_dx = length_arrow * np.cos(theta_east_rad)
        east_dy = length_arrow * np.sin(theta_east_rad)

        sx = 0.975
        sy = 0.025

        if (45. < theta_deg <= 180.) or theta_deg < -135.:
            sy = 0.975

        x = image_shape[1]
        y = image_shape[0]

        x0 = sx * x
        y0 = sy * y

        x1 = x0
        y1 = y0
        if theta_deg <= 45.:
            if y0 - dy < 0:
                y1 = dy
        elif 45 < theta_deg <= 90:
            if dx > y - y0:
                y1 = y0 - dx
        elif 90 < theta_deg:
            x1 = x0 + dx

        if -135 <= theta_deg < 0:
            if x - x0 < -dy:
                x1 = x + dy
            if y0 < -dx:
                y1 = y0 - dx
        elif theta_deg < -135:
            x1 = x + dx
            if y - y0 < -dy:
                y1 = y0 + dy

        # Draw North arrow
        ax.annotate('', xy=(x1, y1), xycoords='data',
                    xytext=(x1 + north_dx, y1 + north_dy), textcoords='data',
                    arrowprops=dict(arrowstyle="<-", lw=1.5), zorder=2)

        # Label North arrow
        ax.text(x1 + north_dx + text_off * north_dx, y1 + north_dy + text_off * north_dy, 'N',
                verticalalignment='center',
                horizontalalignment='center', color=color, fontsize=10, zorder=2)

        # Draw East arrow
        ax.annotate('', xy=(x1, y1), xycoords='data',
                    xytext=(x1 + east_dx, y1 + east_dy), textcoords='data',
                    arrowprops=dict(arrowstyle="<-", color=color, lw=1.5), zorder=2)

        # Label East arrow
        ax.text(x1 + east_dx + text_off * east_dx, y1 + east_dy + text_off * east_dy, 'E', verticalalignment='center',
                horizontalalignment='center', color=color, fontsize=10, zorder=2)

        return ax

    def save_plot(self, fname: str):
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
    def get_mask(src_pos: tuple,
                 width: float,
                 height: float,
                 theta: float,
                 img_size: tuple,
                 scale: float = 3.):
        """Extract single rectangular mask from an image"""

        aperture = RectangularAperture(src_pos,
                                       w=width,
                                       h=scale * height,
                                       theta=theta)

        rect_mask = aperture.to_mask(method='center')
        mask_img = rect_mask.to_image(img_size)
        mask_img = np.where(mask_img == 1., True, False)
        mask = mask_img

        return mask, aperture

    @staticmethod
    def create_mask(image: np.ndarray, reg_data: dict, fac: float = 1.):
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

            rect_mask = aperture.to_mask(method='center')
            mask_img = rect_mask.to_image(image.shape)
            mask_img = np.where(mask_img == 1., True, False)
            mask |= mask_img
            mask_list.append(mask_img)

            del mask_img, rect_mask, aperture, positions

        del image, reg_data

        return mask, mask_list

    @staticmethod
    def add_scale(ax: plt.Axes, image_shape: tuple, scale_x: float, length: float, color: str = 'black'):
        """Make a Ds9 like scale for image"""

        xlemark = length / (scale_x * 60.)  # in pixel

        xstmark = image_shape[1] * 0.025
        xenmark = xstmark + xlemark
        ystmark = image_shape[0] * 0.025

        # Add scale
        ax.plot([xstmark, xenmark], [ystmark, ystmark], color='k', lw=1.25)
        ax.annotate(text=r"{0:.0f}'".format(length),
                    xy=(((xstmark + xenmark) / 2.), ystmark + 0.25 * ystmark), xycoords='data',
                    textcoords='data', ha='center', c=color)

        return ax


def plot_rectangular_apertures(image, src_pos, src_mask, width, height, w_in, w_out, h_in, h_out, theta,
                               img_norm: str = 'lin', cmap: str = 'gray_r'):
    """Plot rectangular apertures for photometry with zoom inset.

    Parameters are the same as before
    """


    # Create apertures
    annulus = RectangularAnnulus(src_pos, w_in=w_in, w_out=w_out,
                                 h_in=h_in, h_out=h_out, theta=theta)
    aperture = RectangularAperture(src_pos, w=width, h=height, theta=theta)

    # Create masked image
    plt_img = image.copy()
    plt_img[plt_img < 0] = 0.
    if src_mask is not None:
        plt_img[src_mask] = np.nan

    # Define normalization and the colormap
    vmin = np.nanpercentile(plt_img, 1.)
    vmax = np.nanpercentile(plt_img, 99.5)
    if img_norm == 'lin':
        nm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=LinearStretch())
    elif img_norm == 'log':
        nm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=LogStretch())
    else:
        nm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=SqrtStretch())
    if cmap is None:
        cmap = 'cividis'

    # Create figure and main plot
    fig, ax1 = plt.subplots(figsize=(8, 8))
    ax1.imshow(plt_img, interpolation='nearest', norm=nm, cmap=cmap)
    ap_patches = aperture.plot(ax=ax1, **{'color': 'lime', 'lw': 1,
                                          'label': 'Photometry aperture'})
    ann_patches = annulus.plot(ax=ax1, **{'color': 'red', 'lw': 1,
                                          'label': 'Background aperture'})

    # Calculate zoom region (0.25 * min image size) centered on src_pos
    ny, nx = image.shape
    zoom_size = int(min(nx, ny) * 0.1)
    x0, y0 = src_pos
    x1 = int(max(0, x0 - zoom_size // 2))
    x2 = int(min(nx, x0 + zoom_size // 2))
    y1 = int(max(0, y0 - zoom_size // 2))
    y2 = int(min(ny, y0 + zoom_size // 2))

    # Show zoom region in main plot
    rect = mpl_patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 fill=False, color='blue', lw=1)
    ax1.add_patch(rect)

    # Create inset axes for zoom
    ax2 = ax1.inset_axes([0.7, 0.7, 0.3, 0.3])
    ax2.imshow(plt_img[y1:y2, x1:x2], interpolation='nearest', norm=nm, cmap=cmap)

    # Add white edge to inset
    for spine in ax2.spines.values():
        spine.set_edgecolor('blue')
        spine.set_linewidth(1.5)

    # Plot apertures in zoom panel with adjusted positions
    aperture_zoom = RectangularAperture(
        (x0 - x1, y0 - y1), w=width, h=height, theta=theta)
    annulus_zoom = RectangularAnnulus(
        (x0 - x1, y0 - y1), w_in=w_in, w_out=w_out,
        h_in=h_in, h_out=h_out, theta=theta)

    aperture_zoom.plot(ax=ax2, **{'color': 'lime', 'lw': 1.5})
    annulus_zoom.plot(ax=ax2, **{'color': 'red', 'lw': 1.5})
    ax2.set_xticks([])
    ax2.set_yticks([])

    # Add lines connecting the corners of the cyan box to the inset
    ax1.annotate('', xy=(x1, y2), xycoords='data', xytext=(0.7, 1.0),
                 textcoords='axes fraction', arrowprops=dict(arrowstyle='-', color='blue'))
    ax1.annotate('', xy=(x2, y1), xycoords='data', xytext=(1.0, 0.7),
                 textcoords='axes fraction', arrowprops=dict(arrowstyle='-', color='blue'))

    # Add legend to main plot
    handles = (ap_patches[0], ann_patches[0])
    ax1.legend(loc='best', facecolor='#458989', labelcolor='white',
               handles=handles, prop={'weight': 'normal', 'size': 11})

    # Format axis
    ax1.tick_params(
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

    minorlocatorx = AutoMinorLocator(5)
    minorlocatory = AutoMinorLocator(5)

    ax1.xaxis.set_minor_locator(minorlocatorx)
    ax1.yaxis.set_minor_locator(minorlocatory)

    ax1.set_ylabel(r'Pixel y direction')
    ax1.set_xlabel(r'Pixel x direction')

    ax1.set_xlim(xmin=0, xmax=plt_img.shape[1])
    ax1.set_ylim(ymin=0, ymax=plt_img.shape[0])

    # plt.tight_layout()
    return fig

# -----------------------------------------------------------------------------


def main():
    """ Main procedure """
    pargs = ParseArguments(prog_typ='satPhotometry')
    args = pargs.args_parsed
    main.__doc__ = pargs.args_doc

    # version check
    bc.check_version(_log)

    AnalyseSatObs(input_path=args.input, args=args,
                  silent=args.silent, verbose=args.verbose)


# -----------------------------------------------------------------------------


# Standard boilerplate to set 'main' as starting function
if __name__ == '__main__':
    main()
# -----------------------------------------------------------------------------
