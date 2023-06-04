#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Name:         calibrateSatObs.py
# Purpose:      Perform astrometric calibration on reduced fits images
#               to determine pixel scale and detector rotation angle.
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
# 27.04.2022
# - transported to project folder, updates and reformatting
#
# -----------------------------------------------------------------------------

""" Modules """

# STDLIB
import os
import gc
import sys
import logging
import warnings
import time
from datetime import (
    timedelta, datetime, timezone)
import collections
import configparser
from pathlib import Path
import argparse

# THIRD PARTY
import numpy as np
import pandas as pd

# astropy
from astropy.io import fits
from astropy import units as u
from astropy.utils.exceptions import AstropyUserWarning
from astropy.wcs import WCS
from astropy.coordinates import (FK5, ICRS, SkyCoord)
from astropy.time import Time
from astropy.visualization import (LinearStretch, LogStretch, SqrtStretch)
from astropy.visualization.mpl_normalize import ImageNormalize

# photutils
from photutils.aperture import CircularAperture

# scipy
from scipy.spatial import KDTree

# plotting; optional
try:
    import matplotlib
except ImportError:
    plt = None
    mpl = None
    warnings.warn('matplotlib not found, plotting is disabled', AstropyUserWarning)
else:
    import matplotlib
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec  # GRIDSPEC !
    from matplotlib.ticker import AutoMinorLocator
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # matplotlib parameter
    mpl.use('Qt5Agg')
    mpl.rc("lines", linewidth=1.2)
    mpl.rc('figure', dpi=150, facecolor='w', edgecolor='k')
    mpl.rc('text.latex', preamble=r'\usepackage{sfmath}')
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    # mpl.rcParams['font.family'] = 'Arial'

# Project modules
from leosatpy.utils.arguments import ParseArguments
from leosatpy.utils.dataset import DataSet
from leosatpy.utils.tables import ObsTables
import leosatpy.utils.sources as sext
import leosatpy.utils.transformations as imtrans
from leosatpy.utils.version import __version__

import leosatpy.utils.base_conf as _base_conf

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

__taskname__ = 'calibrateSatObs'

# -----------------------------------------------------------------------------

""" Parameter used in the script """
# -----------------------------------------------------------------------------
# Logging and console output
logging.root.handlers = []
_log = logging.getLogger()
_log.setLevel(_base_conf.LOG_LEVEL)
stream = logging.StreamHandler()
stream.setFormatter(_base_conf.FORMATTER)
_log.addHandler(stream)
_log_level = _log.level


# -----------------------------------------------------------------------------
# changelog
# version 0.1.0 alpha version


class CalibrateObsWCS(object):
    """Class to calibrate the image world coordinate system."""

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

        if ignore_warnings:
            _base_conf.load_warnings()

        if plt is None or silent or not plot_images:
            plt.ioff()

        # set variables
        self._dataset_object = None
        self._root_dir = _base_conf.ROOT_DIR
        self._input_path = input_path
        self._log = log
        self._converged = bool()
        self._catalog = args.catalog
        self._hdu_idx = args.hdu_idx
        self._dic_rms = {}
        self._log_level = log_level
        self._silent = silent
        self._verbose = verbose
        self._plot_images = plot_images
        self._force_extract = args.force_detection
        self._force_download = args.force_download
        self._instrument = None
        self._telescope = None
        self._obsparams = None
        self._radius = args.radius
        self._rot_scale = args.rotation_scale
        self._xy_transformation = args.xy_transformation
        self._fine_transformation = args.fine_transformation
        self._high_res = args.high_resolution
        self._ignore_header_rot = args.ignore_header_rot
        self._vignette = args.vignette
        self._vignette_rectangular = args.vignette_rectangular
        self._cutouts = args.cutout
        self._src_cat_fname = args.source_cat_fname
        self._ref_cat_fname = args.source_ref_fname
        self._wcs_check = None, None
        self._cat_path = None
        self._config = collections.OrderedDict()
        self._obsTable = None

        # run calibration
        self._run_calibration_all(silent=silent, verbose=verbose)

    def _run_calibration_all(self, silent=False, verbose=False):
        """Run full calibration routine on the input path.

        Find reduced science files and run calibration for a given set of data.
        """

        StartTime = time.perf_counter()
        self._log.info('====> Astrometric calibration init <====')

        if not silent:
            self._log.info("> Search input and prepare data")
        if verbose:
            self._log.debug("  > Check input argument(s)")

        # prepare dataset from input argument
        ds = DataSet(input_args=self._input_path,
                     prog_typ='calibWCS',
                     log=self._log, log_level=self._log_level)

        # load configuration
        ds.load_config()
        self._config = ds.config

        self._obsTable = ObsTables(config=self._config)
        self._obsTable.load_obs_table()

        inst_list = ds.instruments_list
        inst_data = ds.instruments

        time_stamp = self.get_time_stamp()
        fail_fname = Path(self._config['RESULT_TABLE_PATH']) / f'fails_calibrateSatObs_{time_stamp}.log'

        N_inst = len(inst_list)
        for i in range(N_inst):
            inst = inst_list[i]
            self._instrument = inst
            self._telescope = inst_data[inst]['telescope']
            self._obsparams = inst_data[inst]['obsparams']
            ds.get_valid_sci_observations(inst, prog_typ="calibWCS")
            obsfile_list = ds.valid_sci_obs
            self._dataset_object = obsfile_list

            # loop over groups and run reduction for each group
            for src_path, files in obsfile_list:
                if not silent:
                    self._log.info("====> Astrometric calibration run <====")
                    self._log.info(f"> Calibrate WCS of {len(files)} datasets from instrument {inst} "
                                   f"at the {self._telescope} "
                                   "telescope in folder:")
                    self._log.info(f"  {src_path}")
                multiple = False
                if len(list(files)) > 1:
                    multiple = True
                # not_converged = []
                converged_counter = 0
                pass_str = _base_conf.BCOLORS.PASS + "SUCCESSFUL" + _base_conf.BCOLORS.ENDC
                fail_str = _base_conf.BCOLORS.FAIL + "FAILED" + _base_conf.BCOLORS.ENDC
                for row_index, file_df in files.iterrows():

                    self._run_single_calibration(src_path, file_df, catalog=self._catalog,
                                                 hdu_idx=self._hdu_idx)

                    result = self._converged
                    if result:
                        self._log.info(f">> Astrometric calibration was {pass_str}")
                        converged_counter += 1
                    else:
                        self._log.info(f">> Astrometric calibration has {fail_str}")
                        # not_converged.append(file_df['input'])
                        with open(fail_fname, "a", encoding="utf8") as file:
                            file.write('{}\t{}\n'.format(self._telescope, file_df["input"]))

                if multiple:
                    self._log.info(">> Final report:")
                    N = len(files)
                    not_converged_counter = N - converged_counter
                    self._log.info(f"   Processed: {N} file(s), "
                                   f"{pass_str}: {converged_counter}, "
                                   f"{fail_str}: {not_converged_counter}")
                    if not_converged_counter > 0:
                        self._log.info(f"   FAILED calibrations are stored here: {fail_fname}")
                        # self._log.info("   \n".join(not_converged))

        EndTime = time.perf_counter()
        dt = EndTime - StartTime
        td = timedelta(seconds=dt)

        if not silent:
            self._log.info(f"Program execution time in hh:mm:ss: {td}")
        self._log.info('====>  Astrometric calibration finished <====')

    @staticmethod
    def get_time_stamp() -> str:
        """
        Returns time stamp for now: "2021-10-09 16:18:16"
        """

        now = datetime.now(tz=timezone.utc)
        time_stamp = f"{now:%Y-%m-%d_%H_%M_%S}"

        return time_stamp

    def _load_config(self):
        """ Load base configuration file """

        # configuration file name
        configfile = f"{self._root_dir}/leosatpy_config.ini"
        config = configparser.ConfigParser()
        config.optionxform = lambda option: option

        self._log.info('> Read configuration')
        config.read(configfile)

        for group in ['Calibration', 'Detection']:
            items = dict(config.items(group))
            self._config.update(items)
            for key, value in items.items():
                try:
                    val = eval(value)
                except NameError:
                    val = value
                self._config[key] = val

    def _run_single_calibration(self, file_src_path, file_df, catalog="GAIADR3", hdu_idx=0):
        """Run astrometric calibration on a given dataset.

        Create the required folder and run full calibration procedure.

        """

        # check catalog input
        if catalog.upper() not in _base_conf.SUPPORTED_CATALOGS:
            self._log.warning(f"Given catalog '{catalog}' NOT SUPPORTED. "
                              "Defaulting to GAIAdr3")
            catalog = "GAIAdr3"

        report = {}
        obsparams = self._obsparams
        abs_file_path = file_df['input']
        file_name = file_df['file_name']
        fbase = file_name.replace('_red', '')

        # create folder
        if self._verbose:
            self._log.debug("  > Create folder")
        cal_path = Path(file_src_path, 'calibrated')
        if not cal_path.exists():
            cal_path.mkdir(exist_ok=True)
        cat_path = Path(file_src_path, 'catalogs')
        if not cat_path.exists():
            cat_path.mkdir(exist_ok=True)
        plt_path = Path(file_src_path, 'figures')
        if not plt_path.exists():
            plt_path.mkdir(exist_ok=True)

        aux_path = Path(file_src_path, 'auxiliary')
        if not aux_path.exists():
            aux_path.mkdir(exist_ok=True)

        self._log.info(f"> Run astrometric calibration for {file_name} ")

        # set background file name and create folder
        bkg_fname_short = file_name.replace('_red', '_bkg')
        bkg_fname = os.path.join(aux_path, bkg_fname_short)
        estimate_bkg = True
        if os.path.isfile(f'{bkg_fname}.fits'):
            estimate_bkg = False

        # load fits file
        img_mask = None
        with fits.open(abs_file_path) as hdul:
            hdul.verify('fix')
            hdr = hdul[hdu_idx].header
            imgarr = hdul[hdu_idx].data.astype('float32')

            if 'mask' in hdul and obsparams['apply_mask']:
                img_mask = hdul['MASK'].data.astype(bool)

        sat_id, _ = self._obsTable.get_satellite_id(hdr[obsparams['object']])
        plt_path_final = plt_path / sat_id
        if not plt_path_final.exists():
            plt_path_final.mkdir(exist_ok=True)

        # get wcs and an original copy
        wcsprm = WCS(hdr).wcs

        # run checks on wcs and get an initial guess
        self._check_image_wcs(wcsprm=wcsprm, hdr=hdr,
                              obsparams=obsparams,
                              radius=self._radius,
                              ignore_header_rot=self._ignore_header_rot)
        init_wcsprm = self._wcsprm

        # update configuration
        config = obsparams.copy()
        params_to_add = dict(_fov_radius=self._fov_radius,
                             _vignette=self._vignette,
                             _vignette_rectangular=self._vignette_rectangular,
                             _cutouts=self._cutouts,
                             _src_cat_fname=self._src_cat_fname,
                             _ref_cat_fname=self._ref_cat_fname,
                             _photo_ref_cat_fname=None,
                             image_mask=img_mask,
                             estimate_bkg=estimate_bkg,
                             n_ref_sources=self._config['REF_SOURCES_MAX_NO'],
                             ref_cat_mag_lim=self._config['REF_CATALOG_MAG_LIM'],
                             bkg_fname=(bkg_fname, bkg_fname_short),
                             _force_extract=self._force_extract,
                             _force_download=self._force_download,
                             _plot_images=self._plot_images)

        # add to configuration
        config.update(params_to_add)
        for key, value in self._config.items():
            config[key] = value

        # extract sources on detector
        extraction_result, state = sext.get_src_and_cat_info(fbase, cat_path,
                                                             imgarr, hdr, init_wcsprm, catalog,
                                                             mode='astro',
                                                             silent=self._silent,
                                                             **config)
        # unpack extraction result tuple
        (src_tbl, ref_tbl, ref_catalog, _, _,
         src_cat_fname, ref_cat_fname, _,
         kernel_fwhm, psf, segmap, _) = extraction_result

        # check execution state
        if not state:
            self._converged = False
            ra = hdr[obsparams['ra']]
            dec = hdr[obsparams['dec']]

            if obsparams['radec_separator'] == 'XXX':
                ra = round(hdr[obsparams['ra']], _base_conf.ROUND_DECIMAL)
                dec = round(hdr[obsparams['dec']], _base_conf.ROUND_DECIMAL)

            kwargs = {obsparams['exptime']: hdr[obsparams['exptime']],
                      obsparams['object']: hdr[obsparams['object']],
                      obsparams['instrume']: hdr[obsparams['instrume']],
                      obsparams['ra']: ra,
                      obsparams['dec']: dec,
                      'AST_CAL': False}
            self._obsTable.update_obs_table(file=fbase, kwargs=kwargs, obsparams=obsparams)
            return

        # get reference positions before the transformation
        ref_positions_before = init_wcsprm.s2p(ref_tbl[["RA", "DEC"]].values, 0)['pixcrd']

        # test with lower number of stars
        # src_tbl = src_tbl.head(170)
        max_wcs_iter = self._config['MAX_WCS_FUNC_ITER']

        match_radius = self._config['MATCH_RADIUS']
        match_radius = kernel_fwhm[0] if match_radius == -1 else match_radius

        best_wcsprm, converged = self.get_wcs(src_tbl, ref_tbl,
                                              init_wcsprm,
                                              match_radius,
                                              max_wcs_iter)

        # Move scaling to cdelt and out of the pc matrix
        best_wcsprm, scales = imtrans.translate_wcsprm(wcsprm=best_wcsprm)

        status_str = _base_conf.BCOLORS.PASS + "PASS" + _base_conf.BCOLORS.ENDC
        if not converged:
            status_str = _base_conf.BCOLORS.FAIL + "FAIL" + _base_conf.BCOLORS.ENDC
        else:
            wcsprm = best_wcsprm

        if not self._silent:
            self._log.info("  ==> Calibration status: " + status_str)

        # apply the found wcs to the original reference catalog
        ref_tbl_adjusted = ref_tbl.copy()
        pos_on_det = wcsprm.s2p(ref_tbl[["RA", "DEC"]].values, 0)['pixcrd']
        ref_tbl_adjusted["xcentroid"] = pos_on_det[:, 0]
        ref_tbl_adjusted["ycentroid"] = pos_on_det[:, 1]

        # update file header and save
        dic_rms = {"radius_px": None, "matches": None, "rms": None}
        if converged:
            # check goodness
            if not self._silent:
                self._log.info("> Evaluate goodness of transformation")
            dic_rms = self._determine_if_fit_converged(ref_tbl_adjusted, src_tbl, wcsprm,
                                                       match_radius)

            # WCS difference before and after
            wcs_pixscale, rotation_angle, rotation_direction = self._compare_results(imgarr,
                                                                                     hdr,
                                                                                     wcsprm)

            # update report dict
            report["converged"] = converged
            report["catalog"] = ref_catalog
            report["fwhm"] = kernel_fwhm[0]
            report["e_fwhm"] = kernel_fwhm[1]
            report["matches"] = int(dic_rms["matches"])
            report["match_radius"] = dic_rms["radius_px"]
            report["match_rms"] = dic_rms["rms"]
            report["pix_scale"] = np.mean(abs(wcs_pixscale)) * 3600. if wcs_pixscale is not np.nan else np.nan
            report["scale_x"] = abs(wcs_pixscale[1]) * 3600.
            report["scale_y"] = abs(wcs_pixscale[0]) * 3600.
            report["det_rotang"] = rotation_angle if rotation_angle is not np.nan else np.nan

            # save the result to the fits-file
            self._write_wcs_to_hdr(original_filename=abs_file_path,
                                   filename_base=fbase,
                                   destination=cal_path,
                                   wcsprm=wcsprm,
                                   report=report, hdul_idx=hdu_idx)
        else:
            ra = hdr[obsparams['ra']]
            dec = hdr[obsparams['dec']]
            if obsparams['radec_separator'] == 'XXX':
                ra = round(hdr[obsparams['ra']], _base_conf.ROUND_DECIMAL)
                dec = round(hdr[obsparams['dec']], _base_conf.ROUND_DECIMAL)
            kwargs = {obsparams['exptime']: hdr[obsparams['exptime']],
                      obsparams['object']: hdr[obsparams['object']],
                      obsparams['instrume']: hdr[obsparams['instrume']],
                      obsparams['ra']: ra,
                      obsparams['dec']: dec,
                      'AST_CAL': False}
            self._obsTable.update_obs_table(file=fbase, kwargs=kwargs, obsparams=obsparams)

        # plot final figures
        src_positions = list(zip(src_tbl['xcentroid'], src_tbl['ycentroid']))

        # match results within 1 fwhm radius for plot
        matches = imtrans.find_matches(src_tbl,
                                       ref_tbl_adjusted,
                                       wcsprm,
                                       threshold=match_radius)
        _, _, _, ref_positions_after, _, _, _ = matches

        if not self._silent:
            self._log.info("> Plot and save before after comparison")
        self._plot_comparison(imgarr=imgarr, src_pos=src_positions,
                              ref_pos_before=ref_positions_before,
                              ref_pos_after=ref_positions_after,
                              file_name=file_name, fig_path=plt_path_final,
                              wcsprm=wcsprm, **config)

        self._converged = converged
        self._dic_rms = dic_rms

        del imgarr, dic_rms, converged, src_positions, ref_positions_before, ref_positions_after, \
            src_tbl, ref_tbl, ref_catalog, _, src_cat_fname, ref_cat_fname, \
            kernel_fwhm, psf, segmap, extraction_result, report, \
            state, matches
        gc.collect()

    def get_wcs(self, source_df, ref_df, initial_wcsprm,
                match_radius=1, max_iter=6):
        """"""

        # make a copy to work with
        source_cat = source_df.copy()
        reference_cat = ref_df.copy()

        # make sure the catalogs are sorted
        source_cat = source_cat.sort_values(by='mag', ascending=True)
        source_cat.reset_index(inplace=True, drop=True)

        reference_cat = reference_cat.sort_values(by='mag', ascending=True)
        reference_cat.reset_index(inplace=True, drop=True)

        self._log.info("> Run WCS calibration")

        # number of detected sources
        Nobs = len(source_cat)
        Nref = len(reference_cat)
        source_ratio = (Nobs / Nref) * 100.
        percent_to_select = np.ceil(source_ratio * 10)
        n_percent = percent_to_select if percent_to_select < 100 else 100

        # Calculate the number of rows to select based on the percentage
        num_rows = int(Nref * (n_percent / 100))

        # Select at least the first 100 entries
        num_rows = 100 if num_rows < 100 else num_rows

        if not self._silent:
            self._log.info(f'  Number of detected sources: {Nobs}')
            self._log.info(f'  Number of reference sources (total): {Nref}')
            self._log.info(f'  Ratio of detected sources to reference sources: {source_ratio:.3f}%')
            # self._log.info(f'Select: {np.ceil(source_ratio * 10):.2f}%')
            self._log.info(f'  Initial number of reference sources selected: {num_rows}')
            self._log.info(f'  Source match radius = {match_radius:.2f} px')

        # keep track of progress
        progress = 0
        no_changes = 0
        # match_radius = 1
        has_converged = False
        use_initial_ref_cat = False

        # apply the found wcs to the original reference catalog
        ref_cat_orig_adjusted = reference_cat.copy()
        pos_on_det = initial_wcsprm.s2p(reference_cat[["RA", "DEC"]].values, 0)['pixcrd']
        ref_cat_orig_adjusted["xcentroid"] = pos_on_det[:, 0]
        ref_cat_orig_adjusted["ycentroid"] = pos_on_det[:, 1]

        # select initial reference sample based on the current best result
        initial_reference_cat_subset = ref_cat_orig_adjusted.head(num_rows)

        current_reference_cat_subset = initial_reference_cat_subset
        current_wcsprm = initial_wcsprm

        # setup best values
        # best_rms = 1
        # best_score = 0
        best_Nobs = 0
        best_completeness = 0
        best_wcsprm = None

        if not self._silent:
            self._log.info(f'  Find scale, rotation, and offset.')

        while True:
            if progress == max_iter:
                break
            else:
                progress += 1

            if use_initial_ref_cat:
                current_reference_cat_subset = initial_reference_cat_subset

            # get wcs transformation
            try:
                tform_result = self._get_transformations(source_cat=source_cat,
                                                         ref_cat=current_reference_cat_subset,
                                                         wcsprm=current_wcsprm)
            except IndexError:
                current_reference_cat_subset = (initial_reference_cat_subset
                                                .sample(int(len(initial_reference_cat_subset) * 0.95)))
                current_wcsprm = initial_wcsprm
                continue

            # extract variables
            new_wcsprm, new_wcsprm_vals, tform_params, offsets, is_fine_tuned = tform_result

            # match results within 1 pixel radius for plot, consider fwhm
            matches = imtrans.find_matches(source_cat,
                                           current_reference_cat_subset,
                                           new_wcsprm,
                                           threshold=match_radius)
            source_cat_matched, ref_cat_matched, obs_xy, _, distances, _, _ = matches
            # update the wcsprm variable
            current_wcsprm = new_wcsprm

            # check if any matches were found
            if obs_xy is None or len(obs_xy[:, 0]) == 0:
                if not self._silent:
                    self._log.info(f'  Try [ {progress}/{max_iter} ] No matches found. Trying again.')
                use_initial_ref_cat = True
                continue

            # get convergence criteria
            Nobs_matched = len(obs_xy[:, 0])
            new_completeness = Nobs_matched / Nobs
            new_rms = np.sqrt(np.mean(np.square(distances))) / Nobs_matched
            new_score = Nobs_matched / new_rms

            message = f'  Try [ {progress}/{max_iter} ] ' \
                      f'Found matches: {Nobs_matched}/{Nobs} ({new_completeness * 100:.1f}%), ' \
                      f'Score: {new_score:.0f}, ' \
                      f'RMS: {new_rms:.3f}'

            if not self._silent:
                self._log.info(message)

            if Nobs_matched > 3:
                delta_matches = Nobs_matched - best_Nobs

                # update best values
                if Nobs_matched > best_Nobs:
                    best_wcsprm = new_wcsprm
                    best_Nobs = Nobs_matched
                    # best_score = new_score
                    # best_rms = new_rms
                    best_completeness = new_completeness

                # the best case, all sources were matched
                if best_completeness == 1.:
                    has_converged = True
                    break

                if Nobs_matched >= self._config['MIN_SOURCE_NO_CONVERGENCE'] and delta_matches == 0.:
                    if best_completeness > self._config['THRESHOLD_CONVERGENCE'] or best_completeness <= \
                            self._config['THRESHOLD_CONVERGENCE']:
                        has_converged = True
                        break
                elif Nobs_matched >= self._config['MIN_SOURCE_NO_CONVERGENCE'] and delta_matches < 0:
                    no_changes += 1

                if no_changes == 3:
                    if not self._silent:
                        self._log.info(f'  No improvement in the last 3 iterations. Using best result.')
                    has_converged = True
                    break
                # Use the remaining detected sources and find close sources in the ref catalog
                current_reference_cat_subset = self._update_ref_cat_subset(source_cat_matched,
                                                                           ref_cat_matched,
                                                                           source_cat, reference_cat,
                                                                           new_wcsprm)
                # current_wcsprm = initial_wcsprm
                use_initial_ref_cat = False
            else:
                use_initial_ref_cat = True

        return best_wcsprm, has_converged

    def _update_ref_cat_subset(self, source_cat_matched, ref_cat_matched,
                               source_cat_original, ref_cat_original, wcsprm):
        """"""

        # apply the found wcs to the original reference catalog
        ref_cat_orig_adjusted = ref_cat_original.copy()
        pos_on_det = wcsprm.s2p(ref_cat_original[["RA", "DEC"]].values, 0)['pixcrd']
        ref_cat_orig_adjusted["xcentroid"] = pos_on_det[:, 0]
        ref_cat_orig_adjusted["ycentroid"] = pos_on_det[:, 1]

        # get the remaining not matched source from the source catalog
        remaining_sources = source_cat_original[~source_cat_original['id'].isin(source_cat_matched['id'])]

        # get the closest N sources from the reference catalog for the remaining sources
        ref_sources_to_add = self.match_catalogs(remaining_sources,
                                                 ref_cat_orig_adjusted,
                                                 radius=10., N=3)
        # print(ref_sources_to_add)
        #
        ref_cat_orig_matched = self.match_catalogs(ref_cat_matched,
                                                   ref_cat_original,
                                                   ra_dec_tol=1e-4, N=1)

        ref_catalog_subset = pd.concat([ref_cat_orig_matched, ref_sources_to_add],
                                       ignore_index=True)
        # print(ref_catalog_subset)
        # print(len(ref_catalog_subset))
        # self.plot_catalog_positions(source_catalog=remaining_sources,
        #                             reference_catalog=ref_sources_to_add)
        # self.plot_catalog_positions(source_catalog=source_cat_original,
        #                             reference_catalog=ref_catalog_subset,
        #                             wcsprm=wcsprm)
        # plt.show()
        return ref_catalog_subset

    @staticmethod
    def match_catalogs(catalog1, catalog2, ra_dec_tol=None, radius=None, N=3):
        """
        Match catalogs based on the closest points within a tolerance or centroid within a radius.

        Args:
            catalog1 (DataFrame): First catalog.
            catalog2 (DataFrame): Second catalog.
            ra_dec_tol (float): Tolerance in RA/DEC coordinates for matching.
            radius (float): Radius in pixels to consider for matching.
            N (int): Number of the closest points to return.

        Returns:
            DataFrame: Subset of catalog2 with matched points.
        """
        if ra_dec_tol is not None:
            # Build a KDTree for catalog1 based on RA/DEC coordinates
            catalog1_tree = KDTree(catalog1[['RA', 'DEC']].values)

            # Find the indices of the closest points for each row in catalog2
            # based on RA/DEC coordinates
            distances, indices = catalog1_tree.query(catalog2[['RA', 'DEC']].values, k=1)

            # Create a mask for distances within the specified tolerance
            mask = distances < ra_dec_tol
        elif radius is not None:
            # Build a KDTree for catalog1 based on centroid coordinates
            catalog1_tree = KDTree(catalog1[['xcentroid', 'ycentroid']].values)

            # Find the indices and distances of the closest points for each row in catalog2
            # based on centroid coordinates
            distances, indices = catalog1_tree.query(catalog2[['xcentroid', 'ycentroid']].values, k=N)

            # Create a mask for distances within the specified radius
            mask = distances <= radius
        else:
            raise ValueError("Either ra_dec_tol or radius must be specified for matching.")

        # Select the matching rows from catalog2
        matched_catalog2_subset = catalog2[mask]

        return matched_catalog2_subset

    @staticmethod
    def plot_catalog_positions(source_catalog, reference_catalog, wcsprm=None):
        reference_cat = reference_catalog.copy()

        if wcsprm is not None:
            pos_on_det = wcsprm.s2p(reference_catalog[["RA", "DEC"]].values, 0)['pixcrd']
            reference_cat["xcentroid"] = pos_on_det[:, 0]
            reference_cat["ycentroid"] = pos_on_det[:, 1]

        fig, ax = plt.subplots(figsize=(10, 10))

        # Plot the positions of the source catalog
        ax.scatter(source_catalog['xcentroid'], source_catalog['ycentroid'], c='blue', marker='o',
                   label='Source Catalog', alpha=0.7)

        # Plot the positions of the reference catalog
        ax.scatter(reference_cat['xcentroid'], reference_cat['ycentroid'], c='red', marker='x',
                   label='Reference Catalog',
                   alpha=0.7)

        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.legend()
        plt.title('Source Catalog and Reference Catalog Positions')
        # plt.show()

    def _get_transformations(self, source_cat: pd.DataFrame, ref_cat: pd.DataFrame, wcsprm):
        """ Determine rotation, scale and offset using image transformations

        Parameters
        ----------
        source_cat: 'pandas.core.frame.DataFrame'
            Dataframe with pixel positions of detected sources in the observation image
        ref_cat: 'pandas.core.frame.DataFrame'
            Dataframe with coordinates of reference sources
        wcsprm: astropy.wcs.wcsprm
            World coordinate system object describing translation between image and skycoord
        """
        INCREASE_FOV_FLAG, PIXSCALE_UNCLEAR = self._wcs_check
        tform_params = None
        offsets = None

        if self._rot_scale:
            # if not self._silent:
            #     self._log.info("  > Determine pixel scale and rotation angle")
            self._log.debug("  > Determine pixel scale and rotation angle")
            wcsprm, tform_params = imtrans.get_scale_and_rotation(observation=source_cat,
                                                                  catalog=ref_cat,
                                                                  wcsprm=wcsprm,
                                                                  scale_guessed=PIXSCALE_UNCLEAR,
                                                                  dist_bin_size=self._config['DISTANCE_BIN_SIZE'],
                                                                  ang_bin_size=self._config['ANG_BIN_SIZE'])

        if self._xy_transformation:
            # if not self._silent:
            #     self._log.info("  > Determine x, y offset")
            self._log.debug("  > Determine x, y offset")
            wcsprm, offsets, _, _ = imtrans.get_offset_with_orientation(observation=source_cat,
                                                                        catalog=ref_cat,
                                                                        wcsprm=wcsprm)

        # update reference catalog
        ref_cat_corrected = ref_cat.copy()
        pos_on_det = wcsprm.s2p(ref_cat[["RA", "DEC"]].values, 0)['pixcrd']
        ref_cat_corrected["xcentroid"] = pos_on_det[:, 0]
        ref_cat_corrected["ycentroid"] = pos_on_det[:, 1]

        # correct subpixel error
        wcsprm, wcsprm_vals, fine_tune_status = self._correct_subpixel_error(source_cat=source_cat,
                                                                             ref_cat=ref_cat_corrected,
                                                                             wcsprm=wcsprm)

        del source_cat, ref_cat, ref_cat_corrected, _
        gc.collect()

        return wcsprm, wcsprm_vals, tform_params, offsets, fine_tune_status

    def _compare_results(self, imgarr, hdr, wcsprm):
        """Compare WCS difference before and after transformation"""

        wcsprm_original = WCS(hdr).wcs

        if not self._silent:
            self._log.info("> Compared to the input the Wcs was changed by: ")

        # scales_original = utils.proj_plane_pixel_scales(WCS(hdr))

        # if scales is not None:
        #     if not self._silent:
        #         self._log.info("  WCS got scaled by a factor of {} in x direction and "
        #                        "{} in y direction".format(scales[0] / scales_original[0],
        #                                                   scales[1] / scales_original[1]))

        # sources:
        # https://math.stackexchange.com/questions/2113634/comparing-two-rotation-matrices
        # https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python/13849249#13849249

        def unit_vector(vector):
            """ Returns the unit vector of the vector.  """
            return vector / max(np.linalg.norm(vector), 1e-10)

        def matrix_angle(B, A):
            """ comment cos between vectors or matrices """
            Aflat = A.reshape(-1)
            Aflat = unit_vector(Aflat)
            Bflat = B.reshape(-1)
            Bflat = unit_vector(Bflat)

            return np.arccos(np.clip(np.dot(Aflat, Bflat), -1.0, 1.0))

        # bugfix: multiplying by cdelt otherwise the calculated angle is off by a tiny bit
        rotation_angle = matrix_angle(wcsprm.get_pc() @ wcsprm.get_cdelt(),
                                      wcsprm_original.get_pc() @ wcsprm_original.get_cdelt()) / 2. / np.pi * 360.

        if (wcsprm.get_pc() @ wcsprm_original.get_pc())[0, 1] > 0:
            text = "counterclockwise"
            rotation_direction = 'ccw'
        else:
            text = "clockwise"
            rotation_direction = 'cw'

        if not self._silent:
            self._log.info(f"  Rotation of WCS by an angle of {rotation_angle} deg " + text)

        old_central_pixel = [imgarr.shape[0] / 2, imgarr.shape[1] / 2]
        if not self._silent:
            self._log.info("  x offset: {} px, y offset: {} px ".format(wcsprm.crpix[0] - old_central_pixel[0],
                                                                        wcsprm.crpix[1] - old_central_pixel[1]))

        wcs_pixscale = self._get_wcs_pixelscale(wcsprm)

        return wcs_pixscale, rotation_angle, rotation_direction

    @staticmethod
    def _get_wcs_pixelscale(wcsprm):
        """ Get the pixel scale in deg/pixel from the WCS object """

        pc = wcsprm.get_pc()
        cdelt = wcsprm.get_cdelt()
        wcs_pixscale = (pc @ cdelt)

        return wcs_pixscale

    def _correct_subpixel_error(self, source_cat: pd.DataFrame, ref_cat: pd.DataFrame, wcsprm):
        """ Correct the subpixel error using fine transformation.

        Parameters
        ----------
        source_cat: 'pandas.core.frame.DataFrame'
            Dataframe with pixel positions of detected sources in the observation image
        ref_cat: 'pandas.core.frame.DataFrame'
            Dataframe with coordinates of reference sources
        wcsprm: astropy.wcs.wcsprm
            World coordinate system object describing translation between image and skycoord

        Returns
        -------
        wcsprm: astropy.wcs.wcsprm
            New world coordinate system object describing translation between image and skycoord
        """

        compare_threshold = 5
        if self._high_res:
            compare_threshold = 50

        # find matches
        matches = imtrans.find_matches(source_cat,
                                       ref_cat,
                                       wcsprm,
                                       threshold=compare_threshold)
        _, _, _, _, distances, best_score, state = matches

        best_wcsprm_vals = None
        fine_transformation_success = False
        if self._fine_transformation:
            # if not self._silent:
            #     self._log.info("  > Refine scale and rotation")
            self._log.debug("  > Refine scale and rotation")
            lis = [2, 3, 5, 8, 10, 20, 1, 0.5, 0.2, 0.1]  # , 0.5, 0.25, 0.2, 0.1
            if self._high_res:
                lis = [200, 300, 100, 150, 80, 40, 70, 20, 100, 30, 9, 5]
            skip_rot_scale = True
            for i in lis:
                wcsprm_new, score, wcsprm_vals = imtrans.fine_transformation(source_cat,
                                                                             ref_cat,
                                                                             wcsprm,
                                                                             threshold=i,
                                                                             compare_threshold=compare_threshold,
                                                                             skip_rot_scale=skip_rot_scale,
                                                                             silent=self._silent)

                if i == 20:
                    # only allow rot and scaling for the last few tries
                    skip_rot_scale = False

                if score > best_score:
                    wcsprm = wcsprm_new
                    best_score = score
                    best_wcsprm_vals = wcsprm_vals
                    fine_transformation_success = True

            # pass_str = _base_conf.BCOLORS.PASS + "PASS" + _base_conf.BCOLORS.ENDC
            # fail_str = _base_conf.BCOLORS.FAIL + "FAIL" + _base_conf.BCOLORS.ENDC
            # if not fine_transformation_success:
            #     if not self._silent:
            #         self._log.info("    Status: " + fail_str)
            # else:
            #     if not self._silent:
            #         self._log.info("    Status: " + pass_str)

        del source_cat, ref_cat, matches
        gc.collect()

        return wcsprm, best_wcsprm_vals, fine_transformation_success

    def _determine_if_fit_converged(self, catalog, observation, wcsprm, fwhm) -> dict:
        """Cut off criteria for deciding if the astrometry fit worked.
        
        Parameters
        ----------
        catalog :
        observation :
        wcsprm :
        fwhm :

        Returns
        -------
        converged: bool
            Returns whether the fit converged or not
        """

        N_obs = observation.shape[0]

        # get pixel scale
        px_scale = np.abs(wcsprm.cdelt[0]) * 3600.  # in arcsec

        # get the radii list
        r = round(2 * fwhm, 1)
        radii_list = list(imtrans.frange(0.1, r, 0.1))
        result_array = np.zeros((len(radii_list), 6))

        for i, r in enumerate(radii_list):
            matches = imtrans.find_matches(observation,
                                           catalog, wcsprm,
                                           threshold=r)
            _, _, obs_xy, _, distances, _, _ = matches
            len_obs_x = len(obs_xy[:, 0])

            rms = np.sqrt(np.mean(np.square(distances))) / len_obs_x

            result_array[i, :] = [len_obs_x, len_obs_x / N_obs, r, r * px_scale, rms, rms * px_scale]

        # Calculate the differences between consecutive numbers of sources and pixel values
        diff_sources = np.diff(result_array[:, 0])
        diff_pixels = np.diff(result_array[:, 2])

        # Calculate the rate of change
        rate_of_change = diff_sources / diff_pixels

        # Find the point where the rate of change drops below a certain threshold
        threshold = 1.  # This is just an example value, you might need to adjust it based on your data
        index = np.where(np.logical_and((rate_of_change < threshold),
                                        (result_array[1:, 2] > fwhm / 2.)))[0][0]

        result = result_array[index, :]
        len_obs_x = int(result[0])
        r = result[2]
        rms = result[4]
        dic_rms = {"matches": len_obs_x,
                   "complete": result[1],
                   "radius_px": r,
                   "rms": rms}
        # print(dic_rms)
        if not self._silent:
            self._log.info("  Within {} pixel or {:.3g} arcsec {}/{} ({:.1f}%) sources where matched. "
                           "The rms is {:.3g} pixel or "
                           "{:.3g} arcsec".format(r, px_scale * r,
                                                  len_obs_x, N_obs,
                                                  (len_obs_x / N_obs) * 100,
                                                  rms, rms * px_scale))

        return dic_rms

    def _check_image_wcs(self, wcsprm, hdr, obsparams,
                         ra_input=None, dec_input=None,
                         projection_ra=None, projection_dec=None,
                         ignore_header_rot=False, radius=-1.):
        """Check the WCS coordinate system for a given image.

        The function also tries to handle additional or missing data from the header.

        Parameters
        ----------
        wcsprm: astropy.wcs.wcsprm
            World coordinate system object describing translation between image and skycoord
        hdr: header
        """

        # Initialize logging for this function
        log = self._log
        INCREASE_FOV_FLAG = False
        PIXSCALE_UNCLEAR = False

        if not self._silent:
            log.info("> Check header information")

        # check axes keywords
        if "NAXIS1" not in hdr or "NAXIS2" not in hdr:
            log.error("NAXIS1 or NAXIS2 missing in file. Please add!")
            sys.exit(1)
        else:
            axis1 = hdr[obsparams['extent'][0]]
            axis2 = hdr[obsparams['extent'][1]]

        if isinstance(obsparams['binning'][0], str) and isinstance(obsparams['binning'][1], str):
            bin_x = int(hdr[obsparams['binning'][0]])
            bin_y = int(hdr[obsparams['binning'][1]])
        else:
            bin_x = int(obsparams['binning'][0])
            bin_y = int(obsparams['binning'][1])

        if self._telescope in ['DK-1.54']:
            wcs_rebinned = WCS(wcsprm.to_header()).slice((np.s_[::bin_x], np.s_[::bin_y]))
            wcsprm = wcs_rebinned.wcs

        # read out ra and dec from header
        if obsparams['radec_separator'] == 'XXX':
            ra_deg = float(hdr[obsparams['ra']])
            dec_deg = float(hdr[obsparams['dec']])
        else:
            ra_string = hdr[obsparams['ra']].split(
                obsparams['radec_separator'])
            dec_string = hdr[obsparams['dec']].split(
                obsparams['radec_separator'])
            ra_deg = 15. * (float(ra_string[0]) + float(ra_string[1]) / 60. + float(ra_string[2]) / 3600.)
            dec_deg = (abs(float(dec_string[0])) + float(dec_string[1]) / 60. + float(dec_string[2]) / 3600.)
            if dec_string[0].find('-') > -1:
                dec_deg *= -1

        # transform to equinox J2000, if necessary
        wcsprm.equinox = 2000.
        if 'EQUINOX' in hdr:
            equinox = float(hdr['EQUINOX'])
            wcsprm.equinox = equinox
            if equinox != 2000.:
                any_eq = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg,
                                  frame=FK5, equinox=Time(equinox, format='jyear', scale='utc'))
                coo = any_eq.transform_to(ICRS)
                ra_deg = coo.ra.deg
                dec_deg = coo.dec.deg
                wcsprm.equinox = 2000.0

        wcs = WCS(wcsprm.to_header())
        hdr.update(wcs.to_header(), relax=True)

        if 'RADECSYS' not in hdr:
            hdr['RADECSYS'] = 'FK5'
        else:
            wcsprm.radecsys = hdr['RADECSYS']
        if 'RADESYS' not in hdr:
            hdr['RADESYS'] = 'FK5'
        else:
            wcsprm.radesys = hdr['RADESYS']

        wcsprm = WCS(hdr).wcs
        if isinstance(obsparams['secpix'][0], str) and \
                isinstance(obsparams['secpix'][1], str):
            pixscale_x = float(hdr[obsparams['secpix'][0]])
            pixscale_y = float(hdr[obsparams['secpix'][1]])
            if "deg" in hdr.comments[obsparams['secpix'][0]]:
                pixscale_x = pixscale_x * 60. * 60.
                pixscale_y = pixscale_y * 60. * 60.
        else:
            pixscale_x = float(obsparams['secpix'][0])
            pixscale_y = float(obsparams['secpix'][1])

        # apply CCD binning factor
        pixscale_x *= bin_x
        pixscale_y *= bin_y
        x_size = axis1 * pixscale_x / 60.  # arcmin
        y_size = axis2 * pixscale_y / 60.  # arcmin

        # test if the current pixel scale makes sense
        wcs_pixscale = self._get_wcs_pixelscale(wcsprm)

        # print(pc, cdelt, wcs_pixscale)
        test_wcs1 = ((np.abs(wcs_pixscale[0])) < 1e-7 or (np.abs(wcs_pixscale[1])) < 1e-7 or
                     (np.abs(wcs_pixscale[0])) > 5e-3 or (np.abs(wcs_pixscale[1])) > 5e-3)
        if test_wcs1:
            if not self._silent:
                log.info("  Pixelscale appears unrealistic. We will use a guess instead.")
                # wcsprm.pc = [[1, 0], [0, 1]]
                wcsprm.pc = [[0, 1], [-1, 0]]
                guess = pixscale_x / 3600.
                wcsprm.cdelt = [-guess, guess]
                if self._telescope in ['DDOTI 28-cm f/2.2']:
                    # wcsprm.pc = [[-1, 0], [0, 1]]
                    wcsprm.cdelt = [guess, guess]
                elif self._telescope == 'PlaneWave CDK24':
                    wcsprm.pc = [[0, 1], [1, 0]]
                    wcsprm.cdelt = [-guess, guess]
                PIXSCALE_UNCLEAR = True

        if ignore_header_rot:
            wcsprm.pc = [[1, 0], [0, 1]]

        # test if found pixel scale makes sense
        test_wcs2 = (120. > x_size > 0.5) & \
                    (120. > y_size > 0.5)

        if test_wcs2:
            wcs_pixscale = self._get_wcs_pixelscale(wcsprm)

            pixscale_x = pixscale_x / 60. / 60.  # pixel scale now in deg / pixel
            pixscale_y = pixscale_y / 60. / 60.  # pixel scale now in deg / pixel

            if wcs_pixscale[0] / pixscale_x < 0.1 or wcs_pixscale[0] / pixscale_x > 10 \
                    or wcs_pixscale[1] / pixscale_y < 0.1 or wcs_pixscale[1] / pixscale_y > 10:
                # check if there is a huge difference in the scales
                # if yes, then replace the wcs scale with the pixel scale info
                wcsprm.pc = [[0, 1], [-1, 0]]
                wcsprm.cdelt = [-pixscale_x, pixscale_y]

                if self._telescope == 'DK-1.54':
                    wcsprm.pc = [[-1, 0], [0, -1]]
                    wcsprm.cdelt = [pixscale_x, pixscale_y]
                if self._telescope == 'DDOTI 28-cm f/2.2':
                    wcsprm.pc = [[1, 0], [0, -1]]
                    wcsprm.cdelt = [pixscale_x, pixscale_y]

                PIXSCALE_UNCLEAR = True

        if not self._silent:
            log.info("  Changed pixel scale to "
                     "{:.3g} x {:.3g} deg/pixel".format(wcsprm.cdelt[0], wcsprm.cdelt[1]))

        # if the central pixel is not in the header, set to image center pixel
        if np.array_equal(wcsprm.crpix, [0, 0]):
            wcsprm.crpix = [axis1 // 2, axis2 // 2]
        else:
            wcsprm.crpix = [wcsprm.crpix[0] // bin_x, wcsprm.crpix[1] // bin_y]

        # check sky position
        if np.array_equal(wcsprm.crval, [0, 0]):
            INCREASE_FOV_FLAG = True
            # If the sky position is not found check header for RA and DEC info
            wcsprm.crval = [ra_deg, dec_deg]
        else:
            if not self._silent:
                log.info("  RA and DEC information found in the header.")
                log.info("  {}".format(" ".join(np.array(wcsprm.crval, str))))

        if ra_input is not None:  # use user input if provided
            wcsprm.crval = [ra_input, wcsprm.crval[1]]
            wcsprm.crpix = [axis1 // 2, wcsprm.crpix[1]]

        if dec_input is not None:
            wcsprm.crval = [wcsprm.crval[0], dec_input]
            wcsprm.crpix = [wcsprm.crpix[0], axis2 // 2]

        if np.array_equal(wcsprm.ctype, ["", ""]):
            INCREASE_FOV_FLAG = True
            if projection_ra is not None and projection_dec is not None:
                wcsprm.ctype = [projection_ra, projection_dec]
            else:
                wcsprm.ctype = ['RA---TAN', 'DEC--TAN']  # this is a guess

        wcs = WCS(wcsprm.to_header())

        # estimate size of FoV
        fov_radius = sext.compute_radius(wcs, axis1, axis2)
        if radius > 0.:
            fov_radius = radius

        # increase FoV for SPM data
        if self._telescope == 'DDOTI 28-cm f/2.2':
            fov_radius *= 1.5

        if not self._silent:
            log.info("  Using field-of-view radius: {:.3g} deg".format(fov_radius))

        self._wcsprm = wcsprm
        self._fov_radius = fov_radius
        self._wcs_check = (INCREASE_FOV_FLAG, PIXSCALE_UNCLEAR)

    def _write_wcs_to_hdr(self, original_filename, filename_base,
                          destination, wcsprm, report, hdul_idx=0):
        """Update the header of the fits file itself.
        Parameters
        ----------
        original_filename: str
            Original filename of the fits-file.
        wcsprm: astropy.wcs.wcsprm
            World coordinate system object describing translation between image and skycoord
        """

        if not self._silent:
            self._log.info("> Update FITS file.")

        with fits.open(original_filename) as hdul:

            hdu = hdul[hdul_idx]
            hdr_file = hdu.header

            # throughout CD which contains the scaling and separate into pc and Cdelt
            for old_parameter in ['CD1_1', 'CD1_2', 'CD2_1', 'CD2_2',
                                  "PC1_1", "PC1_2", "PC2_1", "PC2_2"]:
                if old_parameter in hdr_file:
                    del hdr_file[old_parameter]

            # update wcs
            wcs = WCS(wcsprm.to_header())
            hdr_file.update(wcs.to_header())

            # adding report
            hdr_file['PIXSCALE'] = (report["pix_scale"], 'Average pixel scale [arcsec/pixel]')
            hdr_file['SCALEX'] = (report["scale_x"], 'Pixel scale in X [arcsec/pixel]')
            hdr_file['SCALEY'] = (report["scale_y"], 'Pixel scale in Y [arcsec/pixel]')
            hdr_file['DETROTANG'] = (report["det_rotang"], 'Detection rotation angel [deg]')
            hdr_file['FWHM'] = (report["fwhm"], 'FWHM [pixel]')
            hdr_file['FWHMERR'] = (report["e_fwhm"], 'FWHM error [pixel]')
            hdr_file['AST_SCR'] = ("Astrometry", 'Astrometric calibration by Christian Adam')
            hdr_file['AST_CAT'] = (report["catalog"], 'Catalog used')
            hdr_file['AST_MAT'] = (report["matches"], 'Number of catalog matches')
            hdr_file['AST_RAD'] = (report["match_radius"], 'match radius in pixel')
            hdr_file['AST_CONV'] = (report["converged"], "T or F for astrometry converged or not")
            hdr_file['AST_CAL'] = (True, "T or F for astrometric calibration executed or not")

            if 'HISTORY' in hdr_file:
                hdr_file.remove('HISTORY')

            if 'COMMENT' in hdr_file:
                hdr_file.remove('COMMENT')

            hdu.header = hdr_file
            hdul[hdul_idx] = hdu

            hdul.writeto(destination / f"{filename_base}_cal.fits", overwrite=True)

            fits_header = hdu.header

        if not self._silent:
            self._log.info("  FITS file updated.")
        obsparams = self._obsparams
        if obsparams['radec_separator'] == 'XXX':
            fits_header[obsparams['ra']] = round(fits_header[obsparams['ra']],
                                                 _base_conf.ROUND_DECIMAL)
            fits_header[obsparams['dec']] = round(fits_header[obsparams['dec']],
                                                  _base_conf.ROUND_DECIMAL)

        self._obsTable.update_obs_table(filename_base, fits_header, obsparams)

    def _plot_comparison(self, imgarr, src_pos, ref_pos_before, ref_pos_after,
                         file_name, fig_path, wcsprm, norm='lin', cmap='Greys', **config):
        """Plot before and after images"""

        # load fits file
        bkg_fname = config['bkg_fname']
        with fits.open(f'{bkg_fname[0]}.fits') as hdul:
            hdul.verify('fix')
            bkg_background = hdul[0].data.astype('float32')

        # subtract background
        imgarr = imgarr - bkg_background

        # get the scale and angle
        scale = wcsprm.cdelt[0]
        angle = np.arccos(wcsprm.pc[0][0])
        theta = angle / 2. / np.pi * 360.
        if wcsprm.pc[0][0] < 0.:
            theta *= -1.

        # get y-axis length as limit
        plim = imgarr.shape[0]
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

        adjusted_src_pos = [(x + 0.5, y + 0.5) for x, y in src_pos]
        adjusted_ref_pos_before = [(x + 0.5, y + 0.5) for x, y in ref_pos_before]
        adjusted_ref_pos_after = [(x + 0.5, y + 0.5) for x, y in ref_pos_after]

        apertures = CircularAperture(adjusted_src_pos, r=7)
        apertures_catalog_before = CircularAperture(adjusted_ref_pos_before, r=10)
        apertures_catalog_after = CircularAperture(adjusted_ref_pos_after, r=10.)

        # Define normalization and the colormap
        vmin = np.percentile(imgarr, 50)
        vmax = np.percentile(imgarr, 99.)
        if norm == 'lin':
            nm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=LinearStretch())
        elif norm == 'log':
            nm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=LogStretch())
        else:
            nm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=SqrtStretch())
        if cmap is None:
            cmap = 'Greys_r'

        figsize = (10, 6)
        fig = plt.figure(figsize=figsize)
        fig.canvas.manager.set_window_title(f'Input for {file_name}')

        gs = gridspec.GridSpec(1, 2)
        ax1 = fig.add_subplot(gs[0, 0])

        im1 = ax1.imshow(imgarr, origin='lower', norm=nm, cmap=cmap)

        apertures.plot(axes=ax1, **{'color': 'blue', 'lw': 1.25, 'alpha': 0.75})
        apertures_catalog_before.plot(axes=ax1, **{'color': 'red', 'lw': 1.25, 'alpha': 0.75})

        ax2 = fig.add_subplot(gs[0, 1])

        im2 = ax2.imshow(imgarr, origin='lower', norm=nm, cmap=cmap)

        # apertures.plot(axes=ax2, color='blue', lw=1.5, alpha=0.75)
        apertures_catalog_after.plot(axes=ax2, **{'color': 'green', 'lw': 1.25, 'alpha': 0.75})

        ax2.plot([xstmark, xenmark], [ystmark, ystmark], color='k', lw=1.25)
        ax2.annotate(text=r"{0:.0f}'".format(length),
                     xy=(((xstmark + xenmark) / 2.), ystmark + 5), xycoords='data',
                     textcoords='data', ha='center', c='k')

        # Add compass
        self._add_compass(ax=ax2, x0=x0, y0=y0, larr=larr, theta=theta)

        # format axes
        ax_list = [ax1, ax2]
        im_list = [im1, im2]
        for i in range(len(ax_list)):
            ax = ax_list[i]
            im = im_list[i]

            # -----------------------------------------------------------------------------
            # Make color bar
            # -----------------------------------------------------------------------------
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("top", size="5%", pad=0.075)
            cbar = plt.colorbar(im, cax=cax, orientation='horizontal')

            cbar.ax.tick_params(
                axis='both',  # changes apply to the x-axis
                which='both',  # both major and minor ticks are affected
                direction='in',
                left=True,  # ticks along the bottom edge are off
                right=True,  # ticks along the top edge are off
                top=True,
                bottom=True,
                width=1.,
                color='w')

            cax.xaxis.set_ticks_position("top")
            cax.xaxis.set_label_position("top")

            cbar.ax.set_xlabel("Flux (counts)")

            plt.setp(cbar.ax.xaxis.get_ticklabels(), fontsize='smaller')
            cbar.update_ticks()

            ax.set_xlim(xmin=0, xmax=imgarr.shape[1])
            ax.set_ylim(ymin=0, ymax=imgarr.shape[0])

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

            if i == 1:
                ax.set_yticklabels([])
            else:
                ax.set_ylabel(r'Pixel y direction')

            ax.set_xlabel(r'Pixel x direction')

        fig.tight_layout(h_pad=0.4, w_pad=0.5, pad=1.)

        # Save the plot
        fname = f"plot.astrometry.final_{file_name}"
        fname = os.path.join(fig_path, fname)

        fig_type = self._config['FIG_TYPE']
        fig_dpi = self._config['FIG_DPI']
        if fig_type == 'png':
            plt.savefig(fname + '.png', format='png', dpi=fig_dpi)
            os.system(f'mogrify -trim {fname}.png ')
        elif fig_type == 'pdf':
            self._log.setLevel("critical".upper())
            plt.savefig(fname + '.pdf', format='pdf', dpi=fig_dpi)
            self._log.setLevel("info".upper())
            os.system('pdfcrop ' + fname + '.pdf ' + fname + '.pdf')
        else:
            self._log.info(f"Figure will be saved as .png and.pdf")
            plt.savefig(fname + '.png', format='png', dpi=fig_dpi)
            os.system(f'mogrify -trim {fname}.png ')
            self._log.setLevel("critical".upper())
            plt.savefig(fname + '.pdf', format='pdf', dpi=fig_dpi)
            self._log.setLevel("info".upper())
            os.system('pdfcrop ' + fname + '.pdf ' + fname + '.pdf')

        if self._plot_images:
            plt.show()

        plt.close(fig=fig)

    @staticmethod
    def _add_compass(ax, x0, y0, larr, theta, color='black'):
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


def main():
    """ Main procedure """
    pargs = ParseArguments(prog_typ='calibWCS')
    args = pargs.args_parsed
    main.__doc__ = pargs.args_doc

    CalibrateObsWCS(input_path=args.input, args=args, silent=args.silent, verbose=args.verbose)


# -----------------------------------------------------------------------------


# standard boilerplate to set 'main' as starting function
if __name__ == '__main__':
    main()
# -----------------------------------------------------------------------------
