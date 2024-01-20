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
import sys
import logging
import warnings
import time
from datetime import (timedelta, datetime, timezone)
import collections
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
# from astropy.visualization import (LinearStretch, LogStretch, SqrtStretch)
# from astropy.visualization.mpl_normalize import ImageNormalize

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
try:
    import leosatpy
except ModuleNotFoundError:
    from utils.arguments import ParseArguments
    from utils.dataset import DataSet
    from utils.tables import ObsTables
    from utils.version import __version__
    import utils.sources as sext
    import utils.transformations as imtrans
    import utils.base_conf as _base_conf
else:

    from leosatpy.utils.arguments import ParseArguments
    from leosatpy.utils.dataset import DataSet
    from leosatpy.utils.tables import ObsTables
    from leosatpy.utils.version import __version__
    import leosatpy.utils.sources as sext
    import leosatpy.utils.transformations as imtrans
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

# Logging and console output
logging.root.handlers = []
_log = logging.getLogger()
_log.setLevel(_base_conf.LOG_LEVEL)
stream = logging.StreamHandler()
stream.setFormatter(_base_conf.FORMATTER)
_log.addHandler(stream)
_log_level = _log.level


# -----------------------------------------------------------------------------


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
        fail_path = Path(self._config['RESULT_TABLE_PATH']).expanduser().resolve()
        fail_fname = fail_path / f'fails_calibrateSatObs_{time_stamp}.log'

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
                    progress = f"{row_index+1}/{len(files)}"
                    self._run_single_calibration(src_path, file_df, progress, catalog=self._catalog,
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

    def _run_single_calibration(self, file_src_path, file_df, progress, catalog="GAIADR3", hdu_idx=0):
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

        self._log.info(f"==> Run astrometric calibration for {file_name} [{progress}] <==")

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

        sat_lim = config['saturation_limit']
        if isinstance(config['saturation_limit'], str) and config['saturation_limit'] in hdr:
            sat_lim = hdr[obsparams['saturation_limit']]
        config['sat_lim'] = sat_lim

        # extract sources on detector
        extraction_result, state = sext.get_src_and_cat_info(fbase, cat_path,
                                                             imgarr, hdr, init_wcsprm,
                                                             silent=self._silent,
                                                             **config)
        # unpack extraction result tuple
        (src_tbl_raw, ref_tbl, ref_catalog, src_cat_fname, ref_cat_fname,
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
        # ref_positions_before = init_wcsprm.s2p(ref_tbl[["RA", "DEC"]].values, 0)['pixcrd']

        # use only the entries with good fwhm values
        src_tbl = src_tbl_raw.query('include_fwhm')
        if len(src_tbl) < 5:
            src_tbl = src_tbl_raw

        max_wcs_iter = self._config['MAX_WCS_FUNC_ITER']

        # set the match radius
        match_radius = self._config['MATCH_RADIUS']
        match_radius = kernel_fwhm[0] if match_radius == 'fwhm' else match_radius

        best_wcsprm, converged = self.get_wcs(src_tbl, ref_tbl,
                                              init_wcsprm,
                                              match_radius,
                                              max_wcs_iter)

        status_str = _base_conf.BCOLORS.PASS + "PASS" + _base_conf.BCOLORS.ENDC
        if not converged or best_wcsprm is None:
            status_str = _base_conf.BCOLORS.FAIL + "FAIL" + _base_conf.BCOLORS.ENDC
        else:
            # Move scaling to cdelt and out of the pc matrix
            best_wcsprm, scales = imtrans.translate_wcsprm(wcsprm=best_wcsprm)
            wcsprm = best_wcsprm

        if not self._silent:
            self._log.info("  ==> Calibration status: " + status_str)

        # apply the found wcs to the original reference catalog
        ref_tbl_adjusted = ref_tbl.copy()
        pos_on_det = wcsprm.s2p(ref_tbl[["RA", "DEC"]].values, 0)['pixcrd']
        ref_tbl_adjusted["xcentroid"] = pos_on_det[:, 0]
        ref_tbl_adjusted["ycentroid"] = pos_on_det[:, 1]

        # update file header and save
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
            # plot final figures
            src_positions = list(zip(src_tbl['xcentroid'], src_tbl['ycentroid']))

            # match results within 1 fwhm radius for plot
            matches = imtrans.find_matches(src_tbl,
                                           ref_tbl_adjusted,
                                           wcsprm,
                                           threshold=match_radius)
            _, _, _, ref_positions_after, _, _, _ = matches

            if not self._silent:
                self._log.info("> Plot WCS calibration result")

            config['match_radius_px'] = match_radius
            self._plot_final_result(imgarr=imgarr, src_pos=src_positions,
                                    ref_pos=ref_positions_after,
                                    file_name=file_name, fig_path=plt_path_final,
                                    wcsprm=wcsprm, **config)

            _base_conf.clean_up(src_positions, ref_positions_after, matches)

        else:
            dic_rms = {"radius_px": None, "matches": None, "rms": None}
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

        self._converged = converged
        self._dic_rms = dic_rms

        _base_conf.clean_up(imgarr, dic_rms, converged,
                            src_tbl, ref_tbl, ref_catalog, _, src_cat_fname, ref_cat_fname,
                            kernel_fwhm, psf, segmap, extraction_result, report,
                            state)

    def get_wcs(self, source_df, ref_df, initial_wcsprm,
                match_radius=1, max_iter=6):
        """"""

        # make a copy to work with
        source_cat = source_df.copy()
        reference_cat = ref_df.copy()

        # source_cat = source_df.query('include_fwhm')

        # make sure the catalogs are sorted
        source_cat = source_cat.sort_values(by='mag', ascending=True)
        source_cat.reset_index(inplace=True, drop=True)

        reference_cat = reference_cat.sort_values(by='mag', ascending=True)
        reference_cat.reset_index(inplace=True, drop=True)

        self._log.info("> Run WCS calibration")

        Nobs = len(source_cat)  # number of detected sources
        Nref = len(reference_cat)  # number of reference stars
        source_ratio = (Nobs / Nref) * 100.  # source ratio in percent
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

        # dist_bin_size = 3
        # ang_bin_size = 0.3
        dist_bin_size = self._config['DISTANCE_BIN_SIZE'],
        ang_bin_size = self._config['ANG_BIN_SIZE'],

        # apply the found wcs to the original reference catalog
        ref_cat_orig_adjusted = reference_cat.copy()
        pos_on_det = initial_wcsprm.s2p(reference_cat[["RA", "DEC"]].values, 0)['pixcrd']
        ref_cat_orig_adjusted["xcentroid"] = pos_on_det[:, 0]
        ref_cat_orig_adjusted["ycentroid"] = pos_on_det[:, 1]

        # select initial reference sample based on the current best result
        initial_reference_cat_subset = ref_cat_orig_adjusted.head(num_rows)

        data = initial_reference_cat_subset['mag'].values
        # Calculate weights inversely proportional to the value
        # Adding a small constant to avoid division by zero for very small values
        weights = np.max(data) - data

        # Normalize weights
        weights /= weights.sum()

        # Draw a weighted random sample
        # sample = np.random.choice(len(initial_reference_cat_subset), size=num_samples, p=weights)

        # # Number of repetitions for drawing samples
        # num_repetitions = 4
        #
        # # Set up the plot
        # fig, axs = plt.subplots(num_repetitions, 2, figsize=(12, num_repetitions * 3))
        #
        # for i in range(num_repetitions):
        #     # Draw a weighted random sample
        #     sample = np.random.choice(data, size=num_samples, p=weights)
        #
        #     # Plot original data
        #     axs[i, 0].hist(data, bins=30, color='blue', alpha=0.7)
        #     axs[i, 0].set_title('Original Data Distribution')
        #     axs[i, 0].set_xlabel('Value')
        #     axs[i, 0].set_ylabel('Frequency')
        #
        #     # Plot sampled data
        #     axs[i, 1].hist(sample, bins=30, color='green', alpha=0.7)
        #     axs[i, 1].set_title('Weighted Sample Distribution')
        #     axs[i, 1].set_xlabel('Value')
        #     axs[i, 1].set_ylabel('Frequency')
        #
        # plt.tight_layout()
        # plt.show()

        current_reference_cat_subset = initial_reference_cat_subset
        current_wcsprm = initial_wcsprm
        current_threshold = match_radius
        # setup best values
        # best_rms = 999
        # best_score = 0
        best_Nobs = 0
        best_completeness = 0
        best_wcsprm = None

        if not self._silent:
            self._log.info(f'  Find scale, rotation, and offset.')

        while not has_converged:
            if progress == max_iter:
                break
            else:
                progress += 1

            if use_initial_ref_cat:
                current_reference_cat_subset = initial_reference_cat_subset
                if progress > 1:
                    current_reference_cat_subset = (initial_reference_cat_subset
                                                    .sample(int(len(initial_reference_cat_subset) * 0.95)))
                    # num_samples = int(len(initial_reference_cat_subset) * 0.95)
                    # sample = np.random.choice(len(initial_reference_cat_subset), size=num_samples, p=weights)
                    # current_reference_cat_subset = initial_reference_cat_subset.iloc[sample]

            # get wcs transformation
            try:
                tform_result = self._get_transformations(source_cat=source_cat,
                                                         ref_cat=current_reference_cat_subset,
                                                         wcsprm=current_wcsprm,
                                                         dist_bin_size=dist_bin_size,
                                                         ang_bin_size=ang_bin_size)
            except IndexError:
                current_reference_cat_subset = (initial_reference_cat_subset
                                                .sample(int(len(initial_reference_cat_subset) * 0.95)))
                # sample = np.random.choice(Nref, size=num_samples, p=weights)
                # current_reference_cat_subset = ref_cat_orig_adjusted.iloc[sample]
                current_wcsprm = initial_wcsprm
                continue

            # extract variables
            new_wcsprm, new_wcsprm_vals, tform_params, offsets, is_fine_tuned = tform_result

            # match catalogs
            matches = imtrans.find_matches(source_cat,
                                           current_reference_cat_subset,
                                           new_wcsprm,
                                           # threshold=1.
                                           threshold=current_threshold
                                           )
            source_cat_matched, ref_cat_matched, obs_xy, _, distances, _, _ = matches

            # check if any matches were found
            if obs_xy is None or len(obs_xy[:, 0]) == 0:
                if not self._silent:
                    self._log.info(f'  Try [ {progress}/{max_iter} ] No matches found. Trying again.')
                use_initial_ref_cat = True
                # best_wcsprm = initial_wcsprm
                current_wcsprm = initial_wcsprm
                continue

            # get convergence criteria
            Nobs_matched = len(obs_xy[:, 0])
            new_completeness = Nobs_matched / Nobs
            new_rms = np.sqrt(np.nanmean(np.square(distances))) / Nobs_matched
            new_score = Nobs_matched / new_rms

            message = f'  Try [ {progress}/{max_iter} ] ' \
                      f'Found matches: {Nobs_matched}/{Nobs} ({new_completeness * 100:.1f}%), ' \
                      f'Score: {new_score:.0f}, ' \
                      f'RMS: {new_rms:.3f}'
            # plt.figure()
            # plt.hist(np.sqrt(np.square(distances))/Nobs_matched, bins='auto')
            # plt.axvline(x=new_rms)
            # # plt.axvline(x=np.sqrt(np.nanmedian(np.square(distances))) / Nobs_matched)
            # plt.title('angles_cat')
            # plt.show()
            if not self._silent:
                self._log.info(message)

            # print(dist_bin_size, ang_bin_size)
            if new_rms < 1e-5:
                use_initial_ref_cat = True
                continue

            # delta_matches = Nobs_matched - best_Nobs
            # print("delta_matches", delta_matches)
            if Nobs_matched > 3:
                delta_matches = Nobs_matched - best_Nobs
                # print("delta_matches", delta_matches)
                # update best values
                update_cat = False
                if Nobs_matched > best_Nobs:
                    best_wcsprm = new_wcsprm
                    best_Nobs = Nobs_matched
                    # best_score = new_score
                    # best_rms = new_rms
                    best_completeness = new_completeness
                    # print(current_threshold)
                    update_cat = True
                    # update the wcsprm variable
                    current_wcsprm = new_wcsprm
                    dist_bin_size = 1
                    ang_bin_size = 0.1
                else:
                    ang_bin_size = np.max([ang_bin_size - 0.1, 0.1])
                    dist_bin_size = np.max([dist_bin_size - 1, 1])
                    # current_threshold = current_threshold + 0.1
                    current_wcsprm = best_wcsprm

                if (Nobs_matched >= self._config['MIN_SOURCE_NO_CONVERGENCE'] and new_completeness >
                        self._config['THRESHOLD_CONVERGENCE']):
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
                        self._log.info(f'    No improvement in the last 3 iterations. Using best result.')
                    has_converged = True
                    break

                if update_cat:
                    # Use the remaining detected sources and find close sources in the ref catalog
                    current_reference_cat_subset = self._update_ref_cat_subset(source_cat_matched,
                                                                               ref_cat_matched,
                                                                               source_cat, reference_cat,
                                                                               new_wcsprm, match_radius)
                    # current_wcsprm = initial_wcsprm
                    use_initial_ref_cat = False

            else:
                dist_bin_size = np.max([dist_bin_size - 1, 1])
                ang_bin_size = np.max([ang_bin_size - 0.1, 0.1])
                use_initial_ref_cat = True
                current_threshold = current_threshold * 1.5

        return best_wcsprm, has_converged

    # def get_wcs_working(self, source_df, ref_df, initial_wcsprm,
    #             match_radius=1, max_iter=6):
    #     """"""
    #
    #     # make a copy to work with
    #     source_cat = source_df.copy()
    #     reference_cat = ref_df.copy()
    #
    #     # make sure the catalogs are sorted
    #     source_cat = source_cat.sort_values(by='mag', ascending=True)
    #     source_cat.reset_index(inplace=True, drop=True)
    #
    #     reference_cat = reference_cat.sort_values(by='mag', ascending=True)
    #     reference_cat.reset_index(inplace=True, drop=True)
    #
    #     self._log.info("> Run WCS calibration")
    #
    #     # number of detected sources
    #     Nobs = len(source_cat)
    #     Nref = len(reference_cat)
    #     source_ratio = (Nobs / Nref) * 100.
    #     percent_to_select = np.ceil(source_ratio * 10)
    #     n_percent = percent_to_select if percent_to_select < 100 else 100
    #
    #     # Calculate the number of rows to select based on the percentage
    #     num_rows = int(Nref * (n_percent / 100))
    #
    #     # Select at least the first 100 entries
    #     num_rows = 100 if num_rows < 100 else num_rows
    #
    #     if not self._silent:
    #         self._log.info(f'  Number of detected sources: {Nobs}')
    #         self._log.info(f'  Number of reference sources (total): {Nref}')
    #         self._log.info(f'  Ratio of detected sources to reference sources: {source_ratio:.3f}%')
    #         # self._log.info(f'Select: {np.ceil(source_ratio * 10):.2f}%')
    #         self._log.info(f'  Initial number of reference sources selected: {num_rows}')
    #         self._log.info(f'  Source match radius = {match_radius:.2f} px')
    #
    #     # keep track of progress
    #     progress = 0
    #     no_changes = 0
    #     # match_radius = 1
    #     has_converged = False
    #     use_initial_ref_cat = False
    #
    #     # apply the found wcs to the original reference catalog
    #     ref_cat_orig_adjusted = reference_cat.copy()
    #     pos_on_det = initial_wcsprm.s2p(reference_cat[["RA", "DEC"]].values, 1)['pixcrd']
    #     ref_cat_orig_adjusted["xcentroid"] = pos_on_det[:, 0]
    #     ref_cat_orig_adjusted["ycentroid"] = pos_on_det[:, 1]
    #
    #     # select initial reference sample based on the current best result
    #     initial_reference_cat_subset = ref_cat_orig_adjusted.head(num_rows)
    #
    #     current_reference_cat_subset = initial_reference_cat_subset
    #     current_wcsprm = initial_wcsprm
    #
    #     # setup best values
    #     # best_rms = 1
    #     # best_score = 0
    #     best_Nobs = 0
    #     best_completeness = 0
    #     best_wcsprm = None
    #
    #     if not self._silent:
    #         self._log.info(f'  Find scale, rotation, and offset.')
    #
    #     while True:
    #         if progress == max_iter:
    #             break
    #         else:
    #             progress += 1
    #
    #         if use_initial_ref_cat:
    #             current_reference_cat_subset = initial_reference_cat_subset
    #             if progress > 1:
    #                 current_reference_cat_subset = (initial_reference_cat_subset
    #                                                 .sample(int(len(initial_reference_cat_subset) * 0.95)))
    #
    #         # get wcs transformation
    #         try:
    #             tform_result = self._get_transformations(source_cat=source_cat,
    #                                                      ref_cat=current_reference_cat_subset,
    #                                                      wcsprm=current_wcsprm)
    #         except IndexError:
    #             current_reference_cat_subset = (initial_reference_cat_subset
    #                                             .sample(int(len(initial_reference_cat_subset) * 0.95)))
    #             current_wcsprm = initial_wcsprm
    #             continue
    #
    #         # extract variables
    #         new_wcsprm, new_wcsprm_vals, tform_params, offsets, is_fine_tuned = tform_result
    #
    #         # match catalogs
    #         matches = imtrans.find_matches(source_cat,
    #                                        current_reference_cat_subset,
    #                                        new_wcsprm,
    #                                        threshold=match_radius)
    #         source_cat_matched, ref_cat_matched, obs_xy, _, distances, _, _ = matches
    #
    #         # check if any matches were found
    #         if obs_xy is None or len(obs_xy[:, 0]) == 0:
    #             if not self._silent:
    #                 self._log.info(f'  Try [ {progress}/{max_iter} ] No matches found. Trying again.')
    #             use_initial_ref_cat = True
    #             best_wcsprm = initial_wcsprm
    #             continue
    #
    #         # get convergence criteria
    #         Nobs_matched = len(obs_xy[:, 0])
    #         new_completeness = Nobs_matched / Nobs
    #         new_rms = np.sqrt(np.nanmean(np.square(distances))) / Nobs_matched
    #         new_score = Nobs_matched / new_rms
    #
    #         message = f'  Try [ {progress}/{max_iter} ] ' \
    #                   f'Found matches: {Nobs_matched}/{Nobs} ({new_completeness * 100:.1f}%), ' \
    #                   f'Score: {new_score:.0f}, ' \
    #                   f'RMS: {new_rms:.3f}'
    #
    #         if not self._silent:
    #             self._log.info(message)
    #
    #         if Nobs_matched > 3:
    #             delta_matches = Nobs_matched - best_Nobs
    #             print("delta_matches", delta_matches)
    #             # update best values
    #             if Nobs_matched > best_Nobs:
    #                 best_wcsprm = new_wcsprm
    #                 best_Nobs = Nobs_matched
    #                 # best_score = new_score
    #                 # best_rms = new_rms
    #                 best_completeness = new_completeness
    #
    #             # the best case, all sources were matched
    #             if new_completeness == 1.:
    #                 has_converged = True
    #                 break
    #
    #             if Nobs_matched >= self._config['MIN_SOURCE_NO_CONVERGENCE'] and delta_matches == 0.:
    #                 if best_completeness > self._config['THRESHOLD_CONVERGENCE'] or best_completeness <= \
    #                         self._config['THRESHOLD_CONVERGENCE']:
    #                     has_converged = True
    #                     break
    #             elif Nobs_matched >= self._config['MIN_SOURCE_NO_CONVERGENCE'] and delta_matches < 0:
    #                 no_changes += 1
    #
    #             if no_changes == 3:
    #                 if not self._silent:
    #                     self._log.info(f'    No improvement in the last 3 iterations. Using best result.')
    #                 has_converged = True
    #                 break
    #             # Use the remaining detected sources and find close sources in the ref catalog
    #             current_reference_cat_subset = self._update_ref_cat_subset(source_cat_matched,
    #                                                                        ref_cat_matched,
    #                                                                        source_cat, reference_cat,
    #                                                                        new_wcsprm)
    #             # current_wcsprm = initial_wcsprm
    #             use_initial_ref_cat = False
    #             # update the wcsprm variable
    #             current_wcsprm = new_wcsprm
    #
    #         else:
    #             use_initial_ref_cat = True
    #
    #     return best_wcsprm, has_converged

    def _update_ref_cat_subset(self, source_cat_matched, ref_cat_matched,
                               source_cat_original, ref_cat_original, wcsprm, radius):
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
                                                 radius=radius, N=3)

        #
        ref_cat_orig_matched = self.match_catalogs(ref_cat_matched,
                                                   ref_cat_original,
                                                   ra_dec_tol=1e-3, N=1)

        ref_catalog_subset = pd.concat([ref_cat_orig_matched, ref_sources_to_add],
                                       ignore_index=True)

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
            distances, indices = catalog1_tree.query(catalog2[['RA', 'DEC']].values, k=N)

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

    def _get_transformations(self, source_cat: pd.DataFrame, ref_cat: pd.DataFrame, wcsprm,
                             dist_bin_size=1, ang_bin_size=0.2):
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
                                                                  # dist_bin_size=self._config['DISTANCE_BIN_SIZE'],
                                                                  # ang_bin_size=self._config['ANG_BIN_SIZE'],
                                                                  dist_bin_size=dist_bin_size,
                                                                  ang_bin_size=ang_bin_size
                                                                  )

        if self._xy_transformation:
            # if not self._silent:
            #     self._log.info("  > Determine x, y offset")
            self._log.debug("  > Determine x, y offset")
            wcsprm, offsets, _, _ = imtrans.get_offset_with_orientation(observation=source_cat,
                                                                        catalog=ref_cat,
                                                                        wcsprm=wcsprm)

        wcsprm, scales = imtrans.translate_wcsprm(wcsprm=wcsprm)

        # update reference catalog
        ref_cat_corrected = ref_cat.copy()
        pos_on_det = wcsprm.s2p(ref_cat[["RA", "DEC"]].values, 0)['pixcrd']
        ref_cat_corrected["xcentroid"] = pos_on_det[:, 0]
        ref_cat_corrected["ycentroid"] = pos_on_det[:, 1]

        # correct subpixel error
        wcsprm, wcsprm_vals, fine_tune_status = self._correct_subpixel_error(source_cat=source_cat,
                                                                             ref_cat=ref_cat_corrected,
                                                                             wcsprm=wcsprm)

        _base_conf.clean_up(source_cat, ref_cat, ref_cat_corrected)
        return wcsprm, wcsprm_vals, tform_params, offsets, fine_tune_status

    def _compare_results(self, imgarr, hdr, wcsprm):
        """Compare WCS difference before and after transformation"""

        wcsprm_original = WCS(hdr).wcs

        if not self._silent:
            self._log.info("> Compared to the input the Wcs was changed by: ")

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
                                      wcsprm_original.get_pc() @ wcsprm_original.get_cdelt()) / np.pi * 180.

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
            lis = [2, 3, 5, 10, 20, 1, 0.5, 0.1]  # , 0.5, 0.25, 0.2, 0.1
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

        _base_conf.clean_up(source_cat, ref_cat, matches)

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

            rms = np.sqrt(np.nanmedian(np.square(distances))) / len_obs_x

            result_array[i, :] = [len_obs_x, len_obs_x / N_obs, r, r * px_scale, rms, rms * px_scale]

        # Calculate the differences between consecutive numbers of sources and pixel values
        diff_sources = np.diff(result_array[:, 0])
        diff_pixels = np.diff(result_array[:, 2])

        # Calculate the rate of change
        rate_of_change = diff_sources / diff_pixels
        # plt.figure()
        # plt.plot(rate_of_change)
        # plt.show()
        # Find the point where the rate of change drops below a certain threshold
        threshold = 1.
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
                # if yes, then replace the wcs scale with the pixel scale information
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
            # If the sky position is not found check header for RA and DEC information
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

            # get y-axis length as limit
            matrix = wcsprm.pc * wcsprm.cdelt[:, np.newaxis]

            # Rotation angle
            theta = np.rad2deg(-np.arctan2(matrix[1, 0], matrix[1, 1]))

            # adding report
            hdr_file['PIXSCALE'] = (report["pix_scale"], 'Average pixel scale [arcsec/pixel]')
            hdr_file['SCALEX'] = (report["scale_x"], 'Pixel scale in X [arcsec/pixel]')
            hdr_file['SCALEY'] = (report["scale_y"], 'Pixel scale in Y [arcsec/pixel]')
            hdr_file['DETROTANG'] = (theta, 'Detection rotation angel [deg]')
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

    def _plot_final_result(self, imgarr, src_pos, ref_pos,
                           file_name, fig_path, wcsprm, cmap='Greys', **config):
        """Plot before and after images"""

        # load fits file
        bkg_fname = config['bkg_fname']
        with fits.open(f'{bkg_fname[0]}.fits') as hdul:
            hdul.verify('fix')
            bkg_background = hdul[0].data.astype('float32')

        # subtract the background from the image
        imgarr -= bkg_background

        # get y-axis length as limit
        matrix = wcsprm.pc * wcsprm.cdelt[:, np.newaxis]

        # Scale
        scale_x = np.sqrt(matrix[0, 0] ** 2 + matrix[0, 1] ** 2)

        # Rotation angle
        theta = -np.arctan2(matrix[1, 0], matrix[1, 1])

        # get x, y-axis length as limits
        plim = imgarr.shape[0]  # Image height
        xplim = imgarr.shape[1]  # Image width

        # # get the scale and angle
        # scale = wcsprm.cdelt[0]
        # angle = np.arccos(wcsprm.pc[0][0])
        # theta = angle / np.pi * 180.
        # if wcsprm.pc[0][0] < 0.:
        #     theta *= -1.

        obs_xy = np.array(src_pos)
        cat_xy = ref_pos

        # calculate the distances
        dist_xy = np.sqrt((obs_xy[:, 0] - cat_xy[:, 0, np.newaxis]) ** 2
                          + (obs_xy[:, 1] - cat_xy[:, 1, np.newaxis]) ** 2)

        idx_arr = np.where(dist_xy == np.min(dist_xy, axis=0))
        min_dist_xy = dist_xy[idx_arr]
        del dist_xy

        mask = min_dist_xy > config['match_radius_px']
        obs_idx = idx_arr[1][mask]
        obs_xy = obs_xy[obs_idx, :]

        # adjust position offset
        adjusted_ref_pos = [(x + 0.5, y + 0.5) for x, y in ref_pos]
        apertures_catalog = CircularAperture(adjusted_ref_pos, r=10.)

        # Define normalization and the colormap
        vmin = np.percentile(imgarr, 50)
        vmax = np.percentile(imgarr, 99.)
        nm = mpl.colors.Normalize(vmin=vmin, vmax=vmax, clip=True)

        figsize = self._config['FIG_SIZE']
        fig = plt.figure(figsize=figsize)
        fig.canvas.manager.set_window_title(f'Input for {file_name}')

        gs = gridspec.GridSpec(1, 1)
        ax = fig.add_subplot(gs[0, 0])

        im = ax.imshow(imgarr, origin='lower', norm=nm, cmap=cmap)

        if list(obs_xy):
            # adjust position offset
            adjusted_src_pos = [(x + 0.5, y + 0.5) for x, y in obs_xy]
            apertures = CircularAperture(adjusted_src_pos, r=10.)
            apertures.plot(axes=ax, **{'color': 'red', 'lw': 1.25, 'alpha': 0.85})

        apertures_catalog.plot(axes=ax, **{'color': 'green', 'lw': 1.25, 'alpha': 0.85})

        # add image scale
        self._add_image_scale(ax=ax, plim_x=xplim, plim_y=plim, scale=scale_x)

        # Add compass
        self._add_compass(ax=ax, image_shape=imgarr.shape,
                          scale_arrow=self._config['ARROW_LENGTH'], theta_rad=theta,
                          color='k')
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

        ax.set_ylabel(r'Pixel y direction')
        ax.set_xlabel(r'Pixel x direction')

        fig.tight_layout(h_pad=0.4, w_pad=0.5, pad=1.)

        # Save the plot
        fname = f"plot.astrometry.final_{file_name}"
        fname = os.path.join(fig_path, fname)

        self._save_plot(fname)

        if self._plot_images:
            plt.show()

        plt.close(fig=fig)

    def _add_image_scale(self, ax, plim_x, plim_y, scale):
        """Make a scale for the image"""

        length = self._config['LINE_LENGTH']
        xlemark = length / (scale * 60.)
        xstmark = 0.025 * plim_x
        xenmark = xstmark + xlemark
        ystmark = plim_y * 0.025

        ax.plot([xstmark, xenmark], [ystmark, ystmark], color='k', lw=1.25)
        ax.annotate(text=r"{0:.0f}'".format(length),
                    xy=(((xstmark + xenmark) / 2.), ystmark + 5), xycoords='data',
                    textcoords='data', ha='center', c='k')

        return ax

    @staticmethod
    def _add_compass(ax: plt.Axes, image_shape, scale_arrow, theta_rad: float, color: str = 'black'):
        """Make a Ds9 like compass for image"""

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

    def _add_compass_old(self, ax, plim_x, plim_y, theta, color='black'):
        """Make a Ds9 like compass for image"""
        #
        # x0 = plim * 0.955
        # y0 = plim * 0.955
        # if self._telescope == 'DK-1.54':
        #     x0 = plim * 0.955
        #     y0 = plim * 0.025
        #     if theta < 0.:
        #         x0 = plim * 0.025
        #         y0 = plim * 0.955
        # if self._telescope == 'Takahashi FSQ 85':
        #     x0 = plim * 1.25
        #     y0 = plim * 0.025
        #     if theta < 0.:
        #         x0 = plim * 0.025
        #         y0 = plim * 0.955

        larr = self._config['ARROW_LENGTH'] * plim_y  # 15% of the image size
        # Default margins
        margin_x = plim_x * 0.04
        margin_y = plim_y * 0.04

        # Determine x0 and y0 without subtracting larr
        if 0 <= theta < np.pi / 2:
            # First quadrant
            x0 = plim_x - margin_x  # Subtract larr to ensure the compass fits within the right edge
            y0 = margin_y
        elif np.pi / 2 <= theta < np.pi:
            # Second quadrant
            x0 = plim_x - margin_x - larr  # No need to subtract larr because the compass orients towards the top left
            y0 = plim_y - margin_y
        elif -np.pi / 2 <= theta < 0:
            # Fourth quadrant
            x0 = plim_x - margin_x - larr  # Subtract larr to ensure the compass fits within the right edge
            y0 = margin_y
        else:
            # Third quadrant
            x0 = plim_x - margin_x  # No need to subtract larr because the compass orients towards the bottom left
            y0 = plim_y - margin_y

        # East
        theta = theta + 90. * np.pi / 180.
        x1, y1 = larr * np.cos(theta), larr * np.sin(theta)
        ax.arrow(x0, y0, x1, y1, head_width=10, color=color, zorder=2)
        ax.text(x0 + 1.25 * x1, y0 + 1.25 * y1, 'N', color=color)

        # North
        theta = theta + 90. * np.pi / 180.
        x1, y1 = larr * np.cos(theta), larr * np.sin(theta)
        ax.arrow(x0, y0, x1, y1, head_width=10, color=color, zorder=2)
        ax.text(x0 + 1.25 * x1, y0 + 1.25 * y1, 'E', color=color)

        return ax

    def _save_plot(self, fname: str):
        """Save plots"""

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
