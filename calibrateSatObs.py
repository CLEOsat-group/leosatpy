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
import gc
import math
# STDLIB
import os
import random
import sys
import logging
import warnings
import time
from datetime import timedelta, datetime, timezone
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
from astropy.utils.exceptions import (AstropyUserWarning, AstropyWarning)
from astropy.wcs import WCS
from astropy.wcs import utils
from astropy.coordinates import (FK5, ICRS, SkyCoord)
from astropy.time import Time
from astropy.visualization import (LinearStretch, LogStretch, SqrtStretch)
from astropy.visualization.mpl_normalize import ImageNormalize
from photutils.aperture import CircularAperture

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

    # matplotlib parameter
    mpl.use('Qt5Agg')
    mpl.rc("lines", linewidth=1.2)
    mpl.rc('figure', dpi=150, facecolor='w', edgecolor='k')
    mpl.rc('text.latex', preamble=r'\usepackage{sfmath}')
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    # mpl.rcParams['font.family'] = 'Arial'

# Project modules
from utils.arguments import ParseArguments
from utils.dataset import DataSet
from utils.tables import ObsTables
import utils.sources as sext
import utils.transformations as imtrans
import config.base_conf as _base_conf

# -----------------------------------------------------------------------------

""" Meta-info """
__author__ = "Christian Adam"
__copyright__ = 'Copyright 2021, UA, LEOSat observations'
__credits__ = ["Christian Adam, Eduardo Unda-Sanzana, Jeremy Tregloan-Reed"]
__license__ = "Free"
__version__ = "0.1.0"
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
        self._load_config()

        self._obsTable = ObsTables(config=self._config)
        self._obsTable.load_obs_table()

        if not silent:
            self._log.info("> Search input and prepare data")
        if verbose:
            self._log.debug("  > Check input argument(s)")

        # prepare dataset from input argument
        ds = DataSet(input_args=self._input_path,
                     prog_typ='calibWCS',
                     log=self._log, log_level=self._log_level)
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
                        with open(fail_fname, "wa", encoding="utf8") as file:
                            file.write('{}\t{}'.format(self._telescope, file_df["input"]))

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

        Create required folder and run full calibration procedure.

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

        # get wcs
        wcsprm = WCS(hdr).wcs
        wcsprm_original = WCS(hdr).wcs

        # run checks on wcs
        self._check_image_wcs(wcsprm=wcsprm, hdr=hdr,
                              obsparams=obsparams,
                              radius=self._radius,
                              ignore_header_rot=self._ignore_header_rot)
        wcsprm = self._wcsprm

        # extract sources on detector
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
                             bkg_fname=(bkg_fname, bkg_fname_short),
                             _force_extract=self._force_extract,
                             _force_download=self._force_download,
                             _plot_images=self._plot_images)

        # add to configuration
        config.update(params_to_add)
        for key, value in self._config.items():
            config[key] = value

        extraction_result, state = sext.get_src_and_cat_info(fbase, cat_path,
                                                             imgarr, hdr, wcsprm, catalog,
                                                             has_trail=False, mode='astro',
                                                             silent=self._silent,
                                                             **config)
        # unpack extraction result tuple
        (src_tbl, ref_tbl, ref_catalog, _, _,
         src_cat_fname, ref_cat_fname, _,
         kernel_fwhm, psf, segmap, _) = extraction_result

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
        ref_positions_before = wcsprm.s2p(ref_tbl[["RA", "DEC"]].values, 1)['pixcrd']

        converged = False
        dic_rms = {}
        wcs_pixscale = np.nan
        rotation_angle = np.nan
        max_iter = 50
        max_ref_src_no = self._config['REF_SOURCES_MAX_NO']

        # select the brightest reference sources
        idx = -1 * max_ref_src_no
        ref_tbl_truncated = ref_tbl[idx:] if len(ref_tbl) > max_ref_src_no else ref_tbl
        ref_tbl_truncated.reset_index(inplace=True, drop=True)

        if 'include_fwhm' in src_tbl.columns:
            idx = src_tbl['include_fwhm']
        else:
            idx = [True] * len(src_tbl)
        src_tbl = src_tbl[idx]

        n_ref_to_select = math.ceil(len(src_tbl) / self._config['THRESHOLD_CONVERGENCE'])
        for i in range(max_iter):
            idx = -1 * n_ref_to_select
            ref_tbl_select = ref_tbl_truncated[idx:]

            # get reference positions before the transformation
            ref_positions_before = wcsprm.s2p(ref_tbl_select[["RA", "DEC"]].values, 1)['pixcrd']
            self._log.info("> Try [ %d/%d ] Find scale, rotation and offset "
                           "using %d reference sources" % (i+1, max_iter, len(ref_tbl_select)))
            # get the scale, rotation and offset
            wcsprm, fine_tune_status = self._get_transformations(source_cat=src_tbl,
                                                                 ref_cat=ref_tbl_select,
                                                                 wcsprm=wcsprm)

            # move scaling to cdelt, out of the pc matrix
            wcsprm, scales = imtrans.translate_wcsprm(wcsprm=wcsprm)

            # WCS difference before and after
            wcs_pixscale, rotation_angle = self._compare_results(imgarr,
                                                                 hdr,
                                                                 wcsprm,
                                                                 wcsprm_original,
                                                                 scales)

            # check convergence
            if not self._silent:
                self._log.info("> Evaluate goodness of transformation")
            converged, dic_rms = self._determine_if_fit_converged(ref_tbl_select, src_tbl, wcsprm,
                                                                  imgarr.shape[0], imgarr.shape[1],
                                                                  kernel_fwhm[0])

            if converged and fine_tune_status:
                self._log.info(f"  Convergence test and refinement was SUCCESSFUL")
                break
            else:
                self._log.warning(f"  Convergence test and refinement has FAILED "
                                  f" - increasing number of reference sources for comparison")

                if n_ref_to_select == len(ref_tbl_truncated):
                    break
            n_ref_to_select += 10

        # update report dict
        report["converged"] = converged
        report["catalog"] = ref_catalog
        report["fwhm"] = kernel_fwhm[0]
        report["e_fwhm"] = kernel_fwhm[1]
        report["matches"] = dic_rms["matches"]
        report["match_radius"] = dic_rms["radius_px"]
        report["match_rms"] = dic_rms["rms"]
        report["pix_scale"] = np.mean(abs(wcs_pixscale)) * 3600. if wcs_pixscale is not np.nan else np.nan
        report["scale_x"] = abs(wcs_pixscale[1]) * 3600.
        report["scale_y"] = abs(wcs_pixscale[0]) * 3600.
        report["det_rotang"] = rotation_angle if rotation_angle is not np.nan else np.nan

        # match results within 1 pixel radius for plot
        matches = imtrans.find_matches(src_tbl,
                                       ref_tbl,
                                       wcsprm,
                                       threshold=1)
        _, _, _, ref_positions_after, _, _, _ = matches
        # update file header and save
        if converged:
            self._write_wcs_to_hdr(original_filename=abs_file_path,
                                   filename_base=fbase,
                                   dest=cal_path,
                                   wcsprm=wcsprm,
                                   report=report, hdul_idx=hdu_idx)
            if not self._silent:
                self._log.info("> Save source and reference catalog.")
            sext.save_catalog(cat=src_tbl, wcsprm=wcsprm, out_name=src_cat_fname,
                              kernel_fwhm=kernel_fwhm)
            sext.save_catalog(cat=ref_tbl, wcsprm=wcsprm, out_name=ref_cat_fname,
                              mode='ref_astro', catalog=ref_catalog)
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
        # ref_positions_after = wcsprm.s2p(ref_tbl[["RA", "DEC"]].values, 1)['pixcrd']

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

    def _get_transformations(self, source_cat: pd.DataFrame, ref_cat, wcsprm):
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
        # print(source_cat[source_cat['include_fwhm']])
        source_cat = source_cat[source_cat['include_fwhm']]

        if self._rot_scale:
            if not self._silent:
                self._log.info("  > Determine pixel scale and rotation angle")
            wcsprm = imtrans.get_scale_and_rotation(observation=source_cat,
                                                    catalog=ref_cat,
                                                    wcsprm=wcsprm,
                                                    scale_guessed=PIXSCALE_UNCLEAR,
                                                    dist_bin_size=self._config['DISTANCE_BIN_SIZE'],
                                                    ang_bin_size=self._config['ANG_BIN_SIZE'],
                                                    silent=self._silent)

        if self._xy_transformation:
            if not self._silent:
                self._log.info("  > Determine x, y offset")
            wcsprm, _, _ = imtrans.get_offset_with_orientation(observation=source_cat,
                                                               catalog=ref_cat,
                                                               wcsprm=wcsprm,
                                                               silent=self._silent)

        # correct subpixel error
        wcsprm, fine_tune_status = self._correct_subpixel_error(source_cat=source_cat,
                                                                ref_cat=ref_cat,
                                                                wcsprm=wcsprm)

        del source_cat, ref_cat, _
        gc.collect()

        return wcsprm, fine_tune_status

    def _compare_results(self, imgarr, hdr, wcsprm, wcsprm_original, scales):
        """Compare WCS difference before and after transformation"""

        if not self._silent:
            self._log.info("> Compared to the input the Wcs was changed by: ")
        scales_original = utils.proj_plane_pixel_scales(WCS(hdr))

        if not self._silent:
            self._log.info("  WCS got scaled by a factor of {} in x direction and "
                           "{} in y direction".format(scales[0] / scales_original[0],
                                                      scales[1] / scales_original[1]))

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
        else:
            text = "clockwise"
        if not self._silent:
            self._log.info(f"  Rotation of WCS by an angle of {rotation_angle} deg " + text)
        old_central_pixel = [imgarr.shape[0] / 2, imgarr.shape[1] / 2]
        if not self._silent:
            self._log.info("  x offset: {} px, y offset: {} px ".format(wcsprm.crpix[0] - old_central_pixel[0],
                                                                        wcsprm.crpix[1] - old_central_pixel[1]))

        pc = wcsprm.get_pc()
        cdelt = wcsprm.get_cdelt()
        wcs_pixscale = (pc @ cdelt)

        return wcs_pixscale, rotation_angle

    def _correct_subpixel_error(self, source_cat, ref_cat, wcsprm):
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

        compare_threshold = 7
        if self._high_res:
            compare_threshold = 50

        # find matches
        matches = imtrans.find_matches(source_cat,
                                       ref_cat,
                                       wcsprm,
                                       threshold=compare_threshold)
        _, _, _, _, distances, best_score, state = matches

        fine_transformation_success = False
        if self._fine_transformation:
            if not self._silent:
                self._log.info("  > Refine scale and rotation")
            lis = [2, 3, 5, 8, 10, 6, 4, 20, 2, 1, 0.5, 0.25, 0.2]
            if self._high_res:
                lis = [200, 300, 100, 150, 80, 40, 70, 20, 100, 30, 9, 5]
            skip_rot_scale = True
            for i in lis:
                wcsprm_new, score = imtrans.fine_transformation(source_cat,
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
                    fine_transformation_success = True
            pass_str = _base_conf.BCOLORS.PASS + "PASS" + _base_conf.BCOLORS.ENDC
            fail_str = _base_conf.BCOLORS.FAIL + "FAIL (result will be discarded)" + _base_conf.BCOLORS.ENDC
            if not fine_transformation_success:
                if not self._silent:
                    self._log.info("    Status: " + fail_str)
            else:
                if not self._silent:
                    self._log.info("    Status: " + pass_str)

        del source_cat, ref_cat, matches
        gc.collect()

        return wcsprm, fine_transformation_success

    def _determine_if_fit_converged(self, catalog, observation, wcsprm, NAXIS1, NAXIS2, fwhm):
        """Cut off criteria for deciding if the astrometry fit worked.
        
        Parameters
        ----------
        catalog :
        observation :
        wcsprm :
        NAXIS1 :
        NAXIS2 :

        Returns
        -------
        converged: bool
            Returns whether the fit converged or not
        """
        converged = False

        N_obs = observation.shape[0]

        catalog_on_sensor = wcsprm.s2p(catalog[["RA", "DEC"]], 1)['pixcrd']
        cat_x = np.array([catalog_on_sensor[:, 0]])
        cat_y = np.array([catalog_on_sensor[:, 1]])
        mask = (cat_x >= 0) & (cat_x <= NAXIS1) & (cat_y >= 0) & (cat_y <= NAXIS2)
        N_cat = mask.sum()

        N = min([N_obs, N_cat])

        # get pixel scale
        px_scale = wcsprm.cdelt[0] * 3600.  # in arcsec

        # get radii list
        r = round(2 * fwhm, 1)
        radii_list = imtrans.frange(0.1, r, 0.1)

        dic_rms = {"radius_px": None, "matches": None, "rms": None}
        for r in radii_list:
            matches = imtrans.find_matches(observation,
                                           catalog, wcsprm,
                                           threshold=r)
            _, _, obs_xy, _, distances, _, _ = matches
            len_obs_x = len(obs_xy[:, 0])

            rms = np.sqrt(np.mean(np.square(distances))) / len_obs_x
            self._log.debug("  Within {} pixel or {:.3g} arcsec {} sources where matched. "
                            "The rms is {:.3g} pixel or "
                            "{:.3g} arcsec".format(r,
                                                   px_scale * r,
                                                   len_obs_x,
                                                   rms,
                                                   rms * px_scale))
            self._log.debug("  Conditions for convergence of fit: at least 3 matches within "
                            "{}px and at least 0.25 * minimum ( observed sources, catalog sources in fov) "
                            "matches with the same radius".format(len_obs_x))

            if len_obs_x >= 3 and len_obs_x >= self._config['THRESHOLD_CONVERGENCE'] * N:
                converged = True
                dic_rms = {"radius_px": r, "matches": len_obs_x, "rms": rms}
                if not self._silent:
                    self._log.info("  Within {} pixel or {:.3g} arcsec {} sources where matched. "
                                   "The rms is {:.3g} pixel or "
                                   "{:.3g} arcsec".format(r, px_scale * r,
                                                          len_obs_x,
                                                          rms, rms * px_scale))
                break

        return converged, dic_rms

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
                anyeq = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg,
                                 frame=FK5, equinox=Time(equinox, format='jyear', scale='utc'))
                coo = anyeq.transform_to(ICRS)
                ra_deg = coo.ra.deg
                dec_deg = coo.dec.deg
                wcsprm.equinox = 2000.0

        wcs = WCS(wcsprm.to_header())
        hdr.update(wcs.to_header())

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
        pc = wcsprm.get_pc()
        cdelt = wcsprm.get_cdelt()
        wcs_pixscale = (pc @ cdelt)

        test_wcs1 = ((np.abs(wcs_pixscale[0])) < 1e-7 or (np.abs(wcs_pixscale[1])) < 1e-7 or
                     (np.abs(wcs_pixscale[0])) > 5e-3 or (np.abs(wcs_pixscale[1])) > 5e-3)
        if test_wcs1:
            if not self._silent:
                log.info("  Pixelscale is unrealistic. Will guess")
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

        # test if found pixel scale makes sense size +- 10%
        test_wcs2 = (120. > x_size > 0.5) & \
                    (120. > y_size > 0.5)

        if test_wcs2:
            pc = wcsprm.get_pc()
            cdelt = wcsprm.get_cdelt()
            wcs_pixscale = (pc @ cdelt)
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

                    # rotation = np.deg2rad(90. - wcsprm.crota[1])
                    # rot = imtrans.rotation_matrix(rotation)
                    # wcsprm = imtrans.rotate(wcsprm, np.array(rot))
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
            fov_radius *= 0.5

        if not self._silent:
            log.info("  Using field-of-view radius: {:.3g} deg".format(fov_radius))

        self._wcsprm = wcsprm
        self._fov_radius = fov_radius
        self._wcs_check = (INCREASE_FOV_FLAG, PIXSCALE_UNCLEAR)

    def _write_wcs_to_hdr(self, original_filename, filename_base,
                          dest, wcsprm, report, hdul_idx=0):
        """Update the header of the fits file itself.
        Parameters
        ----------
        original_filename: str
            Original filename of the fits file.
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

            hdul.writeto(dest / f"{filename_base}_cal.fits", overwrite=True)

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

        bkg_fname = config['bkg_fname']
        # load fits file
        with fits.open(f'{bkg_fname[0]}.fits') as hdul:
            hdul.verify('fix')
            bkg_background = hdul[0].data.astype('float32')

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

        apertures = CircularAperture(src_pos, r=7)
        apertures_catalog_before = CircularAperture(ref_pos_before, r=10)
        apertures_catalog_after = CircularAperture(ref_pos_after, r=10.)

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

            minorLocatorx = AutoMinorLocator(5)
            minorLocatory = AutoMinorLocator(5)

            ax.xaxis.set_minor_locator(minorLocatorx)
            ax.yaxis.set_minor_locator(minorLocatory)

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
