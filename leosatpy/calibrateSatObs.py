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

# astropy
from astropy.io import fits
from astropy import units as u
from astropy.utils.exceptions import AstropyUserWarning
from astropy.wcs import WCS
from astropy.coordinates import (FK5, ICRS, SkyCoord)
from astropy.time import Time

# photutils
from photutils.aperture import CircularAperture

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
                    progress = f"{row_index + 1}/{len(files)}"
                    self._run_calibration_single(src_path, file_df, progress,
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

    def _run_calibration_single(self, file_src_path, file_df, progress, hdu_idx=0):
        """Run astrometric calibration on a given dataset.

        Create the required folder and run full calibration procedure.

        """

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
        init_wcs = self._wcsprm
        init_wcsprm = self._wcsprm.wcs

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
                             ref_cat_mag_lim=self._config['REF_CATALOG_MAG_LIM'],
                             bkg_fname=(bkg_fname, bkg_fname_short),
                             _force_extract=self._force_extract,
                             _force_download=self._force_download,
                             _plot_images=self._plot_images)

        # add to configuration
        config.update(params_to_add)
        for key, value in self._config.items():
            config[key] = value

        # get detector saturation limit
        sat_lim = config['saturation_limit']
        if isinstance(config['saturation_limit'], str) and config['saturation_limit'] in hdr:
            sat_lim = hdr[obsparams['saturation_limit']]
        config['sat_lim'] = sat_lim

        # extract sources on detector
        extraction_result, state, _ = sext.get_src_and_cat_info(fbase, cat_path,
                                                                imgarr, hdr, init_wcsprm,
                                                                silent=self._silent,
                                                                **config)
        # unpack extraction result tuple
        (src_tbl_raw, ref_tbl, ref_catalog, src_cat_fname, ref_cat_fname,
         kernel_fwhm, psf, segmap, _) = extraction_result

        # check execution state and update result table
        if not state or len(src_tbl_raw) == 0:
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

        # use only the entries with good fwhm values
        src_tbl = src_tbl_raw.query('include_fwhm')
        if len(src_tbl) < 5:
            src_tbl = src_tbl_raw

        # set the match radius either based on the fwhm or as absolute value
        match_radius = self._config['MATCH_RADIUS']
        match_radius = kernel_fwhm[0] if match_radius == 'fwhm' else match_radius

        # get the default PC matrix
        self._config['pc_matrix'] = obsparams['pc_matrix']

        # initialize find
        get_wcs = imtrans.FindWCS(src_tbl, ref_tbl, self._config, self._log)

        # run wcs analysis
        wcs_res = get_wcs.find_wcs(init_wcs, match_radius, imgarr)
        has_solution, best_wcsprm, dict_rms = wcs_res

        # check result sate
        status_str = _base_conf.BCOLORS.PASS + "PASS" + _base_conf.BCOLORS.ENDC
        if not has_solution or best_wcsprm is None:
            status_str = _base_conf.BCOLORS.FAIL + "FAIL" + _base_conf.BCOLORS.ENDC
        else:
            # Move scaling to cdelt and out of the pc matrix
            best_wcsprm, scales = imtrans.translate_wcsprm(wcsprm=best_wcsprm)
            wcsprm = best_wcsprm

        if not self._silent:
            self._log.info("  ==> Calibration status: " + status_str)

        # get updated reference catalog positions
        ref_tbl_adjusted = imtrans.update_catalog_positions(ref_tbl, wcsprm, origin=0)

        # update file header and save
        if has_solution:

            # get pixel scale
            wcs_pixscale = imtrans.get_wcs_pixelscale(wcsprm)

            # update report dict
            report["converged"] = has_solution
            report["catalog"] = ref_catalog
            report["fwhm"] = kernel_fwhm[0]
            report["e_fwhm"] = kernel_fwhm[1]
            report["matches"] = int(dict_rms["matches"])
            report["match_radius"] = dict_rms["radius_px"]
            report["match_rms"] = dict_rms["rms"]
            report["pix_scale"] = np.mean(abs(wcs_pixscale)) * 3600. if wcs_pixscale is not np.nan else np.nan
            report["scale_x"] = abs(wcs_pixscale[1]) * 3600.
            report["scale_y"] = abs(wcs_pixscale[0]) * 3600.

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
                                           threshold=dict_rms["radius_px"])
            _, _, _, ref_positions_after, _, _, _ = matches

            if not self._silent:
                self._log.info("> Plot WCS calibration result")

            config['match_radius_px'] = dict_rms["radius_px"]
            self._plot_final_result(imgarr=imgarr, src_pos=src_positions,
                                    ref_pos=ref_positions_after,
                                    file_name=file_name, fig_path=plt_path_final,
                                    wcsprm=wcsprm, **config)

            _base_conf.clean_up(src_positions, ref_positions_after, matches)

        else:
            # dic_rms = {"radius_px": None, "matches": None, "rms": None}
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

        self._converged = has_solution
        self._dic_rms = dict_rms

        _base_conf.clean_up(imgarr, dict_rms, has_solution,
                            src_tbl, ref_tbl, ref_catalog, _, src_cat_fname, ref_cat_fname,
                            kernel_fwhm, psf, segmap, extraction_result, report,
                            state)

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

        if self._telescope in ['DK-1.54', 'PlaneWave CDK24']:
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
        wcs_pixscale = imtrans.get_wcs_pixelscale(wcsprm)

        test_wcs1 = ((np.abs(wcs_pixscale[0])) < 1e-7 or (np.abs(wcs_pixscale[1])) < 1e-7 or
                     (np.abs(wcs_pixscale[0])) > 5e-3 or (np.abs(wcs_pixscale[1])) > 5e-3)
        if test_wcs1:
            if not self._silent:
                log.info("  Pixelscale appears unrealistic. We will use a guess instead.")
                wcsprm.pc = [[1, 0], [0, 1]]
                guess = pixscale_x / 3600.
                wcsprm.cdelt = [guess, guess]
                PIXSCALE_UNCLEAR = True

        if ignore_header_rot:
            wcsprm.pc = [[1, 0], [0, 1]]

        # test if found pixel scale makes sense
        test_wcs2 = (120. > x_size > 0.5) & \
                    (120. > y_size > 0.5)

        if test_wcs2:
            wcs_pixscale = imtrans.get_wcs_pixelscale(wcsprm)

            pixscale_x = pixscale_x / 60. / 60.  # pixel scale now in deg / pixel
            pixscale_y = pixscale_y / 60. / 60.  # pixel scale now in deg / pixel

            if wcs_pixscale[0] / pixscale_x < 0.1 or wcs_pixscale[0] / pixscale_x > 10 \
                    or wcs_pixscale[1] / pixscale_y < 0.1 or wcs_pixscale[1] / pixscale_y > 10:
                # check if there is a huge difference in the scales
                # if yes, then replace the wcs scale with the pixel scale information
                wcsprm.cdelt = [pixscale_x, pixscale_y]
                wcsprm.pc = [[1, 0], [0, 1]]
                PIXSCALE_UNCLEAR = True

        if not self._silent:
            log.info("  Changed pixel scale to "
                     "{:.3g} x {:.3g} deg/pixel".format(wcsprm.cdelt[0], wcsprm.cdelt[1]))

        # if the central pixel is not in the header, set to image center pixel
        # if np.array_equal(wcsprm.crpix, [0, 0]):
        #     wcsprm.crpix = [axis1 // 2, axis2 // 2]
        wcsprm.crpix = [axis1 // 2, axis2 // 2]

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
        else:
            # increase FoV for SPM data
            # ['DDOTI 28-cm f/2.2', 'PlaneWave CDK24']
            if self._telescope in ['DDOTI 28-cm f/2.2', 'CTIO 0.9 meter telescope']:
                fov_radius *= 3.

        if not self._silent:
            log.info("  Using field-of-view radius: {:.3g} deg".format(fov_radius))

        self._wcsprm = wcs
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

    @staticmethod
    def get_time_stamp() -> str:
        """
        Returns time stamp for now: "2021-10-09 16:18:16"
        """

        now = datetime.now(tz=timezone.utc)
        time_stamp = f"{now:%Y-%m-%d_%H_%M_%S}"

        return time_stamp


def main():
    """ Main procedure """
    pargs = ParseArguments(prog_typ='calibWCS')
    args = pargs.args_parsed
    main.__doc__ = pargs.args_doc

    # version check
    _base_conf.check_version(_log)

    CalibrateObsWCS(input_path=args.input, args=args, silent=args.silent, verbose=args.verbose)


# -----------------------------------------------------------------------------


# standard boilerplate to set 'main' as starting function
if __name__ == '__main__':
    main()
# -----------------------------------------------------------------------------
