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
    import utils.base_conf as bc
else:

    from leosatpy.utils.arguments import ParseArguments
    from leosatpy.utils.dataset import DataSet
    from leosatpy.utils.tables import ObsTables
    from leosatpy.utils.version import __version__
    import leosatpy.utils.sources as sext
    import leosatpy.utils.transformations as imtrans
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

__taskname__ = 'calibrateSatObs'

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
            bc.load_warnings()

        if plt is None or silent or not plot_images:
            plt.ioff()

        # Set variables
        self._dataset_object = None
        self._root_dir = bc.ROOT_DIR
        self._input_path = input_path
        self._log = log
        self._converged = bool()
        self._catalog = args.catalog
        self._hdu_idx = args.hdu_idx
        self._dic_rms = {}
        self._log_level = log_level
        self._silent = silent
        self._verbose = verbose
        self.plot_images = plot_images
        self.force_extract = args.force_detection
        self.force_download = args.force_download
        self._instrument = None
        self._telescope = None
        self._obsparams = None
        self._radius = args.radius
        self._src_cat_fname = args.source_cat_fname
        self._ref_cat_fname = args.source_ref_fname
        self._cat_path = None
        self._config = collections.OrderedDict()
        self._obsTable = None
        self._wcsprm = None
        self._fov_radius = 0.5
        self._bin_str = None

        # Run calibration
        self.run_calibration_all(silent=silent, verbose=verbose)

    def run_calibration_all(self, silent=False, verbose=False):
        """Run full calibration routine on the input path.

        Find reduced science files and run calibration for a given set of data.
        """

        StartTime = time.perf_counter()
        self._log.info('====> Astrometric calibration init <====')

        if not silent:
            self._log.info("> Search input and prepare data")
        if verbose:
            self._log.debug("  > Check input argument(s)")

        # Check the input arguments and prepare the dataset
        ds = DataSet(input_args=self._input_path,
                     prog_typ='calibWCS',
                     log=self._log, log_level=self._log_level)

        # Load configuration
        ds.load_config()
        self._config = ds.config

        self._obsTable = ObsTables(config=self._config)
        self._obsTable.load_obs_table()

        inst_list = ds.instruments_list
        inst_data = ds.instruments

        time_stamp = self.get_time_stamp()
        fail_path = Path(self._config['WORKING_DIR_PATH']).expanduser().resolve()
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

            # Loop over groups and run reduction for each group
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
                progress_counter = 0
                pass_str = bc.BCOLORS.PASS + "SUCCESSFUL" + bc.BCOLORS.ENDC
                fail_str = bc.BCOLORS.FAIL + "FAILED" + bc.BCOLORS.ENDC
                for _, file_df in files.iterrows():
                    progress = f"{progress_counter + 1}/{len(files)}"
                    self.run_calibration_single(src_path, file_df, progress,
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
                    progress_counter += 1

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

    def run_calibration_single(self, file_src_path, file_df, progress, hdu_idx=0):
        """Run astrometric calibration on a given dataset.

        Create the required folder and run full calibration procedure.

        """

        report = {}
        obsparams = self._obsparams
        abs_file_path = file_df['input']
        file_name = file_df['file_name']
        fbase = file_name.replace('_red', '')

        # Create needed folder
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

        # Set background file name and create folder
        bkg_fname_short = file_name.replace('_red', '_bkg')
        bkg_fname = os.path.join(aux_path, bkg_fname_short)
        estimate_bkg = True
        if os.path.isfile(f'{bkg_fname}.fits') and not self.force_extract:
            estimate_bkg = False

        # Load FITS file
        img_mask = None
        with fits.open(abs_file_path) as hdul:
            hdul.verify('fix')
            hdr = hdul[hdu_idx].header
            imgarr = hdul[hdu_idx].data.astype('float32')

            if 'mask' in hdul or 'MASK' in hdul and obsparams['apply_mask']:
                img_mask = hdul['MASK'].data.astype(bool)

        # detect_mask = None
        # if self._telescope in ['CTIO 0.9 meter telescope', 'CBNUO-JC']:
        #     detect_mask = np.zeros(imgarr.shape)
        #     ccd_mask_list = ccd_mask_dict[self._telescope]
        #     for yx in ccd_mask_list:
        #         detect_mask[yx[0]:yx[1], yx[2]:yx[3]] = 1
        #     detect_mask = np.where(detect_mask == 1, True, False)
        #
        # if detect_mask is not None:
        #     if img_mask is None:
        #         img_mask = detect_mask
        #     else:
        #         img_mask |= detect_mask
        #
        # vignette_mask = None
        # if hdr['FILTER'] in ['U'] and self._telescope == 'DK-1.54':
        #
        #     bin_x, bin_y = map(int, hdr['CCDSUM'].split())
        #     vignette_mask = sext.create_vignette_mask(imgarr,
        #                                               (2048, 2048),
        #                                               0.9,
        #                                               hdr[obsparams['cropsec']],
        #                                               bin_x)
        #
        # if vignette_mask is not None:
        #     if img_mask is None:
        #         img_mask = vignette_mask
        #     else:
        #         img_mask |= vignette_mask

        # Get the satellite ID
        sat_id, _ = self._obsTable.get_satellite_id(hdr[obsparams['object']])
        plt_path_final = plt_path / sat_id
        if not plt_path_final.exists():
            plt_path_final.mkdir(exist_ok=True)

        # Get WCS from header and an original copy
        wcsprm = WCS(hdr).wcs

        # Run checks on wcs and get an initial guess
        self.check_image_wcs(wcsprm=wcsprm, hdr=hdr,
                             obsparams=obsparams,
                             radius=self._radius)
        init_wcs = self._wcsprm
        init_wcsprm = self._wcsprm.wcs

        # Update configuration
        config = obsparams.copy()
        params_to_add = dict(fov_radius=self._fov_radius,
                             src_cat_fname=self._src_cat_fname,
                             ref_cat_fname=self._ref_cat_fname,
                             image_mask=img_mask,
                             bin_str=self._bin_str,
                             estimate_bkg=estimate_bkg,
                             ref_cat_mag_lim=self._config['REF_CATALOG_MAG_LIM'],
                             bkg_fname=(bkg_fname, bkg_fname_short),
                             force_extract=self.force_extract,
                             force_download=self.force_download)

        # Add the parameters to the configuration
        config.update(params_to_add)
        for key, value in self._config.items():
            config[key] = value

        # Get detector saturation limit
        sat_lim = config['saturation_limit']
        if isinstance(config['saturation_limit'], str) and config['saturation_limit'] in hdr:
            sat_lim = hdr[obsparams['saturation_limit']]
        config['sat_lim'] = sat_lim

        config['image_shape'] = imgarr.shape

        # Extract sources and create a source catalog
        extraction_result, state, _ = sext.get_src_and_cat_info(fbase, cat_path,
                                                                imgarr, hdr, init_wcsprm,
                                                                silent=self._silent,
                                                                **config)
        # Unpack extraction result
        (src_tbl_raw, ref_tbl, ref_catalog, src_cat_fname, ref_cat_fname,
         kernel_fwhm) = extraction_result

        # Check execution state and update result table
        if not state or len(src_tbl_raw) == 0:
            self._converged = False

            # Make sure that the RA and DEC are consistent and rounded to the same decimal
            ra = hdr[obsparams['ra']]
            dec = hdr[obsparams['dec']]
            if obsparams['radec_separator'] == 'XXX':
                ra = round(hdr[obsparams['ra']], bc.ROUND_DECIMAL)
                dec = round(hdr[obsparams['dec']], bc.ROUND_DECIMAL)

            kwargs = {obsparams['exptime']: hdr[obsparams['exptime']],
                      obsparams['object']: hdr[obsparams['object']],
                      obsparams['instrume']: hdr[obsparams['instrume']],
                      obsparams['ra']: ra,
                      obsparams['dec']: dec,
                      'AST_CAL': False}

            self._obsTable.update_obs_table(file=fbase, kwargs=kwargs, obsparams=obsparams)

            return

        # Use only the entries with good fwhm values
        src_tbl = src_tbl_raw.query('include_fwhm')
        if len(src_tbl) < 5:
            src_tbl = src_tbl_raw

        # Set the match radius either based on the fwhm or as absolute value
        match_radius = self._config['MATCH_RADIUS']
        match_radius = kernel_fwhm[0] if match_radius == 'fwhm' else match_radius

        # Get the default PC matrix
        self._config['pc_matrix'] = obsparams['pc_matrix']

        # Initialize find WCS object
        get_wcs = imtrans.FindWCS(src_tbl, ref_tbl, hdr, config, self._log)

        # Run the WCS analysis
        wcs_res = get_wcs.find_wcs(init_wcs, imgarr, match_radius)
        has_solution, best_wcsprm, dict_rms = wcs_res

        # Check the result sate
        status_str = bc.BCOLORS.PASS + "PASS" + bc.BCOLORS.ENDC
        if not has_solution or best_wcsprm is None:
            status_str = bc.BCOLORS.FAIL + "FAIL" + bc.BCOLORS.ENDC
        else:
            # Move scaling to cdelt and out of the pc matrix
            best_wcsprm, scales = imtrans.translate_wcsprm(wcsprm=best_wcsprm)
            wcsprm = best_wcsprm

        if not self._silent:
            self._log.info("  ==> Calibration status: " + status_str)

        # Get updated reference catalog positions
        ref_tbl_adjusted = imtrans.update_catalog_positions(ref_tbl, wcsprm, origin=0)

        # Update file header and save
        if has_solution:

            # Get the pixel scale
            wcs_pixscale = imtrans.get_wcs_pixelscale(wcsprm)

            # Update the report dictionary
            report["converged"] = has_solution
            report["catalog"] = ref_catalog
            report["fwhm"] = kernel_fwhm[0]
            report["e_fwhm"] = kernel_fwhm[1]
            report["matches"] = int(dict_rms["matches"])
            report["match_radius"] = dict_rms["radius_px"]
            report["match_rms"] = dict_rms["rms"]
            report["std_x"] = dict_rms["std_x"]
            report["std_y"] = dict_rms["std_y"]
            report["pix_scale"] = np.mean(abs(wcs_pixscale)) * 3600. if wcs_pixscale is not np.nan else np.nan
            report["scale_x"] = abs(wcs_pixscale[0]) * 3600.
            report["scale_y"] = abs(wcs_pixscale[1]) * 3600.

            # Save the result to the FITS-file
            self.write_wcs_to_hdr(original_filename=abs_file_path,
                                  filename_base=fbase,
                                  destination=cal_path,
                                  wcsprm=wcsprm,
                                  report=report, hdul_idx=hdu_idx)

            # Get the source positions
            src_positions = list(zip(src_tbl['xcentroid'], src_tbl['ycentroid']))

            # Match results within 1 fwhm radius for plot
            matches = imtrans.find_matches(src_tbl,
                                           ref_tbl_adjusted,
                                           wcsprm,
                                           threshold=dict_rms["radius_px"])
            _, _, _, ref_positions_after, _, _, _ = matches

            if not self._silent:
                self._log.info("> Plot WCS calibration result")

            # Plot the final result
            config['match_radius_px'] = dict_rms["radius_px"]
            self.plot_final_result(imgarr=imgarr, src_pos=src_positions,
                                   ref_pos=ref_positions_after,
                                   file_name=file_name, fig_path=plt_path_final,
                                   wcsprm=wcsprm,
                                   **config)

            bc.clean_up(src_positions, ref_positions_after, matches)

        else:

            # Make sure that the RA and DEC are consistent and rounded to the same decimal
            ra = hdr[obsparams['ra']]
            dec = hdr[obsparams['dec']]
            if obsparams['radec_separator'] == 'XXX':
                ra = round(hdr[obsparams['ra']], bc.ROUND_DECIMAL)
                dec = round(hdr[obsparams['dec']], bc.ROUND_DECIMAL)
            kwargs = {obsparams['exptime']: hdr[obsparams['exptime']],
                      obsparams['object']: hdr[obsparams['object']],
                      obsparams['instrume']: hdr[obsparams['instrume']],
                      obsparams['ra']: ra,
                      obsparams['dec']: dec,
                      'AST_CAL': False}

            self._obsTable.update_obs_table(file=fbase, kwargs=kwargs, obsparams=obsparams)

        self._converged = has_solution
        self._dic_rms = dict_rms

        bc.clean_up(imgarr, dict_rms, has_solution,
                            src_tbl, ref_tbl, ref_catalog, _, src_cat_fname, ref_cat_fname,
                            kernel_fwhm, extraction_result, report,
                            state)

    def check_image_wcs(self, wcsprm, hdr, obsparams,
                        radius=-1.):
        """Check the WCS coordinate system for a given image.

        The function also tries to handle additional or missing data from the header.

        Parameters
        ----------
        wcsprm: WCS.wcs
            World coordinate system object describing translation between image and skycoord
        hdr: header
        obsparams
        radius
        """

        # Initialize logging for this function
        log = self._log

        if not self._silent:
            log.info("> Gather header information")

        # Check binning keyword
        if isinstance(obsparams['binning'][0], str) and isinstance(obsparams['binning'][1], str):
            bin_x = int(hdr[obsparams['binning'][0]])
            bin_y = int(hdr[obsparams['binning'][1]])
        else:
            bin_x = int(obsparams['binning'][0])
            bin_y = int(obsparams['binning'][1])

        self._bin_str = f'{bin_x}x{bin_y}'

        if self._telescope in ['DK-1.54', 'PlaneWave CDK24']:
            wcs_rebinned = WCS(wcsprm.to_header()).slice((np.s_[::bin_x], np.s_[::bin_y]))
            wcsprm = wcs_rebinned.wcs

        # Check axes keywords
        if "NAXIS1" not in hdr or "NAXIS2" not in hdr:
            log.error("NAXIS1 or NAXIS2 missing in file. Please add!")
            sys.exit(1)
        else:
            axis1 = obsparams['image_size_1x1'][0] // bin_x
            axis2 = obsparams['image_size_1x1'][1] // bin_y

        # Read out RA and Dec from header
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

        # Transform to equinox J2000, if necessary
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

        # Recompile the WCS object
        wcsprm = WCS(hdr).wcs

        # Get the pixel scale in arcseconds from the header or the telescope configuration
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

        # Apply the CCD binning factor
        pixscale_x *= bin_x
        pixscale_y *= bin_y

        # Estimate the size of the FoV in arcmin
        x_size = axis1 * pixscale_x / 60.  # arcmin
        y_size = axis2 * pixscale_y / 60.  # arcmin

        # Test if the current pixel scale makes sense
        wcs_pixscale = imtrans.get_wcs_pixelscale(wcsprm)

        test_wcs1 = ((np.abs(wcs_pixscale[0])) < 1e-7 or (np.abs(wcs_pixscale[1])) < 1e-7 or
                     (np.abs(wcs_pixscale[0])) > 5e-3 or (np.abs(wcs_pixscale[1])) > 5e-3)
        if test_wcs1:
            if not self._silent:
                log.info("  Pixelscale appears unrealistic. We will use a guess instead.")
                wcsprm.pc = [[1, 0], [0, 1]]
                guess = pixscale_x / 3600.
                wcsprm.cdelt = [guess, guess]

        # Test if found pixel scale makes sense
        test_wcs2 = (120. > x_size > 0.5) & \
                    (120. > y_size > 0.5)

        if test_wcs2:
            wcs_pixscale = imtrans.get_wcs_pixelscale(wcsprm)

            pixscale_x = pixscale_x / 60. / 60.  # pixel scale now in deg / pixel
            pixscale_y = pixscale_y / 60. / 60.  # pixel scale now in deg / pixel

            if wcs_pixscale[0] / pixscale_x < 0.1 or wcs_pixscale[0] / pixscale_x > 10 \
                    or wcs_pixscale[1] / pixscale_y < 0.1 or wcs_pixscale[1] / pixscale_y > 10:
                # Check if there is a huge difference in the scales
                # if yes, then replace the wcs scale with the pixel scale information
                wcsprm.cdelt = [pixscale_x, pixscale_y]
                wcsprm.pc = [[1, 0], [0, 1]]

        # Set the central pixel to the image center pixel
        wcsprm.crpix = [axis1 // 2, axis2 // 2]

        # Check sky position
        if np.array_equal(wcsprm.crval, [0, 0]):
            # If the sky position is not found check header for RA and DEC information
            wcsprm.crval = [ra_deg, dec_deg]

        if np.array_equal(wcsprm.ctype, ["", ""]):
           wcsprm.ctype = ['RA---TAN', 'DEC--TAN']  # this is a guess

        # Recompile the WCS object
        wcs_n = WCS(wcsprm.to_header())

        # Estimate the size of the FoV
        fov_radius = sext.compute_radius(wcs_n, axis1, axis2)

        if radius > 0.:
            fov_radius = radius / 60.
        else:
            # Increase the FoV for SPM data
            if self._telescope in ['DDOTI 28-cm f/2.2', 'CTIO 0.9 meter telescope']:
                fov_radius *= 2.

        if not self._silent:
            log.info(f"{' ':<2}{'Pointing (deg)':<24}: RA={wcsprm.crval[0]:.8g}, "
                     f"DEC={wcsprm.crval[1]:.8g}")
            log.info(f"{' ':<2}{'Pixel scale (deg/pixel)':<24}: "
                     f"{wcsprm.cdelt[0]:.3g} x {wcsprm.cdelt[1]:.3g}")
            log.info(f"{' ':<2}{'FoV radius (deg)':<24}: {fov_radius:.3g}")

        self._wcsprm = wcs_n
        self._fov_radius = fov_radius

    def write_wcs_to_hdr(self, original_filename, filename_base,
                         destination, wcsprm, report, hdul_idx=0):
        """
        Update the header of the FITS file itself.

        Parameters
        ----------
        original_filename : str
            Original filename of the FITS file.
        filename_base : str
            Base name for the output file.
        destination : pathlib.Path
            Destination directory for the output file.
        wcsprm : astropy.wcs.WCS.wcs
            World coordinate system object describing translation between image and sky coordinates.
        report : dict
            Dictionary containing the calibration report.
        hdul_idx : int, optional
            Index of the HDU list to update, by default 0.

        Returns
        -------
        None
        """
        if not self._silent:
            self._log.info("> Update FITS file.")

        with fits.open(original_filename) as hdul:

            hdu = hdul[hdul_idx]
            hdr_file = hdu.header

            # Throughout CD which contains the scaling and separate into PC and CDELT
            for old_parameter in ['CD1_1', 'CD1_2', 'CD2_1', 'CD2_2',
                                  "PC1_1", "PC1_2", "PC2_1", "PC2_2"]:
                if old_parameter in hdr_file:
                    del hdr_file[old_parameter]

            # Update WCS
            wcs = WCS(wcsprm.to_header())
            hdr_file.update(wcs.to_header())

            # Get the transformation matrix
            matrix = wcsprm.pc * wcsprm.cdelt[:, np.newaxis]

            # Rotation angle
            theta = np.rad2deg(-np.arctan2(matrix[1, 0], matrix[1, 1]))

            # Add report to the header
            hdr_file['PIXSCALE'] = (f'{report["pix_scale"]:.5f}', 'Average pixel scale [arcsec/pixel]')
            hdr_file['SCALEX'] = (f'{report["scale_x"]:.5f}', 'Pixel scale in X [arcsec/pixel]')
            hdr_file['SCALEY'] = (f'{report["scale_y"]:.5f}', 'Pixel scale in Y [arcsec/pixel]')
            hdr_file['DETROTANG'] = (f'{theta:.2f}', 'Detection rotation angel [deg]')
            hdr_file['FWHM'] = (f'{report["fwhm"]:.3f}', 'FWHM [pixel]')
            hdr_file['FWHMERR'] = (f'{report["e_fwhm"]:.3f}', 'FWHM error [pixel]')
            hdr_file['POSERRX'] = (f'{report["std_x"]:.3f}', 'Position error in X [pixel]')
            hdr_file['POSERRY'] = (f'{report["std_y"]:.3f}', 'Position error in Y [pixel]')
            hdr_file['AST_SCR'] = ("leosatpy", 'Program used for astrometric calibration')
            hdr_file['AST_VER'] = (__version__, 'Program version')
            hdr_file['AST_CAT'] = (report["catalog"], 'Catalog used')
            hdr_file['AST_MAT'] = (report["matches"], 'Number of catalog matches')
            hdr_file['AST_RAD'] = (f'{report["match_radius"]:.3f}', 'match radius in pixel')
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

        # Make sure that the RA and DEC are consistent and rounded to the same decimal
        obsparams = self._obsparams
        if obsparams['radec_separator'] == 'XXX':
            fits_header[obsparams['ra']] = round(fits_header[obsparams['ra']],
                                                 bc.ROUND_DECIMAL)
            fits_header[obsparams['dec']] = round(fits_header[obsparams['dec']],
                                                  bc.ROUND_DECIMAL)

        self._obsTable.update_obs_table(filename_base, fits_header, obsparams)

    def plot_final_result(self, imgarr, src_pos, ref_pos,
                          file_name, fig_path, wcsprm, cmap='Greys', **config):
        """Plot before and after images

        Parameters
        ----------
        imgarr
        src_pos
        ref_pos
        file_name
        fig_path
        wcsprm
        cmap
        config

        Returns
        -------

        """

        # Load fits file
        bkg_fname = config['bkg_fname']
        with fits.open(f'{bkg_fname[0]}.fits') as hdul:
            hdul.verify('fix')
            bkg_background = hdul[0].data.astype('float32')

        # Subtract the background from the image
        imgarr -= bkg_background

        # Get y-axis length as limit
        matrix = wcsprm.pc * wcsprm.cdelt[:, np.newaxis]

        # Scale
        scale_x = np.sqrt(matrix[0, 0] ** 2 + matrix[0, 1] ** 2)

        # Rotation angle
        theta = -np.arctan2(matrix[1, 0], matrix[1, 1])

        # Get x, y-axis length as limits
        plim = imgarr.shape[0]  # Image height
        xplim = imgarr.shape[1]  # Image width

        # # Get the scale and angle
        # scale = wcsprm.cdelt[0]
        # angle = np.arccos(wcsprm.pc[0][0])
        # theta = angle / np.pi * 180.
        # if wcsprm.pc[0][0] < 0.:
        #     theta *= -1.

        obs_xy = np.array(src_pos)
        cat_xy = ref_pos

        # Calculate the distances
        dist_xy = np.sqrt((obs_xy[:, 0] - cat_xy[:, 0, np.newaxis]) ** 2
                          + (obs_xy[:, 1] - cat_xy[:, 1, np.newaxis]) ** 2)

        idx_arr = np.where(dist_xy == np.min(dist_xy, axis=0))
        min_dist_xy = dist_xy[idx_arr]
        del dist_xy

        mask = min_dist_xy > config['match_radius_px']
        obs_idx = idx_arr[1][mask]
        obs_xy = obs_xy[obs_idx, :]

        # Adjust position offset
        px_offset = -0.
        adjusted_ref_pos = [(x + px_offset, y + px_offset) for x, y in ref_pos]
        apertures_catalog = CircularAperture(adjusted_ref_pos, r=10.)

        image_mask = config['image_mask']

        if image_mask is not None:
            imgarr = np.where(image_mask, 0., imgarr)

        # Define normalization and the colormap
        vmin = np.percentile(imgarr, 50)
        vmin = 0 if vmin < 0 else vmin
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
            adjusted_src_pos = [(x + px_offset, y + px_offset) for x, y in obs_xy]
            apertures = CircularAperture(adjusted_src_pos, r=10.)
            apertures.plot(axes=ax, **{'color': 'red', 'lw': 1.25, 'alpha': 0.85})

        apertures_catalog.plot(axes=ax, **{'color': 'green', 'lw': 1.25, 'alpha': 0.85})

        # Add an image scale
        self.add_image_scale(ax=ax, plim_x=xplim, plim_y=plim, scale=scale_x)

        # Add a compass
        self.add_compass(ax=ax, image_shape=imgarr.shape,
                         scale_arrow=self._config['ARROW_LENGTH'], theta_rad=theta,
                         color='k')
        # -----------------------------------------------------------------------------
        # Make the color bar
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

        self.save_plot(fname)

        if self.plot_images:
            plt.show()

        plt.close(fig=fig)

    def add_image_scale(self, ax, plim_x, plim_y, scale):
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

    def save_plot(self, fname: str):
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
    def add_compass(ax: plt.Axes, image_shape, scale_arrow, theta_rad: float, color: str = 'black'):
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
    bc.check_version(_log)

    CalibrateObsWCS(input_path=args.input, args=args, silent=args.silent, verbose=args.verbose)


# -----------------------------------------------------------------------------


# Standard boilerplate to set 'main' as starting function
if __name__ == '__main__':
    main()
# -----------------------------------------------------------------------------
