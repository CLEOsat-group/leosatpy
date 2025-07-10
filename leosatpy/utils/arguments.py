#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Name:         arguments.py
# Purpose:      Script to handle the arguments parsed
#               when calling a given program from the command line.
#
#
#
#
# Author:       p4adch (cadam)
#
# Created:      04/15/2022
# Copyright:    (c) p4adch 2010-
#
# -----------------------------------------------------------------------------

""" Modules """
from __future__ import annotations
import argparse

from .version import __version__

# -----------------------------------------------------------------------------

""" Meta-info """
__author__ = "Christian Adam"
__copyright__ = 'Copyright 2021-2025, CLEOSat group'
__credits__ = ["Eduardo Unda-Sanzana, Jeremy Tregloan-Reed, Christian Adam"]
__license__ = "GPL-3.0 license"
__maintainer__ = "Christian Adam"
__email__ = "christian.adam84@gmail.com"
__status__ = "Production"


# -----------------------------------------------------------------------------
# changelog
# version 0.1.0 added first method's
# version 0.2.0 added methods for calibration and satellite analysis


class ParseArguments(object):
    """ Main class"""
    prog_typ: str | None

    def __init__(self, prog_typ: str | None):
        """ Constructor with default values """

        self._args = None
        self.prog_typ = prog_typ

        # Call program specific argument functions
        if prog_typ == 'reduce_sat_obs':
            self._parse_reduce_sat_obs()
        elif prog_typ == 'reduce_calib_obs':
            self._parse_reduce_calib_obs()
        elif prog_typ == 'calibWCS':
            self._parse_calibration()
        elif prog_typ == 'satPhotometry':
            self._parse_sat_photometry()

        parser = self._parser

        # Default arguments across scripts
        progressArgs = parser.add_argument_group('progress')

        progressArgs.add_argument("-s", "--silent", default=False, action='store_true',
                                  help="Almost no console output. Use when running many files. "
                                       "turns verbose off, turns plots off, turns ignore warning on. ")

        progressArgs.add_argument("-v", "--verbose", default=False, action='store_true',
                                  help="More console output about what is happening. Helpful for debugging.")

        progressArgs.add_argument("-w", "--ignore_warnings", default=True, action='store_false',
                                  help="Set False to see all Warnings about the header if there is problems. "
                                       "Default is True.")

        parser.add_argument('--version', action='version', version=__version__)

        self._parser = parser

        # Parse arguments
        self._args = parser.parse_args()

    @property
    def args_parsed(self):
        return self._args

    def args_doc(self):
        doc_string = """
        ::
          
          {}
          
        """.format(self._parser.format_help().replace('\n', '\n  '))
        return doc_string

    def _parse_sat_photometry(self):
        """Specific arguments for satellite trail detection"""

        description = """ This program can be used to detect and analyse satellite trails in FITS-images.
        """
        epilog = """
        Examples:
        
        Use default settings and analyse all science images found in the given directory with:
        
        python analyseSatObs.py /path/to/reduced/data
        
        """
        # Create argument parser
        parser = argparse.ArgumentParser(prog='leosat-analyseSatObs',
                                         epilog=epilog, description=description,
                                         formatter_class=argparse.RawDescriptionHelpFormatter)

        # Positional mandatory argument
        parser.add_argument("input", nargs='+', type=str,
                            help="Input path(s) or image(s) separated by whitespace. "
                                 "A combination of folder or multiple images is supported.")

        # Image related args
        imageArgs = parser.add_argument_group('image control')

        imageArgs.add_argument("-hdu_idx", "--hdu_idx", type=int, default=0,
                               help="Index of the image data in the fits file. Default is 0")

        photoArgs = parser.add_argument_group('photometry control')

        photoArgs.add_argument("-b", "--band", type=str, default=None,
                               help="Photometric band to use. Options are B,V,R,...")
        photoArgs.add_argument("-c", "--catalog", type=str, default="auto",
                               help="Catalog to use for photometric reference ('GSC243'). This can be left to the standard 'auto' "
                                    "to let the program choose automatically.")
        photoArgs.add_argument("-f", "--force-detection", dest='force_detection', action='store_true', default=False,
                               help="Set True to force the source and reference catalog extraction. "
                                    "Default is False. If False, the corresponding present .cat files are used.")
        photoArgs.add_argument("-d", "--force-download", dest='force_download', action='store_true', default=False,
                               help="Set True to force the reference catalog download. "
                                    "Default is False. If False, the corresponding present .cat files are used.")
        photoArgs.add_argument("-m", "--manual-select", dest='select_faint_trail', action='store_true', default=False,
                               help="Set True to manually select the faint trail. Default is False.")
        photoArgs.add_argument("-photo_ref_fname", "--photo_ref_fname", dest='ref_cat_phot_fname',
                               type=str, default=None,
                               help="Save the sky positions from the photometry catalog of the reference sources "
                                    "to file for later use if output is True. "
                                    "Set to the filename without extension. Default: None.")

        # Output related args
        outputArgs = parser.add_argument_group('output control')
        outputArgs.add_argument("-p", "--plot_images", action='store_true', default=False,
                                help="Set True to show plots during process.")

        self._parser = parser

    def _parse_calibration(self):
        """Arguments specific to astrometric calibration"""

        description = """ This program can be used to calibrate the image world coordinate system (WCS).
        """
        epilog = """
        Examples:
        
        Use default settings and calibrate all science images found in the given directory with:
        
        python calibrateSatObs.py /path/to/reduced/data
        
        """
        # Create argument parser
        parser = argparse.ArgumentParser(prog='leosat-calibrateSatObs',
                                         epilog=epilog, description=description,
                                         formatter_class=argparse.RawDescriptionHelpFormatter)

        # Positional mandatory argument
        parser.add_argument("input", nargs='+', type=str,
                            help="Input path(s) or image(s) separated by whitespace. "
                                 "A combination of folder or multiple images is supported.")

        # Image related args
        imageArgs = parser.add_argument_group('image control')

        imageArgs.add_argument("-hdu_idx", "--hdu_idx", type=int, default=0,
                               help="Index of the image data in the fits file. Default is 0.")

        imageArgs.add_argument("-r", "--radius", type=float, default=-1,
                               help="Set download radius in arcmin for catalog objects download. "
                                    "Otherwise automatically determine the FoV from image footprint. "
                                    "Default is 'auto'.")
        # imageArgs.add_argument("-vignette", "--vignette", type=float, default=-1,
        #                        help="Do not use corner of the image. "
        #                             "Only use the data in a circle around the center with certain radius. "
        #                             "Default: not used. Set to 1 for circle that touches the sides. "
        #                             "Less to cut off more.")

        # Calibration related arguments
        calibArgs = parser.add_argument_group('calibration control')
        calibArgs.add_argument("-c", "--catalog",
                               help="Catalog to use for position reference. Defaults to `GAIAdr3`.",
                               type=str, default="GAIAdr3")
        calibArgs.add_argument("-f", "--force-detection", dest='force_detection', action='store_true', default=False,
                               help="Set True to force the source and reference catalog extraction. "
                                    "Default is False. If False, the corresponding present .cat files are used.")
        calibArgs.add_argument("-d", "--force-download", dest='force_download', action='store_true', default=False,
                               help="Set True to force the reference catalog download. "
                                    "Default is False. If False, the corresponding present .cat files are used.")
        calibArgs.add_argument("-source_cat_fname", "--source_cat_fname", type=str, default=None,
                               help="Save the detector positions of the sources "
                                    "to file for later use if output is True. "
                                    "Set to the filename without extension. Default: None")
        calibArgs.add_argument("-source_ref_fname", "--source_ref_fname", type=str, default=None,
                               help="Save the sky positions from the catalog of the sources "
                                    "to file for later use if output is True. "
                                    "Set to the filename without extension. Default: None")

        # Output related args
        outputArgs = parser.add_argument_group('output control')
        outputArgs.add_argument("-p", "--plot_images", action='store_true', default=False,
                                help="Set True to show plots during process.")

        self._parser = parser

    def _parse_reduce_sat_obs(self):
        """Specific arguments for reduction"""

        description = """This program reduces astronomical observations from different telescopes 
        using astropy ccdproc."""
        epilog = """ 
        Examples:
        
        Use default settings and reduce all science images found in the given directory with:
        
        python reduceSatObs.py /path/to/data
        
        """
        # Create argument parser
        parser = argparse.ArgumentParser(prog='leosat-reduceSatObs',
                                         epilog=epilog, description=description,
                                         formatter_class=argparse.RawDescriptionHelpFormatter)

        # Positional mandatory argument
        parser.add_argument("input", nargs='+', type=str,
                            help="Input path(s) or image(s) separated by whitespace. "
                                 "A combination of folder or multiple images is supported.")

        # Reduction related args
        reduceArgs = parser.add_argument_group('reduction control')

        reduceArgs.add_argument("-f", "--force-reduction", dest='force_reduction', action='store_true',
                                default=False,
                                help="Set True to force the creation of master calibration images, even if present. "
                                     "Default is False.")

        self._parser = parser

    def _parse_reduce_calib_obs(self):
        """Specific arguments for reduction"""
        description = """This program reduces calibration observations from different telescopes 
        using astropy ccdproc."""
        epilog = """ 
        Examples:
        
        Use default settings and reduce all science images found in the given directory with:
        python reduceCalibObs.py /path/to/data
        
        
        Notes:
        
        """
        # Create argument parser
        parser = argparse.ArgumentParser(prog='leosat-reduceCalibObs',
                                         epilog=epilog, description=description,
                                         formatter_class=argparse.RawDescriptionHelpFormatter)

        # Positional mandatory argument
        parser.add_argument("input", nargs='+', type=str,
                            help="Input path(s) or image(s) separated by whitespace. "
                                 "A combination of folder or multiple images is also supported.")

        # Reduction related args
        reduceArgs = parser.add_argument_group('reduction mode')

        reduceArgs.add_argument("-b", "--bias", action="store_true", default=False,
                                help="If set only bias frames are reduced and a master_bias.fits is created. "
                                     "Defaults to False.")
        reduceArgs.add_argument("-d", "--dark", action="store_true", default=False,
                                help="If set only dark frames are reduced and a master_dark_exptime.fits "
                                     "for each exposure time is created. "
                                     "Defaults to False.")
        reduceArgs.add_argument("-f", "--flat", action="store_true", default=False,
                                help="If set only flat frames are reduced and a master_flat_filter.fits for each "
                                     "filter are created. "
                                     "Defaults to False.")

        self._parser = parser
