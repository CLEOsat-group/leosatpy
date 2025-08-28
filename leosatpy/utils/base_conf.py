#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Name:         base_config.py
# Purpose:      Base configuration file for reduction, astrometric calibration
#               and satellite trail analysis
#
#
# Author:       p4adch (cadam)
#
# Created:      04/15/2022
# Copyright:    (c) p4adch 2010-
#
# -----------------------------------------------------------------------------

""" Modules """
import gc
import os
import sys
import logging
import warnings
from dateutil.parser import parse
from packaging import version
from urllib import request
import numpy as np
from colorlog import ColoredFormatter
from astropy import wcs
from astropy.io import fits
from astropy.utils.exceptions import (AstropyUserWarning, AstropyWarning)

from threading import Lock

# from .telescope_conf import *
from .version import __version__

""" Meta-info """
__author__ = "Christian Adam"
__copyright__ = 'Copyright 2021, UA, LEOSat observations'
__credits__ = ["Christian Adam, Eduardo Unda-Sanzana, Jeremy Tregloan-Reed"]
__license__ = "Free"
__maintainer__ = "Christian Adam"
__email__ = "christian.adam84@gmail.com"
__status__ = "Production"

# Logging and console output
LOG_LEVEL = logging.INFO
LOGFORMAT = '[%(log_color)s%(levelname)8s%(reset)s] %(log_color)s%(message)s%(reset)s'
FORMATTER = ColoredFormatter(LOGFORMAT)


def load_warnings():
    warnings.simplefilter(action="ignore", category=RuntimeWarning)
    warnings.simplefilter(action='ignore', category=UserWarning)
    warnings.simplefilter(action='ignore', category=AstropyWarning)
    warnings.simplefilter(action='ignore', category=AstropyUserWarning)
    warnings.filterwarnings(action='ignore', category=wcs.FITSFixedWarning)
    warnings.filterwarnings(action='ignore', category=fits.column.VerifyWarning)
    warnings.filterwarnings(action='ignore', category=fits.card.VerifyWarning)

    # the following warning gets cast by Gaia query: XXX.convert_unit_to(u.deg)
    warnings.filterwarnings(action='ignore', category=np.ma.core.MaskedArrayFutureWarning)
    warnings.filterwarnings(action='ignore', category=UserWarning)
    warnings.filterwarnings(action='ignore', category=FutureWarning)

    # numpy warnings
    np.seterr(divide='ignore', invalid='ignore')
    np.errstate(invalid='ignore')


# Lock for thread-safe printing
print_lock = Lock()

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '../..'))


# Color definitions for console output
class BCOLORS:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[32m'
    WARNING = '\033[93m'
    PASS = '\033[92m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


# Color strings for console output
pass_str = BCOLORS.PASS + "SUCCESSFUL" + BCOLORS.ENDC
fail_str = BCOLORS.FAIL + "FAILED" + BCOLORS.ENDC

# List of potential FITS header keywords that can be used to identify the instrument
# Any unique header keyword serves as an identifier
INSTRUMENT_KEYS = ['PPINSTRU', 'LCAMMOD', 'INSTRUME', 'INSTRUM',
                   'TELESCOP', 'FLI', 'FPA', 'CAM_NAME']


FRMT_FS = "%Y-%m-%dT%H:%M:%S.%f"
FRMT = "%Y-%m-%dT%H:%M:%S"


def has_fractional_seconds(time_string):
    # Parse the time string into a datetime object
    dt = parse(time_string)

    # Check if the datetime object has fractional seconds
    if dt.microsecond > 0:
        return FRMT_FS
    else:
        return FRMT


# Clean up function to delete variables and free memory
def clean_up(*args):
    """Delete variables and free memory."""

    for _ in args:
        del _

    gc.collect()


# Default number of decimal places used in rounding
ROUND_DECIMAL = 5

IMAGETYP_LIGHT = 'science'
IMAGETYP_BIAS = 'bias'
IMAGETYP_DARK = 'dark'
IMAGETYP_FLAT = 'flat'

IMAGETYP_REDUCE_ORDER = np.array(['bias', 'darks', 'flats'])
IMAGETYP_COMBOS = {'bias': "zero|bias|bias frame",
                   'darks': "dark|dark frame",
                   'flats': "flat|flat frame|dflat|sflat",
                   'light': "science|light|object|sky|light frame"}

BINNING_DKEYS = ('BINX', 'BINY')

# Default result table column names.
DEF_RES_TBL_COL_NAMES = ['File', 'Object', 'Sat-Name', 'AltID', 'UniqueID',
                         'Instrument', 'HDU_idx', 'DetPosID',
                         'Telescope', 'RA', 'DEC', 'Date-Obs', 'Filter', 'ExpTime',
                         'Airmass', 'Binning',
                         'Obs-Start', 'Obs-Mid', 'Obs-Stop',
                         'HasTrail', 'NTrail', 'HasRef', 'NRef',
                         'UT Date', 'UT time',
                         'SatLon', 'SatLat', 'SatAlt',
                         'SatAz', 'SatElev', 'SatRA', 'SatDEC',
                         'SunRA', 'SunDEC', 'SunZenithAngle', 'SatAngularSpeed', 'e_SatAngularSpeed',
                         'ObsSatRange',
                         'ToD', 'e_ToD',
                         'ObsTrailLength', 'e_ObsTrailLength', 'EstTrailLength', 'e_EstTrailLength',
                         'SunSatAng', 'SunPhaseAng', 'SunIncAng', 'ObsAng',
                         'ObsMag', 'e_ObsMag', 'EstMag', 'e_EstMag',
                         'EstScaleMag', 'e_EstScaleMag',
                         'HasZp', 'MagZp',
                         'ObsMag_zp', 'e_ObsMag_zp',
                         'EstMag_zp', 'e_EstMag_zp',
                         'EstScaleMag_zp', 'e_EstScaleMag_zp',
                         'SunAzAng', 'SunElevAng',
                         'FluxScale', 'MagScale', 'MagCorrect', 'e_MagCorrect', 'MagCorrectSat',
                         'dt_tle-obs',
                         'TrailCX', 'e_TrailCX', 'TrailCY', 'e_TrailCY',
                         'TrailCRA', 'e_TrailCRA', 'TrailCDEC', 'e_TrailCDEC',
                         'TrailANG', 'e_TrailANG', 'OptAperHeight',
                         'CalRA', 'CalDEC', 'CentX', 'e_CentX', 'CentY', 'e_CentY', 'PixScale', 'DetRotAng', 'FWHM', 'e_FWHM',
                         'bias_cor', 'dark_cor', 'flat_cor', 'WCS_cal', 'mag_conv', 'QlfAperRad', 'TrailDetMethod']

# To be implemented. Default result table column units.
DEF_RES_TBL_COL_UNITS = ['', '', '', '', '',
                         '', '', '',
                         '', 'hh:mm:ss.sss', 'deg:mm:ss.sss', '', '', 's',
                         '', '',
                         'hh:mm:ss.sss', 'hh:mm:ss.sss', 'hh:mm:ss.sss',
                         '', '', '', '',
                         'YYYY-MM-DD', 'hh:mm:ss.sss',
                         'deg', 'deg', 'km',
                         'deg', 'deg', 'hh:mm:ss.sss', 'deg',
                         'hr', 'deg', 'deg', 'arcsec/s', 'arcsec/s',
                         'km',
                         's', 's',
                         'arcsec', 'arcsec', 'arcsec', 'arcsec',
                         'deg', 'deg', 'deg', 'deg',
                         'mag', 'mag', 'mag', 'mag',
                         'mag', 'mag',
                         '', '',
                         'mag', 'mag',
                         'mag', 'mag',
                         'mag', 'mag',
                         'deg', 'deg',
                         '', '', 'mag', 'mag', 'mag', 's',
                         'px', 'px', 'px', 'px',
                         'deg', 'arcsec', 'deg', 'arcsec',
                         'deg', 'arcsec', 'px',
                         'deg', 'deg', 'px', 'px', 'px', 'px', 'arcsec/px', 'deg', 'px', 'px',
                         '', '', '', '', '', '', '']

# Default visibility file table column names.
DEF_VIS_TBL_COL_NAMES = ['ID', 'AltID', 'UniqueID', 'UT Date', 'UT time',
                         'SatLon', 'SatLat', 'SatAlt',
                         'SatAz', 'SatElev', 'SatRA', 'SatDEC',
                         'SunRA', 'SunDEC', 'SunZenithAngle',
                         'SatAngularSpeed']

# Translations between table and fits header keywords
DEF_KEY_TRANSLATIONS = {
    'Object': ['OBJECT', 'BLKNM'],
    'Instrument': ['INSTRUME', 'INSTRUM'],
    'Telescope': ['TELESCOP', 'OBSERVAT'],
    'HDU_idx': ['HDU_IDX'],
    'DetPosID': ['DETPOS'],
    'Filter': ['FILTER', 'BAND'],
    'ExpTime': ['EXPTIME'],
    'RA': ['RA', 'OBSRA', 'OBJRA', 'OBJCTRA', 'STRRQRA', 'OBJ-RA'],
    'DEC': ['DEC', 'OBSDEC', 'OBJDEC', 'OBJCTDEC', 'STRRQDE', 'OBJ-DEC'],
    'CalRA': ['CRVAL1'],
    'CalDEC': ['CRVAL2'],
    'CentX': ['CRPIX1'],
    'e_CentX': ['POSERRX'],
    'CentY': ['CRPIX2'],
    'e_CentY': ['POSERRY'],
    'PixScale': ['PIXSCALE'],
    'DetRotAng': ['DETROTANG'],
    'FWHM': ['FWHM'],
    'e_FWHM': ['FWHMERR'],
    'Airmass': ['AIRMASS', 'STROBAM'],
    'Date-Obs': ['DATE-OBS', 'DATE-BEG'],
    'Binning': ['BINNING'],
    'bias_cor': ['BIAS_COR'],
    'dark_cor': ['DARK_COR'],
    'flat_cor': ['FLAT_COR'],
    'WCS_cal': ['AST_CAL', 'WCSCAL', 'WCS-STAT'],
    'HasTrail': ['HASTRAIL'],
    'NTrail': ['NTRAIL']
}

# List of available catalogs for photometry
ALLCATALOGS = ['GAIADR1', 'GAIADR2', 'GAIADR3', 'GAIAEDR3',
               '2MASS', 'PS1DR1', 'PS1DR2',
               'GSC241', 'GSC242', 'GSC243']

# Astroquery catalogs
DEF_ASTROQUERY_CATALOGS = {
    # 'GAIADR3': 'gaiadr3.gaia_source',
    'GAIADR3': 'gaiadr3.gaia_source_lite',
    'GSC242': 'I/353/gsc242'
}

# Dictionary of supported astrometric catalogs with column name translations.
# These are the minimum columns necessary for alignment to work properly while
# taking into account all available information from the catalogs.
# Blank column names indicate the catalog does not have that column,
# and will set to None in the output table.
SUPPORTED_CATALOGS = {
    'GAIADR3': {'RA': 'ra',
                # 'RA_error': 'ra_error',
                'DEC': 'dec',
                # 'DEC_error': 'dec_error',
                'pmra': 'pmra', 'pmra_error': 'pmra_error',
                'pmdec': 'pmdec', 'pmdec_error': 'pmdec_error',
                'mag': 'phot_g_mean_mag', 'objID': 'SOURCE_ID',
                # 'epoch': 'ref_epoch'
                },
    'GAIAEDR3': {'RA': 'ra', 'RA_error': 'ra_error',
                 'DEC': 'dec', 'DEC_error': 'dec_error',
                 'pmra': 'pmra', 'pmra_error': 'pmra_error',
                 'pmdec': 'pmdec', 'pmdec_error': 'pmdec_error',
                 'mag': 'mag', 'objID': 'source_id', 'epoch': 'epoch'},
    'GAIADR2': {'RA': 'ra', 'RA_error': 'ra_error',
                'DEC': 'dec', 'DEC_error': 'dec_error',
                'pmra': 'pmra', 'pmra_error': 'pmra_error',
                'pmdec': 'pmdec', 'pmdec_error': 'pmdec_error',
                'mag': 'mag', 'objID': 'objID', 'epoch': 'epoch'},
    'GAIADR1': {'RA': 'ra', 'RA_error': 'ra_error',
                'DEC': 'dec', 'DEC_error': 'dec_error',
                'pmra': 'pmra', 'pmra_error': 'pmra_error',
                'pmdec': 'pmdec', 'pmdec_error': 'pmdec_error',
                'mag': 'mag', 'objID': 'objID', 'epoch': 'epoch'},
    '2MASS': {'RA': 'ra', 'RA_error': '',
              'DEC': 'dec', 'DEC_error': '',
              'pmra': '', 'pmra_error': '',
              'pmdec': '', 'pmdec_error': '',
              'mag': 'mag', 'objID': 'objid', 'epoch': 'jdate'},
    'PS1DR2': {'RA': 'ra', 'RA_error': 'raMeanErr',
               'DEC': 'dec', 'DEC_error': 'decMeanErr',
               'pmra': '', 'pmra_error': '',
               'pmdec': '', 'pmdec_error': '',
               'mag': 'mag', 'objID': 'objid', 'epoch': 'epochMean'},
    'PS1DR1': {'RA': 'ra', 'RA_error': 'RaMeanErr',
               'DEC': 'dec', 'DEC_error': 'DecMeanErr',
               'pmra': '', 'pmra_error': '',
               'pmdec': '', 'pmdec_error': '',
               'mag': 'mag', 'objID': 'objID', 'epoch': ''},
    'GSC243': {'RA': 'ra', 'RA_error': 'raErr',
               'DEC': 'dec', 'DEC_error': 'decErr',
               'pmra': 'rapm', 'pmra_error': 'rapmErr',
               'pmdec': 'decpm', 'pmdec_error': 'decpmErr',
               'mag': 'mag', 'objID': 'objID', 'epoch': 'epoch'},
    'GSC242': {'RA': 'ra', 'RA_error': 'raErr',
               'DEC': 'dec', 'DEC_error': 'decErr',
               'pmra': 'rapm', 'pmra_error': 'rapmErr',
               'pmdec': 'decpm', 'pmdec_error': 'decpmErr',
               'mag': 'mag', 'objID': 'objID', 'epoch': 'epoch'},
    'GSC241': {'RA': 'ra', 'RA_error': 'raErr',
               'DEC': 'dec', 'DEC_error': 'decErr',
               'pmra': 'rapm', 'pmra_error': 'rapmErr',
               'pmdec': 'decpm', 'pmdec_error': 'decpmErr',
               'mag': 'mag', 'objID': 'objID', 'epoch': 'epoch'}
}

# Dictionary of supported filter bands available in the supported catalogs
SUPPORTED_BANDS = {
    "U": ["GSC243", "GSC242"], "B": ["GSC243", "GSC242"],
    "V": ["GSC243", "GSC242"], "R": ["GSC243", "GSC242"], "I": ["GSC243", "GSC242"],
    "J": ["2MASS"], "H": ["2MASS"], "K": ["2MASS"], "Ks": ["2MASS"],
    "g": ["GSC243", "GSC242"], "r": ["GSC243", "GSC242"], "i": ["GSC243", "GSC242"], "u": ["GSC243", "GSC242"],
    "w": ["GSC243", "GSC242"], "y": ["GSC243", "GSC242"], "z": ["GSC243", "GSC242"]
}

# Dictionary of translations for the filter in the supported catalogs
CATALOG_FILTER_EXT = {
    'GAIAEDR3': None,
    'GAIADR3': None,
    'GAIADR2': None,
    'GAIADR1': None,
    '2MASS': {"J": {'Prim': [['j_m', 'j_cmsig']], 'Alt': None},
              "H": {'Prim': [['h_m', 'h_cmsig']], 'Alt': None},
              "K": {'Prim': [['k_m', 'k_cmsig']], 'Alt': None},
              "Ks": {'Prim': [['k_m', 'k_cmsig']], 'Alt': None}},
    'PS1DR2': {"u": {'Prim': [['SDSSuMag', 'SDSSuMagErr']], 'Alt': None},
               "g": {'Prim': [['SDSSgMag', 'SDSSgMagErr']], 'Alt': None},
               "r": {'Prim': [['SDSSrMag', 'SDSSrMagErr']], 'Alt': None},
               "i": {'Prim': [['SDSSiMag', 'SDSSiMagErr']], 'Alt': None},
               "z": {'Prim': [['SDSSzMag', 'SDSSzMagErr']], 'Alt': None},
               "y": {'Prim': [['PS1yMag', 'PS1ymagErr']], 'Alt': None}},
    'PS1DR1': None,
    'GSC243': {"U": {'Prim': [['Umag', 'UmagErr']],
                     'Alt': [['SDSSuMag', 'SDSSuMagErr'],
                             ['SDSSgMag', 'SDSSgMagErr'],
                             ['SDSSrMag', 'SDSSrMagErr']]},
               "B": {'Prim': [['Bmag', 'BmagErr']],
                     'Alt': [['SDSSuMag', 'SDSSuMagErr'],
                             ['SDSSgMag', 'SDSSgMagErr'],
                             ['SDSSrMag', 'SDSSrMagErr']]},
               "V": {'Prim': [['Vmag', 'VmagErr']],
                     'Alt': [['SDSSuMag', 'SDSSuMagErr'],
                             ['SDSSgMag', 'SDSSgMagErr'],
                             ['SDSSrMag', 'SDSSrMagErr']]},
               "R": {'Prim': [['Rmag', 'RmagErr']],
                     'Alt': [['SDSSgMag', 'SDSSgMagErr'],
                             ['SDSSrMag', 'SDSSrMagErr'],
                             ['SDSSiMag', 'SDSSiMagErr']]},
               "I": {'Prim': [['Imag', 'ImagErr']],
                     'Alt': [['SDSSrMag', 'SDSSrMagErr'],
                             ['SDSSiMag', 'SDSSiMagErr'],
                             ['SDSSzMag', 'SDSSzMagErr']]},
               "g": {'Prim': [['SDSSgMag', 'SDSSgMagErr']], 'Alt': None},
               "r": {'Prim': [['SDSSrMag', 'SDSSrMagErr']], 'Alt': None},
               "i": {'Prim': [['SDSSiMag', 'SDSSiMagErr']], 'Alt': None},
               "u": {'Prim': [['SDSSuMag', 'SDSSuMagErr']], 'Alt': None},
               "w": {'Prim': [['SDSSgMag', 'SDSSgMagErr'],
                              ['SDSSrMag', 'SDSSrMagErr']], 'Alt': None},
               "z": {'Prim': [['SDSSzMag', 'SDSSzMagErr']], 'Alt': None}},
    'GSC242': {"U": {'Prim': [['Umag', 'UmagErr']],
                     'Alt': [['SDSSuMag', 'SDSSuMagErr'],
                             ['SDSSgMag', 'SDSSgMagErr'],
                             ['SDSSrMag', 'SDSSrMagErr']]},
               "B": {'Prim': [['Bmag', 'BmagErr']],
                     'Alt': [['SDSSuMag', 'SDSSuMagErr'],
                             ['SDSSgMag', 'SDSSgMagErr'],
                             ['SDSSrMag', 'SDSSrMagErr']]},
               "V": {'Prim': [['Vmag', 'VmagErr']],
                     'Alt': [['SDSSuMag', 'SDSSuMagErr'],
                             ['SDSSgMag', 'SDSSgMagErr'],
                             ['SDSSrMag', 'SDSSrMagErr']]},
               "R": {'Prim': [['Rmag', 'RmagErr']],
                     'Alt': [['SDSSgMag', 'SDSSgMagErr'],
                             ['SDSSrMag', 'SDSSrMagErr'],
                             ['SDSSiMag', 'SDSSiMagErr']]},
               "I": {'Prim': [['Imag', 'ImagErr']],
                     'Alt': [['SDSSrMag', 'SDSSrMagErr'],
                             ['SDSSiMag', 'SDSSiMagErr'],
                             ['SDSSzMag', 'SDSSzMagErr']]},
               "g": {'Prim': [['SDSSgMag', 'SDSSgMagErr']], 'Alt': None},
               "r": {'Prim': [['SDSSrMag', 'SDSSrMagErr']], 'Alt': None},
               "i": {'Prim': [['SDSSiMag', 'SDSSiMagErr']], 'Alt': None},
               "w": {'Prim': [['SDSSgMag', 'SDSSgMagErr'],
                              ['SDSSrMag', 'SDSSrMagErr']], 'Alt': None},
               "z": {'Prim': [['SDSSzMag', 'SDSSzMagErr']], 'Alt': None}},
    'GSC241': None
}

# Color conversion factors for transformation from SSDS mag to BVRI following Lupton (2005)
CONV_SSDS_BVRI = {'Bmag': [[-0.8116, 0.1313, 0.0095],
                           [0.3130, 0.2271, 0.0107]],
                  'Vmag': [[-0.2906, 0.0885, 0.0129],
                           [-0.5784, -0.0038, 0.0054]],
                  'Rmag': [[-0.1837, -0.0971, 0.0106],
                           [-0.2936, -0.1439, 0.0072]],
                  'Imag': [[-1.2444, -0.3820, 0.0078],
                           [-0.3780, -0.3974, 0.0063]]}

# Dictionary of satellite properties
SATELLITE_PROPERTIES = {
    'ONEWEB': {
        'identifiers': ['ONEWEB', 'OW'],
        'format_number': lambda x: f"ONEWEB-{int(x):04d}",
        'orbit_height': 1200.0
    },
    'STARLINK': {
        'identifiers': ['STARLINK'],
        'format_number': lambda x: f"STARLINK-{int(x):04d}" if len(x) == 4 else f"STARLINK-{int(x):05d}",
        'orbit_height': 550.0
    },
    'BLUEWALKER': {
        'identifiers': ['BLUEWALKER', 'BW'],
        'format_number': lambda x: f"BLUEWALKER-{int(x)}",
        'orbit_height': 500.0
    },
    'KUIPER': {
        'identifiers': ['KUIPER'],
        'format_number': lambda x: f"KUIPER-P{int(x)}",
        'orbit_height': 630.0
    },
    'SPACEMOBILE': {
        'identifiers': ['SPACEMOBILE'],
        'format_number': lambda x: f"SPACEMOBILE-{int(x):03d}",
        'orbit_height': 550.0
    }
}


def get_nominal_orbit_altitude(sat_id: str) -> float:
    """
    Get the nominal orbit altitude for a given satellite ID.

    Args:
        sat_id (str): The satellite ID.

    Returns:
        float: The nominal orbit altitude in kilometers.
    """
    for key, props in SATELLITE_PROPERTIES.items():
        if key in sat_id:
            return props['orbit_height']
    # Default to ONEWEB orbit height if no match is found
    return SATELLITE_PROPERTIES['ONEWEB']['orbit_height']


REARTH_EQU = 6378.137  # Radius at sea level at the equator in km
REARTH_POL = 6356.752  # Radius at poles in km

AU_TO_KM = 149597892.  # km/au

# URL configuration for the photometry pipeline
ASTROMETRIC_CAT_ENVVAR = "ASTROMETRIC_CATALOG_URL"
DEF_CAT_URL = 'https://gsss.stsci.edu/webservices'

if ASTROMETRIC_CAT_ENVVAR in os.environ:
    SERVICELOCATION = os.environ[ASTROMETRIC_CAT_ENVVAR]
else:
    SERVICELOCATION = DEF_CAT_URL


def check_version(log):
    """Crosscheck the current version with the GitHub version"""
    url_github_version = 'https://raw.githubusercontent.com/CLEOsat-group/leosatpy/master/leosatpy/utils/version.py'

    log.info("> Crosscheck local version with GitHub")

    try:
        version_string = str(request.urlopen(url_github_version).readlines()[0], 'utf-8').strip()
        version_github = version_string[version_string.rfind('=') + 1:].replace(' ', '').replace('\'', '')
    except (Exception, ValueError):
        version_github = ''

    if version_github and version.parse(__version__) < version.parse(version_github):
        s = ' ' * 15
        log.warning(f'  ==> A newer version of LEOSatpy is available (v{__version__} --> v{version_github}).\n '
                    f'{s} Please consider updating LEOSatpy by typing "pip install leosatpy --upgrade" in the terminal.')
        ask_str = BCOLORS.OKGREEN + "[    INFO]   >> Press Enter to continue <<" + BCOLORS.ENDC
        input(ask_str)
    elif version_github and version.parse(__version__) > version.parse(version_github):
        log.info("  ==> You are using a newer version of LEOSatpy than the official release.")
    else:
        all_good = BCOLORS.PASS + "ALL GOOD" + BCOLORS.OKGREEN
        log.info(f"  ==> {all_good}: You are using the latest official release of LEOSatpy." + BCOLORS.ENDC)


def print_progress_bar(iteration, total, prefix='', length=50, fill='\u2588', color=None, use_lock=False):
    """
    Print a progress bar in the terminal with ANSI color formatting.

    Parameters
    ----------
    iteration : int
        Current iteration step.
    total : int
        Total number of iterations.
    prefix : str, optional
        Prefix to display before the progress bar.
    length : int, optional
        Length of the progress bar in characters.
    fill : str, optional
        Character used to fill the progress bar.
    color : str, optional
        ANSI color code to colorize the progress bar.
    use_lock : bool, optional
        Whether to use a lock for printing (useful in multithreaded contexts).
    """

    # Ensure the percentage reaches 100% at the final iteration
    if iteration >= total:
        percent = "100.0"
        filled_length = length
    else:
        percent = "{0:.1f}".format(100 * (iteration / float(total)))
        filled_length = int(length * iteration // total)

    bar = fill * filled_length + '-' * (length - filled_length)

    color_prefix = color if color else ''
    reset_color = BCOLORS.ENDC if color else ''
    output = (f'\r[{color_prefix}PROGRESS{reset_color}]'
              f'{color_prefix}{prefix}{reset_color} |{bar}| '
              f'{color_prefix}{percent}%{reset_color}')

    if use_lock:
        with print_lock:
            sys.stdout.write(output)
            sys.stdout.flush()
    else:
        sys.stdout.write(output)
        sys.stdout.flush()
