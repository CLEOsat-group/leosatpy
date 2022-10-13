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
import os
import logging
import warnings
import numpy as np
from colorlog import ColoredFormatter
from astropy import wcs
from astropy.io import fits
from astropy.utils.exceptions import (AstropyUserWarning, AstropyWarning)

from config.telescope_conf import *

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


ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))

# # default path to the result .csv containing all the info on an observation.
# # !!! Change this to: /path/to/my_result_file_name.csv
# DEF_RES_TBL_PATH = '/home/cadam/Arbeit/projects/UniversidadAntofagasta/LEOSat/data'
# DEF_RES_TBL_FNAME = 'results_LEOSat.csv'


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


# potential FITS header keywords for looking up the instrument
# any unique header keyword works as a potential identifier
INSTRUMENT_KEYS = ['PPINSTRU', 'LCAMMOD', 'FPA', 'CAM_NAME', 'INSTRUME',
                   'TELESCOP', 'FLI']

# default timeout for selection
DEF_TIMEOUT = 3  # in seconds

# time delta in days for folder search in reduction of calibration files
TIMEDELTA_DAYS = 7  # +-days
FRMT = "%Y-%m-%dT%H:%M:%S.%f"

# # keywords relevant for reduction
# CORRECT_GAIN = False
# EST_UNCERTAINTY = False
# COMBINE_METHOD = 'average'  # Bias, dark and flat combine method; can be median as well
# COMBINE_METHOD_FLAT = 'median'  # Bias, dark and flat combine method; can be median as well

IMAGETYP_LIGHT = 'science'
IMAGETYP_BIAS = 'bias'
IMAGETYP_DARK = 'dark'
IMAGETYP_FLAT = 'flat'

BINNING_DKEYS = ('BINX', 'BINY')

IMAGETYP_REDUCE_ORDER = np.array(['bias', 'darks', 'flats'])
IMAGETYP_COMBOS = {'bias': "zero|bias|bias frame",
                   'darks': "dark|dark frame",
                   'flats': "flat|flat frame"}


# Default result table column names.
# This CAN be modified, by adding, removing, or renaming keywords.
# !!!
# Caution: Keywords extracted from header values have to be specified in the DEF_KEY_TRANSLATIONS and
# the telescope configuration
DEF_RES_TBL_COL_NAMES = ['File', 'Object', 'Sat-Name', 'AltID', 'UniqueID',
                         'Telescope', 'RA', 'DEC', 'Date-Obs', 'Filter', 'ExpTime',
                         'Airmass', 'Binning',
                         'Obs-Start', 'Obs-Mid', 'Obs-Stop',
                         'HasTrail',
                         'UT Date', 'UT time',
                         'SatLon', 'SatLat', 'SatAlt',
                         'SatAz', 'SatElev', 'SatRA', 'SatDEC',
                         'SunRA', 'SunDEC', 'SunZenithAngle', 'SatAngularSpeed',
                         'ObsSatRange',
                         'ToD', 'e_ToD',
                         'ObsTrailLength', 'e_ObsTrailLength', 'EstTrailLength', 'e_EstTrailLength',
                         'SunSatAng', 'SunPhaseAng', 'SunIncAng', 'ObsAng',
                         'ObsMag', 'e_ObsMag', 'EstMag', 'e_EstMag',
                         'EstScaleMag', 'e_EstScaleMag', 'SunElevAng',
                         'FluxScale', 'MagScale', 'MagCorrect', 'e_MagCorrect', 'dt_tle-obs',
                         'TrailCX', 'e_TrailCX', 'TrailCY', 'e_TrailCY',
                         'TrailCRA', 'e_TrailCRA', 'TrailCDEC', 'e_TrailCDEC',
                         'TrailANG', 'e_TrailANG', 'OptAperHeight',
                         'CalRA', 'CalDEC', 'CentX', 'CentY', 'PixScale', 'DetRotAng',
                         'bias_cor', 'dark_cor', 'flat_cor', 'WCS_cal', 'mag_conv']

DEF_RES_TBL_COL_UNITS = ['', '', '', '',
                         '', '', '', '(J2000)', '', '(s)',
                         '', '',
                         '(hh:mm:ss.sss)', '(hh:mm:ss.sss)', '(hh:mm:ss.sss)',
                         '',
                         '', '',
                         '(deg)', '(deg)', '(km)',
                         '(deg)', '(deg)', '(hr)', '(deg)',
                         '(hr)', '(deg)', '(deg)', '(arcsec/px)',
                         '(s)', '(s)',
                         '(arcsec)', '(arcsec)', '(arcsec)', '(arcsec)',
                         '(deg)', '(deg)', '(deg)', '(deg)',
                         '(mag)', '(mag)', '(mag)', '(mag)',
                         '(mag)', '(mag)', '(deg)',
                         '', '(mag)', '(s)',
                         '(deg)', '(deg)', '(px)', '(px)', '(arcsec/px)', '(deg)',
                         '(px)', '(px)', '(px)', '(px)',
                         '(deg)', '(arcsec)', '(deg)', '(arcsec)',
                         '(deg)', '(px)',
                         '', '', '', '']

DEF_VIS_TBL_COL_NAMES = ['ID', 'AltID', 'UniqueID', 'UT Date', 'UT time',
                         'SatLon', 'SatLat', 'SatAlt',
                         'SatAz', 'SatElev', 'SatRA', 'SatDEC',
                         'SunRA', 'SunDEC', 'SunZenithAngle',
                         'SatAngularSpeed']

# Translations between table and fits header keywords
DEF_KEY_TRANSLATIONS = {
    'Object': ['OBJECT'],
    'Telescope': ['TELESCOP', 'OBSERVAT'],
    'Filter': ['FILTER'],
    'ExpTime': ['EXPTIME'],
    'RA': ['RA', 'OBSRA', 'OBJRA', 'OBJCTRA'],
    'DEC': ['DEC', 'OBSDEC', 'OBJDEC', 'OBJCTDEC'],
    'CalRA': ['CRVAL1'],
    'CalDEC': ['CRVAL2'],
    'CentX': ['CRPIX1'],
    'CentY': ['CRPIX2'],
    'PixScale': ['PIXSCALE'],
    'DetRotAng': ['DETROTANG'],
    'Airmass': ['AIRMASS'],
    'Date-Obs': ['DATE-OBS'],
    'Binning': ['BINNING'],
    'bias_cor': ['BIAS_COR'],
    'dark_cor': ['DARK_COR'],
    'flat_cor': ['FLAT_COR'],
    'WCS_cal': ['AST_CAL'],
    'HasTrail': ['HASTRAIL']
}

# list of available catalogs for photometry
ALLCATALOGS = ['GAIADR1', 'GAIADR2', 'GAIADR3', 'GAIAEDR3',
               '2MASS', 'PS1DR1', 'PS1DR2',
               'GSC241', 'GSC242']

DEF_PHOTOMETRY_CATALOG = 'GSC242'

# Dictionary of supported astrometric catalogs with column name translations
# These are the minimum columns necessary for alignment to work properly while
# taking into account all available information from the catalogs.
# Blank column names indicate the catalog does not have that column,
# and will set to None in the output table.
SUPPORTED_CATALOGS = {
    'GAIAEDR3': {'RA': 'ra', 'RA_error': 'ra_error',
                 'DEC': 'dec', 'DEC_error': 'dec_error',
                 'pmra': 'pmra', 'pmra_error': 'pmra_error',
                 'pmdec': 'pmdec', 'pmdec_error': 'pmdec_error',
                 'mag': 'mag', 'objID': 'objID', 'epoch': 'epoch'},
    'GAIADR3': {'RA': 'ra', 'RA_error': 'ra_error',
                'DEC': 'dec', 'DEC_error': 'dec_error',
                'pmra': 'pmra', 'pmra_error': 'pmra_error',
                'pmdec': 'pmdec', 'pmdec_error': 'pmdec_error',
                'mag': 'mag', 'objID': 'objID', 'epoch': 'epoch'},
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
    "U": "GSC242", "B": "GSC242", "V": "GSC242", "R": "GSC242", "I": "GSC242",
    "J": "2MASS", "H": "2MASS", "K": "2MASS", "Ks": "2MASS",
    "g": "GSC242", "r": "GSC242", "i": "GSC242", "z": "GSC242", "y": "GSC242"
}

# Dictionary of translations for the filter in the supported catalogs
CATALOG_FILTER_EXT = {
    'GAIAEDR3': None,
    'GAIADR3': None,
    'GAIADR2': None,
    'GAIADR1': None,
    '2MASS': {"J": {'Prim': ['j_m', 'j_cmsig'], 'Alt': None},
              "H": {'Prim': ['h_m', 'h_cmsig'], 'Alt': None},
              "K": {'Prim': ['k_m', 'k_cmsig'], 'Alt': None},
              "Ks": {'Prim': ['k_m', 'k_cmsig'], 'Alt': None}},
    'PS1DR2': {"g": {'Prim': ['SDSSgMag', 'SDSSgMagErr'], 'Alt': None},
               "r": {'Prim': ['SDSSrMag', 'SDSSrMagErr'], 'Alt': None},
               "i": {'Prim': ['SDSSiMag', 'SDSSiMagErr'], 'Alt': None},
               "z": {'Prim': ['SDSSzMag', 'SDSSzMagErr'], 'Alt': None},
               "y": {'Prim': ['PS1yMag', 'PS1ymagErr'], 'Alt': None}},
    'PS1DR1': None,
    'GSC242': {"U": {'Prim': ['UMag', 'UMagErr'], 'Alt': None},
               "B": {'Prim': ['BMag', 'BMagErr'],
                     'Alt': [['SDSSuMag', 'SDSSuMagErr'],
                             ['SDSSgMag', 'SDSSgMagErr'],
                             ['SDSSrMag', 'SDSSrMagErr']]},
               "V": {'Prim': ['VMag', 'VMagErr'],
                     'Alt': [['SDSSuMag', 'SDSSuMagErr'],
                             ['SDSSgMag', 'SDSSgMagErr'],
                             ['SDSSrMag', 'SDSSrMagErr']]},
               "R": {'Prim': ['RMag', 'RMagErr'],
                     'Alt': [['SDSSgMag', 'SDSSgMagErr'],
                             ['SDSSrMag', 'SDSSrMagErr'],
                             ['SDSSiMag', 'SDSSiMagErr']]},
               "I": {'Prim': ['IMag', 'IMagErr'],
                     'Alt': [['SDSSrMag', 'SDSSrMagErr'],
                             ['SDSSiMag', 'SDSSiMagErr'],
                             ['SDSSzMag', 'SDSSzMagErr']]},
               "g": {'Prim': ['SDSSgMag', 'SDSSgMagErr'], 'Alt': None},
               "r": {'Prim': ['SDSSrMag', 'SDSSrMagErr'], 'Alt': None},
               "i": {'Prim': ['SDSSiMag', 'SDSSiMagErr'], 'Alt': None},
               "z": {'Prim': ['SDSSzMag', 'SDSSzMagErr'], 'Alt': None}},
    'GSC241': None
}

# Color conversion factors for transformation from SSDS mag to BVRI following Lupton (2005)
CONV_SSDS_BVRI = {'BMag': [[-0.8116, 0.1313, 0.0095],
                           [0.3130, 0.2271, 0.0107]],
                  'VMag': [[-0.2906, 0.0885, 0.0129],
                           [-0.5784, -0.0038, 0.0054]],
                  'RMag': [[-0.1837, -0.0971, 0.0106],
                           [-0.2936, -0.1439, 0.0072]],
                  'IMag': [[-1.2444, -0.3820, 0.0078],
                           [-0.3780, -0.3974, 0.0063]]}

# background estimation
# RING_RADIUS_IN = 15  # ring filter radius
# RING_WIDTH = 7  # ring filter width

# resolution of scale and rotation estimation.
NUM_BINS_DIST = 7500  # number of distance bins
BINS_ANG_FAC = 10  # multiplier for rotation resolution

MAX_AREA_LIMIT = 512  # area in pixel above which the source is considered to be saturated

# frequency cut off threshold
FREQU_THRESHOLD = 0.02

# source detection
NUM_SOURCES_MAX = 500  # maximum number of sources to consider
RMS_PX_THRESHOLD = 10  # threshold for rms calculation
USE_N_SOURCES = 150  # number of sources to be used in fast mode
OFFSET_BINWIDTH = 1  # width in pixel for the offset determination

# maximum number of trails expected in the observation
N_TRAILS_MAX = 1

# number of angles to use in hough transform
NUM_THETAS = 6200

# Hough space sub-window size
SUB_WIN_SIZE_HOUGH = (150, 80)

# standard star and trail photometry
NUM_STD = 100  # maximum number of standard stars sources to consider
NUM_STD_MIN = 3  # minimum number of standard stars sources in a given band

# start, stop and step size for optimal aperture estimation from S/N in multiple of fwhm
APER_START = 0.1
APER_STOP = 5
APER_STEP_SIZE = 0.1

# inner and outer radius of circular aperture for sky estimation in aperture photometry
RSKYIN_STD = 20
RSKYOUT_STD = 30

# inner and outer height of rectangular aperture for sky estimation in aperture photometry
HSKYIN_STD = 25
HSKYOUT_STD = 35

# centroid box size in pixels
BOX_SIZE = 11

# plot options
# FIG_TYPE = 'png'
# LINE_LENGTH = 5  # scale line length in arcmin
# ARRW_LENGTH = 0.15  # compass arrow length; 15% of image size

# URL configuration for photometry pipeline
ASTROMETRIC_CAT_ENVVAR = "ASTROMETRIC_CATALOG_URL"
DEF_CAT_URL = 'http://gsss.stsci.edu/webservices'

if ASTROMETRIC_CAT_ENVVAR in os.environ:
    SERVICELOCATION = os.environ[ASTROMETRIC_CAT_ENVVAR]
else:
    SERVICELOCATION = DEF_CAT_URL

SAT_HORB_REF = {'STARLINK': 550., 'ONEWEB': 1200.}

REARTH_EQU = 6378.137  # Radius at sea level at the equator in km
REARTH_POL = 6356.752  # Radius at poles in km

AU_TO_KM = 149597892.  # km/au
