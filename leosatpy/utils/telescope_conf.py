#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Name:         telescope_conf.py
# Purpose:      Configuration file with observatory, telescope and
#               instrument information required for processing and analysis.
#
#               Telescope details were taken from the Satellite tracking script
#               constants.py by Edgar Ortiz edgar.ortiz@uamail.cl and
#               Jeremy Tregloan-Reed jeremy.tregloan-reed@uda.cl
#
#
#
# Author:       p4adch (cadam)
#
# Created:      04/15/2022
# Copyright:    (c) p4adch 2010-
#
# Changelog:    added Ckoirama telescope
# -----------------------------------------------------------------------------

""" Modules """
from collections import OrderedDict

""" Meta-info """
__author__ = "Christian Adam"
__copyright__ = 'Copyright 2021, UA, LEOSat observations'
__credits__ = ["Christian Adam, Eduardo Unda-Sanzana, Jeremy Tregloan-Reed"]
__license__ = "Free"
__version__ = "0.1.0"
__maintainer__ = "Christian Adam"
__email__ = "christian.adam84@gmail.com"
__status__ = "Production"

# Danish 1.54m, DFOSC
dk154_params = {

    # observatory name and location information
    "name": "Danish 1.54m telescope at La Silla Observatory",
    "longitude": -70.7403,
    "latitude": -29.263,
    "altitude": 2363,
    "tz": -4,

    # telescope keywords
    'telescope_instrument': 'DFOSC_FASU',  # telescope/instrument name
    'telescope_keyword': 'DK-1.54',  # telescope/instrument keyword
    'observatory_code': '809',  # MPC observatory code

    # instrument-specific FITS header keywords
    'telescop': 'TELESCOP',  # telescope keyword
    'instrume': 'INSTRUME',  # instrument keyword

    # detector keywords
    'extent': ('NAXIS1', 'NAXIS2'),  # N_pixels in x/y
    'n_amps': 1,  # the number of amplifiers on the chip
    'namps_yx': {'1x1': None},
    'amplist': {'1x1': None},  # amplifier list keyword
    'secpix': ('SECPPIX', 'SECPPIX'),  # unbinned pixel size (arcsec)
    'binning': ('BINX', 'BINY'),  # binning in x/y
    'image_size_1x1': (2148, 2048),
    'gain': 0.25,  # CD gain in el/DN
    'readnoise': 4.5,  # CCD Read Out Noise (e-)

    # telescope pointing keywords
    'ra': 'OBJRA',  # telescope pointing, RA
    'dec': 'OBJDEC',  # telescope pointing, Dec
    'radec_separator': ':',  # RA/Dec hms separator, use 'XXX' if already in degrees
    'equinox': 'EQUINOX',

    # observation keywords
    'date_keyword': 'DATE-OBS',  # obs date/time

    'imagetyp': 'IMAGETYP',
    'object': 'OBJECT',  # object name keyword
    'filter': 'FILTB',  # filter keyword

    # filter name translation dictionary
    'filter_translations': {'U': 'U', 'B': 'B', 'V': 'V', 'R': 'R',
                            'I': 'I', 'empty': None},

    'exptime': 'EXPTIME',  # exposure time keyword (s)
    'airmass': 'AIRMASS',  # airmass keyword

    # reduction keywords
    # define the trim section for trimming of unwanted areas
    'ampsec': {'1x1': None,
               '2x2': None,
               '4x4': None},
    'oscansec': {'1x1': None,
                 # '1x1': {'11': [6, 22, 0, 6144]},
                 '2x2': None,
                 '4x4': None},
    'trimsec': {'1x1': {'11': [0, 2060, 66, 2096]},
                '2x2': {'11': [10, 1020, 49, 1049]},
                '4x4': {'11': [0, 512, 15, 525]}},

    # ccd mask for regions to exclude from all processes
    'ccd_mask': {'1x1': None,
                 '2x2': None,
                 '4x4': None},

    # telescope specific image types
    'imagetyp_light': 'light',
    'imagetyp_bias': 'bias',
    'imagetyp_dark': 'dark',
    'imagetyp_flat': 'flat',
    'add_imagetyp_keys': False,
    'add_imagetyp': {'flat': {'object': 'flat_v'}},

    # defaults for reduction
    # 'bias_cor': True,
    # 'dark_cor': False,
    # 'flat_cor': True,
    # 'flat_dark_cor': False,
    # 'cosmic_correct': True,

    # keywords to exclude from the header update while preparing the file
    'keys_to_exclude': None,

    # keywords for info_dict
    'obs_keywords': ['telescop', 'instrume', 'object', 'date-obs', 'filter', 'exptime',
                     'objra', 'objdec', 'airmass', 'binning'],

    # source extractor settings
    'saturation_limit': 6e5,
    'apply_mask': True
}

# ChungBuk National University Observatory South Korea, CBNUO-JC
cbnuo_params = {

    # observatory name and location information
    "name": "ChungBuk National University Observatory",
    # "longitude": 232.524644889,
    "longitude": 127.475355111,
    "latitude": 36.7815,
    "altitude": 86.92,
    "tz": 9,

    # telescope keywords
    'telescope_instrument': 'SBIG STX-16803 CCD Camera',  # telescope/instrument name
    'telescope_keyword': 'CBNUO-JC',  # telescope/instrument keyword
    'observatory_code': None,  # MPC observatory code

    # instrument-specific FITS header keywords
    'telescop': 'OBSERVAT',  # telescope keyword
    'instrume': 'INSTRUME',  # instrument keyword

    # detector keywords
    'extent': ('NAXIS1', 'NAXIS2'),  # N_pixels in x/y
    'n_amps': 1,  # number of chips on detector
    'namps_yx': {'1x1': None},
    'amplist': {'1x1': None},  # amplifier list keyword
    'secpix': (1.05, 1.05),  # pixel size (arcsec)
    'binning': ('BINX', 'BINY'),  # binning
    'image_size_1x1': (4096, 4096),
    'gain': 'GAIN',  # CD gain in el/DN
    'readnoise': 'RDNOISE',  # CCD Read Out Noise (e-)

    # telescope pointing keywords
    'ra': 'RA',  # telescope pointing, RA
    'dec': 'DEC',  # telescope pointing, Dec
    'radec_separator': ':',  # RA/Dec hms separator, use 'XXX' if already in degrees
    'equinox': None,

    # observation keywords
    'date_keyword': 'DATE-OBS',  # obs date/time

    'imagetyp': 'IMAGETYP',
    'object': 'OBJECT',  # object name keyword
    'filter': 'FILTER',  # filter keyword
    'filter_translations': {'V': 'V', 'R': 'R',
                            'I': 'I', 'B': 'B', 'N': None},  # filter name translation dictionary

    'exptime': 'EXPTIME',  # exposure time keyword (s)
    'airmass': 'AIRMASS',  # airmass keyword

    # reduction keywords
    # define the trim section for trimming of unwanted areas
    'ampsec': {'1x1': None,
               '2x2': None,
               '4x4': None},
    'oscansec': {'1x1': None,
                 # '1x1': {'11': [6, 22, 0, 6144]},
                 '2x2': None,
                 '4x4': None},
    'trimsec': {'1x1': None,
                '2x2': None,
                '4x4': None},

    # ccd mask for regions to exclude from all processes
    'ccd_mask': {'1x1': None,
                 '2x2': None,
                 '4x4': None},

    # telescope specific image types
    'imagetyp_light': 'object',
    'imagetyp_bias': 'zero',
    'imagetyp_dark': 'dark',
    'imagetyp_flat': 'flat',
    'add_imagetyp_keys': False,

    # keywords to exclude from the header update while preparing the file
    'keys_to_exclude': ['DELMAG', 'FWHMH', 'FWHMV', 'FWHMHS', 'FWHMVS', 'NSTAR'],

    # keywords for info_dict
    'obs_keywords': ['observat', 'instrume', 'object', 'date-obs', 'time-obs', 'filter', 'exptime',
                     'ra', 'dec', 'airmass', 'binning'],

    # source extractor settings
    'saturation_limit': 6.5e4,
    'apply_mask': True

}

# Calar Alto 1.23m, DLR-MKIII
ca123dlrmkiii_param = {

    # observatory name and location information
    "name": "Calar Alto Observatory",
    "longitude": -2.540997836,
    "latitude": 37.22083245,
    "altitude": 2168,
    "tz": -1,

    # telescope keywords
    'telescope_instrument': 'Calar Alto 1.23m/DLR-MKIII',  # telescope/instrument
    'telescope_keyword': 'CA 1.23m',  # telescope/instrument keyword
    'observatory_code': '493',  # MPC observatory code

    # instrument-specific FITS header keywords
    'telescop': 'TELESCOP',  # telescope keyword
    'instrume': 'INSTRUME',  # instrument keyword

    # detector keywords
    'extent': ('NAXIS1', 'NAXIS2'),  # N_pixels in x/y
    'n_amps': 1,  # number of chips on detector
    'namps_yx': {'1x1': None},
    'amplist': {'1x1': None},  # amplifier list keyword
    'secpix': ('SCALE', 'SCALE'),  # unbinned pixel size (arcsec)
    'binning': ('CCDBINX', 'CCDBINY'),  # binning keyword
    'image_size_1x1': (4096, 4112),
    'gain': 'GAIN',  # CD gain in el/DN
    'readnoise': None,  # CCD Read Out Noise (e-)

    # telescope pointing keywords
    'ra': 'RA',  # telescope pointing, RA
    'dec': 'DEC',  # telescope pointing, Dec
    'radec_separator': 'XXX',  # RA/Dec hms separator, use 'XXX' if already in degrees
    'equinox': 'EQUINOX',

    # observation keywords
    'date_keyword': 'DATE-OBS',  # obs date/time

    'imagetyp': 'IMAGETYP',
    'object': 'OBJECT',  # object name keyword
    'filter': 'FILTER',  # filter keyword
    # filtername translation dictionary
    'filter_translations': {'V': 'V',
                            'R': 'R',
                            'free': None},
    'exptime': 'EXPTIME',  # exposure time keyword (s)
    'airmass': 'AIRMASS',  # airmass keyword

    # reduction keywords
    # define the trim section for trimming of unwanted areas
    'ampsec': {'1x1': None,
               '2x2': None,
               '4x4': None},
    'oscansec': {'1x1': None,
                 # '1x1': {'11': [6, 22, 0, 6144]},
                 '2x2': None,
                 '4x4': None},
    'trimsec': {'1x1': None,
                '2x2': None,
                '4x4': None},

    # ccd mask for regions to exclude from all processes
    'ccd_mask': {'1x1': None,
                 '2x2': None,
                 '4x4': None},

    # telescope specific image types
    'imagetyp_light': 'science',
    'imagetyp_bias': 'bias',
    'imagetyp_dark': 'dark',
    'imagetyp_flat': 'flat',
    'add_imagetyp_keys': False,

    # defaults for reduction
    # 'bias_cor': True,
    # 'dark_cor': False,
    # 'flat_cor': False,
    # 'flat_dark_cor': False,
    # 'cosmic_correct': True,

    # keywords to exclude from the header update while preparing the file
    'keys_to_exclude': None,

    # keywords for info_dict
    'obs_keywords': ['telescop', 'instrume', 'object', 'date-obs', 'filter', 'exptime',
                     'ra', 'dec', 'airmass', 'binning'],

    # source extractor settings
    'saturation_limit': 5e4,
    'apply_mask': True
}

# Chakana 0.6m telescope ckoirama
ckoir_param = {

    # telescope location information
    "name": "Ckoirama Observatory, Universidad de Antofagasta, Chile",
    "longitude": -69.93058889,
    "latitude": -24.08913333,
    "altitude": 966.0,
    "tz": -4,

    # telescope keywords
    'telescope_instrument': 'FLI',  # telescope/instrument name
    'telescope_keyword': 'PlaneWave CDK24',  # telescope/instrument keyword
    'observatory_code': '',  # MPC observatory code

    # instrument-specific FITS header keywords
    'telescop': 'TELESCOP',  # telescope keyword
    'instrume': 'INSTRUME',  # instrument keyword

    'extent': ('NAXIS1', 'NAXIS2'),  # N_pixels in x/y
    'n_amps': 1,  # number of chips on detector
    'namps_yx': {'1x1': None},
    'amplist': {'1x1': None},  # amplifier list keyword
    'secpix': (0.475, 0.475),  # pixel size (arcsec)
    'binning': ('XBINNING', 'YBINNING'),  # binning in x/y
    'image_size_1x1': (4096, 4096),
    # gain and readout noise taken from https://people.bsu.edu/rberring/observing-facilities/
    # for FLI PL16801 camera
    'gain': 1.43,  # CD gain in el/DN
    'readnoise': 10.83,  # CCD Read Out Noise (e-)

    # telescope pointing keywords
    'ra': 'OBJCTRA',  # telescope pointing, RA
    'dec': 'OBJCTDEC',  # telescope pointing, Dec
    'radec_separator': ' ',  # RA/Dec hms separator, use 'XXX' if already in degrees
    'equinox': 'EQUINOX',

    # keyword; use 'date|time' if separate
    'date_keyword': 'DATE-OBS',  # obs date/time

    'imagetyp': 'IMAGETYP',
    'object': 'OBJECT',  # object name keyword
    'filter': 'FILTER',  # filter keyword

    # filtername translation dictionary
    'filter_translations': {'u\'': 'u', 'g\'': 'g',
                            'r\'': 'r', 'i\'': 'i',
                            'z\'': 'z'},

    'exptime': 'EXPTIME',  # exposure time keyword (s)
    'airmass': 'AIRMASS',  # airmass keyword

    # reduction keywords
    # define the trim section for trimming of unwanted areas
    'ampsec': {'1x1': None,
               '2x2': None,
               '4x4': None},
    'oscansec': {'1x1': None,
                 # '1x1': {'11': [6, 22, 0, 6144]},
                 '2x2': None,
                 '4x4': None},
    'trimsec': {'1x1': None,
                '2x2': None,
                '4x4': None},

    # ccd mask for regions to exclude from all processes
    'ccd_mask': {'1x1': None,
                 '2x2': None,
                 '4x4': None},

    # 'imagetyp_light': 'Light Frame',
    # 'imagetyp_bias': 'Bias Frame',
    # 'imagetyp_dark': 'Dark Frame',
    # 'imagetyp_flat': 'Flat Field',
    'imagetyp_light': 'LIGHT',
    'imagetyp_bias': 'BIAS',
    'imagetyp_dark': 'DARK',
    'imagetyp_flat': 'FLAT',
    'add_imagetyp_keys': True,
    'add_imagetyp': {'light': {'imagetyp': 'Light Frame'}, 'bias': {'imagetyp': 'Bias Frame'},
                     'dark': {'imagetyp': 'Dark Frame'}, 'flat': {'imagetyp': 'Flat Field'}},

    # 'bias_cor': True,
    # 'dark_cor': True,
    # 'flat_cor': True,
    # 'flat_dark_cor': False,
    # 'cosmic_correct': True,

    # keywords to exclude from the header update while preparing the file
    'keys_to_exclude': None,

    # keywords for info_dict
    'obs_keywords': ['telescop', 'instrume', 'object', 'date-obs', 'filter', 'exptime',
                     'objctra', 'objctdec', 'airmass', 'binning'],

    # source extractor settings
    'saturation_limit': 6.5e4,
    'apply_mask': True
}

# CTIO 0.9 m telescope
ctio_params = {

    # telescope location information
    "name": "Cerro Tololo Interamerican Observatory",
    "longitude": -70.815,
    "latitude": -30.16527778,
    "altitude": 2215.0,
    "tz": -4,

    # telescope keywords
    'telescope_instrument': 'CTIO',  # telescope/instrument name
    'telescope_keyword': 'CTIO 0.9 meter telescope',  # telescope/instrument keyword
    'observatory_code': '807',  # MPC observatory code

    # instrument-specific FITS header keywords
    'telescop': 'TELESCOP',  # telescope keyword
    'instrume': 'INSTRUME',  # instrument keyword

    'extent': ('NAXIS1', 'NAXIS2'),  # N_pixels in x/y
    'n_amps': 4,  # number of chips/amplifier on detector
    'namps_yx': {'1x1': {4: [2, 2]}},
    'amplist': {'1x1': ['11', '12', '21', '22']},  # amplifier list keyword
    'secpix': (0.40, 0.40),  # pixel size (arcsec)
    'binning': ('BINX', 'BINY'),  # binning
    'image_size_1x1': (2168, 2048),

    'gain': 3.,  # CD gain in el/DN
    'readnoise': 12.,  # CCD Read Out Noise (e-)

    # telescope pointing keywords
    'ra': 'RA',  # telescope pointing, RA
    'dec': 'DEC',  # telescope pointing, Dec
    'radec_separator': ':',  # RA/Dec hms separator, use 'XXX' if already in degrees
    'equinox': 'EQUINOX',

    # keyword; use 'date|time' if separate
    'date_keyword': 'DATE-OBS',  # obs date/time

    'imagetyp': 'IMAGETYP',
    'object': 'OBJECT',  # object name keyword
    'filter': 'FILTER2',  # filter keyword

    # filtername translation dictionary
    'filter_translations': {'u': 'U', 'b': 'B',
                            'nv': 'V', 'i': 'I',
                            'r': 'R'},

    'exptime': 'EXPTIME',  # exposure time keyword (s)
    'airmass': 'AIRMASS',  # airmass keyword

    # reduction keywords
    # define the trim section for trimming of unwanted areas
    # 'trimsec': None,
    'ampsec': {'1x1': {'11': [0, 1023, 0, 1084],
                       '12': [0, 1023, 1084, 2168],
                       '21': [1025, 2048, 0, 1084],
                       '22': [1025, 2048, 1084, 2168]},
               '2x2': None,
               '4x4': None},
    'oscansec': {'1x1': {'11': [0, 1023, 1044, 1084],
                         '12': [0, 1023, 0, 40],
                         '21': [0, 1023, 1044, 1084],
                         '22': [0, 1023, 0, 40]},
                 '2x2': None,
                 '4x4': None},
    'trimsec': {'1x1': {'11': [0, 1023, 10, 1034],
                        '12': [0, 1023, 50, 1074],
                        '21': [0, 1023, 10, 1034],
                        '22': [0, 1023, 50, 1074]},
                '2x2': None,
                '4x4': None},

    # ccd mask for regions to exclude from all processes
    'ccd_mask': {'1x1': [[0, 2046, 1957, 1959],
                         [316, 1023, 100, 101],
                         [1010, 1023, 437, 438],
                         [1023, 1061, 482, 483],
                         [1023, 1145, 535, 536],
                         [1023, 1166, 547, 546],
                         [1023, 1347, 818, 819],
                         ],
                 '2x2': None,
                 '4x4': None},

    # telescope specific image types
    'imagetyp_light': 'object',
    'imagetyp_bias': 'zero',
    'imagetyp_dark': 'dark',
    'imagetyp_flat': 'dflat',
    'add_imagetyp_keys': True,
    'add_imagetyp': {'flat': {'imagetyp': 'sflat'}},

    # keywords to exclude from the header update while preparing the file
    'keys_to_exclude': None,

    # keywords for info_dict
    'obs_keywords': ['telescop', 'instrume', 'object', 'date-obs', 'filter', 'exptime',
                     'ra', 'dec', 'airmass', 'binning'],

    # source extractor settings
    'saturation_limit': 6.5e4,
    'apply_mask': True

}

spm_params = {

    # telescope location information
    "name": "Observatorio Astronomico Nacional, San Pedro Martir",
    "longitude": -115.48694444,
    "latitude": 31.02916667,
    "altitude": 2830.0,
    "tz": -7,

    # telescope keywords
    'telescope_instrument': 'OAN/SPM',  # telescope/instrument name
    'telescope_keyword': 'DDOTI 28-cm f/2.2',  # telescope/instrument keyword
    'observatory_code': '679',  # MPC observatory code

    # instrument-specific FITS header keywords
    'telescop': 'TELESCOP',  # telescope keyword
    'instrume': 'INSTRUME',  # instrument keyword

    'extent': ('NAXIS1', 'NAXIS2'),  # N_pixels in x/y
    'n_amps': 1,  # number of chips/amplifier on detector
    'namps_yx': {'1x1': None},
    'amplist': {'1x1': None},  # amplifier list keyword
    'secpix': (2., 2.),  # pixel size (arcsec)
    'binning': ('BINNING', 'BINNING'),  # binning
    'image_size_1x1': (6144, 6220),

    'gain': 'SOFTGAIN',  # CD gain in el/DN
    'readnoise': 12,  # CCD Read Out Noise (e-)

    # telescope pointing keywords
    'ra': 'STRRQRA',  # telescope pointing, RA
    'dec': 'STRRQDE',  # telescope pointing, Dec
    'radec_separator': 'XXX',  # RA/Dec hms separator, use 'XXX' if already in degrees
    'equinox': 'STRRQEQ',

    # keyword; use 'date|time' if separate
    'date_keyword': 'DATE-OBS',  # obs date/time

    'object': 'BLKNM',  # object name keyword
    'filter': 'FILTER',  # filter keyword
    'imagetyp': 'EXPTYPE',

    # filtername translation dictionary
    'filter_translations': {'w': 'w'},

    'exptime': 'EXPTIME',  # exposure time keyword (s)
    'airmass': 'STROBAM',  # airmass keyword

    # 'trimsec': None,
    'ampsec': {'1x1': None,
               '2x2': None,
               '4x4': None},
    'oscansec': {'1x1': None,
                 # '1x1': {'11': [6, 22, 0, 6144]},
                 '2x2': None,
                 '4x4': None},
    'trimsec': {'1x1': {'11': [54, 6198, 0, 6144]},
                '2x2': None,
                '4x4': None},

    # ccd mask for regions to exclude from all processes
    'ccd_mask': {'1x1': None,
                 '2x2': None,
                 '4x4': None},

    # telescope specific image types
    'imagetyp_light': 'object',
    'imagetyp_bias': 'bias',
    'imagetyp_dark': 'dark',
    'imagetyp_flat': 'flat',
    'add_imagetyp_keys': False,
    # 'add_imagetyp': {'flat': {'imagetyp': 'sflat'}},

    # keywords to exclude from the header update while preparing the file
    'keys_to_exclude': None,

    # keywords for info_dict
    'obs_keywords': ['telescop', 'instrume', 'blknm', 'date-obs', 'filter', 'exptime',
                     'strrqra', 'strrqde', 'strobam', 'binning'],

    # source extractor settings
    'saturation_limit': 6.5e4,
    'apply_mask': True
}

ouka_params = {
    "name": "Oukaimeden observatory",
    "longitude": -7.866,
    "latitude": 31.206389,
    "altitude": 2700.0,
    "tz": -1,

    # telescope keywords
    'telescope_instrument': 'ZWO ASI2600MM Pro',  # telescope/instrument name
    'telescope_keyword': 'Takahashi FSQ 85',  # telescope/instrument keyword
    'observatory_code': 'J43',  # MPC observatory code

    # instrument-specific FITS header keywords
    'telescop': 'TELESCOP',  # telescope keyword
    'instrume': 'INSTRUME',  # instrument keyword

    'extent': ('NAXIS1', 'NAXIS2'),  # N_pixels in x/y
    'n_amps': 1,  # number of chips/amplifier on detector
    'namps_yx': {'1x1': None},
    'amplist': {'1x1': None},  # amplifier list keyword
    'secpix': (1.205, 1.205),  # pixel size (arcsec)
    'binning': ('XBINNING', 'YBINNING'),  # binning
    'image_size_1x1': (6248, 4176),

    'gain': 'EGAIN',  # CD gain in el/DN
    'readnoise': None,  # CCD Read Out Noise (e-)

    # telescope pointing keywords
    'ra': 'RA',  # telescope pointing, RA
    'dec': 'DEC',  # telescope pointing, Dec
    'radec_separator': 'XXX',  # RA/Dec hms separator, use 'XXX' if already in degrees
    'equinox': None,

    # keyword; use 'date|time' if separate
    'date_keyword': 'DATE-OBS',  # obs date/time

    'object': 'OBJECT',  # object name keyword
    'filter': 'FILTER',  # filter keyword
    'imagetyp': 'IMAGETYP',

    # filtername translation dictionary
    'filter_translations': {'L': 'V'},

    'exptime': 'EXPTIME',  # exposure time keyword (s)
    'airmass': 'AIRMASS',  # airmass keyword

    # 'trimsec': None,
    'ampsec': {'1x1': None,
               '2x2': None,
               '4x4': None},
    'oscansec': {'1x1': None,
                 '2x2': None,
                 '4x4': None},
    'trimsec': {'1x1': None,
                '2x2': None,
                '4x4': None},

    # ccd mask for regions to exclude from all processes
    'ccd_mask': {'1x1': None,
                 '2x2': None,
                 '4x4': None},

    # telescope specific image types
    'imagetyp_light': 'light',
    'imagetyp_bias': 'bias',
    'imagetyp_dark': 'dark',
    'imagetyp_flat': 'flat',
    'add_imagetyp_keys': False,
    # 'add_imagetyp': {'flat': {'imagetyp': 'sflat'}},

    # keywords to exclude from the header update while preparing the file
    'keys_to_exclude': None,

    # keywords for info_dict
    'obs_keywords': ['telescop', 'instrume', 'object', 'date-obs', 'filter', 'exptime',
                     'ra', 'dec', 'airmass', 'binning'],

    # source extractor settings
    'saturation_limit': 6.5e4,
    'apply_mask': True

}

# currently available telescopes
IMPLEMENTED_TELESCOPES = ['CA123DLRMKIII', 'CBNUO-JC', 'CKOIR', 'DK154_DFOSC', 'CTIO09',
                          'OUKA', 'SPM']

# translate INSTRUME header keyword
INSTRUMENT_IDENTIFIERS = OrderedDict({'DFOSC_FASU': 'DK154_DFOSC',
                                      'DLR-MKIII': 'CA123DLRMKIII',
                                      'FLI': 'CKOIR',
                                      'SBIG STX-16803 CCD Camera': 'CBNUO-JC',
                                      'cfccd': 'CTIO09', 'ZWO ASI2600MM Pro': 'OUKA',
                                      'C1': 'SPM', 'C2': 'SPM', 'C3': 'SPM', 'C4': 'SPM'
                                      })

# translate telescope keyword into parameter set defined here
TELESCOPE_PARAMETERS = OrderedDict({'CA123DLRMKIII': ca123dlrmkiii_param,
                                    'CBNUO-JC': cbnuo_params,
                                    'CKOIR': ckoir_param,
                                    'DK154_DFOSC': dk154_params,
                                    'CTIO09': ctio_params,
                                    'OUKA': ouka_params,
                                    'SPM': spm_params
                                    })
