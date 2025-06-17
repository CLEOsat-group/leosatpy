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
    'n_ext': 0,  # Number of Extensions in the calibrated image
    'multiple_amps': False,
    'n_amps': 1,  # the number of amplifiers on the chip
    'namps_yx': None,
    'amplist': None,  # amplifier list keyword
    'secpix': ('SECPPIX', 'SECPPIX'),  # unbinned pixel size (arcsec)
    'binning': ('BINX', 'BINY'),  # binning in x/y
    'image_size_1x1': (2148, 2048),
    'pc_matrix': None,  # Preferred CCD PC matrix
    # 'pc_matrix': [[[-1, 0], [0, -1]]],  # Preferred CCD PC matrix

    'gain': 0.25,  # CD gain in el/DN
    'readnoise': 4.5,  # CCD Read Out Noise (e-)

    # telescope pointing keywords
    'ra': 'OBJRA',  # telescope pointing, RA
    'dec': 'OBJDEC',  # telescope pointing, Dec
    'radec_separator': ':',  # RA/Dec hms separator, use 'XXX' if already in degrees
    'equinox': 'EQUINOX',

    # Observation keywords
    'date_keyword': 'DATE-OBS',  # obs date/time

    'imagetyp': 'IMAGETYP',
    'object': 'OBJECT',  # object name keyword
    'filter': 'FILTB',  # Filter keyword

    # Filter name translation dictionary
    'filter_translations': {'U': 'U', 'B': 'B', 'V': 'V', 'R': 'R',
                            'I': 'I', 'empty': None},

    'exptime': 'EXPTIME',  # exposure time keyword (s)
    'airmass': 'AIRMASS',  # airmass keyword

    # Reduction keywords
    'ampsec': None,
    'oscansec': None,
    'trimsec': None,
    'cropsec' : 'DETSEC',

    # ccd mask for regions to exclude from all processes
    'ccd_mask': {'1x1': ['[1:86,1:2048]',
                         '[2096:2148,1:2048]',
                         '[1:2148,1:4]',
                         '[1:2148,2034:2048]'],
                 '2x2': ['[1:43,1:1024]',
                         '[1048:1074,1:1024]',
                         '[1:1074,1017:1024]'],
                 '4x4': ['[1:15,1:516]',
                         '[524:537,1:516]']},

    # telescope specific image types
    'imagetyp_light': 'light',
    'imagetyp_bias': 'bias',
    'imagetyp_dark': 'dark',
    'imagetyp_flat': 'flat',
    'add_imagetyp_keys': False,
    'add_imagetyp': {'flat': {'object': 'flat_v'}},

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
    'n_ext': 0,  # Number of Extensions in the calibrated image
    'multiple_amps': False,
    'n_amps': 1,  # number of chips on detector
    'namps_yx': None,
    'amplist': None,  # amplifier list keyword
    'secpix': (1.05, 1.05),  # pixel size (arcsec)
    'binning': ('BINX', 'BINY'),  # binning
    'image_size_1x1': (4096, 4096),
    'pc_matrix': None,  # Preferred CCD PC matrix

    'gain': 'GAIN',  # CD gain in el/DN
    'readnoise': 'RDNOISE',  # CCD Read Out Noise (e-)

    # telescope pointing keywords
    'ra': 'RA',  # telescope pointing, RA
    'dec': 'DEC',  # telescope pointing, Dec
    'radec_separator': ':',  # RA/Dec hms separator, use 'XXX' if already in degrees
    'equinox': None,

    # Observation keywords
    'date_keyword': 'DATE-OBS',  # obs date/time

    'imagetyp': 'IMAGETYP',
    'object': 'OBJECT',  # object name keyword
    'filter': 'FILTER',  # Filter keyword
    'filter_translations': {'V': 'V', 'R': 'R',
                            'I': 'I', 'B': 'B', 'N': None},  # Filter name translation dictionary

    'exptime': 'EXPTIME',  # exposure time keyword (s)
    'airmass': 'AIRMASS',  # airmass keyword

    # Reduction keywords
    'ampsec': None,
    'oscansec': None,
    'trimsec': None,
    'cropsec': None,

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
    'n_ext': 0,  # Number of Extensions in the calibrated image
    'n_amps': 1,  # number of chips on detector
    'multiple_amps': False,
    'namps_yx': None,
    'amplist': None,  # amplifier list keyword
    'secpix': ('SCALE', 'SCALE'),  # unbinned pixel size (arcsec)
    'binning': ('CCDBINX', 'CCDBINY'),  # binning keyword
    'image_size_1x1': (4096, 4112),
    'pc_matrix': None,  # Preferred CCD PC matrix

    'gain': 'GAIN',  # CD gain in el/DN
    'readnoise': None,  # CCD Read Out Noise (e-)

    # telescope pointing keywords
    'ra': 'RA',  # telescope pointing, RA
    'dec': 'DEC',  # telescope pointing, Dec
    'radec_separator': 'XXX',  # RA/Dec hms separator, use 'XXX' if already in degrees
    'equinox': 'EQUINOX',

    # Observation keywords
    'date_keyword': 'DATE-OBS',  # obs date/time

    'imagetyp': 'IMAGETYP',
    'object': 'OBJECT',  # object name keyword
    'filter': 'FILTER',  # Filter keyword
    # filtername translation dictionary
    'filter_translations': {'V': 'V',
                            'R': 'R',
                            'free': None},
    'exptime': 'EXPTIME',  # exposure time keyword (s)
    'airmass': 'AIRMASS',  # airmass keyword

    # Reduction keywords
    'ampsec': None,
    'oscansec': None,
    'trimsec': None,
    'cropsec': None,

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
    'n_ext': 0,  # Number of Extensions in the calibrated image
    'n_amps': 1,  # number of chips on detector
    'multiple_amps': False,
    'namps_yx': None,
    'amplist': None,  # amplifier list keyword
    'secpix': (0.465, 0.465),  # pixel size (arcsec)
    'binning': ('XBINNING', 'YBINNING'),  # binning in x/y
    'image_size_1x1': (4096, 4096),
    'pc_matrix': None,  # Preferred CCD PC matrix

    # Gain and readout noise taken from https://people.bsu.edu/rberring/observing-facilities/
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
    'filter': 'FILTER',  # Filter keyword

    # filtername translation dictionary
    'filter_translations': {'u\'': 'u', 'g\'': 'g',
                            'r\'': 'r', 'i\'': 'i',
                            'z\'': 'z'},

    'exptime': 'EXPTIME',  # exposure time keyword (s)
    'airmass': 'AIRMASS',  # airmass keyword

    # Reduction keywords
    'ampsec': None,
    'oscansec': None,
    'trimsec': None,
    'cropsec' : None,

    # ccd mask for regions to exclude from all processes
    'ccd_mask': {'1x1': None,
                 '2x2': None,
                 '4x4': None},

    'imagetyp_light': 'LIGHT',
    'imagetyp_bias': 'BIAS',
    'imagetyp_dark': 'DARK',
    'imagetyp_flat': 'FLAT',
    'add_imagetyp_keys': True,
    'add_imagetyp': {'light': {'imagetyp': 'Light Frame'}, 'bias': {'imagetyp': 'Bias Frame'},
                     'dark': {'imagetyp': 'Dark Frame'}, 'flat': {'imagetyp': 'Flat Field'}},

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
ctio_90cm_params = {

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
    'n_ext': 0,  # Number of Extensions in the calibrated image
    'multiple_amps': True,
    'n_amps': None,  # number of chips/amplifier on detector
    'namps_yx': 'NAMPSYX', # number of amplifiers in y and x (eg. '2 2=quad')
    'amplist': 'AMPLIST',  # amplifier list keyword, readout order in y,x
    'secpix': (0.401, 0.401),  # pixel size (arcsec)
    'binning': ('BINX', 'BINY'),  # binning
    'image_size_1x1': (2168, 2048),
    'pc_matrix': None,  # Preferred CCD PC matrix
    # 'pc_matrix': [[[-1, 0], [0, -1]]],  # Preferred CCD PC matrix

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
    'filter': 'FILTER2',  # Filter keyword

    # filtername translation dictionary
    'filter_translations': {'u': 'U', 'b': 'B',
                            'nv': 'V', 'ov': 'V', 'i': 'I',
                            'r': 'R'},

    'exptime': 'EXPTIME',  # exposure time keyword (s)
    'airmass': 'AIRMASS',  # airmass keyword

    # Reduction keywords
    'ampsec': 'ADSEC',
    'oscansec': 'BSEC',
    'trimsec': 'TSEC',
    'cropsec' : 'ROISEC00',

    # define the trim section for trimming of unwanted areas
    # 'ampsec': {'1x1': {'11': [0, 1023, 0, 1084],
    #                    '12': [0, 1023, 1084, 2168],
    #                    '21': [1025, 2048, 0, 1084],
    #                    '22': [1025, 2048, 1084, 2168]},
    #            '2x2': None,
    #            '4x4': None},
    # 'oscansec': {'1x1': {'11': [0, 1023, 1044, 1084],
    #                      '12': [0, 1023, 0, 40],
    #                      '21': [0, 1023, 1044, 1084],
    #                      '22': [0, 1023, 0, 40]},
    #              '2x2': None,
    #              '4x4': None},
    # 'trimsec': {'1x1': {'11': [0, 1023, 10, 1034],
    #                     '12': [0, 1023, 50, 1074],
    #                     '21': [0, 1023, 10, 1034],
    #                     '22': [0, 1023, 50, 1074]},
    #             '2x2': None,
    #             '4x4': None},

    # ccd mask for regions to exclude from all processes
    'ccd_mask': {'1x1': [[0, 2046, 1957, 1961],
                         [316, 1023, 100, 101],
                         [1010, 1023, 437, 438],
                         [1023, 1061, 482, 483],
                         [1023, 1145, 535, 536],
                         [1023, 1166, 546, 548],
                         [1023, 1347, 818, 819],
                         [1023, 1911, 273, 274],
                         [1023, 1610, 1828, 1829],
                         [1023, 1736, 1845, 1846],
                         [1023, 1766, 1950, 1961],
                         [1520, 1766, 1900, 1961],
                         [273, 275, 1809, 1856]
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

# CTIO 4 m telescope
ctio_400cm_params = {

    # telescope location information
    "name": "Cerro Tololo Interamerican Observatory",
    "longitude": -70.81489,
    "latitude": -30.16606,
    "altitude": 2215.0,
    "tz": -4,

    # telescope keywords
    'telescope_instrument': 'CTIO',  # telescope/instrument name
    'telescope_keyword': 'CTIO 4.0-m telescope',  # telescope/instrument keyword
    'observatory_code': '807',  # MPC observatory code

    # instrument-specific FITS header keywords
    'telescop': 'TELESCOP',  # telescope keyword
    'instrume': 'INSTRUME',  # instrument keyword
    'detector': 'DETPOS',  # detector name keyword

    'extent': ('NAXIS1', 'NAXIS2'),  # Number of pixel in x/y,
    'n_ext': 61,  # Number of Extensions in the calibrated image
    'n_amps': 1,  # number of chips on detector
    'multiple_amps': False,
    'namps_yx': None,
    'amplist': None,  # amplifier list keyword
    'secpix': (0.27, 0.27),  # pixel size (arcsec)
    'binning': ('CCDBIN1', 'CCDBIN2'),  # binning in x/y
    'pc_matrix': None,  # Preferred CCD PC matrix

    'gain': 'GAINA',  # CD gain in el/DN
    'readnoise': 'RDNOISEA',  # CCD Read Out Noise (e-)

    # telescope pointing keywords
    'ra': 'RA',  # telescope pointing, RA
    'dec': 'DEC',  # telescope pointing, Dec
    'radec_separator': ':',  # RA/Dec hms separator, use 'XXX' if already in degrees
    'equinox': 'EQUINOX',

    # keyword; use 'date|time' if separate
    'date_keyword': 'DATE-OBS',  # obs date/time

    'imagetyp': 'IMAGETYP',
    'object': 'OBJECT',  # object name keyword
    'filter': 'BAND',  # Filter keyword

    # filtername translation dictionary
    'filter_translations': {'u': 'u', 'g': 'g',
                            'r': 'r', 'i': 'i',
                            'z': 'z'},

    'exptime': 'EXPTIME',  # exposure time keyword (s)
    'airmass': 'AIRMASS',  # airmass keyword

    # keywords to exclude from the header update while preparing the file
    'keys_to_exclude': None,

    # keywords for info_dict
    'obs_keywords': ['telescop', 'instrume', 'object', 'date-obs', 'filter', 'exptime',
                     'ra', 'dec', 'airmass', 'binning'],

    # source extractor settings
    'saturation_limit': 'SATURATE',
    'apply_mask': False

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
    'n_ext': 0,  # Number of Extensions in the calibrated image
    'n_amps': 1,  # number of chips/amplifier on detector
    'multiple_amps': False,
    'namps_yx': None,
    'amplist': None,  # amplifier list keyword
    'secpix': (2., 2.),  # pixel size (arcsec)
    'binning': ('BINNING', 'BINNING'),  # binning
    'image_size_1x1': (6144, 6220),
    'pc_matrix': None,  # Preferred CCD PC matrix

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
    'filter': 'FILTER',  # Filter keyword
    'imagetyp': 'EXPTYPE',

    # filtername translation dictionary
    'filter_translations': {'w': 'w'},

    'exptime': 'EXPTIME',  # exposure time keyword (s)
    'airmass': 'STROBAM',  # airmass keyword

    'trimsec': {'1x1': {'11': [54, 6198, 0, 6144]},
                '2x2': None,
                '4x4': None},
    'ampsec': None,
    'oscansec': None,
    # 'trimsec': None,
    'cropsec' : None,

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
    'n_ext': 0,  # Number of Extensions in the calibrated image
    'n_amps': 1,  # number of chips/amplifier on detector
    'multiple_amps': False,
    'namps_yx': None,
    'amplist': None,  # amplifier list keyword
    'secpix': (1.205, 1.205),  # pixel size (arcsec)
    'binning': ('XBINNING', 'YBINNING'),  # binning
    'image_size_1x1': (6248, 4176),
    'pc_matrix': None,  # Preferred CCD PC matrix

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
    'filter': 'FILTER',  # Filter keyword
    'imagetyp': 'IMAGETYP',

    # filtername translation dictionary
    'filter_translations': {'L': 'V'},

    'exptime': 'EXPTIME',  # exposure time keyword (s)
    'airmass': 'AIRMASS',  # airmass keyword

    'ampsec': None,
    'oscansec': None,
    'trimsec': None,
    'cropsec': None,

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

# Fjernstyrede Undervisnings Teleskop FUT, Mt. Kent Observatory, Australia
fut_params = {

    # -27.7977, 151.8554, 682 m
    "name": "Mt. Kent Observatory",
    "longitude": 151.8554,
    "latitude": -27.7977,
    "altitude": 682.0,
    "tz": 10,

    # telescope keywords
    'telescope_instrument': 'FLI Kepler4040 cmos KL2441220',  # telescope/instrument name
    # 'telescope_instrument': 'FUT',  # telescope/instrument name
    'telescope_keyword': 'FUT',  # telescope/instrument keyword
    'observatory_code': '',  # MPC observatory code

    # instrument-specific FITS header keywords
    'telescop': 'TELESCOP',  # telescope keyword
    'instrume': 'INSTRUM',  # instrument keyword

    'extent': ('NAXIS1', 'NAXIS2'),  # N_pixels in x/y
    'n_ext': 0,  # Number of Extensions in the calibrated image
    'n_amps': 1,  # number of chips/amplifier on detector
    'multiple_amps': False,
    'namps_yx': None,
    'amplist': None,  # amplifier list keyword
    'secpix': (0.465, 0.465),  # pixel size (arcsec)
    'binning': (1, 1),  # binning
    'image_size_1x1': (4096, 4096),
    'pc_matrix': None,  # Preferred CCD PC matrix

    'gain': 'GAIN',  # CD gain in el/DN
    'readnoise': 'READNOIS',  # CCD Read Out Noise (e-)

    # telescope pointing keywords
    'ra': 'OBJ-RA',  # telescope pointing, RA
    'dec': 'OBJ-DEC',  # telescope pointing, Dec
    'radec_separator': 'XXX',  # RA/Dec hms separator, use 'XXX' if already in degrees
    'equinox': 'EQUINOX',

    # keyword; use 'date|time' if separate
    'date_keyword': 'DATE-BEG',  # obs date/time

    'object': 'OBJECT',  # object name keyword
    'filter': 'FILTER',  # Filter keyword
    'imagetyp': 'IMAGETYP',

    # filtername translation dictionary
    'filter_translations': {'V': 'V'},

    'exptime': 'EXPTIME',  # exposure time keyword (s)
    'airmass': None,  # airmass keyword

    'ampsec': None,
    'oscansec': None,
    'trimsec': None,
    'cropsec': None,

    # ccd mask for regions to exclude from all processes
    'ccd_mask': {'1x1': None,
                 '2x2': None,
                 '4x4': None},

    # telescope specific image types
    'imagetyp_light': 'sky',
    'imagetyp_bias': 'bias',
    'imagetyp_dark': 'dark',
    'imagetyp_flat': 'flat',
    'add_imagetyp_keys': False,
    # 'add_imagetyp': {'flat': {'imagetyp': 'sflat'}},

    # keywords to exclude from the header update while preparing the file
    'keys_to_exclude': None,

    # keywords for info_dict
    'obs_keywords': ['telescop', 'instrum', 'object', 'date-beg', 'filter', 'exptime',
                     'obj-ra', 'obj-dec', 'airmass', 'binning'],

    # source extractor settings
    'saturation_limit': 'SATLEVEL',
    'apply_mask': False
}

# currently available telescopes
IMPLEMENTED_TELESCOPES = ['CA123DLRMKIII', 'CBNUO-JC', 'CKOIR', 'DK154_DFOSC', 'CTIO90', 'CTIO400',
                          'OUKA', 'SPM', 'FUT']

# translate INSTRUME header keyword
INSTRUMENT_IDENTIFIERS = OrderedDict({'DFOSC_FASU': 'DK154_DFOSC',
                                      'DLR-MKIII': 'CA123DLRMKIII',
                                      'FLI': 'CKOIR',
                                      'SBIG STX-16803 CCD Camera': 'CBNUO-JC',
                                      'cfccd': 'CTIO90', 'DECam': 'CTIO400',
                                      'ZWO ASI2600MM Pro': 'OUKA',
                                      'C1': 'SPM', 'C2': 'SPM', 'C3': 'SPM', 'C4': 'SPM',
                                      'FLI Kepler4040 cmos KL2441220': 'FUT'
                                      })

# translate telescope keyword into parameter set defined here
TELESCOPE_PARAMETERS = OrderedDict({'CA123DLRMKIII': ca123dlrmkiii_param,
                                    'CBNUO-JC': cbnuo_params,
                                    'CKOIR': ckoir_param,
                                    'DK154_DFOSC': dk154_params,
                                    'CTIO90': ctio_90cm_params,
                                    'CTIO400': ctio_400cm_params,
                                    'OUKA': ouka_params,
                                    'SPM': spm_params,
                                    'FUT': fut_params
                                    })
