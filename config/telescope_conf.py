#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Name:         telescope_configuration.py
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
    'secpix': ('SECPPIX', 'SECPPIX'),  # unbinned pixel size (arcsec)
    'binning': ('BINX', 'BINY'),  # binning in x/y
    'image_size_1x1': (2148, 2048),
    'gain': 0.25,  # CD gain in el/DN
    'readnoise': 4.5,  # CCD Read Out Noise (e-)

    # telescope pointing keywords
    'ra': 'OBJRA',  # telescope pointing, RA
    'dec': 'OBJDEC',  # telescope pointin, Dec
    'radec_separator': ':',  # RA/Dec hms separator, use 'XXX' if already in degrees

    # observation keywords
    'date_keyword': 'DATE-OBS',  # obs date/time

    'object': 'OBJECT',  # object name keyword
    'filter': 'FILTB',  # filter keyword

    # filter name translation dictionary
    'filter_translations': {'U': 'U', 'B': 'B', 'V': 'V', 'R': 'R',
                            'I': 'I', 'empty': None},

    'exptime': 'EXPTIME',  # exposure time keyword (s)
    'airmass': 'AIRMASS',  # airmass keyword

    # reduction keywords
    # trim section for trimming of unwanted areas
    'trimsec': {'1x1': '[67:2096, 1:2060]', '2x2': '[50:1049, 11:1020]', '4x4': '[16:525, 1:512]'},
    'oscan_cor': None,

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
    'average_fwhm': 2.75,
    'saturation_limit': 6.0e5,
    'nsigma': 1.25,
    'source_box': 9,
    'isolation_size': 11,
    'box_size': 11,
    'win_size': 7,

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
    'secpix': (1.05, 1.05),  # pixel size (arcsec)
    'binning': ('BINX', 'BINY'),  # binning
    'image_size_1x1': (4096, 4096),
    'gain': 'GAIN',  # CD gain in el/DN
    'readnoise': 'RDNOISE',  # CCD Read Out Noise (e-)

    # telescope pointing keywords
    'ra': 'RA',  # telescope pointing, RA
    'dec': 'DEC',  # telescope pointing, Dec
    'radec_separator': ':',  # RA/Dec hms separator, use 'XXX' if already in degrees

    # observation keywords
    'date_keyword': 'DATE-OBS',  # obs date/time

    'object': 'OBJECT',  # object name keyword
    'filter': 'FILTER',  # filter keyword
    'filter_translations': {'V': 'V', 'R': 'R',
                            'I': 'I', 'B': 'B', 'N': None},  # filter name translation dictionary

    'exptime': 'EXPTIME',  # exposure time keyword (s)
    'airmass': 'AIRMASS',  # airmass keyword

    # reduction keywords
    # trim section for trimming of unwanted areas
    'trimsec': None,
    # 'trimsec': {'1x1': '[67:2096, 1:2060]', '2x2': '[50:1049, 11:1020]', '4x4': '[16:525, 1:512]'},
    'oscan_cor': None,  # correct overscan

    # telescope specific image types
    'imagetyp_light': 'object',
    'imagetyp_bias': 'zero',
    'imagetyp_dark': 'dark',
    'imagetyp_flat': 'flat',
    'add_imagetyp_keys': False,

    # defaults for reduction
    # 'bias_cor': True,  # correct bias
    # 'dark_cor': True,  # correct dark
    # 'flat_cor': True,  # correct flat
    # 'flat_dark_cor': False,  # flat-dark correction
    # 'cosmic_correct': True,  # correct for cosmic rays

    # keywords to exclude from the header update while preparing the file
    'keys_to_exclude': ['DELMAG', 'FWHMH', 'FWHMV', 'FWHMHS', 'FWHMVS', 'NSTAR'],

    # keywords for info_dict
    'obs_keywords': ['observat', 'instrume', 'object', 'date-obs', 'time-obs', 'filter', 'exptime',
                     'ra', 'dec', 'airmass', 'binning'],

    # source extractor settings
    'average_fwhm': 2.5,
    'saturation_limit': 6e4,
    'nsigma': 1.5,
    'source_box': 9,
    'isolation_size': 11,
    'box_size': 11,
    'win_size': 7

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
    'secpix': ('SCALE', 'SCALE'),  # unbinned pixel size (arcsec)
    'binning': ('CCDBINX', 'CCDBINY'),  # binning keyword
    'image_size_1x1': (4096, 4112),
    'gain': 'GAIN',  # CD gain in el/DN
    'readnoise': None,  # CCD Read Out Noise (e-)

    # telescope pointing keywords
    'ra': 'RA',  # telescope pointing, RA
    'dec': 'DEC',  # telescope pointing, Dec
    'radec_separator': 'XXX',  # RA/Dec hms separator, use 'XXX' if already in degrees

    # observation keywords
    'date_keyword': 'DATE-OBS',  # obs date/time

    'object': 'OBJECT',  # object name keyword
    'filter': 'FILTER',  # filter keyword
    # filtername translation dictionary
    'filter_translations': {'V': 'V',
                            'R': 'R',
                            'free': None},
    'exptime': 'EXPTIME',  # exposure time keyword (s)
    'airmass': 'AIRMASS',  # airmass keyword

    # reduction keywords
    # trim section for trimming of unwanted areas
    'trimsec': None,
    'oscan_cor': None,

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
    'average_fwhm': 5,
    'saturation_limit': 5e4,
    'nsigma': 2.0,
    'source_box': 15,
    'isolation_size': 5,
    'box_size': 27,
    'win_size': 11
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
    'secpix': (0.47, 0.47),  # pixel size (arcsec)
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

    # keyword; use 'date|time' if separate
    'date_keyword': 'DATE-OBS',  # obs date/time

    'object': 'OBJECT',  # object name keyword
    'filter': 'FILTER',  # filter keyword

    # filtername translation dictionary
    'filter_translations': {'u\'': 'u', 'g\'': 'g',
                            'r\'': 'r', 'i\'': 'i',
                            'z\'': 'z'},

    'exptime': 'EXPTIME',  # exposure time keyword (s)
    'airmass': 'AIRMASS',  # airmass keyword

    # reduction keywords
    # trim section for trimming of unwanted areas
    'trimsec': None,
    'oscan_cor': None,

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
    'average_fwhm': 3,
    'saturation_limit': 6e5,
    'nsigma': 1.5,
    'source_box': 15,
    'isolation_size': 5,
    'box_size': 27,
    'win_size': 11}

# currently available telescopes
IMPLEMENTED_TELESCOPES = ['CA123DLRMKIII', 'CBNUO-JC', 'CKOIR', 'DFOSC']

# translate INSTRUME header keyword
INSTRUMENT_IDENTIFIERS = OrderedDict({'DFOSC_FASU': 'DFOSC',
                                      'DLR-MKIII': 'CA123DLRMKIII',
                                      'FLI': 'CKOIR',
                                      'SBIG STX-16803 CCD Camera': 'CBNUO-JC'
                                      })

# translate telescope keyword into parameter set defined here
TELESCOPE_PARAMETERS = OrderedDict({'CA123DLRMKIII': ca123dlrmkiii_param,
                                    'CBNUO-JC': cbnuo_params,
                                    'CKOIR': ckoir_param,
                                    'DFOSC': dk154_params
                                    })
