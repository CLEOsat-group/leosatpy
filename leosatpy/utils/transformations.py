#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Name:         transformations.py
# Purpose:      Utilities to determine the transformation, detector orientation,
#               and scale to match the detected sources with a reference catalog
#
#
#
#
# Author:       p4adch (cadam)
#
# Created:      04/29/2022
# Copyright:    (c) p4adch 2010-
#
# History:
#
# 29.04.2022
# - file created and basic methods
#
# -----------------------------------------------------------------------------

""" Modules """
import math
import os
from copy import copy
import inspect
import logging
import fast_histogram as fhist

# scipy
from scipy.spatial import KDTree
from scipy.ndimage import maximum_filter, gaussian_filter, label
from scipy.ndimage.measurements import center_of_mass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.wcs import WCS, utils

from . import base_conf

# -----------------------------------------------------------------------------

""" Meta-info """
__author__ = "Christian Adam"
__copyright__ = 'Copyright 2021-2023, CLEOSat group'
__credits__ = ["Eduardo Unda-Sanzana, Jeremy Tregloan-Reed, Christian Adam"]
__license__ = "GPL-3.0 license"
__maintainer__ = "Christian Adam"
__email__ = "christian.adam84@gmail.com"
__status__ = "Production"

__taskname__ = 'transformations'
# -----------------------------------------------------------------------------

""" Parameter used in the script """

log = logging.getLogger(__name__)

MODULE_PATH = os.path.dirname(inspect.getfile(inspect.currentframe()))

pc_matrix_list = np.array([[[1, 0], [0, 1]], [[-1, 0], [0, -1]],
                           [[-1, 0], [0, 1]], [[1, 0], [0, -1]],
                           [[0, 1], [1, 0]], [[0, -1], [-1, 0]],
                           [[0, -1], [1, 0]], [[0, 1], [-1, 0]]])


# -----------------------------------------------------------------------------


class FindWCS(object):
    """"""

    def __init__(self, source_df, ref_df, config, _log: logging.Logger = log):
        """ Constructor with default values """

        self.config = config
        self.log = _log
        self.dist_bin_size = self.config['DISTANCE_BIN_SIZE']
        self.ang_bin_size = self.config['ANG_BIN_SIZE']
        self.source_cat = None

        # make a copy to work with
        source_cat_full = copy(source_df)
        reference_cat_full = copy(ref_df)

        # make sure the catalogs are sorted
        self.source_cat_full = source_cat_full.sort_values(by='mag', ascending=True)
        self.source_cat_full.reset_index(inplace=True, drop=True)

        self.reference_cat = reference_cat_full.sort_values(by='mag', ascending=True)
        self.reference_cat.reset_index(inplace=True, drop=True)

        self.obs_x = None
        self.obs_y = None

        self.log_dist_obs = None
        self.angles_obs = None

        self.current_wcsprm = None
        self.current_ref_cat = None
        self.current_score = 0
        self.current_Nobs_matched = 0

        self.source_cat_matched = None
        self.ref_cat_matched = None

        self.rotations = config['pc_matrix'] if config['pc_matrix'] is not None else pc_matrix_list

        self.apply_rot = True

    def get_source_cat_variables(self):
        """"""

        self.obs_x = np.array([self.source_cat["xcentroid"].values])
        self.obs_y = np.array([self.source_cat["ycentroid"].values])

        self.log_dist_obs = calculate_log_dist(self.obs_x, self.obs_y)
        self.angles_obs = calculate_angles(self.obs_x, self.obs_y)

    def find_wcs(self, init_wcsprm, match_radius_fwhm=1, image=None):
        """Run the plate-solve algorithm to find the best WCS transformation"""

        has_solution = False
        dict_rms = {"radius_px": None, "matches": None, "rms": None}
        final_wcsprm = None

        init_wcsprm = init_wcsprm.wcs
        max_wcs_iter = self.config['MAX_WCS_FUNC_ITER']

        self.log.info("> Run WCS calibration")

        self.source_cat = self.source_cat_full
        self.get_source_cat_variables()

        # update the reference catalog with the initial wcs parameters
        ref_cat_orig = update_catalog_positions(self.reference_cat, init_wcsprm)
        ref_cat_orig = ref_cat_orig.head(self.config['REF_SOURCES_MAX_NO'])

        Nobs = len(self.source_cat)  # number of detected sources
        Nref = len(ref_cat_orig)  # number of reference stars
        source_ratio = (Nobs / Nref) * 100.  # source ratio in percent
        percent_to_select = np.ceil(source_ratio * 3)
        n_percent = percent_to_select if percent_to_select < 100 else 100
        num_rows = int(Nref * (n_percent / 100))
        wcs_match_radius_px = 3. * match_radius_fwhm

        self.log.info(f'  Number of detected sources: {Nobs}')
        self.log.info(f'  Number of reference sources (total): {Nref}')
        self.log.info(f'  Ratio of detected sources to reference sources: {source_ratio:.3f}%')
        self.log.info(f'  Initial number of reference sources selected: {num_rows}')
        self.log.info(f'  Input radius: r_in = {match_radius_fwhm:.2f} px')
        self.log.info(f'  WCS source match radius: 3 x r_in  = {wcs_match_radius_px:.2f} px')

        # create a range of samples
        num_samples = np.logspace(np.log10(num_rows), np.log10(Nref), max_wcs_iter,
                                  base=10, endpoint=False, dtype=int)
        num_samples = np.unique(num_samples)

        for i, n in enumerate(num_samples):
            c = 0  # possible solution counter
            self.log.info(f"  > Run [{i + 1}/{len(num_samples)}]: Using {n} reference sources")

            # select rows and sort by magnitude
            selected_rows = ref_cat_orig.head(n)
            ref_cat = selected_rows.sort_values(by='mag', ascending=True)
            ref_cat.reset_index(inplace=True, drop=True)
            self.current_ref_cat = ref_cat

            best_score = -np.inf
            best_wcsprm = None
            best_rms_dist = np.inf
            best_Nobs = -1
            best_completeness = 0
            for j, init_rot in enumerate(self.rotations):

                self.log.info(
                    f"    > [{j + 1}/{len(self.rotations)}] Test rotation matrix: [{','.join(map(str, init_rot))}]")
                self.current_wcsprm = None
                self.current_ref_cat = ref_cat

                r = self.get_scale_rotation(init_wcsprm, init_rot, wcs_match_radius_px)
                if r is None:
                    self.log.info("      >> NO matches found. Moving on.")
                    continue

                Nobs_matched, completeness, score, wcsprm, result_info, source_cat_matched, ref_cat_matched = r
                self.current_wcsprm = wcsprm
                if self.current_wcsprm is None or Nobs_matched < self.config['MIN_SOURCE_NO_CONVERGENCE']:
                    self.log.info("      >> NO matches found. Moving on.")
                    continue

                self.log.info(f"      >> {Nobs_matched} matches found")

                self.log.info(f"      > X-match with original reference catalog (r={wcs_match_radius_px:.2f} px):")
                ref_cat_new = self.get_updated_ref_cat(source_cat_matched,
                                                       ref_cat_matched,
                                                       wcs_match_radius_px)

                delta_matches = len(ref_cat_new) - Nobs_matched
                delta_str = f"{delta_matches} potential sources have been added."

                self.log.info(f"        >> {len(ref_cat_new)} matches found. {delta_str}")
                self.current_ref_cat = ref_cat_new

                self.log.info(f"      > Refine scale and rotation using {len(ref_cat_new)} sources")
                state, corrected_wcsprm = self.refine_transformation(ref_cat_new,
                                                                     self.current_wcsprm,
                                                                     match_radius_fwhm,
                                                                     wcs_match_radius_px)
                if state:
                    self.current_wcsprm = corrected_wcsprm
                    self.log.info("        >> Solution has improved")
                else:
                    self.log.info("        >> NO improvement found.")

                # find matches in the full reference
                _, _, final_obs_pos, _, _, final_score, _ = find_matches(self.source_cat,
                                                                         self.reference_cat,
                                                                         self.current_wcsprm,
                                                                         threshold=wcs_match_radius_px)

                current_Nobs = len(final_obs_pos)
                current_rms_dist = 1. / (final_score * len(final_obs_pos))
                if best_score <= final_score and current_rms_dist <= best_rms_dist:
                    best_score = final_score
                    best_rms_dist = current_rms_dist
                    best_wcsprm = self.current_wcsprm
                    best_Nobs = current_Nobs
                    best_completeness = best_Nobs / Nobs
                    c += 1

            if best_Nobs == -1 or best_Nobs < self.config['MIN_SOURCE_NO_CONVERGENCE']:
                self.log.info("    >> NO solution found. ... ")
                continue

            # plot_catalog_positions(self.source_cat,
            #                        ref_cat_orig,
            #                        best_wcsprm, image)
            # plt.show()
            # get rms estimate
            dict_rms = get_rms(self.reference_cat,
                               self.source_cat,
                               best_wcsprm,
                               match_radius_fwhm,
                               best_Nobs)

            self.log.info(f"    >> Best solution out of {c} possible results:")
            self.log.info(f"       {'Matches':<20} {f'{best_Nobs}/{Nobs}'} ")
            self.log.info(f"       {'Completeness':<20} {best_completeness:.2f} ({best_completeness * 100:.1f}%)")
            self.log.info(f"       {'Match radius (px)':<20} {dict_rms['radius_px']:.1f} ")
            self.log.info(f"       {'RMS (arcsec)':<20} {dict_rms['rms']:.2f} ")

            has_solution = True
            final_wcsprm = best_wcsprm
            break

        return has_solution, final_wcsprm, dict_rms

    def get_scale_rotation(self, init_wcsprm, init_rot, match_radius):
        """"""

        # apply rotation
        current_wcsprm = rotate(copy(init_wcsprm), init_rot)

        # get the catalog positions
        cat_x, cat_y = process_catalog(self.current_ref_cat, current_wcsprm)

        # calculate the distances and angles
        log_dist_cat = calculate_log_dist(cat_x, cat_y)
        angles_cat = calculate_angles(cat_x, cat_y)

        del cat_x, cat_y

        log_dist_obs = self.log_dist_obs
        angles_obs = self.angles_obs

        # use FFT to find the rotation and scaling first
        cross_corr, binwidth_dist, binwidth_ang = self.calc_cross_corr(log_dist_obs,
                                                                       angles_obs,
                                                                       log_dist_cat,
                                                                       angles_cat)

        result_scale_rot = analyse_cross_corr(cross_corr, binwidth_dist, binwidth_ang)

        if result_scale_rot is None:

            base_conf.clean_up(log_dist_obs, angles_obs, log_dist_cat, angles_cat,
                               cross_corr, binwidth_dist, binwidth_ang)
            return

        best_score = 0
        best_scaling = 1
        best_rotation = 0
        best_Nobs_matched = -1
        best_wcsprm = None
        best_completeness = 0
        best_source_cat_match = None
        best_ref_cat_match = None

        for row in result_scale_rot[:5, :]:
            current_signal = row[0]
            current_scaling = row[1]
            current_rot = row[2]

            if current_signal == 0. or current_scaling < 0.9 or 1.01 < current_scaling:
                continue

            rot_mat = rotation_matrix(current_rot)
            wcsprm_new = rotate(copy(current_wcsprm), rot_mat)
            wcsprm_new = scale(wcsprm_new, current_scaling)
            shift_signal, x_shift, y_shift = self.find_offset(self.current_ref_cat,
                                                              wcsprm_new,
                                                              offset_binwidth=1)

            # apply the found shift
            wcsprm_shifted = shift_wcs_central_pixel(wcsprm_new, x_shift, y_shift)

            # update the current reference catalog positions
            ref_cat_updated = update_catalog_positions(self.current_ref_cat, wcsprm_shifted)

            # find matches
            matches = find_matches(self.source_cat, ref_cat_updated,
                                   wcsprm_shifted,
                                   threshold=match_radius)

            source_cat_matched, ref_cat_matched, obs_xy, cat_xy, _, current_score, _ = matches
            Nobs_matched = len(obs_xy)
            # print(Nobs_matched, current_score, init_rot, current_scaling, current_rot)

            if best_Nobs_matched <= Nobs_matched and best_score <= current_score:
                best_Nobs_matched = Nobs_matched
                best_score = current_score
                best_scaling = current_scaling
                best_rotation = current_rot
                best_wcsprm = copy(wcsprm_shifted)
                best_completeness = Nobs_matched / len(self.source_cat)
                best_source_cat_match = source_cat_matched
                best_ref_cat_match = ref_cat_matched

        # print(best_Nobs_matched, best_completeness, best_score, best_scaling, best_rotation)
        base_conf.clean_up(log_dist_obs, angles_obs, log_dist_cat, angles_cat,
                           cross_corr, binwidth_dist, binwidth_ang)

        return (best_Nobs_matched, best_completeness, best_score, best_wcsprm, [best_scaling, best_rotation],
                best_source_cat_match, best_ref_cat_match)

    def find_offset(self, catalog, wcsprm_in, offset_binwidth=1):
        """"""

        # apply the wcs to the catalog
        cat_x, cat_y = process_catalog(catalog, wcsprm_in)

        # get distances
        dist_x = (self.obs_x - cat_x.T).flatten()
        dist_y = (self.obs_y - cat_y.T).flatten()

        del cat_x, cat_y

        # Compute the minimum and maximum values for both axes
        min_x, max_x = dist_x.min(), dist_x.max()
        min_y, max_y = dist_y.min(), dist_y.max()

        # Compute the number of bins for both axes
        num_bins_x = int(np.ceil((max_x - min_x) / offset_binwidth)) + 1
        num_bins_y = int(np.ceil((max_y - min_y) / offset_binwidth)) + 1

        # Compute the edges for both axes
        x_edges = np.linspace(min_x, max_x, num_bins_x)
        y_edges = np.linspace(min_y, max_y, num_bins_y)

        # prepare histogram
        vals = [dist_x, dist_y]
        bins = [num_bins_x, num_bins_y]
        ranges = [[min_x, max_x], [min_y, max_y]]

        # Compute the histogram
        H = fhist.histogram2d(*vals, bins=bins, range=ranges)

        # find the peak for the x and y distance where the two sets overlap and take the first peak
        peak = np.argwhere(H == np.max(H))[0]

        # sum up the signal in the fixed aperture 1 pixel in each direction around the peak,
        # so a 3x3 array, total 9 pixel
        signal = np.sum(H[peak[0] - 1:peak[0] + 2, peak[1] - 1:peak[1] + 2])
        # signal_wide = np.sum(H[peak[0] - 4:peak[0] + 5, peak[1] - 4:peak[1] + 5])

        x_shift = (x_edges[peak[0]] + x_edges[peak[0] + 1]) / 2
        y_shift = (y_edges[peak[1]] + y_edges[peak[1] + 1]) / 2

        # print(signal, signal_wide - signal, x_shift, y_shift)

        del x_edges, y_edges, H, vals, dist_x, dist_y
        return signal, x_shift, y_shift

    def calc_cross_corr(self, log_dist_obs, angles_obs, log_dist_cat, angles_cat):
        """"""

        dist_bin_size = self.dist_bin_size
        ang_bin_size = self.ang_bin_size

        bins, ranges, binwidth_dist, binwidth_ang = prepare_histogram_bins_and_ranges(
            log_dist_obs, angles_obs,
            log_dist_cat, angles_cat,
            dist_bin_size, ang_bin_size)

        H_obs = create_histogram(log_dist_obs, angles_obs, bins, ranges)
        H_cat = create_histogram(log_dist_cat, angles_cat, bins, ranges)

        ff_obs = apply_fft_and_normalize(H_obs)
        ff_cat = apply_fft_and_normalize(H_cat)

        # calculate cross-correlation
        cross_corr = ff_obs * np.conj(ff_cat)

        base_conf.clean_up(H_obs, H_cat, ff_obs, ff_cat, log_dist_obs, angles_obs,
                           log_dist_cat, angles_cat, angles_obs, angles_cat, bins, ranges)

        return cross_corr, binwidth_dist, binwidth_ang

    def get_updated_ref_cat(self, src_cat_matched, ref_cat_matched, radius):
        """"""

        src_cat_orig = self.source_cat
        ref_cat_orig = self.reference_cat

        ref_cat_orig_updated = update_catalog_positions(ref_cat_orig, self.current_wcsprm)

        # get the remaining not matched source from the source catalog
        remaining_sources = src_cat_orig[~src_cat_orig['id'].isin(src_cat_matched['id'])]

        # get the closest N sources from the reference catalog for the remaining sources
        ref_sources_to_add = match_catalogs(remaining_sources,
                                            ref_cat_orig_updated,
                                            radius=radius, N=1)

        # get the closest N sources from the reference catalog
        # for the matched reference sources
        ref_cat_orig_matched = match_catalogs(ref_cat_matched,
                                              ref_cat_orig_updated,
                                              ra_dec_tol=1e-3, N=1)

        # combine the datasets
        ref_cat_subset_new = pd.concat([ref_cat_orig_matched, ref_sources_to_add],
                                       ignore_index=True)

        base_conf.clean_up(src_cat_matched, ref_cat_matched, ref_cat_orig, ref_cat_orig_updated,
                           ref_sources_to_add, ref_cat_orig_matched)

        return ref_cat_subset_new

    def refine_transformation(self, ref_cat, wcsprm_in, match_radius, compare_threshold):
        """
        Final improvement of registration. This method requires that the wcs is already accurate to a few pixels.
        """

        wcsprm_copy = copy(wcsprm_in)

        # find matches
        _, _, init_obs, _, _, init_score, _ = find_matches(self.source_cat, ref_cat,
                                                           None,
                                                           threshold=compare_threshold)
        # print(compare_threshold, len(init_obs), init_score)

        lis = [0.1, 0.25, 0.5, 1, 2, 3, 4]
        for i in lis:

            threshold = i * match_radius

            # find matches
            _, _, obs_xy, cat_xy, _, s1, _ = \
                find_matches(self.source_cat, ref_cat, None, threshold=threshold)

            if len(obs_xy[:, 0]) < self.config['MIN_SOURCE_NO_CONVERGENCE']:
                continue

            # angle
            angle_offset = -calculate_angles([obs_xy[:, 0]],
                                             [obs_xy[:, 1]]) + calculate_angles([cat_xy[:, 0]],
                                                                                [cat_xy[:, 1]])

            log_dist_obs = calculate_log_dist([obs_xy[:, 0]], [obs_xy[:, 1]])
            log_dist_cat = calculate_log_dist([cat_xy[:, 0]], [cat_xy[:, 1]])

            del obs_xy, cat_xy

            scale_offset = -log_dist_obs + log_dist_cat

            rotation = np.nanmedian(angle_offset)
            scaling = np.e ** (np.nanmedian(scale_offset))

            rot = rotation_matrix(rotation)
            wcsprm_rotated = rotate(wcsprm_copy, np.array(rot))
            if 0.9 < scaling < 1.1:
                wcsprm_scaled = scale(copy(wcsprm_rotated), scaling)
            else:
                continue

            # find matches
            _, _, obs_xy, cat_xy, _, s2, _ = \
                find_matches(self.source_cat, ref_cat,
                             wcsprm_scaled, threshold=threshold)

            if len(obs_xy[:, 0]) < self.config['MIN_SOURCE_NO_CONVERGENCE']:
                continue

            # Get offset and update the central reference pixel
            x_shift = np.nanmean(obs_xy[:, 0] - cat_xy[:, 0])
            y_shift = np.nanmean(obs_xy[:, 1] - cat_xy[:, 1])

            del obs_xy, cat_xy

            wcsprm_shifted = shift_wcs_central_pixel(copy(wcsprm_scaled), x_shift, y_shift)

            # Recalculate score
            _, _, compare_obs, cat_xy, _, compare_score, _ = \
                find_matches(self.source_cat, ref_cat, wcsprm_shifted,
                             threshold=compare_threshold)

            # print(threshold, len(init_obs[:, 0]), len(compare_obs[:, 0]), compare_score)
            if ((len(init_obs[:, 0]) <= len(compare_obs[:, 0])) and
                    (init_score < compare_score or threshold <= compare_threshold)):
                del log_dist_obs, log_dist_cat, scale_offset, angle_offset
                return True, wcsprm_shifted

            del log_dist_obs, log_dist_cat, scale_offset, angle_offset

        return False, wcsprm_copy


def find_peaks_2d(inv_cross_power_spec, size=5, apply_gaussian_filter=False):
    """
    Find peaks in a 2D array with a given size for the neighborhood and threshold.

    :param inv_cross_power_spec: 2D array of cross-correlations.
    :param size: Size of the neighborhood to consider for the local maxima.
    :param apply_gaussian_filter: Boolean flag to apply Gaussian filter for smoothing (default is False).

    :return: List of peak coordinates.
    """

    mean_val = np.median(inv_cross_power_spec)
    std_val = np.std(inv_cross_power_spec)
    adaptive_threshold = mean_val + 3. * std_val  # Adjust multiplier as needed

    if apply_gaussian_filter:
        # Apply a Gaussian filter for smoothing
        smoothed = gaussian_filter(inv_cross_power_spec, sigma=1)
    else:
        # Skip Gaussian filtering
        smoothed = inv_cross_power_spec

    # Find local maxima
    local_max = maximum_filter(smoothed, size=size) == smoothed

    # Apply threshold
    detected_peaks = smoothed > adaptive_threshold

    # Combine conditions
    peaks = local_max & detected_peaks

    # Label peaks
    labeled, num_features = label(peaks)

    # Find the center of mass for each labeled region (peak)
    peak_coords = center_of_mass(smoothed, labels=labeled,
                                 index=np.arange(1, num_features + 1))

    if peak_coords:
        peak_coords = np.array(peak_coords, dtype=int)

    del inv_cross_power_spec, smoothed

    return peak_coords


def analyse_cross_corr(cross_corr, binwidth_dist, binwidth_ang):
    """"""

    # Inverse transform from frequency to spatial domain
    inv_fft = np.real(np.fft.ifft2(cross_corr))
    inv_fft_shifted = np.fft.fftshift(inv_fft)  # the zero shift is at (0,0), this moves it to the middle

    inv_fft_row_median = np.median(inv_fft_shifted, axis=1)[:, np.newaxis]

    # subtract row median for peak detection
    inv_fft_med_sub = inv_fft_shifted - inv_fft_row_median

    # find peaks in the inverse FT
    peak_coords = find_peaks_2d(inv_fft_med_sub, apply_gaussian_filter=True)
    # print('peak_coords', peak_coords)

    if not list(peak_coords):
        base_conf.clean_up(cross_corr, inv_fft, inv_fft_shifted, inv_fft_med_sub)
        return None

    peak_results = np.zeros((len(peak_coords), 3))
    for i, peak in enumerate(peak_coords):
        x_shift_bins, y_shift_bins, signal = get_shift_from_peak(inv_fft_med_sub, peak)

        x_shift = x_shift_bins * binwidth_dist
        y_shift = y_shift_bins * binwidth_ang

        # extract scale and rotation from shift
        scaling = np.e ** (-x_shift)
        rotation = y_shift

        # print(signal, scaling, rotation)
        peak_results[i, :] = np.array([signal, scaling, rotation])

    peak_results_sorted = peak_results[peak_results[:, 0].argsort()[::-1]]

    base_conf.clean_up(cross_corr, inv_fft, inv_fft_shifted, inv_fft_med_sub,
                       inv_fft_row_median, peak_results)

    return peak_results_sorted


def get_shift_from_peak(inv_cross_power_spec, peak):
    """"""

    # sum up the signal in a fixed aperture 1 pixel in each direction around the peak,
    # so a 3x3 array, total 9 pixel
    signal = np.nansum(inv_cross_power_spec[peak[0] - 1:peak[0] + 2,
                       peak[1] - 1:peak[1] + 2])

    around_peak = inv_cross_power_spec[peak[0] - 1:peak[0] + 2, peak[1] - 1:peak[1] + 2]

    # find the sub pixel shift of the true peak
    peak_x_subpixel = np.nansum(np.nansum(around_peak, axis=1)
                                * (np.arange(around_peak.shape[0]) + 1)) / np.nansum(around_peak) - 2
    peak_y_subpixel = np.nansum(np.nansum(around_peak, axis=0)
                                * (np.arange(around_peak.shape[1]) + 1)) / np.nansum(around_peak) - 2

    # get mid-point
    middle_x = inv_cross_power_spec.shape[0] // 2
    middle_y = inv_cross_power_spec.shape[1] // 2

    # calculate final shift
    x_shift = (peak[0] + peak_x_subpixel - middle_x)
    y_shift = (peak[1] + peak_y_subpixel - middle_y)

    # print(signal)
    # print(peak_x_subpixel, peak_y_subpixel)
    # print(middle_x, middle_y, x_shift, y_shift)

    del inv_cross_power_spec

    return x_shift, y_shift, signal


def get_rms(ref_cat, obs, wcsprm_in, radius_px, N_obs_matched):
    """"""
    wcsprm = copy(wcsprm_in)
    N_obs = obs.shape[0]

    # get the radii list
    r = round(3 * radius_px, 1)
    radii_list = list(frange(0.1, r, 0.1))
    result_array = np.zeros((len(radii_list), 4))
    for i, r in enumerate(radii_list):
        matches = find_matches(obs, ref_cat, wcsprm,
                               threshold=r)
        _, _, obs_xy, _, distances, _, _ = matches
        len_obs_x = len(obs_xy[:, 0])

        rms = np.sqrt(np.nanmedian(np.square(distances)))
        # print(r, N_obs_matched, N_obs, N_obs_matched / N_obs, len_obs_x, rms)
        result_array[i, :] = [len_obs_x, len_obs_x / N_obs, r, rms]
        if N_obs_matched <= len_obs_x:
            break

    data = result_array[:, 0]  # number of matches
    _, index = np.unique(data, return_index=True)
    result_array_clean = result_array[np.sort(index)]

    diff_sources = N_obs_matched - result_array_clean[:, 0]

    idx = np.argmin(diff_sources)
    result = result_array_clean[idx, :]
    len_obs_x = int(result[0])
    r = result[2]
    rms = result[3]

    return {"matches": len_obs_x,
            "complete": len_obs_x / N_obs,
            "radius_px": r,
            "rms": rms}


def match_catalogs(catalog1, catalog2, ra_dec_tol=None, radius=None, N=3):
    """
    Match catalogs based on the closest points within a tolerance or centroid within a radius.

    :param DataFrame catalog1: First catalog.
    :param DataFrame catalog2: Second catalog.
    :param float ra_dec_tol: Tolerance in RA/DEC coordinates for matching.
    :param float radius: Radius in pixels to consider for matching.
    :param int N: Number of the closest points to return.

    :returns: A DataFrame that is a subset of catalog2 with matched points.
    """

    if ra_dec_tol is not None:
        # Build a KDTree for catalog1 based on RA/DEC coordinates
        catalog1_tree = KDTree(catalog1[['RA', 'DEC']].values)

        # Find the indices of the closest points for each row in catalog2
        # based on RA/DEC coordinates
        distances, indices = catalog1_tree.query(catalog2[['RA', 'DEC']].values, k=N)

        # Create a mask for distances within the specified tolerance
        mask = distances <= ra_dec_tol

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


def plot_catalog_positions(source_catalog, reference_catalog, wcsprm=None, image=None):
    reference_cat = copy(reference_catalog)

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

    if image is not None:
        plt.imshow(image)

    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.legend()
    plt.title('Source Catalog and Reference Catalog Positions')
    # plt.show()


def find_matches(obs, cat, wcsprm=None, threshold=10.):
    """Match observation with reference catalog using minimum distance."""

    # check if the input has data
    cat_has_data = cat[["xcentroid", "ycentroid"]].any(axis=0).any()
    obs_has_data = obs[["xcentroid", "ycentroid"]].any(axis=0).any()

    if not cat_has_data or not obs_has_data:
        return None, None, None, None, None, None, False

    # convert obs to numpy
    obs_xy = obs[["xcentroid", "ycentroid"]].to_numpy()

    # set up the catalog data; use RA, Dec if used with wcsprm
    if wcsprm is not None:
        cat_xy = wcsprm.s2p(cat[["RA", "DEC"]], 0)
        cat_xy = cat_xy['pixcrd']
    else:
        cat_xy = cat[["xcentroid", "ycentroid"]].to_numpy()

    # calculate the distances
    dist_xy = np.sqrt((obs_xy[:, 0] - cat_xy[:, 0, np.newaxis]) ** 2
                      + (obs_xy[:, 1] - cat_xy[:, 1, np.newaxis]) ** 2)

    idx_arr = np.where(dist_xy == np.min(dist_xy, axis=0))
    min_dist_xy = dist_xy[idx_arr]
    del dist_xy

    mask = min_dist_xy <= threshold
    cat_idx = idx_arr[0][mask]
    obs_idx = idx_arr[1][mask]
    min_dist_xy = min_dist_xy[mask]

    obs_matched = obs.iloc[obs_idx]
    cat_matched = cat.iloc[cat_idx]

    obs_matched = obs_matched.sort_values(by='mag', ascending=True)
    obs_matched.reset_index(drop=True, inplace=True)

    cat_matched = cat_matched.sort_values(by='mag', ascending=True)
    cat_matched.reset_index(drop=True, inplace=True)

    columns_to_add = ['xcentroid', 'ycentroid', 'fwhm', 'fwhm_err', 'include_fwhm']
    cat_matched = cat_matched.assign(**obs_matched[columns_to_add].to_dict(orient='series'))
    del obs, cat

    obs_xy = obs_xy[obs_idx, :]
    cat_xy = cat_xy[cat_idx, :]

    if len(min_dist_xy) == 0:  # meaning the list is empty
        best_score = 0
    else:
        rms = np.sqrt(np.nanmean(np.square(min_dist_xy)))
        best_score = len(obs_xy) / (rms + 1e-6)

    return obs_matched, cat_matched, obs_xy, cat_xy, min_dist_xy, best_score, True


def cross_corr_to_fourier_space(a):
    """Transform 2D array into fourier space. Use padding and normalization."""

    aa = (a - np.nanmean(a)) / np.nanstd(a)

    # wraps around so half the size should be fine, pads 2D array with zeros
    aaa = np.pad(aa, (2, 2), 'constant')
    ff_a = np.fft.fft2(aaa)

    del a, aa, aaa

    return ff_a


def create_bins(min_value, max_value, bin_size, is_distance=True):
    """ Create bins for histogram """

    bin_step = np.deg2rad(bin_size)
    diff = max_value - min_value

    if is_distance:
        bin_step = bin_size
        diff = (np.e ** max_value - np.e ** min_value)

    N = math.ceil(diff / bin_step)
    N += 1

    bins, binwidth = np.linspace(min_value, max_value, N, retstep=True, dtype='float32')

    return bins, binwidth


def frange(x, y, jump=1.0):
    """https://gist.github.com/axelpale/3e780ebdde4d99cbb69ffe8b1eada92c"""
    i = 0.0
    x = float(x)  # Prevent yielding integers.
    y = float(y)  # Comparison converts y to float every time otherwise.
    x0 = x
    epsilon = jump / 2.0
    yield float("%g" % x)  # yield always first value
    while x + epsilon < y:
        i += 1.0
        x = x0 + i * jump
        yield float("%g" % x)


def prepare_histogram_bins_and_ranges(log_distance_obs, angle_obs,
                                      log_distance_cat, angle_cat,
                                      dist_bin_size, ang_bin_size):
    """
    Prepare the bins and ranges for creating histograms based on distance and angle data.

    :param log_distance_obs: np.ndarray
        The log distance data of observations.
    :param angle_obs: np.ndarray
        The angle data of observations.
    :param log_distance_cat: np.ndarray
        The log distance data of catalog.
    :param angle_cat: np.ndarray
        The angle data of catalog.
    :param dist_bin_size: float
        The size of each bin for distance.
    :param ang_bin_size: float
        The size of each bin for the angle.

    :return: tuple
        Returns a tuple containing bins for distance and angle, binwidth for distance and angle, and ranges for both.
    """
    # Set limits for distances and angles
    minimum_distance = min(log_distance_obs)

    # Broader distance range if the scale is just a guess
    maximum_distance = max(max(log_distance_cat), max(log_distance_obs))

    minimum_ang = min(min(angle_cat), min(angle_obs))
    maximum_ang = max(max(angle_cat), max(angle_obs))

    del log_distance_obs, angle_obs, log_distance_cat, angle_cat

    # Create bins and ranges for histogram
    bins_dist, binwidth_dist = create_bins(minimum_distance, maximum_distance, dist_bin_size, is_distance=True)
    bins_ang, binwidth_ang = create_bins(minimum_ang, maximum_ang, ang_bin_size, is_distance=False)

    bins = [len(bins_dist), len(bins_ang)]
    ranges = [[minimum_distance, maximum_distance], [minimum_ang, maximum_ang]]

    return bins, ranges, binwidth_dist, binwidth_ang


def calculate_dist(data_x, data_y):
    """Calculate the distance between positions."""

    data_x = np.array(data_x)  # [0]
    data_y = np.array(data_y)  # [0]

    dist_x = (data_x - data_x.T)
    dist_y = (data_y - data_y.T)

    # only use off diagonal elements
    dist_x = dist_x[np.where(~np.eye(dist_x.shape[0], dtype=bool))]
    dist_y = dist_y[np.where(~np.eye(dist_y.shape[0], dtype=bool))]

    dist = np.sqrt(dist_x ** 2 + dist_y ** 2)

    base_conf.clean_up(data_x, data_y)

    return dist


def calculate_log_dist(data_x, data_y):
    """Calculate logarithmic distance between points."""

    log_dist = np.log(calculate_dist(data_x, data_y) + np.finfo(float).eps)

    del data_x, data_y
    return log_dist


def calculate_angles(data_x, data_y):
    """Calculate the angle with the x-axis."""

    data_x = np.array(data_x)
    data_y = np.array(data_y)

    # get all pairs: vector differences
    vec_x = data_x - data_x.T
    vec_y = data_y - data_y.T
    vec_x = vec_x[np.where(~np.eye(vec_x.shape[0], dtype=bool))]
    vec_y = vec_y[np.where(~np.eye(vec_y.shape[0], dtype=bool))]

    # get the angle with x-axis.
    angles = np.arctan2(vec_x, vec_y)

    # make sure angles are between 0 and 2 Pi
    angles = angles % (2. * np.pi)

    # shift to -pi to pi
    angles[np.where(angles > np.pi)] = -1 * (2. * np.pi - angles[np.where(angles > np.pi)])

    base_conf.clean_up(data_x, data_y, vec_x, vec_y)

    return angles


def rotation_matrix(angle):
    """Return the corresponding rotation matrix"""

    rot = np.array([[np.cos(angle), np.sin(angle)],
                    [-np.sin(angle), np.cos(angle)]])

    del angle
    return rot


def rotate(wcsprm_rot, rot):
    """Help method for offset_with_orientation.
    Set the different rotations in the header.
    """
    # hdr["PC1_1"] = rot[0][0]
    # hdr["PC1_2"] = rot[1][0]
    # hdr["PC2_1"] = rot[0][1]
    # hdr["PC2_2"] = rot[1][1]

    pc = wcsprm_rot.get_pc()
    pc_rotated = rot @ pc
    wcsprm_rot.pc = pc_rotated

    return wcsprm_rot


def scale(wcsprm, scale_factor):
    """Apply the scale to WCS."""

    pc = wcsprm.get_pc()
    pc_scaled = scale_factor * pc
    wcsprm.pc = pc_scaled

    return wcsprm


def translate_wcsprm(wcsprm):
    """ Moving scaling to cdelt, out of the pc matrix.

    Parameters
    ----------
    wcsprm: astropy.wcs.wcsprm
        World coordinate system object describing translation between image and skycoord

    Returns
    -------
    wcsprm
    """

    # Initialize logging for this user-callable function
    log.setLevel(logging.getLevelName(log.getEffectiveLevel()))

    wcs = WCS(wcsprm.to_header())

    # compute the scales corresponding to celestial axes
    scales = utils.proj_plane_pixel_scales(wcs)
    cdelt = wcsprm.get_cdelt()
    scale_ratio = scales / cdelt

    pc = np.array(wcsprm.get_pc())

    pc[0, 0] = pc[0, 0] / scale_ratio[0]
    pc[1, 0] = pc[1, 0] / scale_ratio[1]
    pc[0, 1] = pc[0, 1] / scale_ratio[0]
    pc[1, 1] = pc[1, 1] / scale_ratio[1]

    wcsprm.pc = pc
    wcsprm.cdelt = scales

    return wcsprm, scales


def shift_wcs_central_pixel(wcsprm, x_shift, y_shift):
    """
    Shift the central pixel of the WCS parameters and return the modified WCS parameter object.

    :param wcsprm: WCS parameter object.
    :param x_shift: Shift in the x-direction.
    :param y_shift: Shift in the y-direction.
    :return: Modified WCS parameter object with updated central pixel.
    """

    # Retrieve the current central pixel
    current_central_pixel = wcsprm.crpix

    # Calculate the new central pixel based on the shifts
    new_central_pixel = [current_central_pixel[0] + x_shift,
                         current_central_pixel[1] + y_shift]

    # Update the WCS parameters with the new central pixel
    wcsprm.crpix = new_central_pixel

    # Return the modified wcsprm object
    return copy(wcsprm)


def process_catalog(catalog, wcsprm=None):
    """
    Process the catalog to obtain x and y coordinates.
    If wcsprm is provided, use it to transform RA and DEC to sensor coordinates.
    Otherwise, use the 'xcentroid' and 'ycentroid' from the catalog.

    :param catalog: Catalog containing either RA, DEC or xcentroid, ycentroid.
    :param wcsprm: WCS parameter object (optional).
    :return: Arrays of x and y coordinates.
    """
    if wcsprm is not None:
        # Transform RA, DEC to sensor coordinates using wcsprm
        catalog_on_sensor = wcsprm.s2p(catalog[["RA", "DEC"]], 0)
        catalog_on_sensor = catalog_on_sensor['pixcrd']
        cat_x = np.array([catalog_on_sensor[:, 0]])
        cat_y = np.array([catalog_on_sensor[:, 1]])
    else:
        # Use 'xcentroid' and 'ycentroid' from the catalog directly
        cat_x = np.array([catalog["xcentroid"].values])
        cat_y = np.array([catalog["ycentroid"].values])
    del catalog
    return cat_x, cat_y


def create_histogram(log_distance, angle, bins, ranges):
    """
    Create a 2D histogram from log distance and angle data.
    """

    vals = [log_distance, angle]
    H = fhist.histogram2d(*vals, bins=bins, range=ranges).astype(dtype=np.complex128)

    del log_distance, angle, bins, ranges
    return H


def apply_fft_and_normalize(H):
    """
    Apply FFT to the histogram and normalize the result.
    """

    ff = cross_corr_to_fourier_space(H)

    del H

    return ff


def get_wcs_pixelscale(wcsprm):
    """ Get the pixel scale in deg/pixel from the WCS object """

    pc = wcsprm.get_pc()
    cdelt = wcsprm.get_cdelt()
    wcs_pixscale = (pc @ cdelt)

    return wcs_pixscale


def update_catalog_positions(cat, wcsprm, origin=0):
    """
    Update the catalog with corrected centroid positions based on WCS transformation.

    :param cat: Catalog containing RA and DEC.
    :param wcsprm: WCS parameter object.
    :param origin: Origin for WCS transformation (0 or 1).
    :return: Updated catalog with corrected 'xcentroid' and 'ycentroid'.
    """

    # Copy the catalog to avoid modifying the original
    cat_corrected = copy(cat)

    # Transform RA, DEC to sensor coordinates using wcsprm and the selected mode
    pos_on_det = wcsprm.s2p(cat[["RA", "DEC"]].values, origin)['pixcrd']

    # Update the 'xcentroid' and 'ycentroid' in the corrected catalog
    cat_corrected["xcentroid"] = pos_on_det[:, 0]
    cat_corrected["ycentroid"] = pos_on_det[:, 1]

    del cat

    return cat_corrected
