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
from scipy.optimize import curve_fit

from skimage.transform import warp, AffineTransform
from skimage.measure import ransac

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

rot_matrix_arr = np.array([[[1, 0], [0, 1]], [[-1, 0], [0, -1]],
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
        self.ref_cat = None
        self.image = None

        # make a copy to work with
        source_cat_full = copy(source_df)
        ref_cat_full = copy(ref_df)

        # make sure the catalogs are sorted
        self.source_cat_full = source_cat_full.sort_values(by='mag', ascending=True)
        self.source_cat_full.reset_index(inplace=True, drop=True)

        self.ref_cat_full = ref_cat_full.sort_values(by='mag', ascending=True)
        self.ref_cat_full.reset_index(inplace=True, drop=True)

        self.obs_x = None
        self.obs_y = None

        self.log_dist_obs = None
        self.angles_obs = None

        self.log_dist_cat = None
        self.angles_cat = None

        self.current_wcsprm = None
        self.current_ref_cat = None
        self.current_score = 0
        self.current_Nobs_matched = 0

        self.source_cat_matched = None
        self.ref_cat_matched = None

        self.binwidth_dist = None
        self.binwidth_ang = None
        self.cross_corr = None

        self.rotations = move_match_to_first(np.array(config['pc_matrix']),
                                             rot_matrix_arr) if config['pc_matrix'] is not None else rot_matrix_arr

        self.apply_rot = True
        self.match_radius_px = 5

    def get_source_cat_variables(self):
        """"""

        self.obs_x = np.array([self.source_cat["xcentroid"].values])
        self.obs_y = np.array([self.source_cat["ycentroid"].values])

        self.log_dist_obs = calculate_log_dist(self.obs_x, self.obs_y)
        self.angles_obs = calculate_angles(self.obs_x, self.obs_y)

    def find_wcs(self, input_wcsprm, match_radius_px=1, image=None):
        """Run the plate-solve algorithm to find the best WCS transformation"""

        def reset_best_params():
            nonlocal best_Nobs, best_score, best_rms_dist, best_completeness, best_wcsprm, best_ref_cat, best_src_cat
            best_Nobs = -1
            best_score = -np.inf
            best_rms_dist = np.inf
            best_completeness = 0
            best_wcsprm = None
            best_ref_cat = None
            best_src_cat = None

        self.log.info("> Run WCS calibration")

        self.image = image
        self.match_radius_px = match_radius_px
        match_radius_limit_px = 3. * match_radius_px

        Nobs_total = len(self.source_cat_full)  # total number of detected sources
        Nref_total = len(self.ref_cat_full)  # total number of reference stars

        rotations = self.rotations
        center_coords = [(image.shape[1] // 2, image.shape[0] // 2),
                         (0, 0),
                         (image.shape[1], 0),
                         (0, image.shape[0]),
                         (image.shape[1], image.shape[0])]

        search_area_shape = image.shape
        search_area_additional_width = (1, 1)

        result_dict = {"radius_px": None, "matches": None, "rms": None}
        src_cat_match = ref_cat_match = None

        initial_wcsprm = input_wcsprm.wcs
        final_wcsprm = None

        center_idx = 0
        rot_matrix_idx = 0
        ref_sample_idx = 0

        solve_attempts_count = 0
        high_completeness_count = 0
        possible_solution_count = 0
        no_solution_count = 0

        max_Nobs_initial = 100
        num_ref_samples = 5
        max_rotations = len(rotations)
        max_centers = len(center_coords)
        max_wcs_solve_attempts = max_rotations * max_centers
        max_no_solution_count = 3
        max_wcs_eval_iter = self.config['MAX_WCS_FUNC_ITER']

        use_initial_wcs_estimate = True
        update_src_cat = False
        update_ref_cat = False
        has_large_src_cat = False if Nobs_total <= max_Nobs_initial else True
        has_possible_solution = False
        has_solution = False

        best_Nobs = -1
        best_score = -np.inf
        best_rms_dist = np.inf
        best_completeness = 0
        best_wcsprm = None
        best_ref_cat = None
        best_src_cat = None

        self.source_cat = self.source_cat_full.head(max_Nobs_initial)
        self.get_source_cat_variables()

        self.log.info(f'  Detected sources (total): {Nobs_total}')
        self.log.info(f'  Reference sources (total): {Nref_total}')
        self.log.info(f'  Match radius: {self.match_radius_px:.2f} px')

        while True:

            # failsafe
            if solve_attempts_count >= max_wcs_solve_attempts:
                has_solution = False
                break

            # set rotation matrix
            current_rotMatrix = self.rotations[rot_matrix_idx]

            # apply rotation matrix to WCS
            if use_initial_wcs_estimate:
                self.current_wcsprm = rotate(copy(initial_wcsprm), current_rotMatrix)

            # apply the WCS to the full reference catalog
            ref_cat_full_updated = update_catalog_positions(self.ref_cat_full,
                                                            self.current_wcsprm,
                                                            origin=0)

            # set the cutout center to current index if no possible solution was found,
            # else use the image center since the wcs was adjusted
            current_cutoutCenter = center_coords[center_idx] if not has_possible_solution else center_coords[0]

            # filter sources according to the current cutout center and include only sources
            # within the given cutout shape plus an additional width
            ref_cat_filtered = filter_df_within_bounds(ref_cat_full_updated,
                                                       current_cutoutCenter,
                                                       search_area_shape,
                                                       search_area_additional_width)

            # if the source catalog has more than 100 objects and a possible solution was found, use the full source
            # catalog
            if update_src_cat:
                self.source_cat = self.source_cat_full
                self.get_source_cat_variables()

            # get current object count
            Nobs_current = len(self.source_cat)
            Nref_current = len(ref_cat_filtered)

            # create a list with the number of samples to draw from the reference catalog, evenly spaced in log-space
            ref_cat_sample_number = np.unique(np.logspace(np.log10(Nobs_current),
                                                          np.log10(Nref_current),
                                                          num=num_ref_samples,
                                                          base=10, endpoint=False, dtype=int))

            if update_ref_cat:

                # update the reference catalog by including those objects from the reference catalog
                # that fall within n times the current match radius
                self.ref_cat = get_updated_reference_catalog(self.source_cat, ref_cat_filtered,
                                                             src_cat_match, ref_cat_match,
                                                             match_radius_limit_px)

                #
                self.source_cat = match_catalogs(self.ref_cat,
                                                 self.source_cat,
                                                 radius=match_radius_limit_px, N=1)
            else:
                # select the N brightest reference sources
                self.ref_cat = ref_cat_filtered.head(ref_cat_sample_number[ref_sample_idx])

            if no_solution_count == 0 and not has_possible_solution:
                self.log.info(
                    f"  > Testing configuration [{solve_attempts_count + 1}/{max_wcs_solve_attempts}]:")
                self.log.info(
                    f"{' ':<4}Rotation matrix [{rot_matrix_idx + 1}/{len(self.rotations)}]: "
                    f"[{', '.join(map(str, current_rotMatrix))}]")
                self.log.info(
                    f"{' ':<4}Cutout center position [{center_idx + 1}/{len(center_coords)}] (x, y): "
                    f"({', '.join(map(str, current_cutoutCenter))})")

                self.log.info(f"{' ':<4}Reference sources (within cutout): {Nref_current}")
                if has_large_src_cat:
                    self.log.info(f"{' ':<4}Detected sources (brightest): {Nobs_current} "
                                  f"out of {Nobs_total}")
                else:
                    self.log.info(f"{' ':<4}Detected sources (all): {Nobs_current}")

                self.log.info(f"{' ':<4}> Try [{no_solution_count+1}/{max_no_solution_count}] "
                              f"Initial number of reference sources: {len(self.ref_cat)}/{Nref_current}")

            elif no_solution_count != 0 and not has_possible_solution:
                self.log.info(f"{' ':<4}> Try [{no_solution_count+1}/{max_no_solution_count}] "
                              f"Increasing number of reference sources: {len(self.ref_cat)}/{Nref_current}")

            # process the current source catalog and the selected reference sources
            sample_result = self.process_sample()

            # check result
            if sample_result is None or sample_result[0] < self.config['MIN_SOURCE_NO_CONVERGENCE']:
                no_solution_count += 1

                if no_solution_count >= max_no_solution_count:
                    self.log.warning(f"{' ':<6}> No solution found after {max_no_solution_count} trails. "
                                     f"Skipping to next configuration.")
                    ref_sample_idx = num_ref_samples - 1  # Force iteration
                    no_solution_count = 0  # Reset counter for the next iteration
                else:
                    if sample_result is None:
                        self.log.warning(f"{' ':<6}> No solution/matches found. Trying again...")
                    else:
                        self.log.warning(f"{' ':<6}> Configuration resulted in less than "
                                         f"{self.config['MIN_SOURCE_NO_CONVERGENCE']} matches. "
                                         f"Trying again...")

                # increase counter
                if center_idx == max_centers - 1:
                    rot_matrix_idx += 1
                    center_idx = 0
                    solve_attempts_count += 1
                else:
                    if ref_sample_idx == num_ref_samples - 1:
                        center_idx += 1
                        ref_sample_idx = 0
                        solve_attempts_count += 1
                    else:
                        # Move to the next sample within the same iteration
                        ref_sample_idx += 1

                # reset switches
                src_cat_match = ref_cat_match = None
                use_initial_wcs_estimate = True
                update_ref_cat = False
                update_src_cat = False
                has_possible_solution = False

                # reset best result
                reset_best_params()
                continue
            else:
                has_possible_solution = True

            # update possible solution count
            possible_solution_count += 1

            # unpack results
            Nobs_matched, score, wcsprm_out, result_info, src_cat_match, ref_cat_match = sample_result

            # set variables
            Nref_matched = len(self.ref_cat)
            completeness = Nobs_matched / Nref_matched
            rms_dist = 1. / (score * Nobs_matched)
            self.current_wcsprm = copy(wcsprm_out)

            # check completeness
            if completeness >= self.config['THRESHOLD_CONVERGENCE']:
                high_completeness_count += 1

            # update best result
            if best_score < score and best_Nobs <= Nobs_matched:
                best_Nobs = Nobs_matched
                best_score = score
                best_completeness = completeness
                best_wcsprm = copy(wcsprm_out)
                best_ref_cat = self.ref_cat
                best_src_cat = self.source_cat

            # print(possible_solution_count, Nobs_matched, score, completeness, high_completeness_count, rms_dist)

            # if there are more than the N brightest sources, increase the number of sources and re-run
            if has_large_src_cat and not update_src_cat:
                update_src_cat = True
                update_ref_cat = True
                use_initial_wcs_estimate = False

                possible_solution_count = 0
                high_completeness_count = 0
                self.log.info(f"{' ':<6}> Possible solution found. Confirming...")
                self.log.info(f"{' ':<6}Increase source number: "
                              f"{Nobs_current} (brightest) --> {len(self.source_cat_full)} (all)")
                continue

            elif not has_large_src_cat and possible_solution_count == 1:
                self.log.info(f"{' ':<6}> Possible solution found. Confirming...")

            if completeness == 1.0 or high_completeness_count >= 3 or possible_solution_count >= max_wcs_eval_iter:

                has_solution = True
                final_wcsprm = best_wcsprm

                # get rms estimate
                result_dict = get_rms(best_ref_cat,
                                      best_src_cat,
                                      best_wcsprm,
                                      self.match_radius_px,
                                      best_Nobs)

                self.log.info(f"{' ':<6}>> " + base_conf.BCOLORS.PASS + "Solution found." + base_conf.BCOLORS.OKGREEN)
                self.log.info(f"{' ':<4}{'-'*50}")
                self.log.info(f"{' ':<4}Summary:")
                self.log.info(f"{' ':<4}{'-'*50}")
                self.log.info(f"{' ':<5}{'Evaluations':<30} {possible_solution_count} ")
                self.log.info(f"{' ':<5}{'Matched (detected sources)':<30} {f'{best_Nobs}/{Nobs_total}'} ")
                self.log.info(f"{' ':<5}{'Matched (reference sources)':<30} {f'{len(best_ref_cat)}/{Nref_total}'} ")
                self.log.info(f"{' ':<5}{'Matched (src/ref)':<30} {f'{best_Nobs}/{len(best_ref_cat)}'} ")
                self.log.info(
                    f"{' ':<5}{'Completeness':<30} {best_completeness:.3f} ({best_completeness * 100:.1f}%)")
                self.log.info(f"{' ':<5}{'Match radius (px)':<30} {result_dict['radius_px']:.2f} ")
                self.log.info(f"{' ':<5}{'RMS (arcsec)':<30} {result_dict['rms']:.2f} ")
                self.log.info(f"{' ':<4}{'-'*50}")

                break
            else:
                # update switches for next iteration
                update_ref_cat = True
                use_initial_wcs_estimate = False

        return has_solution, final_wcsprm, result_dict

    def process_sample(self):
        """"""

        # get the catalog positions
        cat_x, cat_y = process_catalog(self.ref_cat, self.current_wcsprm)

        # calculate the distances and angles
        self.log_dist_cat = calculate_log_dist(cat_x, cat_y)
        self.angles_cat = calculate_angles(cat_x, cat_y)

        del cat_x, cat_y

        best_score = 0
        best_scaling = 1
        best_rotation = 0
        best_Nobs_matched = -1
        best_wcsprm = None
        best_source_cat_match = None
        best_ref_cat_match = None

        rot_multiplier_list = [1., -1.]
        for n in rot_multiplier_list:

            self.angles_obs = n * self.angles_obs

            # use FFT to find the rotation and scaling first
            self.compute_cross_corr()

            # get the highest peak from the cross-power spectrum
            result_scale_rot = analyse_cross_corr(self.cross_corr, self.binwidth_dist, self.binwidth_ang)

            # move on if no solution was found
            if result_scale_rot is None:
                continue

            result_signal = result_scale_rot[0]
            result_scaling = result_scale_rot[1]

            # move on if the solution is not good enough
            if result_signal == 0. or result_scaling < 0.9 or 1.1 < result_scaling:
                continue

            # test for reflection
            for m in rot_multiplier_list:
                result_rot_rad = m * result_scale_rot[2]
                # print(n,m)
                # print('rot', result_rot_rad)
                # apply rotation and scale
                rot_mat = rotation_to_matrix(result_rot_rad)
                wcsprm_new = rotate(copy(self.current_wcsprm), rot_mat)
                wcsprm_new = scale(wcsprm_new, result_scaling)

                # find offset between datasets
                shift_signal, x_shift, y_shift = self.find_offset(self.ref_cat,
                                                                  wcsprm_new,
                                                                  offset_binwidth=1)

                # apply the found shift
                wcsprm_shifted = shift_wcs_central_pixel(wcsprm_new, x_shift, y_shift)
                del wcsprm_new

                # find matches
                matches = find_matches(self.source_cat, self.ref_cat,
                                       wcsprm_shifted,
                                       threshold=self.match_radius_px)

                # unpack result
                source_cat_matched, ref_cat_matched, obs_xy, cat_xy, _, current_score, _ = matches
                Nobs_matched = len(obs_xy)
                # print(Nobs_matched, current_score, result_scaling, result_rot_rad)

                # update best result
                if best_Nobs_matched <= Nobs_matched and best_score <= current_score:
                    best_Nobs_matched = Nobs_matched
                    best_score = current_score
                    best_scaling = result_scaling
                    best_rotation = result_rot_rad
                    best_wcsprm = copy(wcsprm_shifted)
                    best_source_cat_match = source_cat_matched
                    best_ref_cat_match = ref_cat_matched

                del wcsprm_shifted

                # plot_catalog_positions(self.source_cat,
                #                        self.ref_cat,
                #                        best_wcsprm, self.image)
                # plt.show()
        return (best_Nobs_matched, best_score, best_wcsprm, [best_scaling, best_rotation],
                best_source_cat_match, best_ref_cat_match)

    def compute_cross_corr(self):
        """"""

        bins, ranges, self.binwidth_dist, self.binwidth_ang = prepare_histogram_bins_and_ranges(
            self.log_dist_obs, self.angles_obs,
            self.log_dist_cat, self.angles_cat,
            self.dist_bin_size, self.ang_bin_size)

        H_obs = create_histogram(self.log_dist_obs, self.angles_obs, bins, ranges)
        H_cat = create_histogram(self.log_dist_cat, self.angles_cat, bins, ranges)

        ff_obs = apply_fft_and_normalize(H_obs)
        ff_cat = apply_fft_and_normalize(H_cat)

        # calculate cross-correlation
        cross_corr = ff_obs * np.conj(ff_cat)

        base_conf.clean_up(H_obs, H_cat, ff_obs, ff_cat, bins, ranges)

        self.cross_corr = cross_corr

    def find_wcs_working(self, init_wcsprm, match_radius_fwhm=1, image=None):
        """Run the plate-solve algorithm to find the best WCS transformation"""

        has_solution = False
        dict_rms = {"radius_px": None, "matches": None, "rms": None}
        final_wcsprm = None

        init_wcsprm = init_wcsprm.wcs
        max_wcs_iter = self.config['MAX_WCS_FUNC_ITER']

        self.log.info("> Run WCS calibration")

        self.source_cat = self.source_cat_full
        self.source_cat = self.source_cat.head(5)
        self.get_source_cat_variables()
        print(self.source_cat)

        # update the reference catalog with the initial wcs parameters
        ref_cat_orig = update_catalog_positions(self.reference_cat, init_wcsprm)
        # ref_cat_orig = ref_cat_orig.head(self.config['REF_SOURCES_MAX_NO'])
        print(self.reference_cat)
        print(ref_cat_orig)

        Nobs = len(self.source_cat)  # number of detected sources
        Nref = len(ref_cat_orig)  # number of reference stars
        source_ratio = (Nobs / Nref) * 100.  # source ratio in percent
        percent_to_select = np.ceil(source_ratio * 3)
        n_percent = percent_to_select if percent_to_select < 100 else 100
        num_rows = int(Nref * (n_percent / 100))
        wcs_match_radius_px = 1. * match_radius_fwhm

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
            ref_cat = selected_rows  #.sort_values(by='mag', ascending=True)
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

            if current_signal == 0. or current_scaling < 0.9 or 1.1 < current_scaling:
                continue

            rot_mat = rotation_to_matrix(current_rot)
            wcsprm_new = rotate(copy(current_wcsprm), rot_mat)
            wcsprm_new = scale(wcsprm_new, current_scaling)
            shift_signal, x_shift, y_shift = self.find_offset(self.current_ref_cat,
                                                              wcsprm_new,
                                                              offset_binwidth=2)

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

        ref_cat_orig_updated = update_catalog_positions(ref_cat_orig, self.current_wcsprm, 1)

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
                                                           wcsprm_in,
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

            rot = rotation_to_matrix(rotation)
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

def get_updated_reference_catalog(src_cat, ref_cat, src_cat_matched, ref_cat_matched, radius):
    """"""

    # get the remaining not matched source from the source catalog
    remaining_sources = src_cat[~src_cat['id'].isin(src_cat_matched['id'])]
    # print(len(remaining_sources))

    # get the closest N sources from the reference catalog for the remaining sources
    ref_sources_to_add = match_catalogs(remaining_sources,
                                        ref_cat,
                                        radius=radius, N=1)
    # print(len(ref_sources_to_add))

    # get the closest N sources from the reference catalog
    # for the matched reference sources
    ref_cat_orig_matched = match_catalogs(ref_cat_matched,
                                          ref_cat,
                                          ra_dec_tol=1e-4, N=1)
    # print(len(ref_cat_orig_matched))

    # combine the datasets
    ref_cat_subset_new = pd.concat([ref_cat_orig_matched, ref_sources_to_add],
                                   ignore_index=True)
    # print(len(ref_cat_subset_new))

    return ref_cat_subset_new


def find_peaks_2d(inv_cross_power_spec, threshold_multiplier=5., apply_gaussian_filter=False,
                  gauss_sigma=1, size=3):
    """
    Find peaks in a 2D array with a given size for the neighborhood and threshold.

    :param inv_cross_power_spec: 2D array of cross-correlations.
    :param threshold_multiplier:
    :param apply_gaussian_filter: Boolean flag to apply Gaussian filter for smoothing (default is False).
    :param size: Size of the neighborhood to consider for the local maxima.
    :param gauss_sigma:

    :return: List of peak coordinates.
    """

    median_val = np.median(inv_cross_power_spec)
    std_val = np.std(inv_cross_power_spec)
    adaptive_threshold = median_val + threshold_multiplier * std_val  # Adjust multiplier as needed

    if apply_gaussian_filter:
        # Apply a Gaussian filter for smoothing
        smoothed = gaussian_filter(inv_cross_power_spec, sigma=gauss_sigma)
    else:
        # Skip Gaussian filtering
        smoothed = inv_cross_power_spec

    # Find local maxima
    local_max = maximum_filter(smoothed, size=size,
                               mode='constant', origin=0) == smoothed

    # Apply the threshold
    detected_peaks = smoothed > adaptive_threshold

    # Combine conditions
    peaks = local_max & detected_peaks

    # Label peaks
    labeled, num_features = label(peaks)

    # Find the center of mass for each labeled region (peak)
    peak_coords = center_of_mass(smoothed, labels=labeled,
                                 index=np.arange(1, num_features + 1))

    # convert to array if not empty
    if peak_coords:
        peak_coords = np.array(peak_coords, dtype=int)

    base_conf.clean_up(inv_cross_power_spec, smoothed, detected_peaks, labeled, num_features)

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
    peak_coords = find_peaks_2d(inv_fft_med_sub,
                                threshold_multiplier=5.,
                                gauss_sigma=1,
                                apply_gaussian_filter=True)
    # print('peak_coords', peak_coords)

    if not list(peak_coords):
        base_conf.clean_up(cross_corr, inv_fft, inv_fft_shifted, inv_fft_med_sub)
        return None

    peak_results = np.zeros((len(peak_coords), 3))
    for i, peak in enumerate(peak_coords):
        x_shift_bins, y_shift_bins, signal = get_shift_from_peak(inv_fft_shifted, peak)

        x_shift = x_shift_bins * binwidth_dist
        y_shift = y_shift_bins * binwidth_ang

        # extract scale and rotation from shift
        scaling = np.e ** (-x_shift)
        rotation = y_shift

        # print(signal, scaling, rotation)
        peak_results[i, :] = np.array([signal, scaling, rotation])

    # sort by signal strength
    peak_results_sorted = peak_results[peak_results[:, 0].argsort()[::-1]][0]

    base_conf.clean_up(cross_corr, inv_fft, inv_fft_shifted, inv_fft_med_sub,
                       inv_fft_row_median, peak_results)

    return peak_results_sorted


def get_shift_from_peak(inv_cross_power_spec, peak):
    """"""
    n_1 = 1
    n_2 = 2
    n = n_1 + n_2

    # sum up the signal in a fixed aperture 1 pixel in each direction around the peak,
    # so a 3x3 array, total 9 pixel
    signal = np.nansum(inv_cross_power_spec[peak[0] - n_1:peak[0] + n_2,
                       peak[1] - n_1:peak[1] + n_2])

    around_peak = inv_cross_power_spec[peak[0] - n_1:peak[0] + n_2, peak[1] - n_1:peak[1] + n_2]
    # print(signal, around_peak)
    if around_peak.shape != (n, n):
        return 0, 0, -1.
    peak_x_subpixel = cosine_interpolation_1d(around_peak[:, 1]) + 0.5
    peak_y_subpixel = cosine_interpolation_1d(around_peak[1, :]) + 0.5

    # get mid-point
    middle_x = inv_cross_power_spec.shape[0] / 2.
    middle_y = inv_cross_power_spec.shape[1] / 2.

    # find the sub pixel shift of the true peak
    # peak_x_subpixel, peak_y_subpixel = subpixel_peak_position(around_peak)
    # print(peak_x_subpixel, peak_y_subpixel)

    # calculate final shift
    x_shift = peak[0] - middle_x + peak_x_subpixel
    y_shift = peak[1] - middle_y + peak_y_subpixel

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

        rms = np.sqrt(np.nanmean(np.square(distances)))
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


def plot_catalog_positions(source_catalog, reference_catalog, wcsprm_in=None, image=None):
    ref_cat = copy(reference_catalog)

    if wcsprm_in is not None:
        pos_on_det = wcsprm_in.s2p(reference_catalog[["RA", "DEC"]], 0)
        pos_on_det = pos_on_det['pixcrd']
        ref_cat["xcentroid"] = pos_on_det[:, 0]
        ref_cat["ycentroid"] = pos_on_det[:, 1]

    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot the positions of the source catalog
    ax.scatter(source_catalog['xcentroid'], source_catalog['ycentroid'], c='blue', marker='o',
               label='Source Catalog', alpha=0.7)

    # Plot the positions of the reference catalog
    ax.scatter(ref_cat['xcentroid'], ref_cat['ycentroid'], c='red', marker='x',
               label='Reference Catalog',
               alpha=0.7)

    if image is not None:
        plt.imshow(image, vmin=0, vmax=2000)

    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.legend()
    plt.title('Source Catalog and Reference Catalog Positions')
    # plt.show()


def find_matches(obs, cat, wcsprm_in=None, threshold=10.):
    """Match observation with reference catalog using minimum distance."""

    # check if the input has data
    cat_has_data = cat[["xcentroid", "ycentroid"]].any(axis=0).any()
    obs_has_data = obs[["xcentroid", "ycentroid"]].any(axis=0).any()

    if not cat_has_data or not obs_has_data:
        return None, None, None, None, None, None, False

    # convert obs to numpy
    obs_xy = obs[["xcentroid", "ycentroid"]].to_numpy()

    # set up the catalog data; use RA, Dec if used with wcsprm
    if wcsprm_in is not None:
        cat_xy = wcsprm_in.s2p(cat[["RA", "DEC"]], 0)
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


def rotation_to_matrix(angle):
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


def process_catalog(catalog, wcsprm=None, origin=0):
    """
    Process the catalog to get x and y coordinates.
    If wcsprm is provided, use it to transform RA and DEC to sensor coordinates.
    Otherwise, use the 'xcentroid' and 'ycentroid' from the catalog.

    :param catalog: Catalog containing either RA, DEC or xcentroid, ycentroid.
    :param wcsprm: WCS parameter object (optional).
    :return: Arrays of x and y coordinates.
    """
    if wcsprm is not None:
        # Transform RA, DEC to sensor coordinates using wcsprm
        catalog_on_sensor = wcsprm.s2p(catalog[["RA", "DEC"]], origin)['pixcrd']
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
    pos_on_det = wcsprm.s2p(cat[["RA", "DEC"]], origin)
    pos_on_det = pos_on_det['pixcrd']

    # Update the 'xcentroid' and 'ycentroid' in the corrected catalog
    cat_corrected["xcentroid"] = pos_on_det[:, 0]
    cat_corrected["ycentroid"] = pos_on_det[:, 1]

    del cat

    return cat_corrected


def move_match_to_first(array1, array2):
    """
    Move the first match of array1 in array2 to the first position of array2.

    :param array1: numpy array of shape (1, 2, 2)
    :param array2: numpy array of shape (n, 2, 2)
    :return: Modified array2 with the match moved to the first position
    """

    array1 = array1.reshape(2, 2)  # Reshape array1 for comparison
    n = array2.shape[0]

    for i in range(n):
        if np.array_equal(array2[i], array1):
            # Swap the found match with the first element
            array2[[0, i]] = array2[[i, 0]]
            break
    return array2


def filter_df_within_bounds(df, center_coords, shape, additional_width):
    """"""
    x_coord, y_coord = center_coords
    y_shape, x_shape = shape
    x_add_width, y_add_width = additional_width

    x_min = x_coord - x_shape / 2 - x_add_width
    x_max = x_coord + x_shape / 2 + x_add_width

    y_min = y_coord - y_shape / 2 - y_add_width
    y_max = y_coord + y_shape / 2 + y_add_width

    filtered_df = df[(df['xcentroid'] >= x_min) & (df['xcentroid'] <= x_max) &
                     (df['ycentroid'] >= y_min) & (df['ycentroid'] <= y_max)]
    return filtered_df


def gaussian_2d(xy, amp, xo, yo, sigma, theta, offset):
    x, y = xy
    xo = float(xo)
    yo = float(yo)
    # theta = 0
    sigma_x = sigma_y = sigma
    a = (np.cos(theta) ** 2) / (2 * sigma_x ** 2) + (np.sin(theta) ** 2) / (2 * sigma_y ** 2)
    b = -(np.sin(2 * theta)) / (4 * sigma_x ** 2) + (np.sin(2 * theta)) / (4 * sigma_y ** 2)
    c = (np.sin(theta) ** 2) / (2 * sigma_x ** 2) + (np.cos(theta) ** 2) / (2 * sigma_y ** 2)
    g = offset + amp * np.exp(- (a * ((x - xo) ** 2) + 2 * b * (x - xo) * (y - yo) + c * ((y - yo) ** 2)))
    return g.ravel()


def subpixel_peak_position(around_peak):
    """"""
    x = np.linspace(0, around_peak.shape[0] - 1, around_peak.shape[0])
    y = np.linspace(0, around_peak.shape[1] - 1, around_peak.shape[1])
    x, y = np.meshgrid(x, y)

    try:
        initial_guess = (around_peak.max(), 1, 1, 1, 0, 0)
        result = curve_fit(gaussian_2d, xdata=(x, y), ydata=around_peak.ravel(), method='lm',
                           p0=initial_guess, nan_policy='omit')
        _, x_peak, y_peak, _, _, _ = result[0]
        peak_x_subpixel = x_peak - 0.5
        peak_y_subpixel = y_peak - 0.5

    except (RuntimeError, ValueError):
        peak_x_subpixel = peak_y_subpixel = 0

    return peak_x_subpixel, peak_y_subpixel


def cosine_interpolation_1d(v):
    """
    Perform cosine interpolation on a 1D array.

    :param v: 1D array of values
    :return: Sub-pixel position of the peak
    """
    if len(v) != 3:
        raise ValueError("Input array must have exactly three elements.")

    denom = v[0] - 2 * v[1] + v[2]
    if denom == 0:
        return 0
    else:
        return (v[0] - v[2]) / (2 * denom)
