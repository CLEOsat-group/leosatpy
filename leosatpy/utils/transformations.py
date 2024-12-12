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
import sys
from copy import copy, deepcopy
import inspect
import logging
import fast_histogram as fhist

# scipy
from scipy.spatial import KDTree, cKDTree
from scipy.ndimage import maximum_filter, gaussian_filter, label
from scipy.ndimage.measurements import center_of_mass
from scipy.optimize import least_squares

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.wcs import WCS, utils

from . import base_conf as bc

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

# rot_matrix_arr = np.array([[[1, 0], [0, 1]], [[-1, 0], [0, -1]],
#                            [[-1, 0], [0, 1]], [[1, 0], [0, -1]],
#                            [[0, 1], [1, 0]], [[0, -1], [-1, 0]],
#                            [[0, -1], [1, 0]], [[0, 1], [-1, 0]]])
rot_matrix_arr = np.array([[[1, 0], [0, 1]], [[-1, 0], [0, -1]],
                           [[-1, 0], [0, 1]], [[1, 0], [0, -1]]])


# -----------------------------------------------------------------------------


class FindWCS(object):
    """"""

    def __init__(self, source_df, ref_df, hdr, config, _log: logging.Logger = log):
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
        self._refine = False

        # check binning keyword
        if isinstance(config['binning'][0], str) and isinstance(config['binning'][1], str):
            bin_x = int(hdr[config['binning'][0]])
            bin_y = int(hdr[config['binning'][1]])
        else:
            bin_x = int(hdr['binning'][0])
            bin_y = int(hdr['binning'][1])

        self.bin_x = bin_x
        self.bin_y = bin_y
        self.image_shape_full = config['image_size_1x1']

        search_area_shape = (self.image_shape_full[0] // self.bin_x,
                             self.image_shape_full[1] // self.bin_y)
        center_coords = [(search_area_shape[1] // 2, search_area_shape[0] // 2)]

        self.search_area_shape = search_area_shape
        self.search_area_additional_width = (10, 10)
        self.current_cutoutCenter = center_coords[0]

        self._debug = False

    def get_source_cat_variables(self):
        """"""

        self.obs_x = np.array([self.source_cat["xcentroid"].values])
        self.obs_y = np.array([self.source_cat["ycentroid"].values])

        self.log_dist_obs = calculate_log_dist(self.obs_x, self.obs_y)
        self.angles_obs = calculate_angles(self.obs_x, self.obs_y)

    def find_wcs(self, input_wcsprm, image, match_radius_px=1):
        """Run the plate-solve algorithm to find the best WCS transformation"""

        # Initialization
        self.log.info("> Run WCS calibration")
        self.image = image
        rotations = self.rotations
        self.match_radius_px = match_radius_px
        match_radius_limit_px = self.config['MATCH_RADIUS_LIM'] * match_radius_px
        num_ref_samples = self.config['NUM_REF_SAMPLES']  # Number of samples to draw from the reference catalog
        num_ref_max = self.config['MAX_REF_NUM']
        max_src_num = self.config['MAX_SOURCE_NUM']
        max_num_top_solution = self.config['MAX_NUM_TOP_SOLUTION']

        Nobs_total = len(self.source_cat_full)  # Total number of detected sources
        Nref_total = len(self.ref_cat_full)  # Total number of reference stars
        self.config['MIN_SOURCE_NO_CONVERGENCE'] = self.config['MIN_SOURCE_NO_CONVERGENCE'] if Nobs_total > 15 else 3

        initial_wcsprm = input_wcsprm.wcs

        solve_attempts_count = 0
        max_no_solution_count = self.config['MAX_NO_SOLUTION_COUNT']

        use_initial_wcs_estimate = True  # Flag to indicate the use of the initial WCS
        has_large_src_cat = False if Nobs_total <= max_src_num else True

        possible_solutions = []

        # Prepare source catalog
        if has_large_src_cat:
            self.log.info("  Using a subset of the source catalog for initial matching.")
            self.source_cat = self.source_cat_full.head(max_src_num)
        else:
            self.source_cat = self.source_cat_full.copy()
        self.get_source_cat_variables()

        self.log.info(f"  {'Detected sources (total)':<28}: {Nobs_total}")
        self.log.info(f"  {'Reference sources (total)':<28}: {Nref_total}")
        self.log.info(f"  {'Match radius (r)':<28}: {self.match_radius_px:.2f} px")
        str_match_radius = f"Match radius limit ({int(self.config['MATCH_RADIUS_LIM'])} x r)"
        self.log.info(f"  {str_match_radius:<28}: {match_radius_limit_px:.2f} px")
        self.log.info(f"  > Search possible solutions (This may take a second.)")

        total_iterations = len(rotations) * num_ref_samples
        rot_matrix_idx = 0
        # Loop possible initial rotations of the WCS
        while rot_matrix_idx < len(rotations):
            ref_sample_idx = 0
            no_solution_count = 0
            solve_attempts_count += 1

            # Apply rotation matrix
            current_rotMatrix = rotations[rot_matrix_idx]

            self.log.debug(f"{' ':<4}Testing PC matrix {rot_matrix_idx + 1}/{len(rotations)}: "
                          f"[{', '.join(map(str, current_rotMatrix))}]")

            while ref_sample_idx <= num_ref_samples:
                iteration = rot_matrix_idx * (num_ref_samples + 1) + ref_sample_idx
                bc.print_progress_bar(iteration, total_iterations, length=80,
                                      color=bc.BCOLORS.OKGREEN)

                # Apply PC matrix to initial WCS
                if use_initial_wcs_estimate:
                    self.current_wcsprm = rotate(copy(initial_wcsprm), current_rotMatrix)

                # Apply current wcs to full reference catalog and filter within image boundaries
                ref_cat_filtered = self.get_filtered_ref_cat(current_wcsprm=self.current_wcsprm, origin=0)

                # Select only the n brightest sources
                ref_cat_filtered = ref_cat_filtered.head(num_ref_max)

                # Determine the number of reference sources to consider
                Nobs_current = len(self.source_cat)
                Nref_current = len(ref_cat_filtered)

                if Nref_current <= 0:
                    self.log.warning("{' ':<4}No reference sources found after filtering.")
                    break  # No reference sources to consider, move to next rotation

                use_endpoint = False
                if Nobs_current < 21:
                    use_endpoint = True

                ref_cat_sample_number = np.unique(
                    np.logspace(np.log10(max(Nobs_current, 1)), np.log10(Nref_current),
                                num=num_ref_samples, base=10,
                                endpoint=use_endpoint, dtype=int))

                # Ensure ref_sample_idx is within bounds
                if ref_sample_idx >= len(ref_cat_sample_number):
                    break  # No more samples to consider, move to next rotation

                # Select the N brightest reference sources
                Nref_sample = ref_cat_sample_number[ref_sample_idx]
                self.ref_cat = ref_cat_filtered.head(Nref_sample)
                Nref_selected = len(self.ref_cat)

                # Process the sample
                sample_result = self.process_sample()
                # print(sample_result[0])

                if sample_result is not None and sample_result[0] >= self.config['MIN_SOURCE_NO_CONVERGENCE']:

                    # Unpack and compute metrics
                    Nobs_matched, score, wcsprm_out, result_info, src_cat_match, ref_cat_match = sample_result

                    completeness = Nobs_matched / Nobs_current

                    self.log.debug(
                        f"{' ':<4}>> Attempt {ref_sample_idx + 1}/{len(ref_cat_sample_number)}: "
                        f"Possible solution found")
                    self.log.debug(f"{' ':<6} Nref={Nref_selected}/{Nref_current}, "
                                   f"Nobs_matched={Nobs_matched}/{Nobs_current} => "
                                   f"completeness={completeness:.3f}, "
                                   f"score={score:.3f}, rms={result_info[1]:.3f} px")

                    # Store the possible solution
                    possible_solutions.append({
                        'rotation_idx': rot_matrix_idx,
                        'method': 'fft',
                        'Nobs_matched': Nobs_matched,
                        'score': score,
                        'completeness': completeness,
                        'wcsprm': copy(wcsprm_out),
                        'src_cat_match': src_cat_match,
                        'ref_cat_match': ref_cat_match,
                        'match_radius': result_info[0],
                        'rms_dist': result_info[1],
                        'std_x': result_info[2][0],
                        'std_y': result_info[2][1]
                    })

                    ref_sample_idx += 1
                else:
                    # No solution found, increase number of reference sources
                    no_solution_count += 1
                    self.log.debug(
                        f"{' ':<4}No solution found for reference sample {ref_sample_idx + 1}. "
                        f"Trying with more reference sources."
                    )
                    if no_solution_count >= max_no_solution_count:
                        # Move to next rotation matrix
                        break
                    else:
                        ref_sample_idx += 1

                    # use initial wcs estimate
                    use_initial_wcs_estimate = True

            # Proceed to next rotation matrix
            rot_matrix_idx += 1

        sys.stdout.write('\n')

        result_dict_fail = {"radius_px": None, "matches": None, "rms": None,
                            'std_x': None, 'std_y': None}

        # After the main loop
        if possible_solutions:
            self.log.info(f"{' ':<2}>> Possible solutions: {len(possible_solutions)}")
            # Sort possible solutions by completeness, number of matches, and score
            possible_solutions.sort(key=lambda x: (x['Nobs_matched'], x['score']),
                                    reverse=True)

            # Select the top n solutions
            top_solutions = possible_solutions[:max_num_top_solution]

            confirmed_solutions = []
            best_overall_completeness = 0.
            best_overall_rms = np.inf
            best_overall_solution = None
            best_solution = None

            self.log.info(f"{' ':<2}> Analysing top {len(top_solutions)} solutions")

            # Loop over top solutions
            for idx, solution in enumerate(top_solutions):

                solution['has_converged'] = False

                self.log.debug(f"{' ':<4}Run WCS fit [{idx + 1}/{len(top_solutions)}]")

                current_wcsprm = solution['wcsprm']

                # Update source catalog to include all sources if necessary
                if has_large_src_cat:
                    self.source_cat = self.source_cat_full
                    self.get_source_cat_variables()

                # Create a new index column
                source_df = self.source_cat.reset_index(drop=False).rename(columns={'index': 'obs_idx'})

                # Apply WCS and get matched data
                matched_src_df, ref_cat_filtered, has_m_before, dist_mask_before = (
                    self.apply_wcs_and_filter_sources(current_wcsprm=current_wcsprm,
                                                      source_df=source_df,
                                                      match_radius=match_radius_limit_px))

                # Prepare data for fitting
                src_df_fit, ref_df_fit = self.extract_sources_subset(matched_src_df,
                                                                     ref_cat_filtered,
                                                                     has_m_before)

                # Fit possible WCS solution using least-squares
                fit_status, refined_wcs = self.refine_wcs_fit(obs_df=src_df_fit,
                                                              ref_df=ref_df_fit,
                                                              wcsprm_in=current_wcsprm,
                                                              match_radius=match_radius_px)

                # Apply WCS and get matched data for fitted WCS
                matched_src_df_after, ref_cat_filtered_after, has_m_after, dist_mask_after = (
                    self.apply_wcs_and_filter_sources(current_wcsprm=refined_wcs,
                                                      source_df=source_df,
                                                      match_radius=match_radius_limit_px))
                #
                src_df_after, ref_df_after = self.extract_sources_subset(matched_src_df_after,
                                                                         ref_cat_filtered_after,
                                                                         has_m_after, dist_mask_after)

                # Compute residual statistics for the fit results
                residual_stats_after = compute_residual_stats(src_df_after, ref_df_after)

                Nobs_matched_r_lim_after = has_m_after.sum()
                Nobs_matched_r_after = dist_mask_after.sum()

                # Update solution metrics
                rms_after = residual_stats_after['rms']
                completeness_after = Nobs_matched_r_after / Nobs_matched_r_lim_after

                solution.update({
                    'Nobs_matched': Nobs_matched_r_after,
                    'Nobs_lim': Nobs_matched_r_lim_after,
                    'rms_dist': rms_after,
                    'completeness': completeness_after,
                    'wcsprm': copy(refined_wcs),
                    # 'src_cat_match': src_df_after,
                    # 'ref_cat_match': ref_df_after,
                    'std_x': residual_stats_after['std_x'],
                    'std_y': residual_stats_after['std_y']
                })

                if fit_status:
                    solution.update({'method': 'WCS_fit'})

                if rms_after <= best_overall_rms or completeness_after >= best_overall_completeness:
                    best_overall_rms = rms_after
                    best_overall_completeness = completeness_after
                    best_overall_solution = copy(solution)

                # Check for convergence
                if completeness_after >= self.config['THRESHOLD_CONVERGENCE']:
                    solution['has_converged'] = True

                confirmed_solutions.append(solution)

            if confirmed_solutions:
                # Check if any solution converged
                converged_solutions = [sol for sol in confirmed_solutions if sol['has_converged']]

                if converged_solutions:
                    # Sort converged solutions and select the best one
                    converged_solutions.sort(key=lambda x: (x['rms_dist'], -x['Nobs_matched'], -x['score']),
                                             reverse=False)
                    best_solution = converged_solutions[0]
                    self.log.info(
                        f"{' ':<2}>> " + bc.BCOLORS.PASS + "Solution found." + bc.BCOLORS.OKGREEN)
                else:
                    # No solution fully converged
                    self.log.warning(f"{' ':<2}>> No solution fully converged.")
                    # Use best overall solution if available
                    if best_overall_solution is not None:
                        best_solution = best_overall_solution
                        self.log.info(f"{' ':<2}>> Using the best available solution.")
                    else:
                        best_solution = None

            if best_solution is not None:
                # Prepare final results
                final_wcsprm = best_solution['wcsprm']
                has_solution = True
                result_dict = {"radius_px": self.match_radius_px,
                               "matches": best_solution['Nobs_matched'],
                               "rms": best_solution['rms_dist'],
                               'std_x': best_solution['std_x'],
                               'std_y': best_solution['std_y']}
                self.print_final_summary(best_solution, Nobs_total, Nref_total)
            else:
                # No solutions found
                has_solution = False
                final_wcsprm = None
                result_dict = result_dict_fail
                self.log.warning("  No solution found after testing all rotation matrices.")
        else:
            # No solutions found
            has_solution = False
            final_wcsprm = None
            result_dict = result_dict_fail
            self.log.warning("  No solution found after testing all rotation matrices.")

        # Return the final results
        return has_solution, final_wcsprm, result_dict

    def print_final_summary(self, solution, Nobs_total, Nref_total):
        """

        Parameters
        ----------
        solution
        Nobs_total
        Nref_total

        Returns
        -------

        """
        Nobs = solution['Nobs_matched']
        Nref = solution['Nobs_lim']
        completeness = solution['completeness']
        rms_dist = solution['rms_dist']

        self.log.info(f"{' ':<2}{'-' * 50}")
        self.log.info(f"{' ':<2}WCS find summary:")
        self.log.info(f"{' ':<2}{'-' * 50}")
        self.log.info(f"{' ':<2}{'Matched (detected sources)':<30} {f'{Nobs}/{Nobs_total}'} ")
        self.log.info(f"{' ':<2}{'Matched (reference sources)':<30} {f'{Nref}/{Nref_total}'} ")
        self.log.info(f"{' ':<2}{'Matched (src/ref)':<30} {f'{Nobs}/{Nref}'} ")
        self.log.info(f"{' ':<2}{'Completeness':<30} {completeness:.3f} ({completeness * 100:.1f}%)")
        self.log.info(f"{' ':<2}{'RMS distance':<30} {rms_dist:.2f} px")
        self.log.info(f"{' ':<2}{'Positional error x':<30} {solution['std_x']:.2f} px")
        self.log.info(f"{' ':<2}{'Positional error y':<30} {solution['std_y']:.2f} px")

    def apply_wcs_and_filter_sources(self, current_wcsprm, source_df, match_radius):
        """Apply the given WCS to the full reference catalog, filter it, then match the observed sources
        to the filtered reference catalog.

        Parameters:
            - self: assumes methods from the current class instance are accessible.
            - current_wcsprm: the WCS object to be applied
            - source_df: DataFrame of observed sources with an 'obs_idx' column
            - ref_cat_full: full reference catalog DataFrame
            - match_radius: maximum matching radius in pixels

        Returns:
            - matched_src_df: DataFrame of observed sources merged with matching results
            - has_m: boolean Series indicating which sources have a match
            - dist_mask: boolean mask for sources with distance < self.match_radius_px
            - ref_cat_filtered: filtered reference catalog after WCS application

        Parameters
        ----------
        current_wcsprm
        source_df
        match_radius

        Returns
        -------

        """

        # Filter reference catalog using WCS
        ref_cat_filtered = self.get_filtered_ref_cat(current_wcsprm=current_wcsprm, origin=0)

        # Match observed to reference
        matches_df = get_source_matches(obs_df=source_df, ref_df=ref_cat_filtered,
                                        match_radius=match_radius)

        # Merge matches with observed sources
        matched_src_df = pd.merge(source_df, matches_df, on='obs_idx', how='left')

        # Create masks
        has_m = matched_src_df["has_match"]
        dist_mask = pd.Series(False, index=matched_src_df.index)
        dist_mask.loc[has_m.index[has_m]] = matched_src_df.loc[has_m, 'closest_distance'] < self.match_radius_px

        return matched_src_df, ref_cat_filtered, has_m, dist_mask

    @staticmethod
    def extract_sources_subset(matched_src_df, ref_cat_filtered, has_m, dist_mask=None):
        """Extract subsets of matched observed sources and their corresponding reference sources.

        Parameters:
            - matched_src_df: DataFrame of matched observed sources (with 'obs_idx', 'chosen_ref_idx', etc.)
            - has_m: boolean Series indicating which observed sources have at least one match
            - ref_cat_filtered: DataFrame of filtered reference sources after WCS application
            - dist_mask: (optional) boolean Series indicating which matched sources pass a distance criterion.
                         If None, the distance criterion is not applied.

        Returns:
            - src_subset_df: DataFrame of observed sources that meet the match (and optional distance) criteria
            - ref_subset_df: DataFrame of reference sources corresponding to 'chosen_ref_idx' in 'src_subset_df'
        """

        # Determine the final selection mask
        if dist_mask is not None:
            selection_mask = has_m & dist_mask
        else:
            selection_mask = has_m

        # Extract the subsets
        src_subset_df = matched_src_df[selection_mask]

        # Extract the chosen reference indices for these sources
        chosen_indices = src_subset_df['chosen_ref_idx'].values.astype(int)

        # Retrieve the corresponding reference sources
        ref_subset_df = ref_cat_filtered.iloc[chosen_indices]

        return src_subset_df, ref_subset_df

    @staticmethod
    def refine_wcs_fit(obs_df, ref_df, wcsprm_in, match_radius):
        """

        Parameters
        ----------
        obs_df
        ref_df
        wcsprm_in
        match_radius

        Returns
        -------

        """

        obs_xy = obs_df[["xcentroid", "ycentroid"]].values  # Source positions
        ref_ra_dec = ref_df[["RA", "DEC"]].values  # Reference positions
        weights = np.array([0.5 if nm > 1 else 1. for nm in obs_df.num_matches])  # Weights according to N matches

        # Initial guesses for parameters
        initial_params = [0., 1., 0., 0.]
        lower_bounds = [-np.pi / 2., 0.9, -match_radius, -match_radius]
        upper_bounds = [np.pi / 2., 1.1, match_radius, match_radius]

        # Perform optimization
        result = least_squares(
            residuals_wcs,
            initial_params,
            args=(obs_xy, ref_ra_dec, wcsprm_in, weights),
            method='trf',  # or dogbox
            loss='huber',  # or cauchy
            f_scale=1.0,
            bounds=(lower_bounds, upper_bounds),
            verbose=0)

        # Apply the optimized transformation to the WCS
        if result.success:
            optimized_params = result.x
            rotation_opt, scaling_opt, x_shift_opt, y_shift_opt = optimized_params

            # Update WCS
            refined_wcs = deepcopy(wcsprm_in)
            rot_matrix_opt = np.array([[np.cos(rotation_opt), -np.sin(rotation_opt)],
                                       [np.sin(rotation_opt), np.cos(rotation_opt)]])
            refined_wcs = rotate(refined_wcs, rot_matrix_opt)
            refined_wcs = scale(refined_wcs, scaling_opt)
            refined_wcs = shift_wcs_central_pixel(refined_wcs, x_shift_opt, y_shift_opt)

            return True, refined_wcs
        else:
            return False, wcsprm_in

    def get_filtered_ref_cat(self, current_wcsprm, origin=0):
        """

        Parameters
        ----------
        current_wcsprm
        origin

        Returns
        -------

        """
        # Update the reference catalog positions
        ref_cat_full_updated = update_catalog_positions(self.ref_cat_full,
                                                        current_wcsprm, origin=origin)

        # Filter the reference catalog within bounds
        ref_cat_filtered = filter_df_within_bounds(ref_cat_full_updated,
                                                   self.current_cutoutCenter,
                                                   self.search_area_shape,
                                                   self.search_area_additional_width)
        return ref_cat_filtered

    def process_sample(self):
        """

        Returns
        -------

        """

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
        best_match_radius = self.match_radius_px
        best_rms = np.inf
        best_std_xy = (np.inf, np.inf)

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

                # apply rotation and scale
                rot_mat = rotation_to_matrix(result_rot_rad)
                wcsprm_new = rotate(copy(self.current_wcsprm), rot_mat)
                wcsprm_new = scale(wcsprm_new, result_scaling)

                # find offset between datasets
                shift_signal, x_shift, y_shift = self.find_offset(self.ref_cat,
                                                                  wcsprm_new,
                                                                  offset_binwidth=1)
                # print(shift_signal, x_shift, y_shift)

                # apply the found shift
                wcsprm_shifted = shift_wcs_central_pixel(wcsprm_new, x_shift, y_shift)

                del wcsprm_new

                # find matches
                ref_cat = self.get_filtered_ref_cat(wcsprm_shifted, origin=0)
                matches = find_matches(obs=self.source_cat,
                                       cat=ref_cat,
                                       wcsprm_in=wcsprm_shifted,
                                       threshold=self.match_radius_px)

                # unpack result
                source_cat_matched, ref_cat_matched, obs_xy, cat_xy, _, current_score, _ = matches
                Nobs_matched = len(obs_xy)

                # update best result
                if best_Nobs_matched <= Nobs_matched or best_score <= current_score[0]:
                    best_Nobs_matched = Nobs_matched
                    best_score = current_score[0]
                    best_scaling = result_scaling
                    best_rotation = result_rot_rad
                    best_wcsprm = copy(wcsprm_shifted)
                    best_source_cat_match = source_cat_matched
                    best_ref_cat_match = ref_cat_matched
                    best_match_radius = self.match_radius_px
                    best_rms = current_score[1]
                    best_std_xy = (current_score[2], current_score[3])

                # if self._debug:
                #     plot_catalog_positions(self.source_cat,
                #                            self.ref_cat,
                #                            wcsprm_shifted, self.image)
                #     plt.show()
                del wcsprm_shifted

        return (best_Nobs_matched, best_score, best_wcsprm,
                [best_match_radius, best_rms, best_std_xy, best_scaling, best_rotation],
                best_source_cat_match, best_ref_cat_match)

    def compute_cross_corr(self):
        """

        Returns
        -------

        """

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

        bc.clean_up(H_obs, H_cat, ff_obs, ff_cat, bins, ranges)

        self.cross_corr = cross_corr

    def get_scale_rotation(self, init_wcsprm, init_rot, match_radius):
        """

        Parameters
        ----------
        init_wcsprm
        init_rot
        match_radius

        Returns
        -------

        """

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
            bc.clean_up(log_dist_obs, angles_obs, log_dist_cat, angles_cat,
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
        bc.clean_up(log_dist_obs, angles_obs, log_dist_cat, angles_cat,
                    cross_corr, binwidth_dist, binwidth_ang)

        return (best_Nobs_matched, best_completeness, best_score, best_wcsprm, [best_scaling, best_rotation],
                best_source_cat_match, best_ref_cat_match)

    def find_offset(self, catalog, wcsprm_in, offset_binwidth=1):
        """Find offset

        Parameters
        ----------
        catalog
        wcsprm_in
        offset_binwidth : int, optional
            test

        Returns
        -------

        """

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
        highest_peak = np.argwhere(H == np.max(H))[0]

        # Check if the highest peak is out of bounds in x_edges or y_edges
        if highest_peak[0] >= len(x_edges) - 1 or highest_peak[1] >= len(y_edges) - 1:
            # Mask the highest peak to discard it
            H[highest_peak[0], highest_peak[1]] = -np.inf

            # Find the next highest peak
            next_peak = np.argwhere(H == np.max(H))[0]

            # Use the next highest peak
            peak = next_peak
        else:
            # Use the highest peak
            peak = highest_peak

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
        """Calculate cross correlation spectrum

        Parameters
        ----------
        log_dist_obs
        angles_obs
        log_dist_cat
        angles_cat

        Returns
        -------

        """

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

        bc.clean_up(H_obs, H_cat, ff_obs, ff_cat, log_dist_obs, angles_obs,
                    log_dist_cat, angles_cat, angles_obs, angles_cat, bins, ranges)

        return cross_corr, binwidth_dist, binwidth_ang

    def refine_transformation(self, ref_cat, wcsprm_in, match_radius, compare_threshold):
        """
        Final improvement of registration. This method requires that the wcs is already accurate to a few pixels.
        """

        wcsprm_copy = copy(wcsprm_in)

        # find matches
        _, _, init_obs, _, _, init_score, _ = find_matches(self.source_cat, ref_cat,
                                                           wcsprm_in,
                                                           threshold=compare_threshold)
        print(compare_threshold, len(init_obs), init_score)

        # lis = [0.1, 0.25, 0.5, 1, 2, 3, 4]
        lis = [1]
        for i in lis:

            threshold = i * match_radius

            # find matches
            _, _, obs_xy, cat_xy, _, s1, _ = \
                find_matches(self.source_cat, ref_cat, wcsprm_in, threshold=threshold)

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
                return True, wcsprm_shifted, match_radius

            del log_dist_obs, log_dist_cat, scale_offset, angle_offset

        return False, wcsprm_copy, match_radius


def compute_residual_stats(obs_df, ref_df):
    """
    Compute residuals and basic statistics between observed and reference positions.

    Parameters:
    - obs_xy: Nx2 array of observed source positions (e.g., in pixels)
    - ref_xy: Nx2 array of reference source positions (e.g., in pixels)

    Returns:
    - stats: dict containing:
        'residuals': Nx2 array of residual vectors (obs_xy - ref_xy)
        'residual_x': Nx array of X-direction residuals
        'residual_y': Nx array of Y-direction residuals
        'positional_uncertainties': Nx array of per-source positional uncertainties (2D residual norm)
        'rms': scalar, RMS of positional uncertainties
        'mean_uncertainty': scalar, mean positional uncertainty
        'median_uncertainty': scalar, median positional uncertainty
        'std_x': scalar, standard deviation of residuals in X
        'std_y': scalar, standard deviation of residuals in Y
    """

    obs_xy = obs_df[["xcentroid", "ycentroid"]].values
    ref_xy = ref_df[["xcentroid", "ycentroid"]].values

    # Compute residuals
    residuals = obs_xy - ref_xy
    residual_x = residuals[:, 0]
    residual_y = residuals[:, 1]

    # Per-source positional uncertainty (distance of residual vector)
    positional_uncertainties = np.sqrt(residual_x ** 2 + residual_y ** 2)

    # Compute RMS of positional uncertainties
    rms = np.sqrt(np.mean(positional_uncertainties ** 2))

    # Compute other statistics
    mean_error = np.mean(positional_uncertainties)
    median_error = np.median(positional_uncertainties)
    std_x = np.std(residual_x)
    std_y = np.std(residual_y)

    # Pack results into a dictionary
    stats = {
        'rms': rms,
        'score': len(obs_xy) / (1 + rms),
        'mean_error': mean_error,
        'median_error': median_error,
        'std_x': std_x,
        'std_y': std_y

    }

    return stats


def get_source_matches(obs_df, ref_df, match_radius):
    """
    Match observed sources to reference sources by finding all candidates within the matching radius.
    For each observed source:
    - Find all reference sources within 'match_radius'.
    - If multiple matches exist, select the closest one.
    - Record information about whether multiple matches were found.

    Parameters:
    - obs_xy: Nx2 array of observed source positions (pixels)
    - ref_xy: Mx2 array of reference source positions (pixels)
    - match_radius: Matching radius in pixels

    Returns:
    - matches: list of dictionaries, one per observed source, containing:
        {
            'obs_idx': index of the observed source,
            'matched_ref_indices': array of reference source indices within the radius,
            'num_matches': number of matched reference sources,
            'chosen_ref_idx': the final chosen reference source index (closest match),
            'closest_distance': the distance to the chosen match,
            'has_multiple_matches': boolean indicating if multiple matches were found
        }
    """

    obs_xy = obs_df[["xcentroid", "ycentroid"]].values
    ref_xy = ref_df[["xcentroid", "ycentroid"]].values

    # Build KD-tree for reference sources
    ref_tree = cKDTree(ref_xy)

    # Query all reference sources within match_radius for each observed source
    obs_all_matches = ref_tree.query_ball_point(obs_xy, r=match_radius, p=1)

    matches = []
    for obs_idx, ref_indices in enumerate(obs_all_matches):
        num_matches = len(ref_indices)
        match_info = {
            'obs_idx': obs_idx,
            'matched_ref_idx': np.array(ref_indices, dtype=int),
            'num_matches': num_matches,
            'chosen_ref_idx': None,
            'closest_distance': None,
            'has_match': num_matches > 0,
            'has_single_match': num_matches == 1,
            'has_multiple_matches': num_matches > 1
        }

        if num_matches == 0:
            # No matches found for this observed source
            matches.append(match_info)
            continue

        # If there are matches (one or multiple), find the closest
        obs_point = obs_xy[obs_idx]
        candidate_ref_points = ref_xy[ref_indices]
        distances = np.linalg.norm(candidate_ref_points - obs_point, axis=1)

        # Find the closest match among candidates
        closest_candidate_idx = np.argmin(distances)
        closest_ref_idx = ref_indices[closest_candidate_idx]
        closest_distance = distances[closest_candidate_idx]

        match_info['chosen_ref_idx'] = int(closest_ref_idx)
        match_info['closest_distance'] = float(closest_distance)

        matches.append(match_info)

    # Cleanup
    bc.clean_up(obs_df, ref_df, obs_xy, ref_xy, ref_tree)

    return pd.DataFrame(matches)


def match_sources_bidirectional(obs_xy, ref_xy, match_radius):
    """

    Parameters
    ----------
    obs_xy
    ref_xy
    match_radius

    Returns
    -------

    """
    # Build KD-trees
    obs_tree = cKDTree(obs_xy)
    ref_tree = cKDTree(ref_xy)

    # Observed to Reference
    dist_obs_to_ref, idx_obs_to_ref = ref_tree.query(obs_xy, distance_upper_bound=match_radius)
    # Reference to Observed
    dist_ref_to_obs, idx_ref_to_obs = obs_tree.query(ref_xy, distance_upper_bound=match_radius)

    # Prepare arrays
    matches = []
    for obs_idx, (ref_idx, dist) in enumerate(zip(idx_obs_to_ref, dist_obs_to_ref)):
        if dist != np.inf:
            # Check for reciprocal match
            if idx_ref_to_obs[ref_idx] == obs_idx:
                matches.append((obs_idx, ref_idx))

    matched_obs_indices = np.array([m[0] for m in matches])
    matched_ref_indices = np.array([m[1] for m in matches])

    return matched_obs_indices, matched_ref_indices


def residuals_wcs(params, obs_xy, ref_ra_dec, wcsprm_in, weights):
    """
    Compute residuals between transformed reference points and observed points.

    Parameters:
    - params: array_like
        Parameters to optimize [rotation (radians), scaling, x_shift, y_shift]
    - obs_xy: array_like
        Observed source positions in pixel coordinates (Nx2 array)
    - ref_ra_dec: array_like
        Reference catalog positions in world coordinates (RA, Dec in degrees)
    - wcsprm_in: astropy.wcs.WCS
        Initial WCS object

    Returns:
    - residuals: array_like
        Residuals between transformed reference points and observed points (flattened array)
    """

    rotation, scaling, x_shift, y_shift = params

    # Make a copy of the WCS to modify
    wcs_n = deepcopy(wcsprm_in)

    # Apply rotation
    rot_matrix = np.array([[np.cos(rotation), -np.sin(rotation)],
                           [np.sin(rotation), np.cos(rotation)]])
    wcs_n = rotate(wcs_n, rot_matrix)

    # Apply scaling
    wcs_n = scale(wcs_n, scaling)

    # Apply shift
    wcs_n = shift_wcs_central_pixel(wcs_n, x_shift, y_shift)

    # Transform reference RA, Dec to pixel coordinates using the updated WCS
    wcs_r = WCS(wcs_n.to_header())
    ref_pixel_coords = wcs_r.all_world2pix(ref_ra_dec, 0)

    # Compute residuals between transformed reference points and observed points
    residuals_r = obs_xy - ref_pixel_coords  # Nx2 array

    # Apply weights
    weighted_residuals = (residuals_r * weights[:, np.newaxis]).ravel()
    return weighted_residuals


def get_matched_reference_catalog(src_cat, ref_cat, src_cat_matched, ref_cat_matched, radius):
    """

    Parameters
    ----------
    src_cat
    ref_cat
    src_cat_matched
    ref_cat_matched
    radius

    Returns
    -------

    """

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

    bc.clean_up(inv_cross_power_spec, smoothed, detected_peaks, labeled, num_features)

    return peak_coords


def analyse_cross_corr(cross_corr, binwidth_dist, binwidth_ang):
    """

    Parameters
    ----------
    cross_corr
    binwidth_dist
    binwidth_ang

    Returns
    -------

    """

    # Inverse transform from frequency to spatial domain
    inv_fft = np.real(np.fft.ifft2(cross_corr))
    inv_fft_shifted = np.fft.fftshift(inv_fft)  # The zero shift is at (0,0), this moves it to the middle

    inv_fft_row_median = np.median(inv_fft_shifted, axis=1)[:, np.newaxis]

    # Subtract row median for peak detection
    inv_fft_med_sub = inv_fft_shifted - inv_fft_row_median

    # Find peaks in the inverse FT
    peak_coords = find_peaks_2d(inv_fft_med_sub,
                                threshold_multiplier=5.,
                                gauss_sigma=1,
                                apply_gaussian_filter=True, size=5)
    # print('peak_coords', peak_coords)

    if not list(peak_coords):
        bc.clean_up(cross_corr, inv_fft, inv_fft_shifted, inv_fft_med_sub)
        return None

    peak_results = np.zeros((len(peak_coords), 3))
    for i, peak in enumerate(peak_coords):
        x_shift_bins, y_shift_bins, signal = get_shift_from_peak(inv_fft_shifted, peak)

        x_shift = x_shift_bins * binwidth_dist
        y_shift = y_shift_bins * binwidth_ang

        # Extract scale and rotation from shift
        scaling = np.e ** (-x_shift)
        rotation = y_shift

        # print(signal, scaling, rotation)
        peak_results[i, :] = np.array([signal, scaling, rotation])

    # print(peak_results)
    # Sort by signal strength
    peak_results_sorted = peak_results[peak_results[:, 0].argsort()[::-1]][0]
    # print(peak_results_sorted)
    # plt.figure()
    # plt.imshow(inv_fft_med_sub)
    # plt.show()

    # Cleanup
    bc.clean_up(cross_corr, inv_fft, inv_fft_shifted, inv_fft_med_sub,
                inv_fft_row_median, peak_results)

    return peak_results_sorted


def get_shift_from_peak(inv_cross_power_spec, peak):
    """

    Parameters
    ----------
    inv_cross_power_spec
    peak

    Returns
    -------

    """
    n_1 = 1
    n_2 = 2
    n = n_1 + n_2

    # Sum up the signal in a fixed aperture 1 pixel in each direction around the peak,
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

    # Find the sub pixel shift of the true peak
    # peak_x_subpixel, peak_y_subpixel = subpixel_peak_position(around_peak)
    # print(peak_x_subpixel, peak_y_subpixel)

    # Calculate final shift
    x_shift = peak[0] - middle_x + peak_x_subpixel
    y_shift = peak[1] - middle_y + peak_y_subpixel

    del inv_cross_power_spec

    return x_shift, y_shift, signal


def get_rms(ref_cat, obs, wcsprm_in, radius_px, N_obs_matched):
    """

    Parameters
    ----------
    ref_cat
    obs
    wcsprm_in
    radius_px
    N_obs_matched

    Returns
    -------

    """
    wcsprm = copy(wcsprm_in)
    N_obs = obs.shape[0]

    # Get the radii list
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
    """Plot image with source and reference catalog positions marked"""

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


def find_matches(obs, cat, wcsprm_in=None, threshold=10., sort=True):
    """Match observation with reference catalog using minimum distance."""

    # Check if the input has data
    cat_has_data = cat[["xcentroid", "ycentroid"]].any(axis=0).any()
    obs_has_data = obs[["xcentroid", "ycentroid"]].any(axis=0).any()

    if not cat_has_data or not obs_has_data:
        return None, None, None, None, None, None, False

    # Convert obs to numpy
    obs_xy = obs[["xcentroid", "ycentroid"]].to_numpy()

    # Set up the catalog data; use RA, Dec if used with wcsprm
    if wcsprm_in is not None:
        cat_xy = wcsprm_in.s2p(cat[["RA", "DEC"]], 0)
        cat_xy = cat_xy['pixcrd']
    else:
        cat_xy = cat[["xcentroid", "ycentroid"]].to_numpy()

    # Calculate the distances
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

    if sort:
        cat_matched = cat_matched.sort_values(by='mag', ascending=True)
    cat_matched.reset_index(drop=True, inplace=True)

    columns_to_add = ['xcentroid', 'ycentroid', 'fwhm', 'include_fwhm']
    cat_matched = cat_matched.assign(**obs_matched[columns_to_add].to_dict(orient='series'))
    del obs, cat

    obs_xy = obs_xy[obs_idx, :]
    cat_xy = cat_xy[cat_idx, :]

    if len(min_dist_xy) == 0:  # meaning the list is empty
        score = 0
        rms = np.inf
        std_x = np.inf
        std_y = np.inf

    else:
        residuals = obs_xy - cat_xy

        # Residuals in x and y directions
        delta_x = residuals[:, 0]
        delta_y = residuals[:, 1]

        std_x = np.std(delta_x)
        std_y = np.std(delta_y)

        # Compute RMS
        rms = np.sqrt(np.mean(delta_x ** 2 + delta_y ** 2))
        score = len(obs_xy) / (rms + 1)

    return obs_matched, cat_matched, obs_xy, cat_xy, min_dist_xy, (score, rms, std_x, std_y), True


def cross_corr_to_fourier_space(a):
    """Transform 2D array into fourier space. Use padding and normalization."""

    aa = (a - np.nanmean(a)) / np.nanstd(a)

    # Wraps around so half the size should be fine, pads 2D array with zeros
    aaa = np.pad(aa, (5, 5), 'constant')
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
    yield float("%g" % x)  # Yield always first value
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

    bc.clean_up(data_x, data_y)

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

    # Get all pairs: vector differences
    vec_x = data_x - data_x.T
    vec_y = data_y - data_y.T
    vec_x = vec_x[np.where(~np.eye(vec_x.shape[0], dtype=bool))]
    vec_y = vec_y[np.where(~np.eye(vec_y.shape[0], dtype=bool))]

    # Get the angle with x-axis.
    angles = np.arctan2(vec_x, vec_y)

    # Make sure angles are between 0 and 2 Pi
    angles = angles % (2. * np.pi)

    # Shift to -pi to pi
    angles[np.where(angles > np.pi)] = -1 * (2. * np.pi - angles[np.where(angles > np.pi)])

    bc.clean_up(data_x, data_y, vec_x, vec_y)

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
    wcsprm, scales
        The updated WCS object
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
    :param origin: Image origin. Either ´0´ or ´1´. Default is 0.
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
    """Create a 2D histogram from log distance and angle data.

    """

    vals = [log_distance, angle]
    H = fhist.histogram2d(*vals, bins=bins, range=ranges).astype(dtype=np.complex128)

    del log_distance, angle, bins, ranges
    return H


def apply_fft_and_normalize(H):
    """Apply FFT to the histogram and normalize the result.

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
    """

    Parameters
    ----------
    df
    center_coords
    shape
    additional_width

    Returns
    -------

    """
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

    # def find_wcs_work(self, input_wcsprm, match_radius_px=1, image=None):
    #     """Run the plate-solve algorithm to find the best WCS transformation"""
    #
    #     def reset_best_params():
    #         nonlocal best_Nobs, best_score, best_rms_dist, best_completeness, best_wcsprm, best_ref_cat, best_src_cat
    #         best_Nobs = -1
    #         best_score = -np.inf
    #         best_rms_dist = np.inf
    #         best_completeness = 0
    #         best_wcsprm = None
    #         best_ref_cat = None
    #         best_src_cat = None
    #
    #     self.log.info("> Run WCS calibration")
    #
    #     self.image = image
    #     self.match_radius_px = match_radius_px
    #     match_radius_limit_px = 3. * match_radius_px
    #
    #     Nobs_total = len(self.source_cat_full)  # total number of detected sources
    #     Nref_total = len(self.ref_cat_full)  # total number of reference stars
    #
    #     rotations = self.rotations
    #     center_coords = [(image.shape[1] // 2, image.shape[0] // 2)]  #,
    #     # (0, 0),
    #     # (image.shape[1], 0),
    #     # (0, image.shape[0]),
    #     # (image.shape[1], image.shape[0])][:1]
    #
    #     search_area_shape = image.shape
    #     search_area_additional_width = (1, 1)
    #
    #     result_dict = {"radius_px": None, "matches": None, "rms": None}
    #     src_cat_match = ref_cat_match = None
    #
    #     initial_wcsprm = input_wcsprm.wcs
    #     final_wcsprm = None
    #
    #     init_dist_bin_size = self.dist_bin_size
    #     init_ang_bin_size = self.ang_bin_size
    #
    #     center_idx = 0
    #     rot_matrix_idx = 0
    #     ref_sample_idx = 0
    #
    #     solve_attempts_count = 0
    #     high_completeness_count = 0
    #     possible_solution_count = 0
    #     no_solution_count = 0
    #
    #     max_Nobs_initial = 100
    #     num_ref_samples = 5
    #     max_rotations = len(rotations)
    #     max_centers = len(center_coords)
    #     max_wcs_solve_attempts = max_rotations * max_centers
    #     max_no_solution_count = 3
    #     max_wcs_eval_iter = self.config['MAX_WCS_FUNC_ITER']
    #
    #     use_initial_wcs_estimate = True
    #     update_src_cat = False
    #     update_ref_cat = False
    #     has_large_src_cat = False if Nobs_total <= max_Nobs_initial else True
    #     has_possible_solution = False
    #     has_solution = False
    #
    #     best_Nobs = -1
    #     best_score = -np.inf
    #     best_rms_dist = np.inf
    #     best_completeness = 0
    #     best_wcsprm = None
    #     best_ref_cat = None
    #     best_src_cat = None
    #
    #     self.source_cat = self.source_cat_full.head(max_Nobs_initial)
    #     self.get_source_cat_variables()
    #
    #     self.log.info(f'  Detected sources (total): {Nobs_total}')
    #     self.log.info(f'  Reference sources (total): {Nref_total}')
    #     self.log.info(f'  Match radius: {self.match_radius_px:.2f} px')
    #
    #     while True:
    #
    #         # failsafe
    #         if solve_attempts_count >= max_wcs_solve_attempts:
    #             has_solution = False
    #             break
    #
    #         # set rotation matrix
    #         current_rotMatrix = self.rotations[rot_matrix_idx]
    #
    #         # apply rotation matrix to WCS
    #         if use_initial_wcs_estimate:
    #             self.current_wcsprm = rotate(copy(initial_wcsprm), current_rotMatrix)
    #
    #         # apply the WCS to the full reference catalog
    #         ref_cat_full_updated = update_catalog_positions(self.ref_cat_full,
    #                                                         self.current_wcsprm,
    #                                                         origin=0)
    #
    #         # set the cutout center to current index if no possible solution was found,
    #         # else use the image center since the wcs was adjusted
    #         current_cutoutCenter = center_coords[center_idx] if not has_possible_solution else center_coords[0]
    #
    #         # filter sources according to the current cutout center and include only sources
    #         # within the given cutout shape plus an additional width
    #         ref_cat_filtered = filter_df_within_bounds(ref_cat_full_updated,
    #                                                    current_cutoutCenter,
    #                                                    search_area_shape,
    #                                                    search_area_additional_width)
    #
    #         # if the source catalog has more than 100 objects and a possible solution was found, use the full source
    #         # catalog
    #         if update_src_cat:
    #             self.source_cat = self.source_cat_full
    #             self.get_source_cat_variables()
    #
    #         # get current object count
    #         Nobs_current = len(self.source_cat)
    #         Nref_current = len(ref_cat_filtered)
    #
    #         # create a list with the number of samples to draw from the reference catalog, evenly spaced in log-space
    #         ref_cat_sample_number = np.unique(np.logspace(np.log10(Nobs_current),
    #                                                       np.log10(Nref_current),
    #                                                       num=num_ref_samples,
    #                                                       base=10, endpoint=False, dtype=int))
    #         # print('update_ref_cat', update_ref_cat)
    #         # print('ref_sample_idx', ref_sample_idx)
    #         if update_ref_cat:
    #
    #             # update the reference catalog by including those objects from the reference catalog
    #             # that fall within n times the current match radius
    #             self.ref_cat = get_matched_reference_catalog(self.source_cat, ref_cat_filtered,
    #                                                          src_cat_match, ref_cat_match,
    #                                                          match_radius_limit_px)
    #
    #             #
    #             self.source_cat = match_catalogs(self.ref_cat,
    #                                              self.source_cat,
    #                                              radius=match_radius_limit_px, N=1)
    #         else:
    #             # select the N brightest reference sources
    #             self.ref_cat = ref_cat_filtered.head(ref_cat_sample_number[ref_sample_idx])
    #         #
    #         if no_solution_count == 0 and not has_possible_solution:
    #             self.log.info(
    #                 f"  > Testing configuration [{solve_attempts_count + 1}/{max_wcs_solve_attempts}]:")
    #             self.log.info(
    #                 f"{' ':<4}Rotation matrix [{rot_matrix_idx + 1}/{len(self.rotations)}]: "
    #                 f"[{', '.join(map(str, current_rotMatrix))}]")
    #             self.log.info(
    #                 f"{' ':<4}Cutout center position (x, y): "
    #                 f"({', '.join(map(str, current_cutoutCenter))})")
    #
    #             self.log.info(f"{' ':<4}Reference sources (within cutout boundaries): {Nref_current}")
    #             if has_large_src_cat:
    #                 self.log.info(f"{' ':<4}Detected sources (brightest): {Nobs_current} "
    #                               f"out of {Nobs_total}")
    #             else:
    #                 self.log.info(f"{' ':<4}Detected sources (all): {Nobs_current}")
    #
    #             self.log.info(f"{' ':<4}> Try [{no_solution_count + 1}/{max_no_solution_count}] "
    #                           f"Number of reference sources: {len(self.ref_cat)}/{Nref_current}")
    #
    #         elif no_solution_count != 0 and not has_possible_solution:
    #             self.log.info(f"{' ':<4}> Try [{no_solution_count + 1}/{max_no_solution_count}] "
    #                           f"Number of reference sources: {len(self.ref_cat)}/{Nref_current}")
    #
    #         # plot_catalog_positions(self.source_cat,
    #         #                        self.ref_cat,
    #         #                        None, self.image)
    #         # process the current source catalog and the selected reference sources
    #         sample_result = self.process_sample()
    #         print(sample_result[0], sample_result[1], 1. / (sample_result[0] * sample_result[1]))
    #
    #         # plot_catalog_positions(self.source_cat,
    #         #                        self.ref_cat,
    #         #                        sample_result[2], self.image)
    #         # plt.show()
    #         sample_result = None
    #         # check result
    #         if sample_result is None or sample_result[0] <= self.config['MIN_SOURCE_NO_CONVERGENCE']:
    #             no_solution_count += 1
    #
    #             if no_solution_count >= max_no_solution_count:
    #                 self.log.warning(f"{' ':<6}> No possible solution found after {max_no_solution_count} trails. "
    #                                  f"Skipping to next configuration.")
    #                 ref_sample_idx = num_ref_samples - 1  # Force iteration
    #                 no_solution_count = 0  # Reset counter for the next iteration
    #             else:
    #                 if sample_result is None:
    #                     self.log.warning(f"{' ':<6}> No solution/matches found. "
    #                                      f"Trying again with increased number of reference sources...")
    #                 else:
    #                     self.log.warning(f"{' ':<6}> Configuration resulted in less than "
    #                                      f"{self.config['MIN_SOURCE_NO_CONVERGENCE']} matches. "
    #                                      f"Trying again with increased number of reference sources...")
    #
    #             if ref_sample_idx == num_ref_samples - 1:
    #                 rot_matrix_idx += 1
    #                 ref_sample_idx = 0
    #                 solve_attempts_count += 1
    #             else:
    #                 # Move to the next sample within the same iteration
    #                 ref_sample_idx += 1
    #
    #             # reset switches
    #             self.ang_bin_size = init_ang_bin_size
    #             self.dist_bin_size = init_dist_bin_size
    #             src_cat_match = ref_cat_match = None
    #             use_initial_wcs_estimate = True
    #             update_ref_cat = False
    #             update_src_cat = False
    #             has_possible_solution = False
    #
    #             # reset best result
    #             reset_best_params()
    #             continue
    #         else:
    #             has_possible_solution = True
    #
    #         # update possible solution count
    #         possible_solution_count += 1
    #
    #         # unpack results
    #         Nobs_matched, score, wcsprm_out, result_info, src_cat_match, ref_cat_match = sample_result
    #
    #         # set variables
    #         Nref_matched = len(self.ref_cat)
    #         completeness = Nobs_matched / Nref_matched
    #         rms_dist = 1. / (score * Nobs_matched)
    #         self.current_wcsprm = copy(wcsprm_out)
    #
    #         # check completeness
    #         if completeness >= self.config['THRESHOLD_CONVERGENCE']:
    #             high_completeness_count += 1
    #
    #         # update best result
    #         if best_Nobs <= Nobs_matched:
    #             if best_score < score or completeness == 1.:
    #                 best_Nobs = Nobs_matched
    #                 best_score = score
    #                 best_completeness = completeness
    #                 best_wcsprm = copy(wcsprm_out)
    #                 best_ref_cat = self.ref_cat
    #                 best_src_cat = self.source_cat
    #
    #         print(possible_solution_count, Nobs_matched, score, completeness, high_completeness_count, rms_dist)
    #         # plot_catalog_positions(self.source_cat,
    #         #                        self.ref_cat,
    #         #                        self.current_wcsprm, self.image)
    #         # plt.show()
    #
    #         # if there are more than the N brightest sources, increase the number of sources and re-run
    #         if has_large_src_cat and not update_src_cat:
    #             update_src_cat = True
    #             update_ref_cat = True
    #             use_initial_wcs_estimate = False
    #
    #             possible_solution_count = 0
    #             high_completeness_count = 0
    #             self.log.info(f"{' ':<6}> Possible solution found. Confirming...")
    #             self.log.info(f"{' ':<6}Increase source number: "
    #                           f"{Nobs_current} (brightest) --> {len(self.source_cat_full)} (all)")
    #             continue
    #
    #         elif not has_large_src_cat and possible_solution_count == 1:
    #             self.log.info(f"{' ':<6}> Possible solution found. Confirming...")
    #
    #         if completeness == 1.0 or high_completeness_count >= 3 or possible_solution_count >= max_wcs_eval_iter:
    #
    #             has_solution = True
    #             final_wcsprm = best_wcsprm
    #
    #             # get rms estimate
    #             result_dict = get_rms(best_ref_cat,
    #                                   best_src_cat,
    #                                   best_wcsprm,
    #                                   self.match_radius_px,
    #                                   best_Nobs)
    #
    #             self.log.info(f"{' ':<6}>> " + bc.BCOLORS.PASS + "Solution found." + bc.BCOLORS.OKGREEN)
    #             self.log.info(f"{' ':<4}{'-' * 50}")
    #             self.log.info(f"{' ':<4}Summary:")
    #             self.log.info(f"{' ':<4}{'-' * 50}")
    #             self.log.info(f"{' ':<5}{'Evaluations':<30} {possible_solution_count} ")
    #             self.log.info(f"{' ':<5}{'Matched (detected sources)':<30} {f'{best_Nobs}/{Nobs_total}'} ")
    #             self.log.info(f"{' ':<5}{'Matched (reference sources)':<30} {f'{len(best_ref_cat)}/{Nref_total}'} ")
    #             self.log.info(f"{' ':<5}{'Matched (src/ref)':<30} {f'{best_Nobs}/{len(best_ref_cat)}'} ")
    #             self.log.info(
    #                 f"{' ':<5}{'Completeness':<30} {best_completeness:.3f} ({best_completeness * 100:.1f}%)")
    #             self.log.info(f"{' ':<5}{'Match radius (px)':<30} {result_dict['radius_px']:.2f} ")
    #             self.log.info(f"{' ':<5}{'RMS (arcsec)':<30} {result_dict['rms']:.2f} ")
    #             self.log.info(f"{' ':<4}{'-' * 50}")
    #
    #             break
    #         else:
    #             # update switches for next iteration
    #             update_ref_cat = True
    #             use_initial_wcs_estimate = False
    #
    #             if possible_solution_count == 1:
    #                 # increase resolution
    #                 self.dist_bin_size = 0.5
    #                 self.ang_bin_size = 0.5
    #
    #     return has_solution, final_wcsprm, result_dict
    #
    # def find_wcs_working_old(self, input_wcsprm, match_radius_px=1, image=None):
    #     """Run the plate-solve algorithm to find the best WCS transformation"""
    #
    #     def reset_best_params():
    #         nonlocal best_Nobs, best_score, best_rms_dist, best_completeness, best_wcsprm, best_ref_cat, best_src_cat
    #         best_Nobs = -1
    #         best_score = -np.inf
    #         best_rms_dist = np.inf
    #         best_completeness = 0
    #         best_wcsprm = None
    #         best_ref_cat = None
    #         best_src_cat = None
    #
    #     self.log.info("> Run WCS calibration")
    #
    #     self.image = image
    #     self.match_radius_px = match_radius_px
    #     match_radius_limit_px = 3. * match_radius_px
    #
    #     Nobs_total = len(self.source_cat_full)  # total number of detected sources
    #     Nref_total = len(self.ref_cat_full)  # total number of reference stars
    #
    #     rotations = self.rotations
    #     center_coords = [(image.shape[1] // 2, image.shape[0] // 2),
    #                      (0, 0),
    #                      (image.shape[1], 0),
    #                      (0, image.shape[0]),
    #                      (image.shape[1], image.shape[0])]
    #
    #     search_area_shape = image.shape
    #     search_area_additional_width = (1, 1)
    #
    #     result_dict = {"radius_px": None, "matches": None, "rms": None}
    #     src_cat_match = ref_cat_match = None
    #
    #     initial_wcsprm = input_wcsprm.wcs
    #     final_wcsprm = None
    #
    #     center_idx = 0
    #     rot_matrix_idx = 0
    #     ref_sample_idx = 0
    #
    #     solve_attempts_count = 0
    #     high_completeness_count = 0
    #     possible_solution_count = 0
    #     no_solution_count = 0
    #
    #     max_Nobs_initial = 100
    #     num_ref_samples = 5
    #     max_rotations = len(rotations)
    #     max_centers = len(center_coords)
    #     max_wcs_solve_attempts = max_rotations * max_centers
    #     max_no_solution_count = 3
    #     max_wcs_eval_iter = self.config['MAX_WCS_FUNC_ITER']
    #
    #     use_initial_wcs_estimate = True
    #     update_src_cat = False
    #     update_ref_cat = False
    #     has_large_src_cat = False if Nobs_total <= max_Nobs_initial else True
    #     has_possible_solution = False
    #     has_solution = False
    #
    #     best_Nobs = -1
    #     best_score = -np.inf
    #     best_rms_dist = np.inf
    #     best_completeness = 0
    #     best_wcsprm = None
    #     best_ref_cat = None
    #     best_src_cat = None
    #
    #     self.source_cat = self.source_cat_full.head(max_Nobs_initial)
    #     self.get_source_cat_variables()
    #
    #     self.log.info(f'  Detected sources (total): {Nobs_total}')
    #     self.log.info(f'  Reference sources (total): {Nref_total}')
    #     self.log.info(f'  Match radius: {self.match_radius_px:.2f} px')
    #
    #     while True:
    #
    #         # failsafe
    #         if solve_attempts_count >= max_wcs_solve_attempts:
    #             has_solution = False
    #             break
    #
    #         # set rotation matrix
    #         current_rotMatrix = self.rotations[rot_matrix_idx]
    #
    #         # apply rotation matrix to WCS
    #         if use_initial_wcs_estimate:
    #             self.current_wcsprm = rotate(copy(initial_wcsprm), current_rotMatrix)
    #
    #         # apply the WCS to the full reference catalog
    #         ref_cat_full_updated = update_catalog_positions(self.ref_cat_full,
    #                                                         self.current_wcsprm,
    #                                                         origin=0)
    #
    #         # set the cutout center to current index if no possible solution was found,
    #         # else use the image center since the wcs was adjusted
    #         current_cutoutCenter = center_coords[center_idx] if not has_possible_solution else center_coords[0]
    #
    #         # filter sources according to the current cutout center and include only sources
    #         # within the given cutout shape plus an additional width
    #         ref_cat_filtered = filter_df_within_bounds(ref_cat_full_updated,
    #                                                    current_cutoutCenter,
    #                                                    search_area_shape,
    #                                                    search_area_additional_width)
    #
    #         # if the source catalog has more than 100 objects and a possible solution was found, use the full source
    #         # catalog
    #         if update_src_cat:
    #             self.source_cat = self.source_cat_full
    #             self.get_source_cat_variables()
    #
    #         # get current object count
    #         Nobs_current = len(self.source_cat)
    #         Nref_current = len(ref_cat_filtered)
    #
    #         # create a list with the number of samples to draw from the reference catalog, evenly spaced in log-space
    #         ref_cat_sample_number = np.unique(np.logspace(np.log10(Nobs_current),
    #                                                       np.log10(Nref_current),
    #                                                       num=num_ref_samples,
    #                                                       base=10, endpoint=False, dtype=int))
    #
    #         if update_ref_cat:
    #
    #             # update the reference catalog by including those objects from the reference catalog
    #             # that fall within n times the current match radius
    #             self.ref_cat = get_matched_reference_catalog(self.source_cat, ref_cat_filtered,
    #                                                          src_cat_match, ref_cat_match,
    #                                                          match_radius_limit_px)
    #
    #             #
    #             self.source_cat = match_catalogs(self.ref_cat,
    #                                              self.source_cat,
    #                                              radius=match_radius_limit_px, N=1)
    #         else:
    #             # select the N brightest reference sources
    #             self.ref_cat = ref_cat_filtered.head(ref_cat_sample_number[ref_sample_idx])
    #
    #         if no_solution_count == 0 and not has_possible_solution:
    #             self.log.info(
    #                 f"  > Testing configuration [{solve_attempts_count + 1}/{max_wcs_solve_attempts}]:")
    #             self.log.info(
    #                 f"{' ':<4}Rotation matrix [{rot_matrix_idx + 1}/{len(self.rotations)}]: "
    #                 f"[{', '.join(map(str, current_rotMatrix))}]")
    #             self.log.info(
    #                 f"{' ':<4}Cutout center position [{center_idx + 1}/{len(center_coords)}] (x, y): "
    #                 f"({', '.join(map(str, current_cutoutCenter))})")
    #
    #             self.log.info(f"{' ':<4}Reference sources (within cutout): {Nref_current}")
    #             if has_large_src_cat:
    #                 self.log.info(f"{' ':<4}Detected sources (brightest): {Nobs_current} "
    #                               f"out of {Nobs_total}")
    #             else:
    #                 self.log.info(f"{' ':<4}Detected sources (all): {Nobs_current}")
    #
    #             self.log.info(f"{' ':<4}> Try [{no_solution_count + 1}/{max_no_solution_count}] "
    #                           f"Initial number of reference sources: {len(self.ref_cat)}/{Nref_current}")
    #
    #         elif no_solution_count != 0 and not has_possible_solution:
    #             self.log.info(f"{' ':<4}> Try [{no_solution_count + 1}/{max_no_solution_count}] "
    #                           f"Increasing number of reference sources: {len(self.ref_cat)}/{Nref_current}")
    #
    #         # process the current source catalog and the selected reference sources
    #         sample_result = self.process_sample()
    #
    #         # check result
    #         if sample_result is None or sample_result[0] < self.config['MIN_SOURCE_NO_CONVERGENCE']:
    #             no_solution_count += 1
    #
    #             if no_solution_count >= max_no_solution_count:
    #                 self.log.warning(f"{' ':<6}> No solution found after {max_no_solution_count} trails. "
    #                                  f"Skipping to next configuration.")
    #                 ref_sample_idx = num_ref_samples - 1  # Force iteration
    #                 no_solution_count = 0  # Reset counter for the next iteration
    #             else:
    #                 if sample_result is None:
    #                     self.log.warning(f"{' ':<6}> No solution/matches found. Trying again...")
    #                 else:
    #                     self.log.warning(f"{' ':<6}> Configuration resulted in less than "
    #                                      f"{self.config['MIN_SOURCE_NO_CONVERGENCE']} matches. "
    #                                      f"Trying again...")
    #
    #             # increase counter
    #             if center_idx == max_centers - 1:
    #                 rot_matrix_idx += 1
    #                 center_idx = 0
    #                 solve_attempts_count += 1
    #             else:
    #                 if ref_sample_idx == num_ref_samples - 1:
    #                     center_idx += 1
    #                     ref_sample_idx = 0
    #                     solve_attempts_count += 1
    #                 else:
    #                     # Move to the next sample within the same iteration
    #                     ref_sample_idx += 1
    #
    #             # reset switches
    #             src_cat_match = ref_cat_match = None
    #             use_initial_wcs_estimate = True
    #             update_ref_cat = False
    #             update_src_cat = False
    #             has_possible_solution = False
    #
    #             # reset best result
    #             reset_best_params()
    #             continue
    #         else:
    #             has_possible_solution = True
    #
    #         # update possible solution count
    #         possible_solution_count += 1
    #
    #         # unpack results
    #         Nobs_matched, score, wcsprm_out, result_info, src_cat_match, ref_cat_match = sample_result
    #
    #         # set variables
    #         Nref_matched = len(self.ref_cat)
    #         completeness = Nobs_matched / Nref_matched
    #         rms_dist = 1. / (score * Nobs_matched)
    #         self.current_wcsprm = copy(wcsprm_out)
    #
    #         # check completeness
    #         if completeness >= self.config['THRESHOLD_CONVERGENCE']:
    #             high_completeness_count += 1
    #
    #         # update best result
    #         if best_score < score and best_Nobs <= Nobs_matched:
    #             best_Nobs = Nobs_matched
    #             best_score = score
    #             best_completeness = completeness
    #             best_wcsprm = copy(wcsprm_out)
    #             best_ref_cat = self.ref_cat
    #             best_src_cat = self.source_cat
    #
    #         # print(possible_solution_count, Nobs_matched, score, completeness, high_completeness_count, rms_dist)
    #
    #         # if there are more than the N brightest sources, increase the number of sources and re-run
    #         if has_large_src_cat and not update_src_cat:
    #             update_src_cat = True
    #             update_ref_cat = True
    #             use_initial_wcs_estimate = False
    #
    #             possible_solution_count = 0
    #             high_completeness_count = 0
    #             self.log.info(f"{' ':<6}> Possible solution found. Confirming...")
    #             self.log.info(f"{' ':<6}Increase source number: "
    #                           f"{Nobs_current} (brightest) --> {len(self.source_cat_full)} (all)")
    #             continue
    #
    #         elif not has_large_src_cat and possible_solution_count == 1:
    #             self.log.info(f"{' ':<6}> Possible solution found. Confirming...")
    #
    #         if completeness == 1.0 or high_completeness_count >= 3 or possible_solution_count >= max_wcs_eval_iter:
    #
    #             has_solution = True
    #             final_wcsprm = best_wcsprm
    #
    #             # get rms estimate
    #             result_dict = get_rms(best_ref_cat,
    #                                   best_src_cat,
    #                                   best_wcsprm,
    #                                   self.match_radius_px,
    #                                   best_Nobs)
    #
    #             self.log.info(f"{' ':<6}>> " + bc.BCOLORS.PASS + "Solution found." + bc.BCOLORS.OKGREEN)
    #             self.log.info(f"{' ':<4}{'-' * 50}")
    #             self.log.info(f"{' ':<4}Summary:")
    #             self.log.info(f"{' ':<4}{'-' * 50}")
    #             self.log.info(f"{' ':<5}{'Evaluations':<30} {possible_solution_count} ")
    #             self.log.info(f"{' ':<5}{'Matched (detected sources)':<30} {f'{best_Nobs}/{Nobs_total}'} ")
    #             self.log.info(f"{' ':<5}{'Matched (reference sources)':<30} {f'{len(best_ref_cat)}/{Nref_total}'} ")
    #             self.log.info(f"{' ':<5}{'Matched (src/ref)':<30} {f'{best_Nobs}/{len(best_ref_cat)}'} ")
    #             self.log.info(
    #                 f"{' ':<5}{'Completeness':<30} {best_completeness:.3f} ({best_completeness * 100:.1f}%)")
    #             self.log.info(f"{' ':<5}{'Match radius (px)':<30} {result_dict['radius_px']:.2f} ")
    #             self.log.info(f"{' ':<5}{'RMS (arcsec)':<30} {result_dict['rms']:.2f} ")
    #             self.log.info(f"{' ':<4}{'-' * 50}")
    #
    #             break
    #         else:
    #             # update switches for next iteration
    #             update_ref_cat = True
    #             use_initial_wcs_estimate = False
    #
    #     return has_solution, final_wcsprm, result_dict
