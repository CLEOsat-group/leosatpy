#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Data-free unit tests for the dark-scaling / pure-dark logic (C1).

These exercise two pure static methods of ``ReduceSatObs`` without touching the
filesystem, FITS data, or the pipeline itself:

* ``_select_exposure_matches`` — chooses the nearest available dark exposure and
  decides whether it is used exactly or scaled.
* ``_dark_needs_bias_removal`` — decides whether a matched dark must be made
  "pure" (bias-subtracted) before use. A dark that will be scaled must be
  bias-free in EITHER scaling direction; this is the C1 regression guard.

Run with:  python3 -m unittest tests.test_dark_scaling
       or:  python3 tests/test_dark_scaling.py
"""
import unittest

from leosatpy.reduceSatObs import ReduceSatObs

sel = ReduceSatObs._select_exposure_matches
needs_bias = ReduceSatObs._dark_needs_bias_removal


def _info(available=True, actual=None, is_exact=True):
    """Build the minimal dark_state entry that _dark_needs_bias_removal reads."""
    return {'available': available, 'actual_exposure': actual, 'is_exact': is_exact}


class TestSelectExposureMatches(unittest.TestCase):
    def test_exact_match_is_exact_no_scale(self):
        m = sel([60.0], [60.0], tolerance=1.0, allow_scaling=True)[60.0]
        self.assertTrue(m['is_exact'])
        self.assertFalse(m['scale_required'])
        self.assertEqual(m['actual_exposure'], 60.0)

    def test_within_tolerance_either_direction_is_exact(self):
        # 0.5 s off in both directions, tolerance 1 s -> treated as exact (unscaled).
        longer = sel([60.0], [60.5], tolerance=1.0, allow_scaling=True)[60.0]
        shorter = sel([60.0], [59.5], tolerance=1.0, allow_scaling=True)[60.0]
        for m in (longer, shorter):
            self.assertTrue(m['is_exact'])
            self.assertFalse(m['scale_required'])

    def test_out_of_tolerance_scale_up_when_allowed(self):
        # Only a shorter dark available, beyond tolerance -> scale up.
        m = sel([60.0], [30.0], tolerance=1.0, allow_scaling=True)[60.0]
        self.assertFalse(m['is_exact'])
        self.assertTrue(m['scale_required'])
        self.assertEqual(m['actual_exposure'], 30.0)

    def test_out_of_tolerance_scale_down_when_allowed(self):
        # Only a longer dark available, beyond tolerance -> scale down.
        m = sel([60.0], [120.0], tolerance=1.0, allow_scaling=True)[60.0]
        self.assertFalse(m['is_exact'])
        self.assertTrue(m['scale_required'])
        self.assertEqual(m['actual_exposure'], 120.0)

    def test_out_of_tolerance_no_scaling_returns_none(self):
        self.assertIsNone(sel([60.0], [30.0], tolerance=1.0, allow_scaling=False)[60.0])

    def test_no_available_returns_none(self):
        self.assertIsNone(sel([60.0], [], tolerance=1.0, allow_scaling=True)[60.0])

    def test_picks_nearest_available(self):
        m = sel([60.0], [10.0, 55.0, 200.0], tolerance=1.0, allow_scaling=True)[60.0]
        self.assertEqual(m['actual_exposure'], 55.0)


class TestDarkNeedsBiasRemoval(unittest.TestCase):
    def test_exact_dark_stays_impure(self):
        # Unscaled dark: bias cancels in the science subtraction, no removal needed.
        self.assertFalse(needs_bias(_info(actual=60.0, is_exact=True)))

    def test_scale_down_requires_bias_removal(self):
        # Dark longer than target, scaled down -> must be pure. (Old code caught this.)
        self.assertTrue(needs_bias(_info(actual=120.0, is_exact=False)))

    def test_scale_up_requires_bias_removal_C1_regression(self):
        # Dark SHORTER than target, scaled up -> must be pure.
        # C1: the previous `actual_exposure > target` check missed this case,
        # leaving the dark impure so its bias pedestal got scaled with the signal.
        self.assertTrue(needs_bias(_info(actual=30.0, is_exact=False)))

    def test_unavailable_dark_not_flagged(self):
        self.assertFalse(needs_bias(_info(available=False, actual=30.0, is_exact=False)))

    def test_missing_actual_exposure_not_flagged(self):
        self.assertFalse(needs_bias(_info(actual=None, is_exact=False)))

    def test_defaults_to_exact_when_flag_absent(self):
        # A malformed entry without 'is_exact' is treated as exact (no removal).
        self.assertFalse(needs_bias({'available': True, 'actual_exposure': 60.0}))


class TestSelectAndNeedsBiasTogether(unittest.TestCase):
    """End-to-end (still data-free): the matcher's output, shaped like a dark_state
    entry, must drive _dark_needs_bias_removal correctly in both scaling directions."""

    @staticmethod
    def _entry_from_match(match):
        # Mirror how _probe_candidate stores an available match into dark_state.
        return {
            'available': True,
            'actual_exposure': match['actual_exposure'],
            'is_exact': match['is_exact'],
        }

    def test_scale_up_path_flags_pure(self):
        match = sel([60.0], [30.0], tolerance=1.0, allow_scaling=True)[60.0]
        self.assertTrue(needs_bias(self._entry_from_match(match)))

    def test_scale_down_path_flags_pure(self):
        match = sel([60.0], [120.0], tolerance=1.0, allow_scaling=True)[60.0]
        self.assertTrue(needs_bias(self._entry_from_match(match)))

    def test_within_tolerance_path_stays_impure(self):
        match = sel([60.0], [60.5], tolerance=1.0, allow_scaling=True)[60.0]
        self.assertFalse(needs_bias(self._entry_from_match(match)))


if __name__ == '__main__':
    unittest.main(verbosity=2)
