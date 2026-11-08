#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Helpers for temperature-aware calibration matching.

The CCD temperature is used as a soft preference when calibration frames are
matched to science frames. Which header keywords hold the temperature, and
whether they can be trusted, depends on the camera; this is configured per
telescope with an optional ``ccd_temperature`` block (see ``telescope_conf.py``):

* ``'match'``: the camera has a regulated setpoint. Frames are grouped by the
  setpoint and calibrations are preferred at the temperature of the frames they
  correct.
* ``'record'``: the temperature is only written to the headers for provenance
  (e.g. LN2-cooled cameras, measured-only or unreliable sensors).
* missing block or ``None``: the feature is off for that telescope.

The functions in this module are pure (no pipeline state) so they can be tested
without data.
"""
import math
import re

import numpy as np

KELVIN_OFFSET = 273.15
MODES = ('match', 'record')

# Header keywords written by the pipeline
KEY_TEMPREF = 'TEMPREF'
KEY_TEMPOK = 'TEMPOK'
KEY_TEMPVAL = 'TEMPVAL'
KEY_TEMPGRP = 'TEMPGRP'
KEY_CALSRC = 'CALSRC'
KEY_CALPROD = 'CALPROD'

UNKNOWN_GROUP = 'Tunknown'


def _is_false(value):
    return value is False or str(value).strip().lower() in ('false', 'off', 'no', '0')


def temperature_settings(obsparams, config):
    """Return the effective temperature settings for a telescope, or None if off.

    Parameters
    ----------
    obsparams : dict
        Telescope parameters; may contain a ``ccd_temperature`` block.
    config : dict
        User configuration (``[Reduction]``); keys are optional.
    """
    block = (obsparams or {}).get('ccd_temperature')
    if not block:
        return None

    global_switch = (config or {}).get('TEMP_MATCH', True)
    if _is_false(global_switch):
        return None

    mode = block.get('mode')
    if mode not in MODES:
        return None
    if str(global_switch).strip().lower() == 'record':
        mode = 'record'
    if mode == 'match' and not block.get('set_key'):
        # Matching needs a regulated setpoint; measured-only values are recorded
        mode = 'record'

    tolerance = block.get('tolerance')
    if tolerance is None:
        tolerance = (config or {}).get('TEMP_TOLERANCE', 1.0)

    policy = str((config or {}).get('TEMP_MISMATCH', 'nearest')).strip().lower()
    if policy not in ('nearest', 'strict'):
        policy = 'nearest'

    return {
        'mode': mode,
        'set_key': block.get('set_key'),
        'ccd_key': block.get('ccd_key'),
        'unit': str(block.get('unit', 'C')).upper(),
        'valid_range': block.get('valid_range'),
        'tolerance': float(tolerance),
        'policy': policy,
        'min_frames': int((config or {}).get('MIN_CALIB_FRAMES', 3)),
    }


def parse_temperature(value, unit='C'):
    """Parse a header temperature value and return it in degrees Celsius.

    Accepts numbers and strings such as ``'164.812'`` or ``'-10.0 C'``.
    Values are treated as Kelvin if ``unit`` is ``'K'``, if the string ends with
    a ``K``, or if the value is above 150 (no CCD runs above 150 °C).
    Returns None for missing, non-numeric or non-finite values.
    """
    if value is None or isinstance(value, bool):
        return None

    is_kelvin = str(unit).upper() == 'K'
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
    else:
        text = str(value).strip()
        match = re.search(r'[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?', text)
        if match is None:
            return None
        number = float(match.group(0))
        if re.search(r'\d\s*K\s*$', text):
            is_kelvin = True

    if not math.isfinite(number):
        return None
    if is_kelvin or number > 150.:
        number -= KELVIN_OFFSET
    return number


def _in_range(value, valid_range):
    if value is None:
        return False
    if not valid_range:
        return True
    low, high = valid_range
    return low <= value <= high


def _header_get(header, key, fallback_header=None):
    if not key:
        return None
    for hdr in (header, fallback_header):
        if hdr is None:
            continue
        if key in hdr:
            return hdr[key]
    return None


def frame_temperature(header, settings, fallback_header=None):
    """Derive the reference temperature of a frame from its header.

    Returns a tuple ``(tempref, tempok, tempval)``:

    * ``tempref``: reference temperature in °C or None if unknown. In ``match``
      mode this is the setpoint; in ``record`` mode the measured value (or the
      setpoint if no measured value is usable).
    * ``tempok``: False when the measured value deviates from the setpoint by
      more than the tolerance (cooling not reached or a faulty sensor), True when
      both agree, None when it cannot be checked.
    * ``tempval``: True if ``tempref`` is known and within the valid range.
    """
    if not settings:
        return None, None, False

    unit = settings.get('unit', 'C')
    valid_range = settings.get('valid_range')
    set_t = parse_temperature(_header_get(header, settings.get('set_key'), fallback_header), unit)
    ccd_t = parse_temperature(_header_get(header, settings.get('ccd_key'), fallback_header), unit)
    set_ok = _in_range(set_t, valid_range)
    ccd_ok = _in_range(ccd_t, valid_range)

    if settings.get('mode') == 'match':
        tempref = set_t if set_ok else None
    else:
        tempref = ccd_t if ccd_ok else (set_t if set_ok else None)

    tempok = None
    if set_ok and ccd_ok:
        tempok = abs(ccd_t - set_t) <= settings.get('tolerance', 1.0)
    elif tempref is not None:
        tempok = True

    if tempref is not None:
        tempref = round(float(tempref), 2)
    return tempref, tempok, tempref is not None


def temperature_header_cards(header, settings, fallback_header=None):
    """Return the provenance cards ``{key: (value, comment)}`` for a frame header."""
    tempref, tempok, tempval = frame_temperature(header, settings, fallback_header)
    cards = {KEY_TEMPVAL: ('T' if tempval else 'F', 'CCD temperature known and plausible')}
    if tempref is not None:
        cards[KEY_TEMPREF] = (tempref, '[degC] reference CCD temperature')
    if tempok is not None:
        cards[KEY_TEMPOK] = ('T' if tempok else 'F', 'measured CCD temperature at setpoint')
    return cards


def header_tempref(header):
    """Return the pipeline reference temperature stored in a header, or None."""
    if header is None or KEY_TEMPREF not in header:
        return None
    try:
        value = float(header[KEY_TEMPREF])
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def group_temperatures(values, tolerance):
    """Group temperatures without chaining.

    Values are sorted; a group starts at its smallest value and takes all values
    up to ``start + tolerance``. Unknown values (None) are returned separately.

    Returns
    -------
    groups : list of tuple
        ``(reference, indices)`` per group, ordered by temperature; the reference
        is the median of the group.
    unknown : list of int
        Indices of values without a temperature.
    """
    known = sorted((float(v), i) for i, v in enumerate(values) if v is not None)
    unknown = [i for i, v in enumerate(values) if v is None]
    groups = []
    current = []
    start = None
    for value, idx in known:
        if start is None or value - start > tolerance + 1e-9:
            if current:
                groups.append(current)
            current = []
            start = value
        current.append((value, idx))
    if current:
        groups.append(current)
    result = []
    for group in groups:
        reference = float(np.median([v for v, _ in group]))
        result.append((round(reference, 2), [i for _, i in group]))
    return result, unknown


def temperature_token(tempref):
    """Return the file name token for a temperature, e.g. -10 -> 'Tm10', -9.6 -> 'Tm9p6'."""
    if tempref is None:
        return None
    value = round(float(tempref), 1)
    if value == 0:
        value = 0.0
    sign = 'm' if value < 0 else ''
    text = f'{abs(value):.1f}'.rstrip('0').rstrip('.')
    return f'T{sign}{text.replace(".", "p")}'


def group_label(tempref):
    """Return the TEMPGRP header label for a science temperature group."""
    token = temperature_token(tempref)
    return token if token is not None else UNKNOWN_GROUP


def source_tag(night_date):
    """Return the master file name tag for a calibration source night, e.g. 's20260914'."""
    if night_date is None:
        return None
    return f"s{night_date.strftime('%Y%m%d')}"


def within_tolerance(t_a, t_b, tolerance):
    """True if both temperatures are known and agree within the tolerance."""
    if t_a is None or t_b is None:
        return False
    return abs(float(t_a) - float(t_b)) <= tolerance + 1e-9


def temperature_distance(t_a, t_b):
    """Absolute temperature difference; 0 if either temperature is unknown."""
    if t_a is None or t_b is None:
        return 0.
    return abs(float(t_a) - float(t_b))


def dark_is_significant(dark, dark_exptime, target_exptime, noise=None, percentile=99.9, nsigma=3.):
    """Decide whether the thermal signal of a dark matters at a target exposure time.

    The hot-pixel excess over the frame median (``percentile`` of
    ``dark - median(dark)``), scaled to the target exposure, is compared with
    ``nsigma`` times the noise of a single frame. A dark with zero exposure has
    no thermal signal.

    Parameters
    ----------
    dark : numpy.ndarray
        Dark frame (single frame or master).
    dark_exptime, target_exptime : float
        Exposure times in seconds.
    noise : float or None
        Noise sigma of a single frame; estimated from the dark if None.
    """
    if dark is None or dark_exptime is None or target_exptime is None:
        return False
    if float(dark_exptime) <= 0. or float(target_exptime) <= 0.:
        return False
    data = np.asarray(dark, dtype=float)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return False
    median = np.median(data)
    excess = np.percentile(data - median, percentile)
    if noise is None:
        noise = 1.4826 * np.median(np.abs(data - median))
    if not noise or not math.isfinite(noise):
        return False
    scaled = excess * float(target_exptime) / float(dark_exptime)
    return bool(scaled > nsigma * noise)


def _robust_sigma(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.
    return float(1.4826 * np.median(np.abs(values - np.median(values))))


def hot_pixel_index(pure_dark, nsigma=8.):
    """Return the flat indices of hot pixels in a bias-free dark (> nsigma robust)."""
    data = np.asarray(pure_dark, dtype=float)
    finite = np.isfinite(data)
    median = np.median(data[finite]) if finite.any() else 0.
    sigma = _robust_sigma(data[finite])
    if sigma <= 0.:
        return np.array([], dtype=int)
    return np.flatnonzero(finite & (data - median > nsigma * sigma))


def local_excess(image, flat_index, size=5):
    """Pixel value minus the median of its ``size`` x ``size`` neighbourhood.

    Evaluated only at ``flat_index`` positions, so the cost scales with the
    number of hot pixels rather than the image size.
    """
    image = np.asarray(image, dtype=float)
    if len(flat_index) == 0:
        return np.array([], dtype=float)
    half = size // 2
    padded = np.pad(image, half, mode='reflect')
    rows, cols = np.unravel_index(flat_index, image.shape)
    offsets = np.arange(-half, half + 1)
    dr, dc = np.meshgrid(offsets, offsets, indexing='ij')
    neigh = padded[rows[:, None] + half + dr.ravel()[None, :],
                   cols[:, None] + half + dc.ravel()[None, :]]
    return image.ravel()[flat_index] - np.median(neigh, axis=1)


def fit_dark_scale(science_excess, dark_excess, nsigma_clip=4., iterations=5):
    """Fit the scale k in ``science_excess ≈ k * dark_excess`` with sigma clipping.

    Both inputs are local excesses at the same hot-pixel positions (see
    ``local_excess``), which removes the sky and most of the star light.

    Returns
    -------
    k : float or None
        Fitted scale, None if it cannot be determined.
    kept_fraction : float
        Fraction of hot pixels kept after clipping.
    """
    x = np.asarray(dark_excess, dtype=float)
    y = np.asarray(science_excess, dtype=float)
    good = np.isfinite(x) & np.isfinite(y)
    if good.sum() < 2:
        return None, 0.
    keep = good.copy()
    k = None
    for _ in range(iterations):
        denom = np.sum(x[keep] ** 2)
        if denom <= 0:
            return None, 0.
        k = float(np.sum(x[keep] * y[keep]) / denom)
        residual = y - k * x
        sigma = _robust_sigma(residual[keep])
        if sigma <= 0:
            break
        new_keep = good & (np.abs(residual - np.median(residual[keep])) <= nsigma_clip * sigma)
        if new_keep.sum() < 2 or np.array_equal(new_keep, keep):
            keep = new_keep if new_keep.sum() >= 2 else keep
            break
        keep = new_keep
    return k, float(keep.sum()) / float(good.sum())


def accept_dark_scale(k, kept_fraction, n_hot, n_pixels,
                      min_hot=50, min_hot_fraction=1e-5, min_kept=0.5, k_range=(0.1, 10.)):
    """Acceptance test for a fitted dark scale (see ``fit_dark_scale``)."""
    if k is None or not math.isfinite(k):
        return False
    if n_hot < max(min_hot, min_hot_fraction * n_pixels):
        return False
    if kept_fraction < min_kept:
        return False
    return k_range[0] <= k <= k_range[1]


def residual_hot_pixels(image, flat_index, nsigma=5., size=5):
    """Flat indices among ``flat_index`` whose local excess in ``image`` exceeds nsigma."""
    if len(flat_index) == 0:
        return np.array([], dtype=int)
    excess = local_excess(image, flat_index, size=size)
    data = np.asarray(image, dtype=float)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return np.array([], dtype=int)
    sample = finite if finite.size <= 200000 else finite[:: max(1, finite.size // 200000)]
    sigma = _robust_sigma(sample - np.median(sample))
    if sigma <= 0:
        return np.array([], dtype=int)
    return np.asarray(flat_index)[np.abs(excess) > nsigma * sigma]
