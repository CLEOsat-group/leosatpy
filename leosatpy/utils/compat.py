#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Name:         compat.py
# Purpose:      Compatibility helpers for differing versions of third-party
#               packages (photutils 2.x/3.x)
#
# History:
#
# 23.10.2026
# - file created; photutils 3 renamed SourceCatalog columns and Background2D
#   arguments
#
# -----------------------------------------------------------------------------

""" Modules """
from __future__ import annotations

import photutils
from packaging.version import Version
from photutils.background import Background2D

PHOTUTILS_GE_3 = Version(photutils.__version__) >= Version('3.0')

# Legacy SourceCatalog column names used throughout the pipeline and their
# photutils >= 3 equivalents
_CATALOG_RENAMES = {
    'xcentroid': 'x_centroid',
    'ycentroid': 'y_centroid',
}


def catalog_to_pandas(catalog, columns):
    """Convert a photutils SourceCatalog to a pandas DataFrame.

    The returned DataFrame always uses the legacy column names
    (e.g. ``xcentroid``), independent of the installed photutils version.

    Parameters
    ----------
    catalog : `~photutils.segmentation.SourceCatalog`
        Source catalog to convert.
    columns : list of str
        Column names to include, using the legacy names.

    Returns
    -------
    pd.DataFrame
        Catalog table with legacy column names.
    """
    if not PHOTUTILS_GE_3:
        return catalog.to_table(columns=list(columns)).to_pandas()

    new_cols = [_CATALOG_RENAMES.get(c, c) for c in columns]
    df = catalog.to_table(columns=new_cols).to_pandas()
    return df.rename(columns={v: k for k, v in _CATALOG_RENAMES.items()})


def make_background2d(data, box_size, bkg_estimator=None, bkg_rms_estimator=None,
                      interpolator=None, **kwargs):
    """Create a `~photutils.background.Background2D` for any photutils version.

    ``edge_method`` is never passed ('pad' is the only behaviour since
    photutils 2.0 and the argument was removed in 3.0). The RMS estimator is
    passed under its version-specific name, and the interpolator only for
    photutils < 3, where it is still supported.

    Parameters
    ----------
    data : np.ndarray
        Input image.
    box_size : int or tuple
        Box size for the background mesh.
    bkg_estimator : `~photutils.background.BackgroundBase`, optional
        Background estimator instance.
    bkg_rms_estimator : `~photutils.background.BackgroundRMSBase`, optional
        Background RMS estimator instance.
    interpolator : `~photutils.background.BkgZoomInterpolator`, optional
        Interpolator instance (ignored for photutils >= 3).
    **kwargs
        Further keyword arguments passed to Background2D.

    Returns
    -------
    `~photutils.background.Background2D`
    """
    kwargs.pop('edge_method', None)
    if bkg_estimator is not None:
        kwargs['bkg_estimator'] = bkg_estimator
    if bkg_rms_estimator is not None:
        key = 'bkg_rms_estimator' if PHOTUTILS_GE_3 else 'bkgrms_estimator'
        kwargs[key] = bkg_rms_estimator
    if interpolator is not None and not PHOTUTILS_GE_3:
        kwargs['interpolator'] = interpolator

    return Background2D(data, box_size, **kwargs)
