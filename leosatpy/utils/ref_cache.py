#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Name:         ref_cache.py
# Purpose:      Shared, coverage-based cache of downloaded reference catalogs
#
# Author:       p4adch (cadam)
#
# Created:      11/08/2026
# Copyright:    (c) p4adch 2010-
#
# -----------------------------------------------------------------------------

""" Shared cache for downloaded reference catalogs (Gaia DR3 for the astrometry,
GSC 2.4.3 for the photometry).

A cached entry is a full, unclipped cone of the catalog. A request is served from
the cache if a cached cone covers it, i.e. sep(centre, request) + r_request <= r_cached,
with a magnitude limit at least as deep and the same query definition. The rows are
then cut to the requested cone and magnitude limit, so a hit returns the rows a
fresh query of that cone returns.

Gaia DR3 positions do not depend on the observing date (the query does not apply
proper motions), so a cone is valid for every night.
"""
# -----------------------------------------------------------------------------
# Standard library imports
import json
import logging
import os
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

# Third party imports
import numpy as np
import pandas as pd
from astropy.io import ascii
from astropy.table import Table

try:
    import fcntl
except ImportError:  # pragma: no cover - not available on Windows
    fcntl = None

# -----------------------------------------------------------------------------

log = logging.getLogger(__name__)

# Bump a version when the query of a catalog changes, so old cones are not used
QUERY_VERSION = {'GAIADR3': 'gaia-dr3-v1', 'GSC243': 'gsc243-v1'}

# Row limit of the GSC web service (MAXOBJ in download_phot_ref_cat)
GSC_MAXOBJ = 5000
# Largest result returned by the Gaia archive for an anonymous query (observed: a dense
# field returned exactly 50000 rows, the brightest ones of the ordered query)
GAIA_MAXROWS = 50000

INDEX_NAME = 'index.json'
TBL_FORMAT = 'ecsv'

# Relative slack for comparing radii and magnitude limits
_EPS = 1e-9


def resolve_cache_dir(config):
    """Return the cache folder from the configuration, or None if the cache is disabled.

    ``REF_CAT_CACHE_PATH = 'auto'`` (default) uses ``<WORKING_DIR_PATH>/ref_catalogs``,
    an empty value disables the shared cache (per-dataset catalogs as before).
    """
    value = config.get('REF_CAT_CACHE_PATH', 'auto')
    if value is None:
        return None
    value = str(value).strip().strip("'\"")
    if value == '' or value.lower() in ('none', 'false', 'off'):
        return None
    if value.lower() == 'auto':
        work_dir = str(config.get('WORKING_DIR_PATH', '~')).strip().strip("'\"") or '~'
        return Path(work_dir).expanduser().resolve() / 'ref_catalogs'
    return Path(value).expanduser().resolve()


def angular_separation(ra1, dec1, ra2, dec2):
    """Great-circle distance in degrees (Vincenty formula, stable at all distances)."""
    ra1, dec1, ra2, dec2 = map(np.radians, (ra1, dec1, ra2, dec2))
    dra = ra2 - ra1
    sin_d1, cos_d1 = np.sin(dec1), np.cos(dec1)
    sin_d2, cos_d2 = np.sin(dec2), np.cos(dec2)
    num1 = cos_d2 * np.sin(dra)
    num2 = cos_d1 * sin_d2 - sin_d1 * cos_d2 * np.cos(dra)
    den = sin_d1 * sin_d2 + cos_d1 * cos_d2 * np.cos(dra)
    return np.degrees(np.arctan2(np.hypot(num1, num2), den))


def enclosing_cone(ras, decs, radii):
    """Cone around several cones: centre at the normalised mean direction, radius max(sep + r)."""
    ras, decs, radii = np.atleast_1d(ras), np.atleast_1d(decs), np.atleast_1d(radii)
    ra_r, dec_r = np.radians(ras), np.radians(decs)
    vec = np.array([np.cos(dec_r) * np.cos(ra_r), np.cos(dec_r) * np.sin(ra_r), np.sin(dec_r)]).sum(axis=1)
    norm = np.linalg.norm(vec)
    if norm < 1e-12:  # antipodal set, take the first cone's centre
        ra_c, dec_c = float(ras[0]), float(decs[0])
    else:
        vec /= norm
        ra_c = float(np.degrees(np.arctan2(vec[1], vec[0])) % 360.)
        dec_c = float(np.degrees(np.arcsin(np.clip(vec[2], -1., 1.))))
    radius = float(np.max(angular_separation(ra_c, dec_c, ras, decs) + radii))
    return ra_c, dec_c, radius


def cluster_pointings(ras, decs, radii, merge_max):
    """Group pointings greedily so that each group fits a cone of at most ``merge_max`` x its largest radius.

    Returns a list of index lists, in order of the first member.
    """
    groups = []
    for i in range(len(ras)):
        placed = False
        for g in groups:
            members = g + [i]
            _, _, r = enclosing_cone([ras[k] for k in members], [decs[k] for k in members],
                                     [radii[k] for k in members])
            if r <= merge_max * max(radii[k] for k in members) + _EPS:
                g.append(i)
                placed = True
                break
        if not placed:
            groups.append([i])
    return groups


def _mag_covers(cached, requested):
    """Whether a cached magnitude limit (-1 = none) contains all rows of the requested one."""
    cached, requested = float(cached), float(requested)
    if cached < 0:
        return True
    if requested < 0:
        return False
    return cached >= requested - _EPS


def _epoch_key(epoch):
    return None if epoch is None else round(float(epoch), 1)


class RefCatalogCache:
    """Coverage-based cache of reference catalog cones in one folder."""

    def __init__(self, cache_dir):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self.cache_dir / INDEX_NAME
        self._lock_path = self.cache_dir / 'index.lock'
        # Cones stored by this process; a forced download (-d) only trusts these
        self.fresh = set()

    # ---- index handling
    @contextmanager
    def _locked(self):
        with open(self._lock_path, 'a+') as fh:
            if fcntl is not None:
                fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(fh, fcntl.LOCK_UN)

    def _read_index(self):
        try:
            with open(self._index_path, encoding='utf8') as fh:
                entries = json.load(fh)['entries']
            return [e for e in entries if (self.cache_dir / e['file']).exists()]
        except FileNotFoundError:
            return self._rebuild_index() if any(self.cache_dir.glob(f'*.{TBL_FORMAT}')) else []
        except (ValueError, KeyError, TypeError):
            log.debug(f"    Reference catalog index unreadable, rebuilt from {self.cache_dir}")
            return self._rebuild_index()

    def _write_index(self, entries):
        tmp = self._index_path.with_suffix(f'.{uuid.uuid4().hex}.tmp')
        with open(tmp, 'w', encoding='utf8') as fh:
            json.dump({'entries': entries}, fh, indent=1)
        os.replace(tmp, self._index_path)

    def _rebuild_index(self):
        """Recreate the index from the meta data stored in every cached file."""
        entries = []
        for path in sorted(self.cache_dir.glob(f'*.{TBL_FORMAT}')):
            try:
                meta = ascii.read(path, format=TBL_FORMAT).meta.get('cone')
            except Exception:  # unreadable file, ignore
                continue
            if isinstance(meta, dict) and meta.get('file') == path.name:
                entries.append(dict(meta))
        return entries

    def entries(self):
        with self._locked():
            return self._read_index()

    # ---- public interface
    def lookup(self, catalog, ra, dec, radius, mag_lim=-1, epoch=None, fresh_only=False):
        """Return the rows of the requested cone as DataFrame and its entry, or (None, None) if no cone covers it."""
        entry = self.find(catalog, ra, dec, radius, mag_lim=mag_lim, epoch=epoch, fresh_only=fresh_only)
        if entry is None:
            return None, None
        try:
            tbl = ascii.read(self.cache_dir / entry['file'], format=TBL_FORMAT)
        except Exception as exc:
            log.debug(f"    Cached reference catalog {entry['file']} unreadable: {exc}")
            return None, None
        df = tbl.to_pandas()
        df = self.cut(df, ra, dec, radius, mag_lim)
        return df, entry

    def find(self, catalog, ra, dec, radius, mag_lim=-1, epoch=None, fresh_only=False, stale_only=False):
        """Smallest cached cone that covers the request, or None."""
        cat = catalog.upper()
        query = QUERY_VERSION.get(cat, cat)
        best = None
        for e in self.entries():
            if e['catalog'] != cat or e['query'] != query or e.get('epoch') != _epoch_key(epoch):
                continue
            if (fresh_only and e['file'] not in self.fresh) or (stale_only and e['file'] in self.fresh):
                continue
            if not _mag_covers(e['mag_lim'], mag_lim):
                continue
            sep = float(angular_separation(e['ra'], e['dec'], ra, dec))
            if e.get('truncated'):
                # A cone cut by the service row limit only stands for exactly this query
                if sep > 1e-6 or abs(e['radius'] - radius) > 1e-6 or float(e['mag_lim']) != float(mag_lim):
                    continue
            elif sep + radius > e['radius'] * (1. + _EPS) + _EPS:
                continue
            if best is None or e['radius'] < best['radius']:
                best = e
        return best

    @staticmethod
    def cut(df, ra, dec, radius, mag_lim=-1):
        """Rows inside the cone and magnitude limit, in the stored order."""
        if df is None or len(df) == 0:
            return df
        sep = angular_separation(ra, dec, df['RA'].to_numpy(float), df['DEC'].to_numpy(float))
        keep = sep <= radius
        if float(mag_lim) >= 0:
            mag = pd.to_numeric(df['mag'], errors='coerce').to_numpy(float)
            keep &= ~(mag > mag_lim)
        return df.loc[keep].reset_index(drop=True)

    def store(self, df, catalog, ra, dec, radius, mag_lim=-1, epoch=None, truncated=False):
        """Save a downloaded cone and register it. Returns the index entry."""
        cat = catalog.upper()
        name = (f"{cat}_{ra:08.4f}{dec:+08.4f}_r{radius:.4f}_m{float(mag_lim):g}_"
                f"{uuid.uuid4().hex[:8]}.{TBL_FORMAT}")
        entry = {'file': name, 'catalog': cat, 'query': QUERY_VERSION.get(cat, cat),
                 'ra': float(ra), 'dec': float(dec), 'radius': float(radius),
                 'mag_lim': float(mag_lim), 'epoch': _epoch_key(epoch),
                 'n_rows': int(len(df)), 'truncated': bool(truncated),
                 'created': time.strftime('%Y-%m-%dT%H:%M:%S')}
        drop = [c for c in ('xcentroid', 'ycentroid') if c in df.columns]
        tbl = Table.from_pandas(df.drop(columns=drop), index=False)
        tbl.meta = {'cone': entry}
        tmp = self.cache_dir / f'.{name}.tmp'
        ascii.write(tbl, tmp, format=TBL_FORMAT, overwrite=True)
        os.replace(tmp, self.cache_dir / name)
        with self._locked():
            entries = self._read_index()
            entries = [e for e in entries if e['file'] != name]
            entries.append(entry)
            self._write_index(entries)
        self.fresh.add(name)
        return entry

    def discard_covering(self, catalog, ra, dec, radius, mag_lim=-1, epoch=None):
        """Remove the cones from earlier runs that would serve this request (used by a forced download)."""
        removed = []
        while True:
            e = self.find(catalog, ra, dec, radius, mag_lim=mag_lim, epoch=epoch, stale_only=True)
            if e is None:
                break
            with self._locked():
                entries = [x for x in self._read_index() if x['file'] != e['file']]
                self._write_index(entries)
            try:
                (self.cache_dir / e['file']).unlink()
            except FileNotFoundError:
                pass
            removed.append(e['file'])
        return removed


_CACHES = {}


def get_cache(config):
    """Cache object for the configuration (one per folder and process), or None if disabled or not writable."""
    cache_dir = resolve_cache_dir(config)
    if cache_dir is None:
        return None
    if cache_dir in _CACHES:
        return _CACHES[cache_dir]
    try:
        _CACHES[cache_dir] = RefCatalogCache(cache_dir)
        return _CACHES[cache_dir]
    except OSError as exc:
        log.warning(f"Reference catalog cache {cache_dir} not usable ({exc}); catalogs are downloaded per frame")
        return None
