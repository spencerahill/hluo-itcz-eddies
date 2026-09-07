"""Finite-difference derivatives on the ERA5 grid.

Moved from ``myfun.py``, where ``ddt``, ``ddx``, ``ddy`` and ``ddp`` are
byte-identical between the two copies.  Each original dispatches on the number
of dimensions, ``len(da.shape(x))``, and the only thing the dispatch decides is
the order of the ``transpose`` that follows.  Here the order is built from the
dimensions actually present, in the canonical order ``(time, level, latitude,
longitude)``, which reproduces every branch of every original.

Two properties of these derivatives are worth stating, because both are
invisible at the call site.

**The horizontal metric is a constant 111.2 km per degree.**  That is a sphere
of radius 6.3714e6 m, against the 6.371e6 m the decomposition script sets as
``a``, a difference of 6e-5 relative.  ``ddx`` divides by an additional
``cos(latitude)``, so it is the zonal derivative on the sphere; ``ddy`` is the
meridional one.

**``ddt`` returns a per-hour rate only because the callers leave time
encoded.**  Every script that calls ``ddt`` opens its files with
``decode_times=False``, so the ``time`` coordinate is a plain count of hours
and ``xr.DataArray.differentiate`` returns a per-hour rate, which the callers
then divide by 3600 to get per second.  Handed a decoded ``datetime64``
coordinate, ``differentiate`` uses its ``datetime_unit`` default instead, and
the answer is wrong by the ratio of that unit to an hour, with nothing raising.
Measured here on xarray 2026.7.0 the default is seconds, so the callers' result
comes out 3600 times too small.  The size of the error therefore depends on the
xarray version, which is the point of F9 in ``code-review/FINDINGS.md``.
``ddt`` raises on a datetime coordinate instead of returning any of them.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from .names import LAT_STR, LEV_STR, LON_STR, TIME_STR

__all__ = ["DEG_TO_METERS", "SECONDS_PER_HOUR", "ddt", "ddx", "ddy", "ddp"]

# Meters per degree of latitude, as hardcoded in myfun.py.
DEG_TO_METERS = 111.2e3
SECONDS_PER_HOUR = 3600.0

_CANONICAL_ORDER = (TIME_STR, LEV_STR, LAT_STR, LON_STR)


def _canonical(arr: xr.DataArray) -> xr.DataArray:
    """Transpose to (time, level, latitude, longitude), keeping what is there."""
    dims = [d for d in _CANONICAL_ORDER if d in arr.dims]
    extra = [d for d in arr.dims if d not in _CANONICAL_ORDER]
    return arr.transpose(*(dims + extra))


def ddt(arr: xr.DataArray, per_second: bool = False) -> xr.DataArray:
    """Time derivative, per hour by default.

    Parameters
    ----------
    arr
        Field with an undecoded ``time`` coordinate, in hours.
    per_second
        When ``True``, divide by 3600 so the result is per second.  Every
        caller of the original ``ddt`` does this division itself.

    Raises
    ------
    TypeError
        If the ``time`` coordinate holds datetimes.  ``myfun.ddt`` returns a
        rate per ``datetime_unit`` in that case, which on xarray 2026.7.0 is
        per second, so every caller's subsequent division by 3600 leaves the
        answer 3600 times too small.  Nothing in the original signals it.
        This is F9 in ``code-review/FINDINGS.md``.
    """
    if np.issubdtype(arr[TIME_STR].dtype, np.datetime64):
        raise TypeError(
            "ddt needs an undecoded time coordinate, in hours. This array's "
            "time coordinate is datetime64, for which xarray's differentiate "
            "returns a rate per its datetime_unit rather than per hour, so "
            "every caller's division by 3600 leaves the answer wrong by the "
            "ratio of that unit to an hour. Open the files with "
            "decode_times=False, or convert the derivative yourself."
        )
    out = _canonical(arr.differentiate(TIME_STR))
    return out / SECONDS_PER_HOUR if per_second else out


def ddx(arr: xr.DataArray) -> xr.DataArray:
    """Zonal derivative, per meter, using 111.2 km per degree times cos(lat).

    The two divisions are kept separate, in the original's order, rather than
    folded into one division by their product.  Floating-point division is not
    associative, so ``(a / b) / c`` and ``a / (b * c)`` differ in the last bit,
    and the equivalence test in ``tests/test_derivatives.py`` demands bitwise
    agreement.
    """
    deriv = _canonical(arr.differentiate(LON_STR))
    return deriv / DEG_TO_METERS / np.cos(arr[LAT_STR] * np.pi / 180.0)


def ddy(arr: xr.DataArray) -> xr.DataArray:
    """Meridional derivative, per meter, using 111.2 km per degree."""
    return _canonical(arr.differentiate(LAT_STR) / DEG_TO_METERS)


def ddp(arr: xr.DataArray) -> xr.DataArray:
    """Pressure derivative, per Pa, from a ``level`` coordinate in hPa.

    The original chunks the level dimension into blocks of 9 before
    differentiating.  That is a memory hint: xarray rechunks along the
    differentiated dimension itself, so the values are unchanged, which
    ``tests/test_derivatives.py`` checks on a 37-level column.  The chunking is
    left to the caller here.
    """
    return _canonical(arr.differentiate(LEV_STR) / 100.0)
