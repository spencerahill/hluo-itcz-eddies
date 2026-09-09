"""Below-ground masking and vertical integration.

Two quadratures live here, and the difference between them is the subject of
decision 3 in ``code-review/FINDINGS.md``.

**The legacy quadrature** is what produced every figure in the current draft.
``col_int_trapz`` is the trapezoid rule of ``nantrapz``, applied to a field
whose below-ground values have been replaced by NaN.  Because ``nantrapz``
sums with ``skipna=True``, a NaN at the lowest valid level removes that whole
trapezoid, so the column integral stops at the lowest level *above* the
surface and the partial layer between that level and the surface contributes
nothing.  Two consequences, both measured by Casper job 5841789 on
2026-09-07: the zonal mean and the vertical integral do not commute, and
neither order reproduces the pointwise identity, one leaving 22.7% of the peak
flux unaccounted for and the other 9.5%.

**The mass-weighted quadrature** is ``col_int``, which is
``puffins.vert_coords.int_dp_g``: the sum of ``arr * dp`` over levels, divided
by gravity.  ``dp_from_sfc_pressure`` builds the layer thicknesses, giving the
layer that contains the surface its partial thickness and giving every layer
below the surface a thickness of zero.  Two properties follow.  The integral
is a linear functional of the field with weights that do not depend on the
field, so it commutes with any other linear operator applied at fixed weights.
And no NaN enters the reduction, so every term of the flux decomposition is
averaged over the same set of gridpoints, which is what the ``skipna=True``
zonal mean in the current code fails to do.

Pressure units: the ``level`` coordinate of the ERA5 files is hPa, and so is
the surface pressure once the scripts divide the stored Pa value by 100.  The
functions here take pressures in hPa, matching the scripts, and convert to Pa
internally where the physics needs it.  ``col_int_trapz`` carries the
scripts' own ``/9.8*100.0``, which is the hPa-to-Pa factor divided by their
value for gravity; ``col_int`` takes gravity from ``puffins.constants``.
"""

from __future__ import annotations

import logging

import numpy as np
import xarray as xr
from puffins.constants import GRAV_EARTH
from puffins.vert_coords import col_avg as _puffins_col_avg
from puffins.vert_coords import int_dp_g
from puffins.vert_coords import subtract_col_avg as _puffins_subtract_col_avg

from .names import LEV_STR

__all__ = [
    "GRAV_HAOCHANG",
    "nantrapz",
    "sfc_pressure_mask",
    "col_int_trapz",
    "dp_from_sfc_pressure",
    "col_int",
    "col_avg",
    "subtract_col_avg",
]

# The value of gravity hardcoded throughout Haochang Luo's scripts.  The
# manuscript's Appendix A states 9.81, a 0.1% disagreement recorded as F8 in
# code-review/FINDINGS.md.  puffins.constants.GRAV_EARTH is 9.81.
GRAV_HAOCHANG = 9.8

# Placeholder for the interface below the lowest level, in hPa.  It only has
# to lie below every physical surface pressure; Earth's record is about
# 1085 hPa.
P_BOTTOM_PLACEHOLDER_HPA = 1100.0


def nantrapz(y, x=None, dx=1.0, dim=None):
    """Trapezoid rule along ``dim`` that skips NaN trapezoids.

    Moved unchanged from ``myfun.nantrapz``, which is byte-identical in the
    two copies of ``myfun.py``.  The only edit is to the signature, whose
    original annotations imported three private names out of
    ``numpy._typing._array_like``.

    A trapezoid is dropped, rather than treated as zero, whenever either of its
    two endpoints is NaN.  That is the behavior the legacy column integral
    depends on and the reason it cannot see the partial layer at the surface.
    """
    if dim is None:
        axis = 1
    else:
        axis = y.dims.index(dim)

    if x is None:
        d = dx
    else:
        d = x.diff(dim=dim)

    nd = y.ndim
    slice1 = [slice(None)] * nd
    slice2 = [slice(None)] * nd
    slice1[axis] = slice(1, None)
    slice2[axis] = slice(None, -1)

    bottom = y[tuple(slice1)].assign_coords({dim: d[dim]})
    top = y[tuple(slice2)].assign_coords({dim: d[dim]})
    ret = (d * (top + bottom) / 2.0).sum(dim=dim, skipna=True)

    return ret


def sfc_pressure_mask(field, p_sfc, below_ground=0.0, lev_str=LEV_STR):
    """One where the level is at or above the surface, ``below_ground`` below it.

    Consolidates the three in-script ``maskout`` bodies, which appear 16 times
    between them and differ in exactly two ways.  Twelve copies fill below
    ground with ``0`` and two with ``np.nan``; ``below_ground`` selects which.
    The third body, in ``MSEadjust/main_v.py``, builds its array of ones from
    ``model`` rather than from the broadcast surface-pressure field, which
    gives the same result whenever the two broadcast to the same shape.

    A fourth body, ``myfun.maskout``, is a different function under the same
    name: it opens the surface-pressure files itself from a hardcoded
    ``ds633.0`` path that no longer resolves, and returns the masked field
    rather than a mask.  It is not reproduced here.  Compose
    ``era5.read_data`` with this function instead.

    Parameters
    ----------
    field
        Field carrying the ``level`` coordinate to be masked, in hPa.
    p_sfc
        Surface pressure in hPa, on the same horizontal and time grid.
    below_ground
        Value to place at levels below the surface.  ``0.0`` reproduces the
        twelve-copy body, ``np.nan`` the two-copy body used by the flux
        decomposition and the vertical-profile script.
    """
    if lev_str not in field.coords:
        raise ValueError(
            f"cannot mask: {field.name!r} has no '{lev_str}' coordinate"
        )
    level_exp, p_sfc_exp = xr.broadcast(field[lev_str], p_sfc)
    level_exp = level_exp.transpose(*field.dims)
    p_sfc_exp = p_sfc_exp.transpose(*field.dims)
    ones = xr.DataArray(
        np.ones(np.shape(p_sfc_exp)), dims=p_sfc_exp.dims, coords=p_sfc_exp.coords
    )
    mask = xr.where(level_exp <= p_sfc_exp, ones, below_ground)
    logging.info("finished masking")
    return mask


def col_int_trapz(data, mask=None, lev_str=LEV_STR, grav=GRAV_HAOCHANG):
    """Legacy column integral: trapezoid rule over the NaN-masked levels.

    Reproduces ``column_integrate(data, mask)``, which appears 15 times
    identically across the scripts, and the one-argument ``myfun``
    version when ``mask`` is ``None``.  The ``100.0`` converts the hPa level
    coordinate to Pa.

    This is kept so that the restructured code can reproduce the published
    products exactly.  New calculations should use ``col_int``.
    """
    if mask is not None:
        data = xr.where(mask == 1, data, np.nan)
    return nantrapz(data, data[lev_str], dim=lev_str) / grav * 100.0


def dp_from_sfc_pressure(level, p_sfc, p_top=None, lev_str=LEV_STR):
    """Pressure thickness of each level in Pa, clipped at the surface.

    Interfaces sit midway between adjacent levels in pressure.  The topmost
    interface is placed at ``p_top``, which defaults to the topmost level
    itself, so the column is integrated from the highest stored level down.
    The bottom of each layer is clipped at the surface pressure, so:

    - a layer wholly above the surface gets its full thickness;
    - the layer containing the surface gets the thickness from its own upper
      interface down to the surface;
    - a layer wholly below the surface gets zero.

    The thicknesses therefore sum to ``p_sfc - p_top`` at every gridpoint and
    every time, which is the column mass times gravity.  That identity is what
    ``tests/test_columns.py`` checks.

    Parameters
    ----------
    level
        Pressure at level centers in hPa, ascending or descending.
    p_sfc
        Surface pressure in hPa, any shape broadcastable against ``level``.
    p_top
        Pressure of the topmost interface in hPa.  Default: the topmost level.

    Returns
    -------
    xarray.DataArray
        Thickness of each level in Pa, named ``dp``.
    """
    level = xr.DataArray(level) if not isinstance(level, xr.DataArray) else level
    vals = np.asarray(level.values, dtype="float64")
    ascending = bool(vals[0] < vals[-1])
    ordered = vals if ascending else vals[::-1]

    interfaces = np.empty(ordered.size + 1, dtype="float64")
    interfaces[1:-1] = 0.5 * (ordered[:-1] + ordered[1:])
    interfaces[0] = ordered[0] if p_top is None else float(p_top)
    # The bottom interface is a placeholder that the clip below replaces with
    # the surface pressure, so it has to lie below every surface pressure the
    # data can hold.  Until 2026-09-08 it was the midpoint-style value one
    # half-spacing below the bottom level, 1012.5 hPa on ERA5's levels, and
    # every column whose surface pressure exceeded that was silently
    # truncated there: up to 27 hPa of the surface layer lost where the
    # surface pressure reached 1039 hPa in July 1997 (F21 in
    # code-review/FINDINGS.md).
    interfaces[-1] = P_BOTTOM_PLACEHOLDER_HPA

    if not ascending:
        interfaces = interfaces[::-1]
        upper = interfaces[1:]
        lower = interfaces[:-1]
    else:
        upper = interfaces[:-1]
        lower = interfaces[1:]

    upper_da = xr.DataArray(upper, coords={lev_str: level}, dims=[lev_str])
    lower_da = xr.DataArray(lower, coords={lev_str: level}, dims=[lev_str])

    upper_clipped = np.minimum(upper_da, p_sfc)
    lower_clipped = np.minimum(lower_da, p_sfc)
    dp_hpa = (lower_clipped - upper_clipped).clip(min=0.0)
    return (dp_hpa * 100.0).rename("dp")


def col_int(arr, dp, lev_str=LEV_STR, grav=GRAV_EARTH):
    """Mass-weighted column integral, ``sum(arr * dp) / g``.

    Thin wrapper on ``puffins.vert_coords.int_dp_g`` that first sets the field
    to zero wherever the layer thickness is zero.  Without that step a field
    carrying NaN below ground would poison the sum, since ``0 * nan`` is
    ``nan``; with it, a below-ground value of any kind contributes nothing.

    A NaN in a layer of nonzero thickness is a different matter: it means the
    field is undefined somewhere the column mass is not, and the integral is
    then undefined too.  ``int_dp_g`` sums with NaN skipped, so left alone it
    would return the integral over the remaining layers, and for a column
    that is NaN throughout, zero.  Found 2026-09-08 when the Lanczos time
    mean, which is NaN for its half-window at each end of the record, came
    out of the column integral as a flux of exactly zero there.  Such a
    column is returned as NaN.
    """
    undefined = (arr.isnull() & (dp > 0)).any(lev_str)
    arr = xr.where(dp > 0, arr, 0.0)
    return int_dp_g(arr, dp, dim=lev_str, grav=grav).where(~undefined)


def col_avg(arr, dp, lev_str=LEV_STR):
    """Mass-weighted column average, from ``puffins.vert_coords.col_avg``."""
    return _puffins_col_avg(xr.where(dp > 0, arr, 0.0), dp, dim=lev_str)


def subtract_col_avg(arr, dp, lev_str=LEV_STR):
    """Remove the column average at every level.

    ``puffins.vert_coords.subtract_col_avg``, whose docstring gives the
    physical argument for the operation: in the time mean and neglecting the
    column mass tendency, the column-integrated meridional mass transport has
    to vanish at each latitude, or mass builds up on one side.

    That argument applies to a wind.  The flux decomposition in
    ``main/adjusted_MMC_stationary_transient_calculation.py`` line 324 applies
    the same correction to the zonal-mean wind alone, after the mass
    adjustment has already been applied to the full wind at line 310, which is
    decision 1 in ``code-review/FINDINGS.md``.
    """
    return _puffins_subtract_col_avg(xr.where(dp > 0, arr, 0.0), dp, dim=lev_str)
