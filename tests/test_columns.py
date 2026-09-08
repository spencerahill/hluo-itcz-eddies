"""Vertical quadrature and below-ground masking.

Three groups of checks:

1. the mass-weighted layer thicknesses satisfy the identity that defines them,
   summing to the pressure difference between the top interface and the
   surface;
2. the mass-weighted column integral is linear and commutes with the zonal
   mean, and the legacy trapezoid integral does neither, which is the
   measurement behind decision 3 in ``code-review/FINDINGS.md``;
3. the package reproduces Haochang Luo's own ``nantrapz``, ``column_integrate``
   and ``maskout`` exactly, checked by importing his modules.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from puffins.constants import GRAV_EARTH

from itcz_eddies.columns import (
    GRAV_HAOCHANG,
    col_int,
    col_int_trapz,
    dp_from_sfc_pressure,
    nantrapz,
    sfc_pressure_mask,
    subtract_col_avg,
)


# ---------------------------------------------------------------- thicknesses


def test_dp_sums_to_surface_minus_top(grid, p_sfc):
    """Layer thicknesses sum to (p_sfc - p_top) in Pa at every gridpoint."""
    dp = dp_from_sfc_pressure(
        xr.DataArray(grid["level"], dims="level",
                     coords={"level": grid["level"]}),
        p_sfc,
    )
    total = dp.sum("level")
    expected = (p_sfc - grid["level"][0]) * 100.0
    xr.testing.assert_allclose(total, expected, rtol=1e-12)


def test_dp_is_zero_below_ground_and_partial_at_the_surface(grid):
    """A hand-computed case: surface at 925 hPa, levels 900, 950 and 1000."""
    level = xr.DataArray(grid["level"], dims="level",
                         coords={"level": grid["level"]})
    dp = dp_from_sfc_pressure(level, xr.DataArray(925.0)) / 100.0

    # Interfaces around 900 hPa are 850 and 925; around 950 they are 925 and
    # 975; around 1000 they are 975 and 1025.
    assert dp.sel(level=900.0).values == pytest.approx(75.0)   # 925 - 850
    assert dp.sel(level=950.0).values == pytest.approx(0.0)    # wholly below
    assert dp.sel(level=1000.0).values == pytest.approx(0.0)
    assert dp.sel(level=800.0).values == pytest.approx(100.0)  # 850 - 750, full


@pytest.mark.parametrize("p_sfc_hpa", [1013.0, 1025.0, 1039.0, 1080.0])
def test_dp_keeps_the_surface_layer_when_the_surface_is_below_1012_hPa(
    grid, p_sfc_hpa
):
    """F21: a surface pressure above the old 1012.5 hPa placeholder lost mass.

    The lowest level's layer must extend from its upper interface, midway
    between the two lowest levels, all the way down to the surface, so the
    thicknesses still sum to (p_sfc - p_top).  July 1997 surface pressures
    reach 1039 hPa.
    """
    level = xr.DataArray(grid["level"], dims="level",
                         coords={"level": grid["level"]})
    upper_interface = 0.5 * (grid["level"][-2] + grid["level"][-1])
    dp = dp_from_sfc_pressure(level, xr.DataArray(p_sfc_hpa)) / 100.0
    assert dp.sel(level=1000.0).values == pytest.approx(p_sfc_hpa - upper_interface)
    assert float(dp.sum("level")) == pytest.approx(p_sfc_hpa - grid["level"][0])


def test_dp_is_independent_of_level_ordering(grid, p_sfc):
    """Descending levels, as ERA5 stores them, give the same thicknesses."""
    level = xr.DataArray(grid["level"], dims="level",
                         coords={"level": grid["level"]})
    dp_up = dp_from_sfc_pressure(level, p_sfc)
    dp_down = dp_from_sfc_pressure(level[::-1], p_sfc)
    xr.testing.assert_allclose(dp_up, dp_down.reindex(level=dp_up.level))


def test_column_integral_of_a_constant(grid, p_sfc):
    """Integrating one gives the column mass, (p_sfc - p_top) / g."""
    level = xr.DataArray(grid["level"], dims="level",
                         coords={"level": grid["level"]})
    dp = dp_from_sfc_pressure(level, p_sfc)
    ones = xr.ones_like(dp)
    expected = (p_sfc - grid["level"][0]) * 100.0 / GRAV_EARTH
    xr.testing.assert_allclose(col_int(ones, dp), expected, rtol=1e-12)


# ------------------------------------------------- linearity and commutation


def test_mass_weighted_integral_is_linear(grid, p_sfc, field):
    """col_int(a + b) equals col_int(a) + col_int(b) to machine precision."""
    level = xr.DataArray(grid["level"], dims="level",
                         coords={"level": grid["level"]})
    dp = dp_from_sfc_pressure(level, p_sfc)
    other = field * 0.3 + 2.0
    lhs = col_int(field + other, dp)
    rhs = col_int(field, dp) + col_int(other, dp)
    scale = float(abs(lhs).max())
    assert float(abs(lhs - rhs).max()) < 1e-12 * scale


def test_mass_weighted_integral_commutes_with_the_zonal_mean(grid, field):
    """With a longitude-independent surface, the two orders agree exactly.

    This is the property the trapezoid rule lacks.  A surface pressure that
    does not vary with longitude makes the layer thicknesses the same at every
    longitude, so the zonal mean and the vertical integral are two linear
    operators on different axes and they commute.
    """
    level = xr.DataArray(grid["level"], dims="level",
                         coords={"level": grid["level"]})
    lat = field.latitude
    p_sfc_zonal = 1005.0 - 5.0 * np.cos(np.deg2rad(lat * 4))
    dp = dp_from_sfc_pressure(level, p_sfc_zonal)

    integrate_then_average = col_int(field, dp).mean("longitude")
    average_then_integrate = col_int(field.mean("longitude"), dp)
    scale = float(abs(integrate_then_average).max())
    difference = float(abs(integrate_then_average - average_then_integrate).max())
    assert difference < 1e-12 * scale


def test_trapezoid_integral_does_not_commute_with_the_zonal_mean(grid, field, p_sfc):
    """The legacy quadrature fails the same test, by a fraction worth naming.

    A regression guard on the legacy path, not an aspiration.  It asserts that
    the two orders differ, so that if a future edit made them agree the change
    would be noticed rather than absorbed.
    """
    mask = sfc_pressure_mask(field, p_sfc, below_ground=np.nan)
    masked = xr.where(mask == 1, field, np.nan)
    integrate_then_average = col_int_trapz(masked).mean("longitude", skipna=True)
    average_then_integrate = col_int_trapz(masked.mean("longitude", skipna=True))
    scale = float(abs(integrate_then_average).max())
    difference = float(abs(integrate_then_average - average_then_integrate).max())
    assert difference > 1e-6 * scale


def test_subtract_col_avg_leaves_zero_column_integral(grid, p_sfc, field):
    """Removing the column average makes the column integral vanish."""
    level = xr.DataArray(grid["level"], dims="level",
                         coords={"level": grid["level"]})
    dp = dp_from_sfc_pressure(level, p_sfc)
    residual = col_int(subtract_col_avg(field, dp), dp)
    scale = float(abs(col_int(field, dp)).max())
    assert float(abs(residual).max()) < 1e-12 * scale


# ------------------------------------------------------- legacy equivalence


def test_nantrapz_matches_the_original(myfun, field):
    """Bitwise agreement with ``main/myfun.py``'s own ``nantrapz``."""
    mine = nantrapz(field, field.level, dim="level")
    theirs = myfun.nantrapz(field, field.level, dim="level")
    assert np.array_equal(mine.values, theirs.values)


def test_nantrapz_matches_the_original_with_nans(myfun, field, p_sfc):
    """Bitwise agreement on a field carrying below-ground NaNs."""
    mask = sfc_pressure_mask(field, p_sfc, below_ground=np.nan)
    masked = xr.where(mask == 1, field, np.nan)
    mine = nantrapz(masked, masked.level, dim="level")
    theirs = myfun.nantrapz(masked, masked.level, dim="level")
    assert np.array_equal(mine.values, theirs.values, equal_nan=True)


def test_col_int_trapz_matches_the_scripts_column_integrate(
    decomp_script, field, p_sfc
):
    """Bitwise agreement with ``column_integrate`` in the decomposition script."""
    mask = sfc_pressure_mask(field, p_sfc, below_ground=np.nan)
    mine = col_int_trapz(field, mask)
    theirs = decomp_script.column_integrate(field, mask)
    assert np.array_equal(mine.values, theirs.values, equal_nan=True)


def test_col_int_trapz_uses_the_scripts_value_of_gravity():
    """The legacy path keeps 9.8, which the manuscript states as 9.81 (F8)."""
    assert GRAV_HAOCHANG == 9.8
    assert GRAV_EARTH != GRAV_HAOCHANG


@pytest.mark.parametrize("below_ground", [0.0, np.nan])
def test_sfc_pressure_mask_matches_the_originals(
    decomp_script, zero_mask_script, field, p_sfc, below_ground
):
    """Bitwise agreement with both in-script ``maskout`` bodies.

    ``MSEadjust/main.py`` fills below ground with 0 and the decomposition
    script with NaN.  Twelve scripts share the first body and two the second;
    those are the only two in use.
    """
    original = (decomp_script.maskout if np.isnan(below_ground)
                else zero_mask_script.maskout)
    mine = sfc_pressure_mask(field, p_sfc, below_ground=below_ground)
    theirs = original(field, p_sfc)
    assert np.array_equal(mine.values, theirs.values, equal_nan=True)


def test_sfc_pressure_mask_rejects_a_field_with_no_level_coordinate(field, p_sfc):
    with pytest.raises(ValueError, match="level"):
        sfc_pressure_mask(field.isel(level=0, drop=True), p_sfc)
