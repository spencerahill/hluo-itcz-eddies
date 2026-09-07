"""Moist static energy, the net energy input, and the mass adjustment."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from puffins.constants import GRAV_EARTH

from itcz_eddies.columns import col_int, dp_from_sfc_pressure
from itcz_eddies.mse import (
    C_P_PUFFINS,
    C_P_SCRIPTS,
    L_V,
    adjust_wind_by_mse_flux,
    adjusted_col_fluxes,
    budget_residual,
    flux_adjustment_per_unit_mass,
    moist_static_energy,
    net_energy_input,
)


@pytest.fixture
def thermo(grid):
    """Temperature, geopotential and specific humidity on the test grid."""
    lev = xr.DataArray(grid["level"], dims="level", coords={"level": grid["level"]})
    lat = xr.DataArray(grid["latitude"], dims="latitude",
                       coords={"latitude": grid["latitude"]})
    temp = (288.0 - 55.0 * (1.0 - lev / 1000.0)
            - 4.0 * np.sin(np.deg2rad(lat)) ** 2)
    geopot = -287.0 * 250.0 * np.log(lev * 100.0 / 1.0e5)
    sphum = 0.018 * (lev / 1000.0) ** 3 * (1.0 - 0.5 * np.sin(np.deg2rad(lat)) ** 2)
    return temp, geopot, sphum


def test_moist_static_energy_matches_the_scripts_formula(thermo):
    """Cp*T + Lv*q + Z, exactly, with the scripts' constants.

    puffins takes geopotential height and multiplies by gravity; ERA5 stores
    geopotential and the scripts add it in directly.  Dividing by the same
    gravity on the way in makes the two identical.
    """
    temp, geopot, sphum = thermo
    theirs = C_P_SCRIPTS * temp + L_V * sphum + geopot
    mine = moist_static_energy(temp, geopot, sphum)
    xr.testing.assert_allclose(mine, theirs, rtol=1e-15)


@pytest.mark.parametrize("grav", [9.8, 9.81, 9.80665])
def test_moist_static_energy_does_not_depend_on_gravity(thermo, grav):
    """The division and multiplication by gravity cancel exactly."""
    temp, geopot, sphum = thermo
    reference = moist_static_energy(temp, geopot, sphum, grav=GRAV_EARTH)
    xr.testing.assert_allclose(
        moist_static_energy(temp, geopot, sphum, grav=grav), reference, rtol=1e-15
    )


def test_the_two_specific_heats_differ_by_a_measured_amount(thermo):
    """1004.0 against puffins' 1003.5: 0.5*T out of an MSE of order 3e5 J/kg.

    The number is here so that adopting puffins' constant is a decision taken
    against a measurement rather than a shrug.
    """
    temp, geopot, sphum = thermo
    scripts = moist_static_energy(temp, geopot, sphum, c_p=C_P_SCRIPTS)
    puffins = moist_static_energy(temp, geopot, sphum, c_p=C_P_PUFFINS)
    difference = float(abs(scripts - puffins).max())
    typical = float(abs(scripts).mean())
    assert difference == pytest.approx(
        (C_P_SCRIPTS - C_P_PUFFINS) * float(temp.max()), rel=1e-12
    )
    # 0.5 J/kg/K times a temperature near 290 K, against an MSE near 3e5 J/kg.
    assert 100.0 < difference < 200.0
    assert difference / typical < 1e-3


def test_net_energy_input_matches_the_scripts_combination():
    """F = TS + TL - SS - SL - latent - sensible, from MSEadjust/main.py."""
    values = {name: float(i + 1) for i, name in enumerate(
        ["sensible", "latent", "sfc_sw", "sfc_lw", "toa_sw", "toa_lw"])}
    expected = (values["toa_sw"] + values["toa_lw"] - values["sfc_sw"]
                - values["sfc_lw"] - values["latent"] - values["sensible"])
    assert net_energy_input(**values) == expected


# ----------------------------------------------- the mass adjustment itself


@pytest.fixture
def global_grid():
    """A global latitude-longitude grid windspharm accepts.

    windspharm requires latitudes that are Gaussian, or equally spaced and
    global.  For an odd count, equally spaced and global means the poles are
    included, which is what ERA5 at half-degree spacing does with its 361
    latitudes.  33 latitudes and 64 longitudes is the same shape and runs in
    well under a second.
    """
    return {
        "latitude": np.linspace(90.0, -90.0, 33),
        "longitude": np.arange(0.0, 360.0, 360.0 / 64),
        "time": np.arange(4.0),
    }


def _global_field(global_grid, func):
    shape = tuple(global_grid[k].size
                  for k in ("latitude", "longitude", "time"))
    lat = global_grid["latitude"][:, None, None]
    lon = global_grid["longitude"][None, :, None]
    time = global_grid["time"][None, None, :]
    values = func(np.deg2rad(lat), np.deg2rad(lon), time) * np.ones(shape)
    return xr.DataArray(
        values, dims=("latitude", "longitude", "time"), coords=global_grid
    )


def test_the_adjustment_closes_the_column_budget_except_its_global_mean(
    global_grid,
):
    """The adjustment removes everything a divergent wind can remove.

    The global integral of a divergence vanishes, so no adjustment built from
    a purely divergent wind can remove an area-weighted global mean imbalance
    from the residual.  Everything else it removes down to the accuracy of the
    spherical-harmonic round trip.

    Measured here on a smooth large-scale residual: the pre-adjustment
    residual has an rms of 681 W/m2 and an area-weighted global mean of
    -5.721 W/m2; after adjustment the rms is 5.700 W/m2 and its global mean is
    -5.708 W/m2, the two global means agreeing to 0.22%; removing the global
    mean leaves 0.047 W/m2, which is 6.8e-5 of the pre-adjustment rms and is
    the accuracy of the spherical-harmonic round trip.
    """
    u_col = _global_field(
        global_grid,
        lambda la, lo, t: 3.0e9 * np.cos(la) * np.sin(2 * lo + 0.1 * t),
    )
    v_col = _global_field(
        global_grid, lambda la, lo, t: 2.0e9 * np.cos(la) ** 2 * np.cos(2 * lo)
    )
    tendency = _global_field(
        global_grid, lambda la, lo, t: 5.0 * np.cos(3 * la) * np.cos(lo)
    )
    source = _global_field(global_grid, lambda la, lo, t: 20.0 * np.cos(la) - 10.0)

    import windspharm.xarray

    div = windspharm.xarray.VectorWind(u_col, v_col).divergence()
    before = tendency + div - source

    weights = np.cos(np.deg2rad(before.latitude))

    def area_mean(arr):
        return ((arr * weights).sum(("latitude", "longitude"))
                / (weights.sum() * arr.sizes["longitude"]))

    u_adj, v_adj = adjusted_col_fluxes(u_col, v_col, tendency, source)
    after = budget_residual(u_adj, v_adj, tendency, source)

    rms_before = float(np.sqrt((before ** 2).mean()))
    rms_after = float(np.sqrt((after ** 2).mean()))
    assert rms_after / rms_before < 1e-2

    # What survives is the global mean the method cannot touch.
    global_mean_before = float(area_mean(before).mean())
    global_mean_after = float(area_mean(after).mean())
    assert abs(global_mean_after - global_mean_before) < 1e-2 * abs(global_mean_before)
    assert abs(global_mean_after) == pytest.approx(rms_after, rel=1e-2)

    # With that removed, what is left is the harmonic round-trip error.
    without_global_mean = after - area_mean(after)
    assert float(np.sqrt((without_global_mean ** 2).mean())) / rms_before < 1e-4


def test_the_wind_correction_integrates_back_to_the_flux_adjustment(grid, p_sfc):
    """The identity behind ``v_adj = v - vMSE_adjust / MSE``.

    With the correction ``dv = V / ((p_s - p_t)/g) / h``, the column integral
    of ``dv * h`` is ``V`` at every gridpoint, whatever the vertical structure
    of the MSE.  That is why dividing by the MSE reproduces exactly the column
    flux adjustment that puffins subtracts from the column flux.
    """
    level = xr.DataArray(grid["level"], dims="level",
                         coords={"level": grid["level"]})
    dp = dp_from_sfc_pressure(level, p_sfc)
    p_top_pa = float(grid["level"][0]) * 100.0
    p_sfc_pa = p_sfc * 100.0

    v_col_adjustment = 1.0e8 * np.sin(np.deg2rad(p_sfc.latitude))
    _, v_per_mass = flux_adjustment_per_unit_mass(
        v_col_adjustment, v_col_adjustment, p_sfc_pa, p_top_pa, grav=GRAV_EARTH
    )

    mse = (3.2e5 + 2.0e4 * np.sin(np.pi * level / 1000.0)
           + 1.0e3 * np.cos(np.deg2rad(p_sfc.longitude)))
    mse = mse.broadcast_like(dp)

    wind = xr.zeros_like(mse)
    corrected = adjust_wind_by_mse_flux(wind, v_per_mass, mse)
    recovered = -col_int(corrected * mse, dp)

    expected = v_col_adjustment.broadcast_like(recovered)
    scale = float(abs(expected).max())
    assert float(abs(recovered - expected).max()) < 1e-10 * scale
