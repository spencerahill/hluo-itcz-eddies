"""The split of the layer mass flux: exact, closing under a moving surface,
and reducing to the split of the wind when the surface does not move.

These reproduce F24 on synthetic data.  The surface pressure fluctuates in
time at every gridpoint, and the wind near the surface covaries with it (an
equatorward wind when the surface pressure is high, as in a cold-air
outbreak), so the transient wind covaries with the thickness of the lowest
layer.  The split of the wind then leaves a record-mean cross term that no
projection time mean removes; the split of the layer mass flux does not.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from puffins.constants import GRAV_EARTH

from itcz_eddies.columns import col_int, dp_from_sfc_pressure
from itcz_eddies.decomp import (
    block_time_mean,
    boxcar_time_mean,
    decompose,
    decompose_mass_flux,
    ideal_time_mean,
    lanczos_time_mean,
)

FIVE = ["mmc", "stationary", "transient",
        "cross_mean_wind_eddy_mse", "cross_eddy_wind_mean_mse"]


@pytest.fixture
def moving_surface(grid):
    """Wind, MSE, layer thickness and surface pressure on a 240-step record.

    The surface pressure carries a 4-day (8-step) oscillation of 12 hPa on
    top of a ridge, and the wind at the two lowest levels carries the same
    oscillation, so the two covary.  A 45-day component sits in the mean.
    """
    lev = xr.DataArray(grid["level"], dims="level", coords={"level": grid["level"]})
    lat = xr.DataArray(grid["latitude"], dims="latitude",
                       coords={"latitude": grid["latitude"]})
    lon = xr.DataArray(grid["longitude"], dims="longitude",
                       coords={"longitude": grid["longitude"]})
    n_times = 240
    time = xr.DataArray(np.arange(float(n_times)), dims="time",
                        coords={"time": np.arange(float(n_times))})
    synoptic = np.cos(2 * np.pi * time / 8.0 + np.deg2rad(4 * lon))
    slow = np.sin(2 * np.pi * time / 90.0)
    ridge = np.exp(-((lon - 40.0) / 8.0) ** 2)
    p_sfc = (1005.0 - 60.0 * ridge + 12.0 * synoptic + 3.0 * slow
             - 2.0 * np.cos(np.deg2rad(lat * 6))
             ).transpose("time", "latitude", "longitude")
    near_surface = np.exp(-((1000.0 - lev) / 60.0))
    shape = np.sin(np.pi * lev / 1000.0)
    v = (2.0 * shape * np.cos(np.deg2rad(3 * lat))
         + 1.0 * shape * np.cos(np.deg2rad(2 * lon))
         + 0.8 * shape * slow
         - 3.0 * near_surface * synoptic)
    mse = (3.2e5 + 2.0e4 * shape + 3.0e3 * np.sin(np.deg2rad(lat))
           + 1.5e3 * np.sin(np.deg2rad(2 * lon))
           + 2.0e3 * np.cos(2 * np.pi * time / 8.0 + np.deg2rad(4 * lon) + 0.7))
    order = ("time", "level", "latitude", "longitude")
    template = time * lev * lat * lon
    v = v.broadcast_like(template).transpose(*order)
    mse = mse.broadcast_like(template).transpose(*order)
    dp = dp_from_sfc_pressure(lev, p_sfc).transpose(*order)
    return v, mse, dp, p_sfc


def _gap(terms, names):
    total = terms["total"]
    summed = sum(terms[name] for name in names)
    return float(abs(summed - total).max()) / float(abs(total).max())


@pytest.mark.parametrize(
    "time_mean", [boxcar_time_mean, block_time_mean, lanczos_time_mean, ideal_time_mean]
)
def test_the_five_pointwise_terms_sum_to_the_layer_flux(moving_surface, time_mean):
    v, mse, dp, _ = moving_surface
    terms = decompose_mass_flux(v, mse, dp, time_mean=time_mean,
                                zonal_mean_terms=False).dropna("time")
    assert _gap(terms, FIVE) < 1e-12


@pytest.mark.parametrize(
    "time_mean", [boxcar_time_mean, block_time_mean, lanczos_time_mean, ideal_time_mean]
)
def test_the_five_zonal_mean_terms_sum_to_the_column_flux(moving_surface, time_mean):
    """Summed over levels and averaged around the circle, five terms suffice:
    there is no sixth term, because the layer mass is inside each term."""
    v, mse, dp, _ = moving_surface
    terms = decompose_mass_flux(v, mse, dp, time_mean=time_mean).dropna("time")
    column = terms.sum("level")
    assert _gap(column, FIVE) < 1e-12


def test_the_cross_terms_vanish_under_a_moving_surface_for_the_ideal_filter(moving_surface):
    """The F24 contrast.  With the surface moving and the wind covarying with
    it, the record mean of the wind split's eddy-wind cross term, column
    integrated with the fluctuating thickness, is not small; the same
    record mean for the split of the layer mass flux is rounding."""
    v, mse, dp, _ = moving_surface
    wind_terms = decompose(v, mse, time_mean=ideal_time_mean, zonal_mean_terms=False)
    wind_cross = col_int(wind_terms["cross_eddy_wind_mean_mse"], dp).mean("time")
    scale = float(abs(col_int(wind_terms["total"], dp).mean("time")).max())
    assert float(abs(wind_cross).max()) > 1e-3 * scale

    mass_terms = decompose_mass_flux(v, mse, dp, time_mean=ideal_time_mean)
    column = mass_terms.sum("level")
    for name in ["cross_mean_wind_eddy_mse", "cross_eddy_wind_mean_mse"]:
        assert float(abs(column[name].mean("time")).max()) < 1e-10 * scale, name


def test_the_mean_circulation_carries_the_record_mean_mass_transport(moving_surface):
    """The sum over levels of the zonal-mean time-mean layer mass flux equals
    the record mean of the zonal-mean column mass transport of the wind,
    for the ideal filter, which keeps the zero frequency exactly."""
    v, mse, dp, _ = moving_surface
    terms = decompose_mass_flux(v, mse, dp, time_mean=ideal_time_mean,
                                zonal_mean_fields=True)
    net = terms["m_bar_zm"].sum("level").mean("time")
    direct = col_int(v, dp).mean(("time", "longitude"))
    scale = float(abs(direct).max())
    assert float(abs(net - direct).max()) < 1e-10 * scale


def test_reduces_to_the_wind_split_when_the_surface_is_flat(grid, moving_surface):
    """With a layer thickness constant in time and longitude, every term is
    the wind split's term times dp over gravity."""
    v, mse, _, _ = moving_surface
    lev = xr.DataArray(grid["level"], dims="level", coords={"level": grid["level"]})
    dp_flat = dp_from_sfc_pressure(lev, xr.DataArray(1005.0)).broadcast_like(v).transpose(*v.dims)
    mass = decompose_mass_flux(v, mse, dp_flat, time_mean=block_time_mean)
    wind = decompose(v, mse, time_mean=block_time_mean)
    weight = dp_flat.isel(time=0, latitude=0, longitude=0) / GRAV_EARTH
    # 1e-10 rather than 1e-12: the stationary term is formed here as the
    # difference of two products near 3e5 times 25 and there as a product of
    # departures, so rounding leaves a few parts in 1e12 of the term.
    for name in FIVE + ["total"]:
        expected = wind[name] * weight
        scale = float(abs(expected).max())
        assert float(abs(mass[name] - expected).max()) < 1e-10 * scale, name
