"""The dry-air mass correction and the polar-cap transport it must reproduce."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from itcz_eddies.columns import col_int, dp_from_sfc_pressure
from itcz_eddies.mass import centered_tendency, dry_air_mass_correction, polar_cap_transport

pytest.importorskip("windspharm")

LEVELS = np.array([100.0, 300.0, 500.0, 700.0, 850.0, 925.0, 1000.0])


@pytest.fixture
def global_fields():
    """A global 33 by 64 grid, latitude descending, four times, with a wind
    whose column dry-mass divergence is nonzero and a tendency of zero."""
    lat = xr.DataArray(np.linspace(90.0, -90.0, 33), dims="latitude")
    lat = lat.assign_coords(latitude=lat)
    lon = xr.DataArray(np.arange(0.0, 360.0, 360.0 / 64), dims="longitude")
    lon = lon.assign_coords(longitude=lon)
    lev = xr.DataArray(LEVELS, dims="level", coords={"level": LEVELS})
    time = xr.DataArray(np.arange(4.0), dims="time", coords={"time": np.arange(4.0)})
    order = ("time", "level", "latitude", "longitude")
    template = time * lev * lat * lon
    shape = np.sin(np.pi * lev / 1000.0)
    u = (8.0 * shape * np.cos(np.deg2rad(lat)) * np.sin(np.deg2rad(2 * lon) + 0.3 * time)
         ).broadcast_like(template).transpose(*order)
    v = (3.0 * shape * np.cos(np.deg2rad(lat)) ** 2 * np.cos(np.deg2rad(2 * lon))
         + 0.5 * np.cos(np.deg2rad(lat)) ** 3   # a barotropic divergent part
         ).broadcast_like(template).transpose(*order)
    q = (0.015 * (lev / 1000.0) ** 3 * np.cos(np.deg2rad(lat)) ** 2
         ).broadcast_like(template).transpose(*order)
    p_sfc = (1000.0 + 5.0 * np.cos(np.deg2rad(lat))).broadcast_like(time * lat * lon)
    p_sfc = p_sfc.transpose("time", "latitude", "longitude")
    dp = dp_from_sfc_pressure(lev, p_sfc).transpose(*order)
    return u, v, q, dp


def _divergence(u_col, v_col):
    import windspharm.xarray
    return windspharm.xarray.VectorWind(u_col, v_col).divergence()


def test_the_correction_closes_the_dry_mass_budget(global_fields):
    u, v, q, dp = global_fields
    tendency = xr.zeros_like(col_int(q, dp))
    du, dv, mass = dry_air_mass_correction(u, v, q, dp, tendency)
    before = _divergence(col_int(u * (1 - q), dp), col_int(v * (1 - q), dp)) + tendency
    after = _divergence(col_int((u + du) * (1 - q), dp), col_int((v + dv) * (1 - q), dp)) + tendency
    rms_before = float(np.sqrt((before ** 2).mean()))
    rms_after = float(np.sqrt((after ** 2).mean()))
    assert rms_before > 1e-5
    assert rms_after < 1e-3 * rms_before
    # the correction is two-dimensional (barotropic) and of the order of the
    # column-mean divergent wind planted, which is a few meters per second
    assert du.dims == ("time", "latitude", "longitude")
    assert 0.1 < float(abs(dv).max()) < 5.0


def test_the_corrected_zonal_mean_mass_transport_is_the_cap_requirement(global_fields):
    """After the correction, the zonal-mean column dry-mass transport across
    each latitude equals the rate of change of the dry mass north of it,
    which is zero here, so the corrected transport is zero to rounding."""
    u, v, q, dp = global_fields
    tendency = xr.zeros_like(col_int(q, dp))
    du, dv, mass = dry_air_mass_correction(u, v, q, dp, tendency)
    v_col_after = col_int((v + dv) * (1 - q), dp)
    zonal = v_col_after.mean(("time", "longitude"))
    interior = zonal.sel(latitude=slice(80.0, -80.0))
    required = polar_cap_transport(tendency.mean(("time", "longitude")))
    scale = float(abs(col_int(v * (1 - q), dp).mean(("time", "longitude"))).max())
    assert float(abs(interior - required.sel(latitude=slice(80.0, -80.0))).max()) < 1e-3 * scale


def test_polar_cap_transport_of_a_uniform_source():
    """A uniform source S over the whole sphere requires a transport
    a S (1 - sin phi) / cos phi northward across latitude phi, with the
    latitude either way round.  The grid includes the poles, as ERA5's
    does, since the integral starts at the first latitude; the pole itself
    is excluded from the comparison because cos phi vanishes there."""
    lat = np.linspace(90.0, -90.0, 181)
    src = xr.DataArray(np.full(lat.size, 2.0e-5), dims="latitude", coords={"latitude": lat})
    phi = np.deg2rad(lat[1:-1])
    expected = 6.371e6 * 2.0e-5 * (1.0 - np.sin(phi)) / np.cos(phi)
    got = polar_cap_transport(src)
    assert np.allclose(got.values[1:-1], expected, rtol=1e-3)
    got_flipped = polar_cap_transport(src.isel(latitude=slice(None, None, -1)))
    assert np.allclose(got_flipped.values[1:-1], expected[::-1], rtol=1e-3)


def test_centered_tendency_of_a_linear_ramp():
    times = np.arange("2000-01-01T00", "2000-01-03T00", np.timedelta64(1, "h"), dtype="datetime64[ns]")
    hourly = xr.DataArray(np.arange(times.size, dtype=float) * 3.0, dims="time", coords={"time": times})
    at = times[[12, 24]]
    tend = centered_tendency(hourly, at, half_window_hours=1)
    assert np.allclose(tend.values, 3.0 / 3600.0)
    tend12 = centered_tendency(hourly, at, half_window_hours=12)
    assert np.allclose(tend12.values, 3.0 / 3600.0)
