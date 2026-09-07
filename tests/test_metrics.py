"""ITCZ position metrics and the flux-to-transport conversion."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from itcz_eddies.metrics import (
    RAD_EARTH,
    cross_equatorial_transport,
    energy_flux_equator,
    flux_to_petawatts,
    precip_centroid,
)


@pytest.fixture
def precip_grid():
    """Daily precipitation on a half-degree grid from 30S to 30N."""
    lat = np.arange(-30.0, 30.01, 0.5)
    lon = np.arange(0.0, 90.0, 0.5)
    time = np.arange(4.0)
    return lat, lon, time


def _precip(lat, lon, time, center):
    """A Gaussian rain band centered at ``center`` degrees, uniform in longitude."""
    profile = np.exp(-((lat - center) / 4.0) ** 2)
    values = np.tile(profile[None, :, None], (time.size, 1, lon.size))
    return xr.DataArray(
        values, dims=("time", "latitude", "longitude"),
        coords={"time": time, "latitude": lat, "longitude": lon},
    )


def test_a_symmetric_rain_band_has_a_centroid_at_the_equator(precip_grid):
    lat, lon, time = precip_grid
    centroid = precip_centroid(_precip(lat, lon, time, center=0.0))
    assert float(abs(centroid).max()) < 0.5


@pytest.mark.parametrize("center", [-6.0, -2.5, 0.0, 3.0, 7.5])
def test_the_centroid_follows_a_shifted_rain_band(precip_grid, center):
    """The centroid lands within one grid spacing of the planted center.

    Half a degree is the grid spacing, and the metric returns a grid latitude,
    so one spacing is the tightest bound available to it.  The cosine weighting
    biases the answer slightly toward the equator, which is why the tolerance
    is one spacing rather than half of one.
    """
    lat, lon, time = precip_grid
    centroid = precip_centroid(_precip(lat, lon, time, center=center))
    assert float(abs(centroid - center).max()) <= 0.5


def test_the_centroid_is_quantized_to_the_grid(precip_grid):
    """Every value the metric returns is a latitude that exists on the grid."""
    lat, lon, time = precip_grid
    centroid = precip_centroid(_precip(lat, lon, time, center=3.7))
    assert np.isin(centroid.values, lat).all()


@pytest.fixture
def flux_profile():
    """A zonal-mean flux crossing zero at 3.4 degrees, between grid latitudes.

    3.4 rather than 3.25, so that the two neighboring grid latitudes are not
    equidistant from the crossing and ``idxmin`` has no tie to break.
    """
    lat = np.arange(-20.0, 20.01, 0.5)
    return xr.DataArray(2.0 * (lat - 3.4), dims="latitude",
                        coords={"latitude": lat})


def test_the_energy_flux_equator_is_quantized_without_interpolation(flux_profile):
    """The published definition returns the nearest grid latitude, 3.5 here."""
    assert float(energy_flux_equator(flux_profile)) == pytest.approx(3.5)



def test_interpolating_the_energy_flux_equator_finds_the_true_crossing(
    flux_profile,
):
    """The planted zero crossing at 3.4 comes back exactly.

    The gap between this and the previous test, 0.1 degrees on a half-degree
    grid, is what the quantization costs on this linear profile.  On a profile
    that is flat near its crossing it costs more, up to half a grid spacing.
    """
    assert float(energy_flux_equator(flux_profile, interpolate=True)) == (
        pytest.approx(3.4)
    )


def test_flux_to_petawatts_uses_the_latitude_circle(flux_profile):
    """W/m times the circumference at that latitude, in petawatts."""
    transport = flux_to_petawatts(flux_profile)
    expected = (flux_profile * 2 * np.pi * RAD_EARTH
                * np.cos(np.deg2rad(flux_profile.latitude)) * 1e-15)
    xr.testing.assert_allclose(transport, expected, rtol=1e-15)
    # At the equator the circle is 2*pi*a = 40030 km.
    circumference_km = 2 * np.pi * RAD_EARTH / 1000.0
    assert circumference_km == pytest.approx(40030.0, abs=1.0)


def test_cross_equatorial_transport_averages_the_named_band(precip_grid):
    lat, lon, time = precip_grid
    field = _precip(lat, lon, time, center=0.0)
    got = cross_equatorial_transport(field, band=(-5.0, 5.0))
    expected = field.sel(latitude=slice(-5.0, 5.0)).mean(
        dim=["latitude", "longitude"])
    xr.testing.assert_allclose(got, expected, rtol=1e-15)
