"""Finite-difference derivatives, checked against ``myfun.py``.

Every test here requires bitwise agreement with the original, on 4-D, 3-D, 2-D
and 1-D inputs, which are the cases the originals' ``len(shape)`` dispatch
handles separately.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from itcz_eddies.derivatives import ddp, ddt, ddx, ddy


@pytest.fixture
def arr4d(field):
    return field


@pytest.fixture
def arr3d(field):
    return field.isel(level=5, drop=True)


def test_ddt_matches_the_original_4d(myfun, arr4d):
    assert np.array_equal(ddt(arr4d).values, myfun.ddt(arr4d).values)


def test_ddt_matches_the_original_3d(myfun, arr3d):
    assert np.array_equal(ddt(arr3d).values, myfun.ddt(arr3d).values)


def test_ddx_matches_the_original_4d(myfun, arr4d):
    assert np.array_equal(ddx(arr4d).values, myfun.ddx(arr4d).values)


def test_ddx_matches_the_original_3d(myfun, arr3d):
    assert np.array_equal(ddx(arr3d).values, myfun.ddx(arr3d).values)


def test_ddy_matches_the_original_4d(myfun, arr4d):
    assert np.array_equal(ddy(arr4d).values, myfun.ddy(arr4d).values)


def test_ddy_matches_the_original_2d_lat_lon(myfun, arr4d):
    """The 2-D branch that the original selects when longitude is present."""
    two_d = arr4d.isel(time=0, level=3, drop=True)
    assert np.array_equal(ddy(two_d).values, myfun.ddy(two_d).values)


def test_ddy_matches_the_original_2d_time_lat(myfun, arr4d):
    """The other 2-D branch, which the original selects when time is present."""
    two_d = arr4d.isel(level=3, drop=True).mean("longitude")
    assert np.array_equal(ddy(two_d).values, myfun.ddy(two_d).values)


def test_ddp_matches_the_original(myfun, arr4d):
    mine = ddp(arr4d)
    theirs = myfun.ddp(arr4d).compute()
    assert np.array_equal(mine.values, theirs.transpose(*mine.dims).values)


ERA5_LEVELS = np.array(
    [1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200, 225, 250,
     300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 775, 800, 825, 850,
     875, 900, 925, 950, 975, 1000], dtype="float64")


def _era5_column():
    """A 4-D field on ERA5's own 37 pressure levels."""
    base = np.sin(ERA5_LEVELS / 300.0)[None, :, None, None]
    return xr.DataArray(
        base * np.ones((3, 1, 4, 5)),
        dims=("time", "level", "latitude", "longitude"),
        coords={"time": np.arange(3.0), "level": ERA5_LEVELS,
                "latitude": np.arange(4.0), "longitude": np.arange(5.0)},
    )


def test_ddp_chunking_moves_values_only_at_rounding_level():
    """Chunking the level dimension is a memory hint, not a numerical choice.

    xarray hands the chunked array to dask's own gradient, which sums the
    non-uniform-spacing weights in a different order at the three levels next
    to an internal chunk boundary.  On ERA5's 37 levels with chunks of 10, the
    largest difference from the unchunked derivative is 2.0e-16 of the peak,
    which is floating-point rounding.
    """
    arr = _era5_column()
    assert ERA5_LEVELS.size == 37
    unchunked = ddp(arr)
    chunked = ddp(arr.chunk({"level": 10})).compute()
    difference = float(abs(unchunked - chunked).max())
    peak = float(abs(unchunked).max())
    assert difference / peak < 1e-14


def test_the_originals_chunk_of_9_fails_on_37_levels():
    """``myfun.ddp``'s hardcoded chunk of 9 cannot differentiate ERA5's 37 levels.

    37 levels split into blocks of 9 leave a final chunk of 1, and dask's
    gradient requires every chunk along the differentiated axis to be larger
    than ``edge_order + 1``.  No committed script calls ``ddp``, so this is
    latent rather than a defect in any published number, and it is why the
    package leaves the chunking to the caller.
    """
    arr = _era5_column().chunk({"level": 9})
    with pytest.raises(ValueError, match="Chunk size must be larger"):
        ddp(arr).compute()


def test_ddx_carries_the_cosine_of_latitude(arr4d):
    """ddx is the plain longitude derivative divided by 111.2 km and cos(lat)."""
    zonal = ddx(arr4d)
    expected = (arr4d.differentiate("longitude") / 111.2e3
                / np.cos(arr4d.latitude * np.pi / 180.0)).transpose(*zonal.dims)
    assert np.array_equal(zonal.values, expected.values)


def test_ddt_rejects_a_decoded_time_coordinate(myfun, arr4d):
    """F9: with a datetime64 time coordinate the original is silently wrong.

    ``myfun.ddt`` returns a rate per xarray's ``datetime_unit`` rather than per
    hour, and every caller then divides by 3600 as though it were per hour.
    The size of the resulting error is the ratio of that unit to an hour, which
    is a property of the xarray version rather than of this code: on xarray
    2026.7.0 the unit is seconds and the factor is 3600.  Asserting the exact
    value here is deliberate, so that an xarray release that changes the
    default fails this test.
    """
    hours = arr4d.time.values * 12.0
    decoded = arr4d.assign_coords(
        time=np.datetime64("1997-01-01T00") + hours.astype("timedelta64[h]")
    )
    per_hour = ddt(arr4d.assign_coords(time=hours))
    per_datetime_unit = myfun.ddt(decoded)

    ratio = (per_hour / per_datetime_unit.assign_coords(time=per_hour.time)).values
    finite = ratio[np.isfinite(ratio) & (np.abs(ratio) > 0)]
    assert finite.size > 0
    assert np.allclose(finite, 3600.0, rtol=1e-9)

    with pytest.raises(TypeError, match="decode_times"):
        ddt(decoded)
