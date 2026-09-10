"""The flux decomposition: what closes, what does not, and why.

The tests are ordered as the argument runs.  The split is exact pointwise; it
survives any reduction that weights longitudes equally; and it fails under a
reduction that weights them unequally, by an amount this file measures.  That
is F11 in ``code-review/FINDINGS.md``, reproduced on synthetic data in a few
seconds rather than on Casper.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from itcz_eddies.columns import col_int, col_int_trapz, dp_from_sfc_pressure, sfc_pressure_mask
from itcz_eddies.decomp import (
    block_time_mean,
    boxcar_time_mean,
    decompose,
    ideal_time_mean,
    lanczos_time_mean,
    lanczos_weights,
    legacy_decompose,
    zonal_mean,
)

FIVE_TERMS = ["mmc", "stationary", "transient",
              "cross_mean_wind_eddy_mse", "cross_eddy_wind_mean_mse"]
SIX_TERMS = FIVE_TERMS + ["zonal_cross"]


@pytest.fixture
def wind_and_mse(grid):
    """A meridional wind and an MSE field with mean, stationary and transient parts."""
    lev = xr.DataArray(grid["level"], dims="level", coords={"level": grid["level"]})
    lat = xr.DataArray(grid["latitude"], dims="latitude",
                       coords={"latitude": grid["latitude"]})
    lon = xr.DataArray(grid["longitude"], dims="longitude",
                       coords={"longitude": grid["longitude"]})
    time = xr.DataArray(np.arange(240.0), dims="time",
                        coords={"time": np.arange(240.0)})
    shape = np.sin(np.pi * lev / 1000.0)

    v = (2.5 * shape * np.cos(np.deg2rad(3 * lat))
         + 1.2 * shape * np.cos(np.deg2rad(2 * lon))
         + 1.5 * shape * np.sin(np.deg2rad(6 * lon) - 2 * np.pi * time / 12.0))
    mse = (3.2e5 + 2.0e4 * shape
           + 3.0e3 * np.sin(np.deg2rad(lat))
           + 1.5e3 * np.sin(np.deg2rad(2 * lon))
           + 2.0e3 * np.cos(np.deg2rad(6 * lon) - 2 * np.pi * time / 12.0))
    order = ("time", "level", "latitude", "longitude")
    return v.transpose(*order), mse.transpose(*order)


def _relative_gap(terms, names):
    total = terms["total"]
    summed = sum(terms[name] for name in names)
    return float(abs(summed - total).max()) / float(abs(total).max())


# ------------------------------------------------------- the split is exact


@pytest.mark.parametrize(
    "time_mean", [boxcar_time_mean, block_time_mean, lanczos_time_mean,
                  ideal_time_mean]
)
def test_the_six_pointwise_terms_sum_to_the_flux(wind_and_mse, time_mean):
    """At every gridpoint, for any time-mean operator."""
    v, mse = wind_and_mse
    terms = decompose(v, mse, time_mean=time_mean, zonal_mean_terms=False).dropna("time")
    assert _relative_gap(terms, SIX_TERMS) < 1e-12


@pytest.mark.parametrize(
    "time_mean", [boxcar_time_mean, block_time_mean, lanczos_time_mean,
                  ideal_time_mean]
)
def test_the_five_zonal_mean_terms_sum_to_the_zonal_mean_flux(wind_and_mse, time_mean):
    """After the zonal mean, the sixth term vanishes and five suffice."""
    v, mse = wind_and_mse
    terms = decompose(v, mse, time_mean=time_mean).dropna("time")
    assert _relative_gap(terms, FIVE_TERMS) < 1e-12


def test_the_cross_terms_vanish_in_the_time_mean_for_a_block_average(wind_and_mse):
    """A block average makes the two cross terms zero once time-averaged.

    Within one block the time mean is constant, so the block average of
    ``v_bar * h_prime`` is ``v_bar`` times the block average of ``h_prime``,
    which is zero by construction.  The three-way split therefore closes in
    the time mean, which is the quantity every figure in the manuscript plots.

    At an individual time the cross terms are not small: on this field the
    larger of the two reaches 9.5e-2 of the peak flux.  That is why the
    published three-term split does not close timestep by timestep.
    """
    v, mse = wind_and_mse
    terms = decompose(v, mse, time_mean=block_time_mean)
    scale = float(abs(terms["total"]).max())

    instantaneous = max(float(abs(terms[name]).max()) for name in
                        ["cross_mean_wind_eddy_mse", "cross_eddy_wind_mean_mse"])
    assert instantaneous / scale > 1e-3

    for name in ["cross_mean_wind_eddy_mse", "cross_eddy_wind_mean_mse"]:
        assert float(abs(block_time_mean(terms[name])).max()) < 1e-12 * scale

    three = terms["mmc"] + terms["stationary"] + terms["transient"]
    assert float(abs(block_time_mean(three - terms["total"])).max()) < 1e-12 * scale


def test_the_ideal_low_pass_is_a_projection_and_annihilates_the_cross_terms(
    wind_and_mse
):
    """The spectral low-pass applied twice is the low-pass applied once, and
    the two cross terms of the exact split vanish in the mean over the
    record.

    The Lanczos, for comparison, is only nearly a projection: on the same
    field its second application moves the result by more than the rounding
    floor, and its record-mean cross terms are nonzero.  Both halves are
    asserted so the contrast is measured rather than described.
    """
    v, mse = wind_and_mse
    once = ideal_time_mean(v)
    twice = ideal_time_mean(once)
    scale = float(abs(once).max())
    assert float(abs(twice - once).max()) < 1e-12 * scale

    terms = decompose(v, mse, time_mean=ideal_time_mean)
    flux_scale = float(abs(terms["total"]).max())
    for name in ["cross_mean_wind_eddy_mse", "cross_eddy_wind_mean_mse"]:
        record_mean = float(abs(terms[name].mean("time")).max())
        assert record_mean < 1e-12 * flux_scale, name

    # A 41-weight Lanczos, so that two applications leave 160 of the 240
    # time steps defined; the default 121 weights would leave none.
    lanczos_once = lanczos_time_mean(v, lobes=20)
    lanczos_twice = lanczos_time_mean(lanczos_once, lobes=20).dropna("time")
    assert lanczos_twice.sizes["time"] == 160
    moved = float(abs(lanczos_twice - lanczos_once.sel(time=lanczos_twice["time"])).max())
    assert moved > 1e-6 * scale
    lanczos_terms = decompose(v, mse, time_mean=lanczos_time_mean,
                              lobes=20).dropna("time")
    lanczos_cross = max(
        float(abs(lanczos_terms[name].mean("time")).max())
        for name in ["cross_mean_wind_eddy_mse", "cross_eddy_wind_mean_mse"])
    assert lanczos_cross > 1e-6 * flux_scale


def test_the_ideal_low_pass_keeps_a_slow_wave_and_removes_a_fast_one(grid):
    """A 90-day wave passes untouched and a 6-day wave is removed entirely.

    The time axis is 12-hourly over 360 days, so both periods are whole
    numbers of cycles in the record and neither sits at the cutoff.
    """
    time = xr.DataArray(np.arange(720.0), dims="time",
                        coords={"time": np.arange(720.0)})
    lat = xr.DataArray(grid["latitude"], dims="latitude",
                       coords={"latitude": grid["latitude"]})
    slow = np.cos(2 * np.pi * time / (2 * 90.0)) * (1.0 + lat / 100.0)
    fast = np.sin(2 * np.pi * time / (2 * 6.0)) * (2.0 - lat / 100.0)
    field = (slow + fast).transpose("time", "latitude")
    filtered = ideal_time_mean(field, temporal_resolution=12, cutoff_days=30)
    xr.testing.assert_allclose(filtered, slow.transpose("time", "latitude"),
                               atol=1e-10)


def test_a_planted_decomposition_is_recovered():
    """Build the three parts separately, then check they come back.

    A full longitude circle, so the zonal mean of each planted wave is zero;
    a block time mean; and a transient part that alternates sign every step,
    so its average over any even-length block is exactly zero.  Each planted
    piece then lands in exactly one term of the split.
    """
    levels = np.array([50.0, 100, 200, 300, 400, 500, 600, 700, 800, 900,
                       950, 1000])
    lev = xr.DataArray(levels, dims="level", coords={"level": levels})
    lats = np.linspace(-8.0, 8.0, 9)
    lat = xr.DataArray(lats, dims="latitude", coords={"latitude": lats})
    lons = np.arange(0.0, 360.0, 15.0)
    lon = xr.DataArray(lons, dims="longitude", coords={"longitude": lons})
    n_times = 120
    window = int(30 * 24 / 12)          # 60 steps in a 30-day block
    times = np.arange(float(n_times))
    time = xr.DataArray(times, dims="time", coords={"time": times})

    flip = xr.DataArray((-1.0) ** np.arange(n_times), dims="time",
                        coords={"time": times})
    shape = np.sin(np.pi * lev / 1000.0)

    v_mean = 2.0 * shape * np.cos(np.deg2rad(3 * lat))
    v_stat = 1.0 * shape * np.cos(np.deg2rad(2 * lon))
    v_trans = 0.7 * shape * flip * np.sin(np.deg2rad(4 * lon))
    h_mean = 3.0e5 + 1.0e4 * shape
    h_stat = 2.0e3 * np.sin(np.deg2rad(2 * lon))
    h_trans = 1.0e3 * flip * np.cos(np.deg2rad(4 * lon))

    order = ("time", "level", "latitude", "longitude")
    template = time * lev * lat * lon
    v = (v_mean + v_stat + v_trans).broadcast_like(template).transpose(*order)
    mse = (h_mean + h_stat + h_trans).broadcast_like(template).transpose(*order)

    terms = decompose(v, mse, time_mean=block_time_mean, window_days=30)
    assert window == 60

    expected_mmc = (v_mean * h_mean).broadcast_like(terms["mmc"])
    expected_stat = (v_stat * h_stat).mean("longitude").broadcast_like(
        terms["stationary"])
    expected_trans = (v_trans * h_trans).mean("longitude").broadcast_like(
        terms["transient"])

    scale = float(abs(terms["total"]).max())
    assert float(abs(terms["mmc"] - expected_mmc).max()) < 1e-12 * scale
    assert float(abs(terms["stationary"] - expected_stat).max()) < 1e-12 * scale
    assert float(abs(terms["transient"] - expected_trans).max()) < 1e-12 * scale


# ------------------------------------------------- the split under reduction


def test_the_split_survives_a_reduction_that_weights_longitudes_equally(
    grid, wind_and_mse
):
    """No terrain: the layer thicknesses are the same at every longitude.

    The column integral is then a linear operator with longitude-independent
    weights, so it commutes with the zonal mean and the five terms still add
    up to the total after it.
    """
    v, mse = wind_and_mse
    level = xr.DataArray(grid["level"], dims="level",
                         coords={"level": grid["level"]})
    p_sfc_flat = xr.DataArray(1005.0)
    dp = dp_from_sfc_pressure(level, p_sfc_flat)

    terms = decompose(v, mse).dropna("time")
    integrated = {name: col_int(terms[name], dp) for name in FIVE_TERMS + ["total"]}
    summed = sum(integrated[name] for name in FIVE_TERMS)
    gap = float(abs(summed - integrated["total"]).max())
    assert gap / float(abs(integrated["total"]).max()) < 1e-12


def test_the_split_fails_under_a_longitude_dependent_column_integral(
    grid, wind_and_mse, p_sfc
):
    """With terrain, the five zonal-mean terms no longer add up, by a measured gap.

    The reduction here is the production one: integrate each longitude down to
    its own surface pressure, then average around the latitude circle.  Its
    weights vary with longitude, so it does not annihilate the sixth term of
    the pointwise split, and it does not commute with the zonal means inside
    the stationary and transient terms either.

    The gap is asserted to be present rather than to have a particular size:
    its size depends on how deeply the terrain cuts the levels, which on the
    real archive is what F11 measures as up to 38% of the peak flux.
    """
    v, mse = wind_and_mse
    level = xr.DataArray(grid["level"], dims="level",
                         coords={"level": grid["level"]})
    p_sfc_time_mean = p_sfc.isel(time=0, drop=True)
    dp = dp_from_sfc_pressure(level, p_sfc_time_mean)

    pointwise = decompose(v, mse, zonal_mean_terms=False).dropna("time")
    reduced = {name: col_int(pointwise[name], dp).mean("longitude")
               for name in SIX_TERMS + ["total"]}

    peak = float(abs(reduced["total"]).max())

    # Six pointwise terms still add up: the reduction is linear.
    six = sum(reduced[name] for name in SIX_TERMS)
    assert float(abs(six - reduced["total"]).max()) / peak < 1e-12

    # Five of them do not, and the shortfall is the sixth term.
    five = sum(reduced[name] for name in FIVE_TERMS)
    shortfall = float(abs(five - reduced["total"]).max())
    assert shortfall / peak > 1e-4
    assert float(abs((five + reduced["zonal_cross"]) - reduced["total"]).max()) / peak < 1e-12


# -------------------------------------------------------- the legacy split


def test_legacy_decompose_matches_the_published_script(
    myfun, decomp_script, grid, wind_and_mse, p_sfc
):
    """Bitwise agreement with lines 318 to 339 of the decomposition script.

    The reference below is transcribed from those lines, and it calls the
    script's own ``running_mean``, ``maskout`` and ``nantrapz`` rather than
    reimplementing them.
    """
    v, mse = wind_and_mse
    p_sfc_long = p_sfc.reindex(time=v.time, method="nearest")
    mask = sfc_pressure_mask(mse, p_sfc_long, below_ground=np.nan)
    temporal_resolution = 12

    # --- reference, transcribed from the script ---
    v_adj = xr.where(mask == 1, v, np.nan)
    mse_ref = xr.where(mask == 1, mse, np.nan)
    v_time_mean = decomp_script.running_mean(v_adj, temporal_resolution)
    v_mean = v_time_mean.mean("longitude", skipna=True)
    v_mean = xr.where(mask == 1, v_mean, np.nan)
    v_mean = v_mean - myfun.nantrapz(v_mean, v_mean.level, dim="level") / p_sfc_long
    mse_time_mean = decomp_script.running_mean(mse_ref, temporal_resolution)
    mse_mean = mse_time_mean.mean("longitude", skipna=True)
    v_time_ano = v_adj - v_time_mean
    v_ano = v_time_ano - v_time_ano.mean("longitude", skipna=True)
    mse_time_ano = mse_ref - mse_time_mean
    mse_ano = mse_time_ano - mse_time_ano.mean("longitude", skipna=True)
    reference = {
        "mmc": (v_mean * mse_mean).broadcast_like(mse_ref),
        "stationary": ((v_time_mean - v_mean) * (mse_time_mean - mse_mean)
                       ).mean("longitude", skipna=True).broadcast_like(mse_ref),
        "transient": v_ano * mse_ano,
    }
    # --- end reference ---

    mine = legacy_decompose(v, mse, mask, p_sfc_long,
                            temporal_resolution=temporal_resolution)
    for name, expected in reference.items():
        got = mine[name].transpose(*expected.dims)
        assert np.array_equal(got.values, expected.values, equal_nan=True), name


def test_the_published_split_does_not_close(grid, wind_and_mse, p_sfc):
    """The three published terms, reduced as published, leave a gap.

    A regression guard on the legacy path rather than an aspiration: it fixes
    the current behavior so that a change to it is visible.
    """
    v, mse = wind_and_mse
    p_sfc_long = p_sfc.reindex(time=v.time, method="nearest")
    mask = sfc_pressure_mask(mse, p_sfc_long, below_ground=np.nan)
    terms = legacy_decompose(v, mse, mask, p_sfc_long)

    total = col_int_trapz(xr.where(mask == 1, v, np.nan)
                          * xr.where(mask == 1, mse, np.nan))
    summed = sum(col_int_trapz(terms[name], mask) for name in
                 ["mmc", "stationary", "transient"])
    gap = float(abs(summed.mean("longitude", skipna=True)
                    - total.mean("longitude", skipna=True)).max())
    peak = float(abs(total.mean("longitude", skipna=True)).max())
    assert gap / peak > 1e-3


# ------------------------------------------- the mass-weighted zonal mean


def test_the_mass_weighted_zonal_mean_closes_the_five_term_split(
    grid, wind_and_mse, p_sfc
):
    """Weighting the zonal mean by layer mass restores closure under terrain.

    Same reduction as the failing test above, integrating each longitude to its
    own surface pressure and then averaging around the latitude circle.  The
    only change is that the zonal means inside the split are weighted by the
    layer thicknesses, so the departure from the mean has zero mass-weighted
    zonal integral at every level.  This is F16 in
    ``code-review/FINDINGS.md``.
    """
    v, mse = wind_and_mse
    level = xr.DataArray(grid["level"], dims="level",
                         coords={"level": grid["level"]})
    p_sfc_time_mean = p_sfc.isel(time=0, drop=True)
    dp = dp_from_sfc_pressure(level, p_sfc_time_mean)

    unweighted = decompose(v, mse, zonal_mean_terms=False).dropna("time")
    weighted = decompose(v, mse, zonal_mean_terms=False,
                         weights=dp).dropna("time")

    def five_term_gap(terms):
        reduced = {name: col_int(terms[name], dp).mean("longitude")
                   for name in FIVE_TERMS + ["total"]}
        five = sum(reduced[name] for name in FIVE_TERMS)
        return (float(abs(five - reduced["total"]).max())
                / float(abs(reduced["total"]).max()))

    assert five_term_gap(unweighted) > 1e-4
    assert five_term_gap(weighted) < 1e-12


def test_the_weighting_leaves_the_transient_and_cross_terms_alone(
    grid, wind_and_mse, p_sfc
):
    """Only the terms built from a zonal mean of a time-mean field can move.

    The transient term is the product of two departures from the time mean and
    the two cross terms pair a time mean with a departure from it, so none of
    the three involves a zonal mean and none can change when the zonal mean is
    reweighted.  Asserting that is what makes the mean-circulation and
    stationary-eddy changes attributable to the reweighting alone.
    """
    v, mse = wind_and_mse
    level = xr.DataArray(grid["level"], dims="level",
                         coords={"level": grid["level"]})
    dp = dp_from_sfc_pressure(level, p_sfc.isel(time=0, drop=True))

    unweighted = decompose(v, mse, zonal_mean_terms=False).dropna("time")
    weighted = decompose(v, mse, zonal_mean_terms=False, weights=dp).dropna("time")

    for name in ["transient", "cross_mean_wind_eddy_mse",
                 "cross_eddy_wind_mean_mse", "total"]:
        assert np.array_equal(unweighted[name].values, weighted[name].values), name

    moved = float(abs(weighted["mmc"] - unweighted["mmc"]).max())
    assert moved > 0.0


def test_zonal_mean_helper_reduces_to_the_arithmetic_mean(wind_and_mse):
    """With uniform weights the weighted mean is the plain one."""
    v, _ = wind_and_mse
    plain = zonal_mean(v)
    uniform = zonal_mean(v, weights=xr.ones_like(v))
    xr.testing.assert_allclose(plain, uniform, rtol=1e-14)


def test_indicator_weights_reproduce_the_published_skipna_mean(
    grid, wind_and_mse, p_sfc
):
    """Weighting by the above-ground indicator is the published zonal mean.

    The published code masks below-ground points to NaN and averages with
    ``skipna=True``, so every above-ground point counts once and the
    denominator is the number of them.  A weighted mean with weights of one
    above ground and zero below is the same number, without a NaN entering
    the calculation.  This is how ``scripts/decomposition.py`` realizes
    ``--zonal-mean plain`` in the exact split.
    """
    v, _ = wind_and_mse
    level = xr.DataArray(grid["level"], dims="level",
                         coords={"level": grid["level"]})
    p_sfc_time_mean = p_sfc.isel(time=0, drop=True)
    above = (level <= p_sfc_time_mean).transpose("level", "latitude", "longitude")
    assert 0.0 < float(above.mean()) < 1.0, "the ridge must put some points below ground"

    published = v.where(above).mean("longitude", skipna=True)
    indicator = zonal_mean(v, weights=xr.where(above, 1.0, 0.0))
    xr.testing.assert_allclose(indicator, published, rtol=1e-13)


def test_zonal_mean_fields_are_the_weighted_means_of_the_time_means(
    grid, wind_and_mse, p_sfc
):
    """``zonal_mean_fields=True`` returns the factors of the MMC term.

    Checked in both output forms and under both weightings: the returned
    fields are the weighted zonal means of the time-mean wind and MSE, their
    product is the ``mmc`` term, and the other terms are unchanged.
    """
    v, mse = wind_and_mse
    level = xr.DataArray(grid["level"], dims="level",
                         coords={"level": grid["level"]})
    dp = dp_from_sfc_pressure(level, p_sfc.isel(time=0, drop=True))
    for weights in (None, dp):
        for zonal_mean_terms in (True, False):
            without = decompose(v, mse, zonal_mean_terms=zonal_mean_terms,
                                weights=weights)
            with_fields = decompose(v, mse, zonal_mean_terms=zonal_mean_terms,
                                    weights=weights, zonal_mean_fields=True)
            assert set(with_fields.data_vars) == (
                set(without.data_vars) | {"v_bar_zm", "h_bar_zm"})
            for name in without.data_vars:
                assert np.array_equal(with_fields[name].values,
                                      without[name].values, equal_nan=True), name
            v_bar_zm = zonal_mean(boxcar_time_mean(v), weights=weights)
            h_bar_zm = zonal_mean(boxcar_time_mean(mse), weights=weights)
            xr.testing.assert_equal(with_fields["v_bar_zm"], v_bar_zm)
            xr.testing.assert_equal(with_fields["h_bar_zm"], h_bar_zm)
            assert "longitude" not in with_fields["v_bar_zm"].dims
            product = (with_fields["v_bar_zm"] * with_fields["h_bar_zm"])
            xr.testing.assert_allclose(
                product.broadcast_like(with_fields["mmc"]).transpose(
                    *with_fields["mmc"].dims),
                with_fields["mmc"], rtol=1e-14)


# ------------------------------------------------ the Lanczos filter by FFT


def test_the_fft_lanczos_matches_the_rolling_dot_product():
    """The FFT convolution of 2026-09-09 gives what the strided dot product
    gave, to rounding, with the same NaN ends, on a record with a trend,
    an annual cycle, fast noise and more than one dimension."""
    from itcz_eddies.decomp import lanczos_time_mean_rolling

    rng = np.random.default_rng(7)
    n_times = 500
    time = np.arange(n_times, dtype="float64")
    signal = (0.01 * time[:, None, None]
              + np.sin(2 * np.pi * time[:, None, None] / 730.0)
              + rng.standard_normal((n_times, 3, 4)))
    arr = xr.DataArray(signal, dims=("time", "level", "latitude"),
                       coords={"time": time, "level": [1.0, 2.0, 3.0],
                               "latitude": [10.0, 5.0, 0.0, -5.0]})
    for lobes in (20, 60):
        fft = lanczos_time_mean(arr, lobes=lobes)
        rolling = lanczos_time_mean_rolling(arr, lobes=lobes)
        assert fft.dims == rolling.dims and fft.shape == rolling.shape
        nan_fft = fft.isnull().all(("level", "latitude")).values
        nan_rolling = rolling.isnull().all(("level", "latitude")).values
        np.testing.assert_array_equal(nan_fft, nan_rolling)
        assert nan_fft[:lobes].all() and nan_fft[-lobes:].all() and not nan_fft[lobes:-lobes].any()
        interior = slice(lobes, n_times - lobes)
        np.testing.assert_allclose(fft.values[interior], rolling.values[interior],
                                   rtol=0, atol=1e-12)
    # the time axis need not come first
    transposed = arr.transpose("level", "time", "latitude")
    fft_t = lanczos_time_mean(transposed, lobes=20)
    np.testing.assert_allclose(fft_t.transpose("time", "level", "latitude").values[20:-20],
                               lanczos_time_mean(arr, lobes=20).values[20:-20], rtol=0, atol=1e-12)
def _lanczos_response(weights, period_days, temporal_resolution):
    """Amplitude the weights retain at a sinusoid of the given period."""
    lobes = (weights.size - 1) // 2
    n = np.arange(-lobes, lobes + 1)
    freq = 1.0 / (period_days * 24.0 / temporal_resolution)  # cycles per sample
    return float((weights * np.exp(-2j * np.pi * freq * n)).real.sum())


@pytest.mark.parametrize("temporal_resolution,lobes", [(6.0, 240), (12.0, 120)])
def test_the_lanczos_low_pass_has_its_designed_response(temporal_resolution, lobes):
    """The two windows the pipeline uses keep the long periods, halve the
    cutoff and reject the short periods.

    Both are the 60-day half window of decision D4 at their own sampling: 481
    weights at four samples a day, 241 at two.  The bounds here are properties
    of a windowed ideal low-pass and hold for any cutoff, so they survive a
    change of the cutoff itself.
    """
    weights = lanczos_weights(30.0, temporal_resolution=temporal_resolution, lobes=lobes)
    assert weights.size == 2 * lobes + 1
    np.testing.assert_allclose(weights.sum(), 1.0, rtol=1e-12, atol=0.0)

    def response(period_days):
        return _lanczos_response(weights, period_days, temporal_resolution)

    # passband: periods at or above twice the cutoff come through, to 0.01
    for period in (60.0, 120.0, 240.0):
        assert abs(response(period) - 1.0) < 0.01, period
    # the cutoff itself is halved, which is what "30-day cutoff" means
    assert abs(response(30.0) - 0.5) < 0.01
    # stopband: periods at or below three quarters of the cutoff are gone
    for period in (22.5, 15.0, 7.5, 2.0):
        assert abs(response(period)) < 0.01, period


def test_the_lanczos_response_is_set_by_the_window_in_days_not_the_sampling():
    """The 481-weight window at four samples a day and the 241-weight window
    at two are the same 120-day window, so they have the same response at
    every physical period.  A units error in the sampling would break this
    and nothing else in the suite would see it."""
    six_hourly = lanczos_weights(30.0, temporal_resolution=6.0, lobes=240)
    twelve_hourly = lanczos_weights(30.0, temporal_resolution=12.0, lobes=120)
    for period in (240.0, 120.0, 60.0, 40.0, 30.0, 22.5, 15.0, 7.5):
        np.testing.assert_allclose(
            _lanczos_response(six_hourly, period, 6.0),
            _lanczos_response(twelve_hourly, period, 12.0),
            # the response is an O(1) amplitude and passes through zero in
            # the stopband, so the tolerance is absolute: the two weight
            # arrays differ only in their floating-point path.
            rtol=0.0, atol=1e-7, err_msg=f"period {period} d")
