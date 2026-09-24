"""The corrected assembly on synthetic fields: what closes and what is stored."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from itcz_eddies.assembly import (
    COLUMN_INTEGRALS,
    EP_ENTHALPY_T_REF,
    SERIES,
    closure_summary,
    column_integrals,
    corrected_fields,
    d3_energy_curve,
    energy_flux_curve,
)
from itcz_eddies.columns import col_int, dp_from_sfc_pressure
from itcz_eddies.mass import RAD_EARTH, mass_correction_from_columns
from itcz_eddies.mse import C_P_SCRIPTS, budget_residual, moist_static_energy

pytest.importorskip("windspharm")

LEVELS = np.array([100.0, 300.0, 500.0, 700.0, 850.0, 925.0, 1000.0])
ORDER = ("time", "level", "latitude", "longitude")


@pytest.fixture
def synthetic():
    """A global 33 by 64 grid, latitude descending, seven levels, four analysis
    times six hours apart, with a wind whose column dry-mass divergence does
    not match the prescribed tendency and an energy budget that does not
    close, so that both corrections have work to do."""
    lat = xr.DataArray(np.linspace(90.0, -90.0, 33), dims="latitude")
    lat = lat.assign_coords(latitude=lat)
    lon = xr.DataArray(np.arange(0.0, 360.0, 360.0 / 64), dims="longitude")
    lon = lon.assign_coords(longitude=lon)
    lev = xr.DataArray(LEVELS, dims="level", coords={"level": LEVELS})
    times = np.datetime64("1997-07-01T00") + np.arange(4) * np.timedelta64(6, "h")
    step = xr.DataArray(np.arange(4.0), dims="time", coords={"time": times})
    template = xr.DataArray(np.zeros((4, LEVELS.size, 33, 64)), dims=ORDER,
                            coords={"time": times, "level": LEVELS,
                                    "latitude": lat.values, "longitude": lon.values})
    phi = np.deg2rad(lat)
    lam = np.deg2rad(lon)
    shape = np.sin(np.pi * lev / 1000.0)

    def field(expr):
        return (expr + 0.0 * template).transpose(*ORDER)

    u = field(8.0 * shape * np.cos(phi) * np.sin(2 * lam + 0.3 * step))
    v = field(3.0 * shape * np.cos(phi) ** 2 * np.cos(2 * lam)
              + 0.5 * np.cos(phi) ** 3 + 0.2 * np.cos(phi) * np.sin(lam + 0.2 * step))
    q = field(0.015 * (lev / 1000.0) ** 3 * np.cos(phi) ** 2)
    temp = field(250.0 + 50.0 * lev / 1000.0 - 20.0 * np.sin(phi) ** 2)
    geopot = field(9.81 * 8000.0 * np.log(1000.0 / lev))
    p_sfc = (1000.0 + 5.0 * np.cos(phi) + 0.5 * np.cos(2 * lam) * step
             + 0.0 * template.isel(level=0)).transpose("time", "latitude", "longitude")
    two_d = 0.0 * template.isel(level=0, drop=True)
    dry_mass_tend_2h = (1e-3 * np.sin(phi) * (1.0 + 0.2 * np.cos(2 * lam)) + two_d
                        ).transpose("time", "latitude", "longitude")
    dry_mass_tend_24h = 0.5 * dry_mass_tend_2h
    energy_tend_2h = (20.0 * np.cos(phi) * np.cos(lam + 0.5 * step) + two_d
                      ).transpose("time", "latitude", "longitude")
    f_net = (50.0 * np.cos(phi) ** 2 * (1.0 + 0.3 * np.sin(lam)) - 30.0 * np.sin(phi) ** 2
             + two_d).transpose("time", "latitude", "longitude")
    return dict(temp=temp, sphum=q, geopot=geopot, u=u, v=v, p_sfc=p_sfc,
                dry_mass_tend_2h=dry_mass_tend_2h, dry_mass_tend_24h=dry_mass_tend_24h,
                energy_tend_2h=energy_tend_2h, f_net=f_net)


@pytest.fixture
def assembled(synthetic):
    keep = {"latitude": np.arange(4, 29), "longitude": np.arange(64)}
    integrals, kept = column_integrals(
        synthetic["temp"], synthetic["sphum"], synthetic["geopot"], synthetic["u"],
        synthetic["v"], synthetic["p_sfc"], interfaces="logp", keep=keep)
    fields = corrected_fields(integrals, synthetic["dry_mass_tend_2h"],
                              synthetic["dry_mass_tend_24h"],
                              synthetic["energy_tend_2h"], synthetic["f_net"])
    return integrals, kept, fields, keep


def test_the_integrals_are_the_column_integrals_of_the_fields(synthetic, assembled):
    """The looped, per-time integrals equal the whole-field column integral."""
    integrals, _, _, _ = assembled
    dp = dp_from_sfc_pressure(synthetic["temp"]["level"], synthetic["p_sfc"],
                              interfaces="logp").transpose(*ORDER)
    h = moist_static_energy(synthetic["temp"], synthetic["geopot"], synthetic["sphum"],
                            c_p=C_P_SCRIPTS)
    expected = {
        "v_dry": col_int(synthetic["v"] * (1 - synthetic["sphum"]), dp),
        "vh": col_int(synthetic["v"] * h, dp),
        "s_h": col_int(h, dp),
        "vapor": col_int(synthetic["sphum"], dp),
    }
    assert set(integrals.data_vars) == set(COLUMN_INTEGRALS)
    for name, arr in expected.items():
        np.testing.assert_allclose(integrals[name].values, arr.values, rtol=1e-12, atol=0)


def test_the_corrected_wind_transports_the_required_dry_mass(assembled):
    """After the correction the zonal-mean column dry-mass transport across
    each latitude is the rate of change of the dry mass north of it."""
    _, _, fields, _ = assembled
    interior = slice(80.0, -80.0)
    # The prescribed tendency has the zonal mean 1e-3 sin(phi), whose polar-cap
    # integral is a * 1e-3 * cos(phi) / 2 exactly.  The stored requirement
    # is the trapezoid rule on the grid's 33 latitudes, whose error on this
    # integrand is a * 1e-3 * h^2 / 6 * cos(phi), 10 kg/m/s at the equator
    # for h of 5.6 degrees and 0.08 on ERA5's half-degree grid; the corrected
    # wind is checked against the exact value.
    phi = np.deg2rad(fields["latitude"])
    exact = RAD_EARTH * 1e-3 * np.cos(phi) / 2.0
    gap = (fields["v_dry_corr_zm"] - exact).sel(latitude=interior)
    raw_gap = (fields["v_dry_zm"] - exact).sel(latitude=interior)
    stored_gap = (fields["req_dry_2h"] - exact).sel(latitude=interior)
    scale = float(abs(fields["v_dry_zm"]).max())
    assert float(abs(raw_gap).max()) > 0.05 * scale
    assert float(abs(gap).max()) < 1e-3 * scale
    assert float(abs(stored_gap).max()) < 1e-2 * scale


def test_the_flux_correction_closes_the_energy_budget(synthetic, assembled):
    """Adding the stored divergent correction to the corrected wind's flux
    removes the residual, up to the global mean no divergent field can carry."""
    _, _, fields, _ = assembled
    before = fields["energy_residual"]
    after = budget_residual(fields["uh_corr"] + fields["u_flux_corr"],
                            fields["vh_corr"] + fields["v_flux_corr"],
                            synthetic["energy_tend_2h"], synthetic["f_net"])
    weights = np.cos(np.deg2rad(before["latitude"]))

    def without_global_mean(resid):
        global_mean = (resid.mean("longitude") * weights).sum("latitude") / weights.sum()
        return resid - global_mean

    rms_before = float(np.sqrt((without_global_mean(before) ** 2).mean()))
    rms_after = float(np.sqrt((without_global_mean(after) ** 2).mean()))
    assert rms_before > 1.0
    assert rms_after < 1e-3 * rms_before


def test_the_net_mass_transport_is_the_column_integral_of_the_corrected_wind(
        synthetic, assembled):
    integrals, _, fields, _ = assembled
    dp = dp_from_sfc_pressure(synthetic["temp"]["level"], synthetic["p_sfc"],
                              interfaces="logp").transpose(*ORDER)
    v_adj = synthetic["v"] + fields["dv_mass"]
    direct = col_int(v_adj, dp).mean("longitude")
    np.testing.assert_allclose(fields["net_mass_corr_zm"].values, direct.values,
                               rtol=1e-10, atol=1e-10 * float(abs(direct).max()))
    np.testing.assert_allclose(
        fields["net_mass_daily"].values,
        (fields["req_dry_24h"] + fields["v_vapor_corr_zm"]).values, rtol=1e-12)
    assert set(SERIES) <= set(fields.data_vars)


def test_the_kept_fields_are_the_wind_and_the_mse_on_the_sub_grid(synthetic, assembled):
    _, kept, _, keep = assembled
    assert kept["v"].dtype == np.float32 and kept["h"].dtype == np.float32
    assert kept["v"].dims == ORDER
    np.testing.assert_allclose(kept["v"].values,
                               synthetic["v"].isel(keep).values, rtol=1e-6, atol=1e-6)
    h = moist_static_energy(synthetic["temp"], synthetic["geopot"], synthetic["sphum"],
                            c_p=C_P_SCRIPTS).isel(keep)
    np.testing.assert_allclose(kept["h"].values, h.values, rtol=1e-6)
    np.testing.assert_allclose(
        kept["latitude"].values,
        synthetic["temp"]["latitude"].isel(latitude=keep["latitude"]).values)


def test_the_summary_reports_the_closure(synthetic, assembled):
    _, _, fields, _ = assembled
    summary = closure_summary(fields, synthetic["p_sfc"], synthetic["dry_mass_tend_2h"])
    assert all(np.isfinite(value) for value in summary.values())
    assert (summary["mass_closure_corrected_cms_rms_band"]
            < 1e-2 * summary["mass_nonclosure_raw_cms_rms_band"])
    # the prescribed tendency has no global mean, so the closure reported
    # with the uniform-source curve removed is the closure itself
    assert abs(summary["dry_mass_tend_global_mean_kg_m2_s"]) < 1e-12
    np.testing.assert_allclose(summary["mass_closure_corrected_less_global_mean_cms_rms_band"],
                               summary["mass_closure_corrected_cms_rms_band"], rtol=1e-6)
    # the cap integral of the zonal-mean residual and the zonal mean of the
    # inverted flux correction are two computations of one curve
    assert (summary["energy_flux_curve_vs_inversion_max_abs_pw"]
            < 1e-2 * summary["energy_flux_curve_rms_pw_band"])


def test_the_integrals_refuse_a_field_on_another_grid(synthetic):
    shifted = synthetic["p_sfc"].assign_coords(
        latitude=synthetic["p_sfc"]["latitude"] + 1.0)
    with pytest.raises(ValueError):
        column_integrals(synthetic["temp"], synthetic["sphum"], synthetic["geopot"],
                         synthetic["u"], synthetic["v"], shifted)
def test_the_series_are_the_sums_they_are_documented_to_be(assembled):
    """``net_mass_corr_zm`` and ``net_mass_daily`` are the sums ``SERIES``
    says they are, exactly.  Both are what the mean-circulation term's net
    mass transport is read from, and a refactor that swapped one for the
    other would change that term with nothing else failing."""
    _, _, fields, _ = assembled
    np.testing.assert_array_equal(
        fields["net_mass_corr_zm"].values,
        (fields["v_dry_corr_zm"] + fields["v_vapor_corr_zm"]).values)
    np.testing.assert_array_equal(
        fields["net_mass_daily"].values,
        (fields["req_dry_24h"] + fields["v_vapor_corr_zm"]).values)


def test_the_mass_correction_is_separable_and_replaceable(synthetic, assembled):
    """A different mass correction can be applied without re-reading ERA5.

    This is what makes decision D2 cheap to revise; see ``RERUN-COSTS.md``.
    Two halves.  The assembly stores the corrected wind ``v + dv`` and stores
    ``dv`` beside it, so the uncorrected wind comes back by subtraction, to
    float32 rounding.  And the correction is a function of the column
    integrals and a tendency alone, all of which ``budget_YYYYMM.nc`` holds,
    so a replacement can be formed without the three-dimensional fields.
    """
    integrals, kept, fields, keep = assembled

    dv = fields["dv_mass"].isel(keep).astype("float32")
    v_adj = (kept["v"] + dv).astype("float32")
    recovered = (v_adj - dv).astype("float32")
    # the wind is order 3 m/s, so float32 rounding of the sum is order 3e-7;
    # 1e-5 m/s is 1e-3 cm/s, and the mean-circulation term moves 1.4 PW per
    # cm/s, so this bounds the recovery error at about 1e-3 PW.
    np.testing.assert_allclose(recovered.values,
                               kept["v"].values.astype("float32"),
                               rtol=0.0, atol=1e-5)

    # a replacement built from the stored integrals and the other tendency
    _, dv_new = mass_correction_from_columns(
        integrals["u_dry"], integrals["v_dry"], integrals["mass_dry"],
        synthetic["dry_mass_tend_24h"])
    fresh = corrected_fields(integrals, synthetic["dry_mass_tend_24h"],
                             synthetic["dry_mass_tend_24h"],
                             synthetic["energy_tend_2h"], synthetic["f_net"])
    np.testing.assert_array_equal(dv_new.values, fresh["dv_mass"].values)
    # and it is a different correction, so the check above is not vacuous
    assert float(abs(dv_new - fields["dv_mass"]).max()) > 0.0


def test_the_mse_gauge_moves_only_the_net_mass_transport(synthetic, assembled):
    """Adding a constant to the MSE shifts the corrected column MSE flux by
    that constant times the corrected column mass transport, and moves
    nothing else the assembly stores.

    This is what makes decision D5 a post-hoc arithmetic change rather than a
    rerun; see ``RERUN-COSTS.md`` and section D5 of the pipeline
    recommendation.  The constant enters through the geopotential, since the
    MSE is ``c_p T + gz + L q``.
    """
    integrals, _, fields, keep = assembled
    c = 2.75e5  # J/kg, the size of c_p times 273.16 K

    integrals_c, _ = column_integrals(
        synthetic["temp"], synthetic["sphum"], synthetic["geopot"] + c,
        synthetic["u"], synthetic["v"], synthetic["p_sfc"],
        interfaces="logp", keep=keep)
    fields_c = corrected_fields(integrals_c, synthetic["dry_mass_tend_2h"],
                                synthetic["dry_mass_tend_24h"],
                                synthetic["energy_tend_2h"], synthetic["f_net"])

    # the correction and every mass series are gauge-free
    np.testing.assert_allclose(fields_c["dv_mass"].values, fields["dv_mass"].values,
                               rtol=1e-12, atol=0.0)
    for name in SERIES:
        np.testing.assert_allclose(fields_c[name].values, fields[name].values,
                                   rtol=1e-12, atol=0.0,
                                   err_msg=f"{name} moved under a change of gauge")

    # and the MSE flux shifts by exactly c times the corrected mass transport
    shift = (fields_c["vh_corr"] - fields["vh_corr"]).mean("longitude")
    expected = c * fields["net_mass_corr_zm"]
    # a tolerance relative to the scale of the shift, not to each value: the
    # transport vanishes at both poles, where a relative tolerance compares
    # two roundings of zero.
    scale = float(abs(expected).max())
    np.testing.assert_allclose(shift.values, expected.values,
                               rtol=0.0, atol=1e-10 * scale)


# ------------------------------------------------ decision D3's curve, 2026-09-24

LAT_1D = np.linspace(90.0, -90.0, 181)


def _on_lat(values):
    return xr.DataArray(values, dims="latitude", coords={"latitude": LAT_1D})


def _cap_curve_numpy(source):
    """The flux in PW that closes a zonal-mean source in W/m2, rebuilt with
    numpy alone: the area-weighted global mean removed, the trapezoid integral
    over the cap north of each latitude, times 2 pi a^2."""
    phi = np.deg2rad(LAT_1D)
    w = np.cos(phi)
    anomaly = source - (source * w).sum() / w.sum()
    steps = 0.5 * (anomaly[1:] * w[1:] + anomaly[:-1] * w[:-1]) * np.abs(np.diff(phi))
    return 2.0 * np.pi * RAD_EARTH ** 2 * np.concatenate([[0.0], np.cumsum(steps)]) * 1e-15


def test_the_d3_curve_is_the_stored_curve_when_nothing_is_replaced():
    """With the 24-hour storage equal to the two-hour one, no E - P and the
    daily net mass transport equal to the sampled one, decision D3's curve is
    the curve of the residual itself."""
    phi = np.deg2rad(LAT_1D)
    residual = _on_lat(30.0 * np.sin(phi) + 10.0 * np.cos(phi) ** 2)
    tend = _on_lat(5.0 * np.sin(2 * phi))
    net = _on_lat(1e3 * np.cos(phi))
    curve = d3_energy_curve(residual, tend, tend, 0.0 * residual, net, net,
                            _on_lat(3.4e5 + 0.0 * phi))
    np.testing.assert_allclose(curve.values, energy_flux_curve(residual).values, rtol=0, atol=1e-12)
    np.testing.assert_allclose(energy_flux_curve(residual).values,
                               _cap_curve_numpy(residual.values), rtol=1e-12, atol=1e-12)
    assert np.array_equal(curve["latitude"].values, LAT_1D)


def test_each_replacement_in_the_d3_curve_enters_with_its_sign():
    """The three replacements, one at a time against a numpy reconstruction,
    and each with the sign the physics sets at the equator:

    * net evaporation in the north, E - P = B sin(phi) with B > 0, adds the
      enthalpy of that water to the northern cap, so the northward flux the
      budget requires across the equator falls;
    * a 24-hour storage tendency above the two-hour one in the north raises
      the northward flux the northern cap requires;
    * a daily-mean net mass transport more northward than the sampled one
      carries more MSE north, so the flux still required falls.
    """
    phi = np.deg2rad(LAT_1D)
    eq = int(np.argmin(np.abs(LAT_1D)))
    zero = _on_lat(0.0 * phi)
    mse = 3.4e5
    b = 2e-5
    storage = 4.0 * np.sin(phi)
    net_extra = 50.0 * np.cos(phi)
    circumference = 2.0 * np.pi * RAD_EARTH * np.cos(phi)

    ep_only = d3_energy_curve(zero, zero, zero, _on_lat(b * np.sin(phi)), zero, zero,
                              _on_lat(mse + 0.0 * phi))
    storage_only = d3_energy_curve(zero, zero, _on_lat(storage), zero, zero, zero,
                                   _on_lat(mse + 0.0 * phi))
    shift_only = d3_energy_curve(zero, zero, zero, zero, zero, _on_lat(net_extra),
                                 _on_lat(mse + 0.0 * phi))

    expected_ep = -_cap_curve_numpy(C_P_SCRIPTS * EP_ENTHALPY_T_REF * b * np.sin(phi))
    expected_storage = _cap_curve_numpy(storage)
    expected_shift = -net_extra * mse * circumference * 1e-15
    for got, expected in ((ep_only, expected_ep), (storage_only, expected_storage),
                          (shift_only, expected_shift)):
        assert abs(expected[eq]) > 0.1
        np.testing.assert_allclose(got.values, expected, rtol=1e-12, atol=1e-12)
    assert ep_only.values[eq] < 0.0
    assert storage_only.values[eq] > 0.0
    assert shift_only.values[eq] < 0.0

    together = d3_energy_curve(zero, zero, _on_lat(storage), _on_lat(b * np.sin(phi)), zero,
                               _on_lat(net_extra), _on_lat(mse + 0.0 * phi))
    np.testing.assert_allclose(together.values, expected_ep + expected_storage + expected_shift,
                               rtol=1e-12, atol=1e-12)
    # the reference temperature is a real parameter, not only its default
    warmer = d3_energy_curve(zero, zero, zero, _on_lat(b * np.sin(phi)), zero, zero,
                             _on_lat(mse + 0.0 * phi), t_ref=300.0)
    np.testing.assert_allclose(warmer.values, expected_ep * 300.0 / EP_ENTHALPY_T_REF,
                               rtol=1e-12, atol=1e-12)


def test_the_summary_reports_the_d3_curve_only_when_given_its_inputs(synthetic, assembled):
    integrals, _, fields, _ = assembled
    base = closure_summary(fields, synthetic["p_sfc"], synthetic["dry_mass_tend_2h"])
    assert "energy_flux_curve_d3_equator_pw" not in base

    phi = np.deg2rad(synthetic["f_net"]["latitude"])
    tend_24h = 0.5 * synthetic["energy_tend_2h"] + 3.0 * np.sin(phi)
    e_minus_p = (2e-5 * np.sin(phi) + 0.0 * synthetic["f_net"]).transpose(
        "time", "latitude", "longitude")
    summary = closure_summary(fields, synthetic["p_sfc"], synthetic["dry_mass_tend_2h"],
                              integrals=integrals, energy_tend_2h=synthetic["energy_tend_2h"],
                              energy_tend_24h=tend_24h, e_minus_p=e_minus_p)
    for key, value in base.items():
        assert summary[key] == value

    def tm_zm(arr):
        return arr.mean(("time", "longitude"))

    col_mse = tm_zm(integrals["s_h"]) / tm_zm(integrals["mass_dry"] + integrals["vapor"])
    curve = d3_energy_curve(tm_zm(fields["energy_residual"]), tm_zm(synthetic["energy_tend_2h"]),
                            tm_zm(tend_24h), tm_zm(e_minus_p),
                            fields["net_mass_corr_zm"].mean("time"),
                            fields["net_mass_daily"].mean("time"), col_mse)
    eq = int(np.argmin(np.abs(curve["latitude"].values)))
    np.testing.assert_allclose(summary["energy_flux_curve_d3_equator_pw"],
                               float(curve.isel(latitude=eq)), rtol=1e-12)
    assert abs(summary["energy_flux_curve_d3_equator_pw"] - base["energy_flux_curve_equator_pw"]) > 0.01
