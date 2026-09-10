"""The corrected assembly on synthetic fields: what closes and what is stored."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from itcz_eddies.assembly import (
    COLUMN_INTEGRALS,
    SERIES,
    closure_summary,
    column_integrals,
    corrected_fields,
)
from itcz_eddies.columns import col_int, dp_from_sfc_pressure
from itcz_eddies.mass import RAD_EARTH
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
