r"""Assembling one month of the corrected fields the flux decomposition consumes.

This is the computation behind ``scripts/assemble_fields.py`` in its
corrected mode, kept in the package so that it runs on synthetic fields in
``tests/test_assembly.py``.  It realizes four of the six decisions of the
pipeline recommendation of 2026-09-09
(``pipeline-2026-09-09/pipeline-recommendation.pdf`` at the project root):

D1  every column integral is the mass-weighted sum with the layer
    thickness of ``columns.dp_from_sfc_pressure`` and interfaces midway in
    log pressure, so the layer between the lowest above-ground level and
    the surface is kept;
D2  the wind is corrected by the divergent barotropic wind that closes the
    dry-air column mass budget, with the tendency from the hourly surface
    pressure and column vapor centered on the analysis time (``mass``);
D3  the residual of the column energy budget with the corrected wind and
    the hourly model-level storage tendency is stored beside the fields as
    the divergent flux correction that would close it, in W/m, and is never
    turned into a wind;
D6  the daily-mean mass requirement, the polar-cap integral of the 24-hour
    centered tendency, is stored as a zonal-mean series so that the
    mean-circulation term's net mass transport can be taken from it.

Three functions.  ``column_integrals`` loops over the analysis times and
forms the column integrals in float64 one time at a time, since a month of
five global three-dimensional fields does not fit in memory at once.
``corrected_fields`` takes those integrals and the tendencies and forms the
corrections, the residual and the zonal-mean series.  ``closure_summary``
reduces the result to the numbers a run reports.
"""

from __future__ import annotations

import time as _time

import numpy as np
import xarray as xr
from puffins.constants import GRAV_EARTH

from .columns import col_int, dp_from_sfc_pressure
from .mass import mass_correction_from_columns, polar_cap_transport
from .metrics import flux_to_petawatts
from .mse import C_P_SCRIPTS, adjusted_col_fluxes, budget_residual, moist_static_energy
from .names import LAT_STR, LEV_STR, LON_STR, TIME_STR

__all__ = [
    "COLUMN_INTEGRALS",
    "SERIES",
    "closure_summary",
    "column_integrals",
    "corrected_fields",
]

COLUMN_INTEGRALS = {
    "u_dry": ("column dry-air mass flux, zonal, <(1 - q) u>", "kg m-1 s-1"),
    "v_dry": ("column dry-air mass flux, meridional, <(1 - q) v>", "kg m-1 s-1"),
    "mass_dry": ("column dry-air mass <1 - q>", "kg m-2"),
    "vapor": ("column water vapor <q>", "kg m-2"),
    "v_vapor": ("column vapor flux, meridional, <q v>", "kg m-1 s-1"),
    "uh": ("column MSE flux, zonal, <h u>", "W m-1"),
    "vh": ("column MSE flux, meridional, <h v>", "W m-1"),
    "s_h": ("column MSE <h>", "J m-2"),
}

SERIES = {
    "req_dry_2h": "zonal-mean dry-mass transport the two-hour centered tendency "
                  "requires: the polar-cap integral of the tendency",
    "req_dry_24h": "zonal-mean dry-mass transport the 24-hour centered tendency "
                   "requires, the daily-mean requirement of decision D6",
    "v_dry_zm": "zonal-mean column dry-mass transport of the raw wind",
    "v_dry_corr_zm": "zonal-mean column dry-mass transport of the corrected wind, "
                     "which equals req_dry_2h to the accuracy of the inversion",
    "v_vapor_corr_zm": "zonal-mean column vapor transport of the corrected wind",
    "net_mass_corr_zm": "zonal-mean column mass transport of the corrected wind, "
                        "v_dry_corr_zm plus v_vapor_corr_zm, which the "
                        "mean-circulation term of the mass-flux split integrates to",
    "net_mass_daily": "net_mass_corr_zm with the two-hour requirement replaced by "
                      "the daily-mean one: req_dry_24h plus v_vapor_corr_zm, the "
                      "mean-circulation term's net mass transport under decision D6",
}


def column_integrals(temp, sphum, geopot, u, v, p_sfc, interfaces="logp",
                     c_p=C_P_SCRIPTS, keep=None, log=None, time_str=TIME_STR,
                     lev_str=LEV_STR, lat_str=LAT_STR, lon_str=LON_STR):
    """Column integrals of the wind, the vapor and the MSE, one time at a time.

    Parameters
    ----------
    temp, sphum, geopot, u, v
        The five pressure-level fields on (time, level, latitude, longitude),
        global and latitude descending as ``windspharm`` needs downstream.
        They are indexed lazily, so a month read with ``era5.read_data`` is
        loaded one analysis time at a time.
    p_sfc
        Surface pressure in hPa on (time, latitude, longitude), at the same
        times and on the same grid.
    interfaces
        Rule for the layer interfaces, passed to ``columns.dp_from_sfc_pressure``.
    c_p
        Specific heat in the MSE.
    keep
        Positional indexers, ``{"latitude": ..., "longitude": ...}``, of the
        sub-grid on which the wind and the MSE themselves are returned as
        float32.  ``None`` returns neither.
    log
        Called as ``log(step, n_steps, seconds)`` after each time step.

    Returns
    -------
    (integrals, kept)
        ``integrals`` is a Dataset on (time, latitude, longitude) in float64
        of the eight quantities in ``COLUMN_INTEGRALS``.  ``kept`` is a
        Dataset of ``v`` and ``h`` on (time, level, kept latitudes, kept
        longitudes) as float32, or ``None``.
    """
    for name, arr in (("sphum", sphum), ("geopot", geopot), ("u", u), ("v", v),
                      ("p_sfc", p_sfc)):
        for dim in (lat_str, lon_str):
            if not np.allclose(arr[dim].values, temp[dim].values):
                raise ValueError(f"{name} is not on the grid of temp along {dim}")
    n_t = temp.sizes[time_str]
    lat = temp[lat_str].values
    lon = temp[lon_str].values
    acc = {name: np.empty((n_t, lat.size, lon.size)) for name in COLUMN_INTEGRALS}
    kept_v = kept_h = sample = None
    if keep is not None:
        sample = temp.isel({time_str: 0}).isel(keep).drop_vars(time_str, errors="ignore")
        kept_v = np.empty((n_t,) + sample.shape, dtype="float32")
        kept_h = np.empty_like(kept_v)

    for it in range(n_t):
        started = _time.monotonic()
        t = temp.isel({time_str: it}).astype("float64").load()
        q = sphum.isel({time_str: it}).astype("float64").load()
        z = geopot.isel({time_str: it}).astype("float64").load()
        uu = u.isel({time_str: it}).astype("float64").load()
        vv = v.isel({time_str: it}).astype("float64").load()
        ps = p_sfc.isel({time_str: it}).astype("float64").load()
        h = moist_static_energy(t, z, q, c_p=c_p)
        dp = dp_from_sfc_pressure(t[lev_str], ps, interfaces=interfaces,
                                  lev_str=lev_str).transpose(*h.dims)
        dry = 1.0 - q
        acc["u_dry"][it] = col_int(uu * dry, dp, lev_str=lev_str).values
        acc["v_dry"][it] = col_int(vv * dry, dp, lev_str=lev_str).values
        acc["mass_dry"][it] = col_int(dry, dp, lev_str=lev_str).values
        acc["vapor"][it] = col_int(q, dp, lev_str=lev_str).values
        acc["v_vapor"][it] = col_int(vv * q, dp, lev_str=lev_str).values
        acc["uh"][it] = col_int(uu * h, dp, lev_str=lev_str).values
        acc["vh"][it] = col_int(vv * h, dp, lev_str=lev_str).values
        acc["s_h"][it] = col_int(h, dp, lev_str=lev_str).values
        if keep is not None:
            kept_v[it] = vv.isel(keep).values
            kept_h[it] = h.isel(keep).values
        if log is not None:
            log(it, n_t, _time.monotonic() - started)

    coords = {time_str: temp[time_str].values, lat_str: lat, lon_str: lon}
    integrals = xr.Dataset(
        {name: ((time_str, lat_str, lon_str), acc[name]) for name in COLUMN_INTEGRALS},
        coords=coords,
    )
    for name, (long_name, units) in COLUMN_INTEGRALS.items():
        integrals[name].attrs.update(long_name=long_name, units=units)
    kept = None
    if keep is not None:
        dims = (time_str,) + sample.dims
        kept_coords = {time_str: temp[time_str].values}
        kept_coords.update({dim: sample[dim].values for dim in sample.dims})
        kept = xr.Dataset({"v": (dims, kept_v), "h": (dims, kept_h)}, coords=kept_coords)
        kept["v"].attrs.update(long_name="meridional wind, uncorrected", units="m s-1")
        kept["h"].attrs.update(long_name="moist static energy", units="J kg-1", c_p=c_p)
    return integrals, kept


def _on_grid_of(arr, reference, name, time_str, lat_str, lon_str):
    if arr.shape != reference.shape:
        raise ValueError(f"{name} has shape {arr.shape}, the integrals {reference.shape}")
    for dim in (lat_str, lon_str):
        if not np.allclose(arr[dim].values, reference[dim].values):
            raise ValueError(f"{name} is not on the grid of the integrals along {dim}")
    return arr.assign_coords({time_str: reference[time_str], lat_str: reference[lat_str],
                              lon_str: reference[lon_str]})


def corrected_fields(integrals, dry_mass_tend_2h, dry_mass_tend_24h, energy_tend_2h,
                     f_net, lat_str=LAT_STR, lon_str=LON_STR, time_str=TIME_STR):
    """The corrections, the residual and the zonal-mean series from the integrals.

    Parameters
    ----------
    integrals
        The Dataset ``column_integrals`` returns.
    dry_mass_tend_2h, dry_mass_tend_24h
        Rate of change of the dry column mass ``(p_s - g W) / g`` in kg/m2/s
        at the analysis times, from ``mass.centered_tendency`` with a
        one-hour and with a twelve-hour half window.
    energy_tend_2h
        Rate of change of the column energy in W/m2 at the analysis times,
        from the hourly model-level integrals ``vipile`` plus ``vike`` with a
        one-hour half window.
    f_net
        Net energy input to the column in W/m2 at the analysis times, from
        ``mse.net_energy_input`` on the forecast rates bracketing each time.

    Returns
    -------
    xarray.Dataset
        On (time, latitude, longitude): ``dv_mass`` and ``du_mass``, the
        barotropic correction in m/s; ``vh_corr`` and ``uh_corr``, the
        column MSE flux of the corrected wind in W/m; ``v_flux_corr`` and
        ``u_flux_corr``, the divergent flux correction in W/m that would
        close the energy budget of the corrected wind; ``energy_residual``,
        the residual ``dE/dt + div<vh_corr> - F_net`` it closes, in W/m2.
        On (time, latitude), in kg/m/s and positive northward, the seven
        series named in ``SERIES``.
    """
    ref = integrals["v_dry"]
    tend_2h = _on_grid_of(dry_mass_tend_2h, ref, "dry_mass_tend_2h", time_str, lat_str, lon_str)
    tend_24h = _on_grid_of(dry_mass_tend_24h, ref, "dry_mass_tend_24h", time_str, lat_str, lon_str)
    energy_tend = _on_grid_of(energy_tend_2h, ref, "energy_tend_2h", time_str, lat_str, lon_str)
    source = _on_grid_of(f_net, ref, "f_net", time_str, lat_str, lon_str)

    du, dv = mass_correction_from_columns(
        integrals["u_dry"], integrals["v_dry"], integrals["mass_dry"], tend_2h,
        lat_str=lat_str, lon_str=lon_str, time_str=time_str)
    vh_corr = integrals["vh"] + dv * integrals["s_h"]
    uh_corr = integrals["uh"] + du * integrals["s_h"]
    uh_adj, vh_adj = adjusted_col_fluxes(uh_corr, vh_corr, energy_tend, source,
                                         lat_str=lat_str, lon_str=lon_str,
                                         time_str=time_str)
    residual = budget_residual(uh_corr, vh_corr, energy_tend, source)

    v_dry_corr = integrals["v_dry"] + dv * integrals["mass_dry"]
    v_vapor_corr = integrals["v_vapor"] + dv * integrals["vapor"]
    req_2h = polar_cap_transport(tend_2h.mean(lon_str), lat_str=lat_str)
    req_24h = polar_cap_transport(tend_24h.mean(lon_str), lat_str=lat_str)
    v_dry_zm = integrals["v_dry"].mean(lon_str)
    v_dry_corr_zm = v_dry_corr.mean(lon_str)
    v_vapor_corr_zm = v_vapor_corr.mean(lon_str)

    out = xr.Dataset({
        "dv_mass": dv,
        "du_mass": du,
        "vh_corr": vh_corr,
        "uh_corr": uh_corr,
        "v_flux_corr": vh_adj - vh_corr,
        "u_flux_corr": uh_adj - uh_corr,
        "energy_residual": residual,
        "req_dry_2h": req_2h,
        "req_dry_24h": req_24h,
        "v_dry_zm": v_dry_zm,
        "v_dry_corr_zm": v_dry_corr_zm,
        "v_vapor_corr_zm": v_vapor_corr_zm,
        "net_mass_corr_zm": v_dry_corr_zm + v_vapor_corr_zm,
        "net_mass_daily": req_24h + v_vapor_corr_zm,
    })
    long_names = {
        "dv_mass": ("barotropic wind correction, meridional, that closes the dry-air "
                    "column mass budget (decision D2)", "m s-1"),
        "du_mass": ("barotropic wind correction, zonal", "m s-1"),
        "vh_corr": ("column MSE flux of the corrected wind, meridional", "W m-1"),
        "uh_corr": ("column MSE flux of the corrected wind, zonal", "W m-1"),
        "v_flux_corr": ("divergent flux correction, meridional, that would close the "
                        "column energy budget of the corrected wind (decision D3)",
                        "W m-1"),
        "u_flux_corr": ("divergent flux correction, zonal", "W m-1"),
        "energy_residual": ("column energy budget residual of the corrected wind, "
                            "dE/dt + div<vh_corr> - F_net", "W m-2"),
    }
    for name, (long_name, units) in long_names.items():
        out[name].attrs.update(long_name=long_name, units=units)
    for name, long_name in SERIES.items():
        out[name].attrs.update(long_name=long_name, units="kg m-1 s-1")
    return out


def closure_summary(fields, p_sfc, band=(-30.0, 30.0), lat_str=LAT_STR,
                    lon_str=LON_STR, time_str=TIME_STR):
    """The numbers a run of the corrected assembly reports, as a dict.

    ``fields`` is the Dataset ``corrected_fields`` returns and ``p_sfc`` the
    surface pressure in hPa on (time, latitude, longitude).  Mass transports
    are given as the equivalent barotropic wind in cm/s, the transport times
    gravity over the time-mean zonal-mean surface pressure, which is the unit
    of the pipeline recommendation: 1 cm/s of net zonal-mean mass transport
    is 1.3 PW of MSE flux at the equator.  Everything is a time mean over the
    record passed in, and the rms values are over ``band``, 30S to 30N by
    default.  Two energy-residual curves are given, the polar-cap integral
    of the zonal-mean time-mean residual with its global mean removed, and
    the zonal mean of the flux correction the spherical-harmonic inversion
    produced; they must agree, and their largest difference is returned as
    a control.
    """
    lat = fields[lat_str]
    ps_zm = (p_sfc * 100.0).mean((time_str, lon_str))
    to_cms = GRAV_EARTH / ps_zm * 100.0
    in_band = ((lat >= min(band)) & (lat <= max(band))).values
    equator = int(np.argmin(np.abs(lat.values)))

    def rms(arr):
        return float(np.sqrt((arr ** 2).mean()))

    def rms_band(arr):
        return rms(arr.isel({lat_str: in_band}))

    corrected = (fields["v_dry_corr_zm"] - fields["req_dry_2h"]).mean(time_str) * to_cms
    raw = (fields["v_dry_zm"] - fields["req_dry_2h"]).mean(time_str) * to_cms
    daily = (fields["net_mass_daily"] - fields["net_mass_corr_zm"]).mean(time_str) * to_cms

    r_bar = fields["energy_residual"].mean(time_str)
    r_zm = r_bar.mean(lon_str)
    weights = np.cos(np.deg2rad(lat))
    global_mean = float((r_zm * weights).sum() / weights.sum())
    curve = flux_to_petawatts(polar_cap_transport(r_zm - global_mean, lat_str=lat_str),
                              lat_str=lat_str)
    inversion = flux_to_petawatts(fields["v_flux_corr"].mean((time_str, lon_str)),
                                  lat_str=lat_str)
    return {
        "mass_closure_corrected_cms_rms_band": rms_band(corrected),
        "mass_nonclosure_raw_cms_rms_band": rms_band(raw),
        "dv_mass_rms_ms": rms(fields["dv_mass"]),
        "energy_residual_time_mean_map_rms_wm2": rms(r_bar),
        "energy_residual_global_mean_wm2": global_mean,
        "energy_flux_curve_rms_pw_band": rms_band(curve),
        "energy_flux_curve_equator_pw": float(curve.isel({lat_str: equator})),
        "energy_flux_curve_vs_inversion_max_abs_pw": float(abs(curve - inversion).max()),
        "daily_minus_sampled_net_mass_cms_rms_band": rms_band(daily),
        "daily_minus_sampled_net_mass_cms_equator": float(daily.isel({lat_str: equator})),
    }
