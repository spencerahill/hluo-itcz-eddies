"""Assemble one month of the fields the flux decomposition consumes.

Two modes.

``--mode corrected``, the default since 2026-09-09, is the pipeline of
``pipeline-2026-09-09/pipeline-recommendation.pdf`` at the project root.
The five pressure-level fields are read globally at 00, 06, 12 and 18 UTC
(``--temporal-resolution 6``, decision D6); the column integrals use the
mass-weighted quadrature with interfaces midway in log pressure
(``--interfaces logp``, decision D1); the wind is corrected barotropically so
that the dry-air column mass budget closes, with the tendency from the
hourly surface pressure and column vapor (decision D2); the residual of the
column energy budget with the corrected wind and the hourly model-level
storage tendency is stored as the divergent flux correction that would
close it, never as a wind (decision D3); and the daily-mean mass
requirement is stored for the mean-circulation term's net mass transport
(decision D6).  The computation is ``itcz_eddies.assembly``; this file reads
and writes.

``--mode archived`` is what this script did before: the archived adjustment
of Appendix A applied to the wind through the MSE at 00 and 12 UTC, the
published construction, kept for the golden master and for comparison.
Never combine its files with the mass-weighted quadrature (F22 in
``code-review/FINDINGS.md``).

What is written, under ``ITCZ_FIELDS_ROOT/<tag>/`` with the tag naming the
configuration, ``corrected_6h_logp`` by default (``--tag`` overrides, and
``paths.fields_tag`` builds it):

``fields_YYYYMM.nc``
    What the decomposition reads: ``v_adj``, ``mse`` and ``p_sfc`` on the
    ``--boundary`` grid, 60S to 60N by default, as float32; in the corrected
    mode also ``dv_mass`` there and the zonal-mean mass-transport series of
    ``assembly.SERIES`` on the boundary latitudes.  About 7 GB a month at
    four samples a day.
``budget_YYYYMM.nc``
    Corrected mode only: the global column integrals, the corrections, the
    energy residual and its two pieces, and the global series, so that every
    budget curve of the pipeline recommendation can be re-formed without the
    three-dimensional fields.  About 3 GB a month.

The hourly two-dimensional fields are read for the month and both
neighboring months, since the tendencies reach twelve hours past the month
ends.  The forecast rates come from the three half-month mean-flux blocks
covering the month, indexed by valid time; the rate at an analysis time is
the mean of the rates valid at that hour and at the next, which brackets
the analysis time by one hour either side.

Never at 09, 10, 21 or 22 UTC: the surface-pressure tendency jumps by 3 to
7.5 cm/s of equivalent barotropic wind across ERA5's assimilation-window
boundaries there (decision D2), and the script refuses those hours.

``--smoke N`` keeps the first N analysis times and writes under a ``smoke``
subdirectory, so a check never overwrites a full month.

Usage:
  python scripts/assemble_fields.py --year 1997 --month 7
  python scripts/assemble_fields.py --year 1997 --month 7 --smoke 4
  python scripts/assemble_fields.py --year 1997 --month 7 --mode archived
"""

from __future__ import annotations

import argparse
import datetime
import gc
import logging
import pathlib
import time as _time

import numpy as np
import xarray as xr
from puffins.constants import GRAV_EARTH

from itcz_eddies import paths
from itcz_eddies.assembly import SERIES, closure_summary, column_integrals, corrected_fields
from itcz_eddies.era5 import read_data, read_meanflux_by_valid_time, stride
from itcz_eddies.mass import centered_tendency
from itcz_eddies.mse import (
    C_P_SCRIPTS,
    adjust_wind_by_mse_flux,
    moist_static_energy,
    net_energy_input,
)

FORBIDDEN_HOURS = {9, 10, 21, 22}
PL_VARS_BY_MODE = {"corrected": ("T", "Q", "Z", "U", "V"), "archived": ("T", "Q", "Z", "V")}
DEFAULT_HOURS = {"corrected": 6.0, "archived": 12.0}
FLUX_ARGS = {"MSSHF": "sensible", "MSLHF": "latent", "MSNSWRF": "sfc_sw",
             "MSNLWRF": "sfc_lw", "MTNSWRF": "toa_sw", "MTNLWRF": "toa_lw"}
HOUR = np.timedelta64(1, "h")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--month", type=int, required=True)
    parser.add_argument("--mode", choices=sorted(PL_VARS_BY_MODE), default="corrected")
    parser.add_argument("--interfaces", choices=["logp", "midpoint"], default="logp",
                        help="where the layer interfaces sit (decision D1); "
                             "the corrected mode only")
    parser.add_argument("--out", type=pathlib.Path, default=None,
                        help="output directory; default is $ITCZ_FIELDS_ROOT/<tag>")
    parser.add_argument("--tag", default="auto",
                        help="configuration subdirectory under the fields root; "
                             "auto builds it from the mode, the sampling and the "
                             "interfaces, e.g. corrected_6h_logp")
    parser.add_argument("--boundary", type=float, nargs=4,
                        default=[-60.0, 60.0, 0.0, 360.0],
                        metavar=("LAT0", "LAT1", "LON0", "LON1"),
                        help="the grid the three-dimensional fields are kept on")
    parser.add_argument("--temporal-resolution", type=float, default=None,
                        help="hours between samples; default 6 in the corrected "
                             "mode (decision D6) and 12 in the archived mode")
    parser.add_argument("--spatial-resolution", type=float, default=0.5)
    parser.add_argument("--smoke", type=int, default=0,
                        help="keep only the first N analysis times and write "
                             "under a smoke subdirectory")
    args = parser.parse_args(argv)
    if args.temporal_resolution is None:
        args.temporal_resolution = DEFAULT_HOURS[args.mode]
    if args.tag == "auto":
        args.tag = paths.fields_tag(args.mode, args.temporal_resolution, args.interfaces)
    return args


def output_directory(args):
    out = args.out or (paths.fields_root() / args.tag)
    if args.smoke:
        out = out / "smoke"
    return out


def pressure_level_files(year, month, variables):
    pl_files = {var: paths.pl_files_month(var, year, month) for var in variables}
    counts = {var: len(files) for var, files in pl_files.items()}
    logging.info("daily pressure-level files found for %d-%02d: %s", year, month, counts)
    expected = paths.last_day(year, month)
    if any(count != expected for count in counts.values()):
        raise SystemExit(
            f"expected {expected} daily files per variable for {year}-{month:02d}, "
            f"found {counts}"
        )
    return pl_files


def read_hourly(file_of, var, year, month, needed, spatial_resolution):
    """One hourly two-dimensional ERA5 field at the given times, global,
    latitude descending, loaded.

    Reads the month and both neighbors, since ``needed`` reaches twelve hours
    past the month ends.  Each file is read as one contiguous block from the
    first hour needed to the last, at full resolution, and strided in memory.
    Measured on Casper on 2026-09-09 (the files are chunked 27 hours deep and
    deflated): one full-resolution hour reads in 0.32 s, the same hour
    strided through the netCDF library in 3.2 s, and six scattered strided
    hours in 20 s, so the smoke run that read scattered strided hours took
    80 s per field for 16 hours.  A month at full resolution is 3.1 GB, held
    only until the stride is taken.
    """
    needed = np.asarray(needed, dtype="datetime64[ns]")
    pieces = []
    for offset in (-1, 0, 1):
        path = file_of(var, *paths.shift_month(year, month, offset))
        if not path.exists():
            raise SystemExit(f"missing {path}")
        with xr.open_dataset(path) as ds:
            step = stride(spatial_resolution,
                          float(ds["longitude"][1] - ds["longitude"][0]))
            hit = np.flatnonzero(np.isin(ds["time"].values.astype("datetime64[ns]"), needed))
            if hit.size:
                block = ds[var].isel(time=slice(int(hit[0]), int(hit[-1]) + 1)).load()
                block = block.isel(latitude=slice(None, None, step),
                                   longitude=slice(None, None, step))
                keep = np.isin(block["time"].values.astype("datetime64[ns]"), needed)
                pieces.append(block.isel(time=np.flatnonzero(keep)).copy())
                del block
    out = xr.concat(pieces, "time").sortby("time")
    missing = np.setdiff1d(needed, out["time"].values.astype("datetime64[ns]"))
    if missing.size:
        raise SystemExit(f"{var}: {missing.size} of the {needed.size} hours needed are "
                         f"not in the files, the first {missing[0]}")
    if out["latitude"][0] < out["latitude"][-1]:
        out = out.isel(latitude=slice(None, None, -1))
    logging.info("read %s at %d hours", var, out.sizes["time"])
    return out


def on_grid(arr, lat, lon, name):
    """``arr`` with the pressure-level grid's coordinates, after checking it is on it."""
    if not (np.allclose(arr["latitude"].values, lat) and np.allclose(arr["longitude"].values, lon)):
        raise SystemExit(f"{name} is not on the pressure-level grid")
    return arr.assign_coords(latitude=lat, longitude=lon)


def hours_of(times):
    return sorted({int(h) for h in ((times - times.astype("datetime64[D]")) / HOUR)})


def common_attrs(args, times):
    return dict(
        year=args.year, month=args.month, mode=args.mode,
        interfaces=args.interfaces if args.mode == "corrected" else "none",
        hours_utc=" ".join(str(h) for h in hours_of(times)),
        n_times=int(times.size),
        smoke=args.smoke,
        boundary=" ".join(str(b) for b in args.boundary),
        temporal_resolution_hours=args.temporal_resolution,
        spatial_resolution_degrees=args.spatial_resolution,
        c_p=C_P_SCRIPTS,
        era5_root=str(paths.era5_root()),
        written=datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
    )


def write(ds, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest.unlink()
    # the readers downstream select latitude bands with ascending slices
    if ds["latitude"][0] > ds["latitude"][-1]:
        ds = ds.isel(latitude=slice(None, None, -1))
    ds.to_netcdf(dest)
    logging.info("wrote %s: %s", dest, dict(ds.sizes))


# ------------------------------------------------------- the corrected mode


def corrected_month(args):
    started = _time.monotonic()
    year, month = args.year, args.month
    g = GRAV_EARTH
    pl_files = pressure_level_files(year, month, PL_VARS_BY_MODE["corrected"])
    read = dict(temporal_resolution=args.temporal_resolution,
                spatial_resolution=args.spatial_resolution,
                boundary=None, lat_descending=True)
    fields = {var: xr.decode_cf(read_data(pl_files[var], **read))[var]
              for var in PL_VARS_BY_MODE["corrected"]}
    if args.smoke:
        fields = {var: da.isel(time=slice(0, args.smoke)) for var, da in fields.items()}
    times = fields["T"]["time"].values.astype("datetime64[ns]")
    hours = hours_of(times)
    if set(hours) & FORBIDDEN_HOURS:
        raise SystemExit(f"the sampled hours {hours} include one of {sorted(FORBIDDEN_HOURS)} "
                         "UTC, across which the analysis increments jump (decision D2)")
    lat = fields["T"]["latitude"].values
    lon = fields["T"]["longitude"].values
    if lat[0] < lat[-1]:
        raise SystemExit("the pressure-level fields must come latitude descending")
    logging.info("%d analysis times at %s UTC from %s to %s on a %d by %d grid",
                 times.size, hours, times[0], times[-1], lat.size, lon.size)

    # ---- the hourly two-dimensional fields and the tendencies
    needed = np.unique(np.concatenate([times + k * HOUR for k in (-12, -1, 0, 1, 12)]))
    hourly = dict(spatial_resolution=args.spatial_resolution)
    sp = on_grid(read_hourly(paths.sfc_file, "SP", year, month, needed, **hourly),
                 lat, lon, "SP") / 100.0
    tcwv = on_grid(read_hourly(paths.sfc_file, "TCWV", year, month, needed, **hourly),
                   lat, lon, "TCWV")
    energy = on_grid(read_hourly(paths.vinteg_file, "VIPILE", year, month, needed, **hourly),
                     lat, lon, "VIPILE")
    energy = energy + on_grid(read_hourly(paths.vinteg_file, "VIKE", year, month, needed,
                                          **hourly), lat, lon, "VIKE")
    dry_mass = sp * 100.0 / g - tcwv
    dry_mass_tend_2h = centered_tendency(dry_mass, times, half_window_hours=1)
    dry_mass_tend_24h = centered_tendency(dry_mass, times, half_window_hours=12)
    energy_tend_2h = centered_tendency(energy, times, half_window_hours=1)
    p_sfc = sp.sel(time=times)
    del dry_mass, energy, tcwv, sp
    gc.collect()
    logging.info("tendencies formed at %.0f s", _time.monotonic() - started)

    # ---- the net energy input from the forecast rates bracketing each time
    bracket = np.unique(np.concatenate([times, times + HOUR]))
    rates = {}
    for var, name in FLUX_ARGS.items():
        files = paths.meanflux_files(var, year, month, pad_december=False)
        for path in files:
            if not path.exists():
                raise SystemExit(f"missing {path}")
        rate = on_grid(read_meanflux_by_valid_time(files, var, args.spatial_resolution,
                                                   bracket, lat_descending=True),
                       lat, lon, var)
        rates[name] = xr.DataArray(
            0.5 * (rate.sel(time=times).values + rate.sel(time=times + HOUR).values),
            dims=("time", "latitude", "longitude"),
            coords={"time": times, "latitude": lat, "longitude": lon})
        del rate
    f_net = net_energy_input(**rates)
    del rates
    gc.collect()
    logging.info("net energy input formed at %.0f s", _time.monotonic() - started)

    # ---- the column integrals, one analysis time at a time
    lat0, lat1, lon0, lon1 = args.boundary
    keep = {"latitude": np.flatnonzero((lat >= min(lat0, lat1)) & (lat <= max(lat0, lat1))),
            "longitude": np.flatnonzero((lon >= min(lon0, lon1)) & (lon <= max(lon0, lon1)))}

    def log(step, n_steps, seconds):
        if step % 10 == 0 or step == n_steps - 1:
            logging.info("column integrals: step %d of %d took %.1f s (elapsed %.0f s)",
                         step + 1, n_steps, seconds, _time.monotonic() - started)

    integrals, kept = column_integrals(
        fields["T"], fields["Q"], fields["Z"], fields["U"], fields["V"], p_sfc,
        interfaces=args.interfaces, c_p=C_P_SCRIPTS, keep=keep, log=log)
    del fields
    gc.collect()

    # ---- the corrections, the residual and the series
    corr = corrected_fields(integrals, dry_mass_tend_2h, dry_mass_tend_24h,
                            energy_tend_2h, f_net)
    summary = closure_summary(corr, p_sfc)
    for key, value in summary.items():
        logging.info("%s: %.4g", key, value)
    logging.info("corrections formed at %.0f s", _time.monotonic() - started)

    attrs = common_attrs(args, times)
    attrs.update({key: float(value) for key, value in summary.items()})
    out_dir = output_directory(args)

    dv = corr["dv_mass"].isel(keep).astype("float32")
    v_adj = (kept["v"] + dv).astype("float32").rename("v_adj")
    v_adj.attrs.update(long_name="meridional wind after the dry-air mass correction",
                       units="m s-1")
    fields_out = xr.Dataset({
        "v_adj": v_adj,
        "mse": kept["h"].rename("mse"),
        "p_sfc": p_sfc.isel(keep).astype("float32"),
        "dv_mass": dv,
    })
    fields_out["p_sfc"].attrs.update(long_name="surface pressure", units="hPa")
    for name in SERIES:
        fields_out[name] = corr[name].isel(latitude=keep["latitude"])
    fields_out.attrs.update(attrs)
    write(fields_out, out_dir / f"fields_{year}{month:02d}.nc")
    del fields_out, kept, v_adj
    gc.collect()

    budget = xr.Dataset()
    for name in integrals.data_vars:
        budget[name] = integrals[name].astype("float32")
    for name in corr.data_vars:
        budget[name] = corr[name].astype("float32") if "longitude" in corr[name].dims else corr[name]
    budget["energy_tendency"] = energy_tend_2h.astype("float32")
    budget["energy_tendency"].attrs.update(
        long_name="rate of change of the column energy from the hourly model-level "
                  "integrals vipile plus vike, two-hour centered", units="W m-2")
    budget["f_net"] = f_net.astype("float32")
    budget["f_net"].attrs.update(long_name="net energy input to the column from the "
                                 "forecast rates bracketing the analysis time", units="W m-2")
    budget["dry_mass_tend_2h"] = dry_mass_tend_2h.astype("float32")
    budget["dry_mass_tend_2h"].attrs.update(
        long_name="rate of change of the dry column mass, two-hour centered", units="kg m-2 s-1")
    budget["dry_mass_tend_24h"] = dry_mass_tend_24h.astype("float32")
    budget["dry_mass_tend_24h"].attrs.update(
        long_name="rate of change of the dry column mass, 24-hour centered", units="kg m-2 s-1")
    budget["p_sfc"] = p_sfc.astype("float32")
    budget["p_sfc"].attrs.update(long_name="surface pressure", units="hPa")
    budget.attrs.update(attrs)
    write(budget, out_dir / f"budget_{year}{month:02d}.nc")
    logging.info("done at %.0f s", _time.monotonic() - started)


# -------------------------------------------------------- the archived mode


def read_month_archived(year, month, boundary, temporal_resolution, spatial_resolution):
    """The wind, its flux adjustment, the MSE and the surface pressure, lazily.

    Same construction, call for call, as ``read_year`` in
    ``scripts/decomposition.py`` before 2026-09-08, restricted to one month.
    The wind is returned unadjusted with the adjustment beside it, so that
    the caller can compute the MSE once and adjust against the computed
    field; adjusting against the lazy MSE would read temperature, humidity
    and geopotential a second time.
    """
    read = dict(temporal_resolution=temporal_resolution,
                spatial_resolution=spatial_resolution, boundary=boundary)
    pl_files = pressure_level_files(year, month, PL_VARS_BY_MODE["archived"])
    sfc = paths.sfc_file("SP", year, month)
    adjust = paths.adjust_file(year, month)
    for path in (sfc, adjust):
        if not path.exists():
            raise SystemExit(f"missing {path}")

    temp = xr.decode_cf(read_data(pl_files["T"], **read))["T"]
    sphum = xr.decode_cf(read_data(pl_files["Q"], **read))["Q"]
    geopot = xr.decode_cf(read_data(pl_files["Z"], **read))["Z"]
    mse = moist_static_energy(temp, geopot, sphum, c_p=C_P_SCRIPTS)

    v = xr.decode_cf(read_data(pl_files["V"], **read))["V"]

    adjust_ds = read_data([adjust], **read)
    adjust_ds["time"] = adjust_ds["time"].assign_attrs(
        units="hours since 1900-01-01 00:00:00", calendar="gregorian")
    v_flux_adjust = xr.decode_cf(adjust_ds)["vMSE"]

    p_sfc = xr.decode_cf(read_data([sfc], **read))["SP"] / 100.0
    p_sfc = p_sfc.sel(time=mse["time"])
    return v, v_flux_adjust, mse, p_sfc, {"adjustment_file": str(adjust),
                                          "surface_pressure_file": str(sfc)}


def archived_month(args):
    v, v_flux_adjust, mse, p_sfc, sources = read_month_archived(
        args.year, args.month, args.boundary,
        args.temporal_resolution, args.spatial_resolution)
    if args.smoke:
        v, v_flux_adjust, mse, p_sfc = (da.isel(time=slice(0, args.smoke))
                                        for da in (v, v_flux_adjust, mse, p_sfc))

    logging.info("computing MSE")
    mse = mse.astype("float32").load().rename("mse")
    gc.collect()
    logging.info("computing the adjusted wind")
    v_adj = adjust_wind_by_mse_flux(v, v_flux_adjust, mse)
    v_adj = v_adj.astype("float32").load().rename("v_adj")
    del v, v_flux_adjust
    gc.collect()
    p_sfc = p_sfc.astype("float32").load().rename("p_sfc")

    out = xr.Dataset({"v_adj": v_adj, "mse": mse, "p_sfc": p_sfc})
    out["v_adj"].attrs.update(long_name="meridional wind after the mass adjustment",
                              units="m s-1")
    out["mse"].attrs.update(long_name="moist static energy", units="J kg-1",
                            c_p=C_P_SCRIPTS)
    out["p_sfc"].attrs.update(long_name="surface pressure", units="hPa")
    out.attrs.update(common_attrs(args, out["time"].values.astype("datetime64[ns]")))
    out.attrs.update(sources)
    write(out, output_directory(args) / f"fields_{args.year}{args.month:02d}.nc")


def main(argv=None):
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    logging.info("mode %s, tag %s, %g-hourly, writing under %s", args.mode, args.tag,
                 args.temporal_resolution, output_directory(args))
    if args.mode == "corrected":
        corrected_month(args)
    else:
        archived_month(args)


if __name__ == "__main__":
    main()
