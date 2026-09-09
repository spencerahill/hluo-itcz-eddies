"""Assemble one month of the fields the flux decomposition consumes.

Reads the daily ERA5 pressure-level files for temperature, specific humidity,
geopotential and meridional wind, the monthly surface-pressure file and the
monthly mass-adjustment file; forms the moist static energy and the adjusted
wind exactly as ``scripts/decomposition.py`` does; and writes them, with the
surface pressure, to one file ``fields_YYYYMM.nc`` under ``ITCZ_FIELDS_ROOT``.

Why this exists.  A yearly decomposition needs 14 months of four variables,
about 1,700 daily files of which only two of the 24 hourly steps and one
gridpoint in four are used, and reading that took Haochang Luo's script most
of a 24-hour walltime.  Assembling each month in its own job runs the fourteen
reads in parallel and leaves a 3 GB file per month that the decomposition, or
any later calculation on the same fields, reads in seconds.  Every month is
independent, so a partial archive is usable and a failed month is re-run
alone.

What is stored, all on the analysis grid (12-hourly, 0.5 degree, 60S to 60N
by default) and all as float32, which is the precision of the ERA5 source:

``v_adj``
    Meridional wind after the mass adjustment of Appendix A, m/s.
``mse``
    Moist static energy, J/kg, with the scripts' specific heat of 1004.0.
``p_sfc``
    Surface pressure in hPa.

Below-ground points are left as ERA5 stores them, extrapolated; nothing here
masks.  The consumer decides how to treat them.

Usage:
  python scripts/assemble_fields.py --year 1997 --month 7
"""

from __future__ import annotations

import argparse
import datetime
import gc
import logging
import pathlib

import xarray as xr

from itcz_eddies import paths
from itcz_eddies.era5 import read_data
from itcz_eddies.mse import C_P_SCRIPTS, adjust_wind_by_mse_flux, moist_static_energy


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--month", type=int, required=True)
    parser.add_argument("--out", type=pathlib.Path, default=None,
                        help="output file; default is $ITCZ_FIELDS_ROOT/fields_YYYYMM.nc")
    parser.add_argument("--boundary", type=float, nargs=4,
                        default=[-60.0, 60.0, 0.0, 360.0],
                        metavar=("LAT0", "LAT1", "LON0", "LON1"))
    parser.add_argument("--temporal-resolution", type=float, default=12.0)
    parser.add_argument("--spatial-resolution", type=float, default=0.5)
    return parser.parse_args(argv)


def read_month(year, month, boundary, temporal_resolution, spatial_resolution):
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

    pl_files = {var: paths.pl_files_month(var, year, month)
                for var in ("T", "Q", "Z", "V")}
    counts = {var: len(files) for var, files in pl_files.items()}
    logging.info("daily pressure-level files found for %d-%02d: %s",
                 year, month, counts)
    expected = paths.last_day(year, month)
    if any(count != expected for count in counts.values()):
        raise SystemExit(
            f"expected {expected} daily files per variable for {year}-{month:02d}, "
            f"found {counts}"
        )
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


def main(argv=None):
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    v, v_flux_adjust, mse, p_sfc, sources = read_month(
        args.year, args.month, args.boundary,
        args.temporal_resolution, args.spatial_resolution)

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
    out.attrs.update(
        year=args.year, month=args.month,
        boundary=" ".join(str(b) for b in args.boundary),
        temporal_resolution_hours=args.temporal_resolution,
        spatial_resolution_degrees=args.spatial_resolution,
        era5_root=str(paths.era5_root()),
        written=datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        **sources,
    )

    dest = args.out or paths.fields_file(args.year, args.month)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest.unlink()
    out.to_netcdf(dest)
    logging.info("wrote %s: %s", dest, dict(out.sizes))


if __name__ == "__main__":
    main()
