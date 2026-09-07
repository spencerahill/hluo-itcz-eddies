"""Split the column-integrated meridional MSE flux into its terms, one year.

Replaces ``main/adjusted_MMC_stationary_transient_calculation.py`` as the entry
point.  All the calculation lives in ``itcz_eddies``; this file reads the year,
assembles the fields and writes the products.

Three choices the published code makes implicitly are flags here, because all
three are open decisions at the top of ``code-review/FINDINGS.md``:

``--quadrature``
    ``mass`` is a mass-weighted sum with a layer thickness that ends at the
    true surface pressure, and ``trapezoid`` is the published trapezoid rule
    over the NaN-masked levels.  Measured against an exactly known integral by
    ``code-review/checks/quadrature_comparison.py``, the trapezoid rule misses
    by up to 2.0% of the column integral when the surface falls between levels
    and jumps by up to 1.9% when the surface moves 1 hPa across a level; the
    mass-weighted sum stays within 0.13% and varies smoothly.
``--time-mean``
    ``boxcar`` is the published 30-day centred rolling mean, ``block`` a
    non-overlapping 30-day average, ``lanczos`` a low-pass at the same cutoff.
``--barotropic-correction``
    Whether to subtract the column mean of the zonal-mean wind from the
    mean-circulation term alone, which the published code does.

Every output file carries the full configuration in its attributes and in its
directory name, so two configurations cannot overwrite one another.

Usage:
  python scripts/decomposition.py --year 1997
  python scripts/decomposition.py --year 1997 --quadrature trapezoid \
      --time-mean boxcar --barotropic-correction
"""

from __future__ import annotations

import argparse
import datetime
import gc
import logging
import pathlib

import numpy as np
import xarray as xr

from itcz_eddies import paths
from itcz_eddies.columns import (
    col_int,
    col_int_trapz,
    dp_from_sfc_pressure,
    sfc_pressure_mask,
)
from itcz_eddies.decomp import (
    block_time_mean,
    boxcar_time_mean,
    decompose,
    lanczos_time_mean,
    legacy_decompose,
)
from itcz_eddies.era5 import read_data
from itcz_eddies.mse import adjust_wind_by_mse_flux, moist_static_energy

TIME_MEANS = {
    "boxcar": boxcar_time_mean,
    "block": block_time_mean,
    "lanczos": lanczos_time_mean,
}

TERM_LONG_NAMES = {
    "mmc": "mean meridional circulation",
    "stationary": "stationary eddies",
    "transient": "transient eddies",
    "cross_mean_wind_eddy_mse": "time-mean wind times eddy MSE",
    "cross_eddy_wind_mean_mse": "eddy wind times time-mean MSE",
    "total": "full flux",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--quadrature", choices=["mass", "trapezoid"],
                        default="mass")
    parser.add_argument("--time-mean", choices=sorted(TIME_MEANS),
                        default="boxcar")
    parser.add_argument("--barotropic-correction", action="store_true",
                        help="reproduce the published treatment of the MMC wind")
    parser.add_argument("--out", type=pathlib.Path, default=None,
                        help="output directory; default is $ITCZ_PRODUCT_ROOT "
                             "plus a subdirectory naming the configuration")
    parser.add_argument("--boundary", type=float, nargs=4,
                        default=[-60.0, 60.0, 0.0, 360.0],
                        metavar=("LAT0", "LAT1", "LON0", "LON1"))
    parser.add_argument("--temporal-resolution", type=float, default=12.0)
    parser.add_argument("--spatial-resolution", type=float, default=0.5)
    return parser.parse_args(argv)


def config_tag(args):
    """A directory name that says which configuration produced the files."""
    barotropic = "withB" if args.barotropic_correction else "noB"
    return f"{args.quadrature}_{args.time_mean}_{barotropic}"


def months_of(year):
    """Every (year, month) the calculation needs, including the two flanks.

    The centred 30-day time mean reaches 15 days either side of the record, so
    December of the preceding year and January of the following one are read
    and then dropped from the output.
    """
    return ([paths.shift_month(year, 1, -1)]
            + [(year, month) for month in range(1, 13)]
            + [paths.shift_month(year, 12, 1)])


def read_year(args):
    """Assemble v, MSE and surface pressure for one year plus its flanks."""
    boundary = args.boundary
    read = dict(temporal_resolution=args.temporal_resolution,
                spatial_resolution=args.spatial_resolution,
                boundary=boundary)

    pl_files = {var: [] for var in ("T", "Q", "Z", "V")}
    sfc_files, adjust_files = [], []
    for year, month in months_of(args.year):
        for var in pl_files:
            pl_files[var].extend(paths.pl_files_month(var, year, month))
        sfc = paths.sfc_file("SP", year, month)
        if sfc.exists():
            sfc_files.append(sfc)
        adjust = paths.adjust_file(year, month)
        if adjust.exists():
            adjust_files.append(adjust)

    counts = {var: len(files) for var, files in pl_files.items()}
    logging.info("daily pressure-level files found: %s", counts)
    if len(set(counts.values())) != 1:
        raise SystemExit(
            f"the four pressure-level variables have different file counts, "
            f"{counts}; the archive is incomplete for {args.year}"
        )
    if not sfc_files:
        raise SystemExit(f"no surface-pressure files found for {args.year}")
    if not adjust_files:
        raise SystemExit(f"no mass-adjustment files found for {args.year}")

    temp = xr.decode_cf(read_data(pl_files["T"], **read))["T"]
    sphum = xr.decode_cf(read_data(pl_files["Q"], **read))["Q"]
    geopot = xr.decode_cf(read_data(pl_files["Z"], **read))["Z"]
    mse = moist_static_energy(temp, geopot, sphum)
    del temp, sphum, geopot
    gc.collect()

    v = xr.decode_cf(read_data(pl_files["V"], **read))["V"]

    adjust = read_data(adjust_files, **read)
    adjust["time"] = adjust["time"].assign_attrs(
        units="hours since 1900-01-01 00:00:00", calendar="gregorian")
    v_flux_adjust = xr.decode_cf(adjust)["vMSE"]
    v = adjust_wind_by_mse_flux(v, v_flux_adjust, mse)
    del adjust, v_flux_adjust
    gc.collect()

    p_sfc = xr.decode_cf(read_data(sfc_files, **read))["SP"] / 100.0
    p_sfc = p_sfc.sel(time=mse.time)
    return v, mse, p_sfc


def reduce_terms(terms, args, p_sfc, mask):
    """Column-integrate every term with the chosen quadrature."""
    level = terms["total"]["level"]
    if args.quadrature == "mass":
        dp = dp_from_sfc_pressure(level, p_sfc)
        return xr.Dataset({name: col_int(terms[name], dp) for name in terms})
    return xr.Dataset({name: col_int_trapz(terms[name], mask) for name in terms})


def main(argv=None):
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)

    if args.barotropic_correction and args.time_mean != "boxcar":
        raise SystemExit(
            "--barotropic-correction reproduces the published calculation, "
            "which uses the 30-day centred rolling mean; it cannot be combined "
            f"with --time-mean {args.time_mean}"
        )

    v, mse, p_sfc = read_year(args)
    mask = sfc_pressure_mask(mse, p_sfc,
                             below_ground=np.nan if args.quadrature == "trapezoid"
                             else 0.0)

    if args.barotropic_correction:
        terms = legacy_decompose(v, mse, mask, p_sfc,
                                 temporal_resolution=args.temporal_resolution)
        terms["total"] = xr.where(mask == 1, v, np.nan) * xr.where(
            mask == 1, mse, np.nan)
    else:
        terms = decompose(v, mse, time_mean=TIME_MEANS[args.time_mean],
                          zonal_mean=False,
                          temporal_resolution=args.temporal_resolution)

    reduced = reduce_terms(terms, args, p_sfc, mask)
    del terms
    gc.collect()

    # Drop the flanking months that only existed to fill the time-mean window.
    in_year = reduced["time"].dt.year == args.year
    reduced = reduced.isel(time=in_year)

    out_dir = args.out or (paths.product_root() / config_tag(args))
    out_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "year": args.year,
        "quadrature": args.quadrature,
        "time_mean": args.time_mean,
        "barotropic_correction": str(args.barotropic_correction),
        "boundary": " ".join(str(b) for b in args.boundary),
        "temporal_resolution_hours": args.temporal_resolution,
        "spatial_resolution_degrees": args.spatial_resolution,
        "written": datetime.datetime.now().astimezone().isoformat(
            timespec="seconds"),
    }

    for name in reduced.data_vars:
        out = out_dir / f"vMSE_col_{name}_{args.year}.nc"
        if out.exists():
            out.unlink()
        ds = reduced[[name]].assign_attrs(
            description=("column-integrated meridional moist static energy "
                         f"flux, {TERM_LONG_NAMES.get(name, name)}"),
            unit="W m^-1",
            **run_config,
        )
        ds.to_netcdf(out)
        logging.info("wrote %s", out)


if __name__ == "__main__":
    main()
