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
    non-overlapping 30-day average, ``lanczos`` a low-pass whose cutoff and
    window are ``--lanczos-cutoff-days`` and ``--lanczos-lobes``, and
    ``ideal`` the spectral low-pass at the same cutoff, which is an exact
    projection and treats the record as periodic.  ``--flank-months`` sets how
    many months either side of the year are read to feed the time mean and
    then dropped: one covers a 121-weight Lanczos, whose undefined half-window
    is 30 days; a 241-weight window needs 60 days, so three months, since two
    give only 59 days after December.
``--barotropic-correction``
    Whether to subtract the column mean of the zonal-mean wind from the
    mean-circulation term alone, which the published code does.
``--zonal-mean``
    ``plain`` is the arithmetic average around a latitude circle over the
    points above ground, which is the published code's ``skipna`` mean.
    ``mass-weighted`` weights each longitude by that level's layer thickness.
    Measured in ``code-review/checks/zonal_mean_weighting.py``, the weighted
    form closes the five-term split to 7.0e-16 of the peak under a column
    integral taken to each longitude's own surface pressure, where the plain
    form leaves 1.0e-3.  It also redefines the stationary eddy as the
    departure from a mass-weighted zonal mean.

A caution that the flags cannot enforce.  The archived adjustment files under
``ITCZ_ADJUST_ROOT`` were derived from a budget integrated with the trapezoid
rule, and four fifths of what they add is the flux of the surface layer that
rule drops (F22 in ``code-review/FINDINGS.md``).  Under ``--quadrature mass``
that layer is kept, so the adjusted total counts it twice: in July 1997 the
adjusted zonal-mean total comes out northward at 5S where the physical
transport is southward.  The terms still sum to the total, and the effect of
the zonal-mean weighting on the split is measured correctly, but the total
itself is not the flux the manuscript needs until the adjustment is rebuilt
on the mass-weighted quadrature.

Where the input comes from.  If ``scripts/assemble_fields.py`` has written
``fields_YYYYMM.nc`` under ``ITCZ_FIELDS_ROOT`` for the year and its two
flanking months, those are read; otherwise the daily ERA5 files are read
directly, which is about 1,700 files for one year.

How the exact split is computed.  The fields for the year plus flanks are
read one band at a time as float32, and the split runs in float64 over bands of
``--lat-band`` latitudes, since every operation in it (the time mean, the
zonal mean, the column integral) acts within a latitude.  A band of ten
latitudes on the 60S to 60N half-degree grid is about 1.8 GB per float64
field at twelve samples a day and twice that at six, and only one band is
held at a time.

Every output file carries the full configuration in its attributes and in its
directory name, so two configurations cannot overwrite one another.  Besides
the column-integrated maps of each term, the exact split writes
``zonal_mean_fields_YYYY.nc``, holding the zonal means of the time-mean wind
and MSE and the zonal-mean layer thickness on (time, level, latitude), from
which any barotropic correction of the mean-circulation term can be formed
without rerunning.

Since 2026-09-09 the assembled fields come from ``scripts/assemble_fields.py``
in its corrected mode, and the defaults here are the pipeline of
``pipeline-2026-09-09/pipeline-recommendation.pdf`` at the project root: the
mass-flux split with the Lanczos low-pass at 30 days over 481 weights
(``--lanczos-lobes 240`` at six-hourly sampling is the 60-day half window
of decision D4, so ``--flank-months 3``), four samples a day (decision D6),
interfaces midway in log pressure (``--interfaces``, passed to the layer
thickness, decision D1), and the fields read from the subdirectory of
``ITCZ_FIELDS_ROOT`` that the assembly names for the configuration
(``--fields-tag``, ``auto`` by default; the empty tag reads the root, where
the archived-mode files of 2026-09-08 sit).  When the fields files carry
the assembly's zonal-mean net mass transport series, their time means go
into ``zonal_mean_fields_YYYY.nc`` as ``net_mass_daily_bar`` and
``net_mass_corr_bar``, from which the reduce step of
``code-review/checks/zonal_mean_weighting_era5.py`` forms the
mean-circulation term with its net mass transport taken from the daily-mean
requirement (decision D6).

Usage:
  python scripts/decomposition.py --year 1997
  python scripts/decomposition.py --year 1997 --quadrature trapezoid --split wind \
      --time-mean boxcar --barotropic-correction
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
    decompose_mass_flux,
    ideal_time_mean,
    lanczos_time_mean,
    legacy_decompose,
)
from itcz_eddies.era5 import read_data
from itcz_eddies.mse import adjust_wind_by_mse_flux, moist_static_energy

TIME_MEANS = {
    "boxcar": boxcar_time_mean,
    "block": block_time_mean,
    "lanczos": lanczos_time_mean,
    "ideal": ideal_time_mean,
}

TERM_LONG_NAMES = {
    "mmc": "mean meridional circulation",
    "stationary": "stationary eddies",
    "transient": "transient eddies",
    "cross_mean_wind_eddy_mse": "time-mean wind times eddy MSE",
    "cross_eddy_wind_mean_mse": "eddy wind times time-mean MSE",
    "zonal_cross": "zonal-mean field times zonal departure, zero in a plain zonal mean",
    "total": "full flux",
}

FIVE_TERMS = ["mmc", "stationary", "transient",
              "cross_mean_wind_eddy_mse", "cross_eddy_wind_mean_mse"]
SIX_TERMS = FIVE_TERMS + ["zonal_cross"]
ZONAL_MEAN_FIELDS = ["v_bar_zm", "h_bar_zm"]
MASS_FLUX_ZONAL_MEAN_FIELDS = ["m_bar_zm", "h_bar_zm", "dp_bar_zm"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--quadrature", choices=["mass", "trapezoid"],
                        default="mass")
    parser.add_argument("--time-mean", choices=sorted(TIME_MEANS),
                        default="lanczos")
    parser.add_argument("--lanczos-cutoff-days", type=float, default=30.0,
                        help="low-pass cutoff period for lanczos and ideal "
                             "(decision 3)")
    parser.add_argument("--flank-months", type=int, default=3,
                        help="months read either side of the year for the "
                             "time mean, then dropped")
    parser.add_argument("--lanczos-lobes", type=int, default=240,
                        help="half-width of the Lanczos window in time steps; "
                             "the window has 2*lobes+1 weights; 240 at six-hourly "
                             "sampling is the 60-day half window of decision D4")
    parser.add_argument("--barotropic-correction", action="store_true",
                        help="reproduce the published treatment of the MMC wind")
    parser.add_argument("--zonal-mean", choices=["plain", "mass-weighted"],
                        default="plain",
                        help="how longitudes are weighted in the zonal mean")
    parser.add_argument("--split", choices=["wind", "mass-flux"], default="mass-flux",
                        help="split the wind and the MSE (five zonal-mean terms plus "
                             "the sixth pointwise term), or the layer mass flux "
                             "v dp/g and the MSE (five terms, exact under a moving "
                             "surface; the pipeline recommendation of 2026-09-09). "
                             "The mass-flux split carries its own zonal-mean "
                             "weighting and ignores --zonal-mean and --quadrature")
    parser.add_argument("--lat-band", type=int, default=10,
                        help="latitudes per band in the exact split")
    parser.add_argument("--out", type=pathlib.Path, default=None,
                        help="output directory; default is $ITCZ_PRODUCT_ROOT "
                             "plus a subdirectory naming the configuration")
    parser.add_argument("--boundary", type=float, nargs=4,
                        default=[-60.0, 60.0, 0.0, 360.0],
                        metavar=("LAT0", "LAT1", "LON0", "LON1"))
    parser.add_argument("--temporal-resolution", type=float, default=6.0)
    parser.add_argument("--spatial-resolution", type=float, default=0.5)
    parser.add_argument("--interfaces", choices=["logp", "midpoint"], default="logp",
                        help="where the layer interfaces sit in the layer thickness "
                             "(decision D1)")
    parser.add_argument("--fields-tag", default="auto",
                        help="subdirectory of $ITCZ_FIELDS_ROOT holding the assembled "
                             "fields; auto names the corrected assembly at this "
                             "sampling and these interfaces, and an empty string "
                             "names the root")
    return parser.parse_args(argv)


def config_tag(args):
    """A directory name that says which configuration produced the files."""
    barotropic = "withB" if args.barotropic_correction else "noB"
    weighting = "massZM" if args.zonal_mean == "mass-weighted" else "plainZM"
    if args.split == "mass-flux":
        weighting = "mflux"
    time_mean = args.time_mean
    if time_mean == "lanczos":
        time_mean = f"lanczos{args.lanczos_cutoff_days:g}d{args.lanczos_lobes}"
    elif time_mean == "ideal":
        time_mean = f"ideal{args.lanczos_cutoff_days:g}d"
    return (f"{args.quadrature}_{time_mean}_{barotropic}_{weighting}"
            f"_{args.temporal_resolution:g}h_{args.interfaces}")


def fields_tag(args):
    """The subdirectory of the fields root the run reads, from ``--fields-tag``."""
    if args.fields_tag != "auto":
        return args.fields_tag
    return paths.fields_tag("corrected", args.temporal_resolution, args.interfaces)


def time_mean_kwargs(args):
    """Keyword arguments the chosen time mean takes, for ``decompose``."""
    kwargs = {"temporal_resolution": args.temporal_resolution}
    if args.time_mean == "lanczos":
        kwargs.update(cutoff_days=args.lanczos_cutoff_days,
                      lobes=args.lanczos_lobes)
    elif args.time_mean == "ideal":
        kwargs.update(cutoff_days=args.lanczos_cutoff_days)
    return kwargs


def months_of(year, flank_months=1):
    """Every (year, month) the calculation needs, including the flanks.

    The time mean reaches into the months either side of the record, so
    ``flank_months`` months before January and after December are read and
    then dropped from the output.
    """
    return ([paths.shift_month(year, 1, -k)
             for k in range(flank_months, 0, -1)]
            + [(year, month) for month in range(1, 13)]
            + [paths.shift_month(year, 12, k)
               for k in range(1, flank_months + 1)])


def read_year_from_era5(args):
    """Assemble v, MSE and surface pressure for one year plus its flanks, lazily."""
    boundary = args.boundary
    read = dict(temporal_resolution=args.temporal_resolution,
                spatial_resolution=args.spatial_resolution,
                boundary=boundary)

    pl_files = {var: [] for var in ("T", "Q", "Z", "V")}
    sfc_files, adjust_files = [], []
    for year, month in months_of(args.year, args.flank_months):
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


def fields_files(args):
    """The assembled monthly files for the year and its flanks, or ``None``.

    The twelve months of the year must all exist; a missing flank is logged
    and skipped, which is the situation at the end of the record, where the
    following January has ERA5 fields and no adjustment file.
    """
    files = []
    for year, month in months_of(args.year, args.flank_months):
        path = paths.fields_file(year, month, tag=fields_tag(args))
        if path.exists():
            files.append(path)
        elif year == args.year:
            return None
        else:
            logging.warning("flank month %d-%02d has no assembled fields file "
                            "%s; the time mean will be one-sided there",
                            year, month, path)
    return files


def read_year_from_fields(args, files):
    """v, MSE and surface pressure from the assembled monthly files, lazily,
    with the assembly's zonal-mean net mass transport series loaded, or
    ``None`` for files of the archived mode, which carry none."""
    logging.info("reading %d assembled fields files from %s", len(files),
                 files[0].parent)
    ds = xr.open_mfdataset(files, concat_dim="time", combine="nested",
                           chunks={"latitude": args.lat_band})
    lat0, lat1, lon0, lon1 = args.boundary
    ds = ds.sel(latitude=slice(min(lat0, lat1), max(lat0, lat1)),
                longitude=slice(min(lon0, lon1), max(lon0, lon1)))
    series = None
    if all(name in ds for name in ("net_mass_daily", "net_mass_corr_zm")):
        series = ds[["net_mass_daily", "net_mass_corr_zm"]].load()
    else:
        logging.warning("the fields files carry no net mass transport series "
                        "(archived mode), so the zonal-mean fields will not "
                        "carry the daily-mean net mass transport of decision D6")
    return ds["v_adj"], ds["mse"], ds["p_sfc"], series


def read_year(args):
    files = fields_files(args)
    if files is None:
        if args.temporal_resolution != 12:
            raise SystemExit(
                f"no assembled fields for {args.year} under "
                f"{paths.fields_root() / fields_tag(args)}; reading ERA5 directly "
                "applies the archived adjustment, which exists at 00 and 12 UTC "
                "only, so run scripts/assemble_fields.py first"
            )
        logging.info("no assembled fields for %d under %s; reading ERA5 directly",
                     args.year, paths.fields_root())
        return read_year_from_era5(args) + (None,), "era5"
    return read_year_from_fields(args, files), "fields"


# ------------------------------------------------------- the exact split


def exact_split_by_band(v, mse, p_sfc, args):
    """Run ``decompose`` band by band in latitude, in float64, in memory.

    Returns the column-integrated maps of the six pointwise terms and the
    total, and the zonal-mean fields the mean-circulation term is built
    from, both concatenated over latitude.
    """
    n_lat = v.sizes["latitude"]
    band = args.lat_band
    kwargs = time_mean_kwargs(args)
    reduced_bands, zm_bands = [], []
    worst_gap = 0.0
    for i0 in range(0, n_lat, band):
        started = _time.monotonic()
        sl = slice(i0, min(i0 + band, n_lat))
        vb = v.isel(latitude=sl).astype("float64").load()
        hb = mse.isel(latitude=sl).astype("float64").load()
        pb = p_sfc.isel(latitude=sl).astype("float64").load()
        dp = dp_from_sfc_pressure(vb["level"], pb, interfaces=args.interfaces
                                  ).transpose(*vb.dims)
        if args.split == "mass-flux":
            # The split of the layer mass flux: the column integral is the
            # sum over levels, since dp/g is inside every term, and five
            # terms are exact with no sixth.
            terms = decompose_mass_flux(vb, hb, dp,
                                        time_mean=TIME_MEANS[args.time_mean],
                                        zonal_mean_terms=False,
                                        zonal_mean_fields=True, **kwargs)
            zm = terms[MASS_FLUX_ZONAL_MEAN_FIELDS]
            terms = terms.drop_vars(MASS_FLUX_ZONAL_MEAN_FIELDS)
            del vb, hb
            reduced = xr.Dataset({name: terms[name].sum("level", skipna=False)
                                  for name in terms.data_vars})
            check_terms = FIVE_TERMS
            del terms, dp
        else:
            if args.zonal_mean == "mass-weighted":
                weights = dp
            else:
                # The published skipna mean: every point above ground counts once.
                weights = xr.where(dp > 0, 1.0, 0.0)

            terms = decompose(vb, hb, time_mean=TIME_MEANS[args.time_mean],
                              zonal_mean_terms=False, weights=weights,
                              zonal_mean_fields=True, **kwargs)
            zm = terms[ZONAL_MEAN_FIELDS]
            zm["dp_zm"] = dp.mean("longitude")
            terms = terms.drop_vars(ZONAL_MEAN_FIELDS)
            del vb, hb

            if args.quadrature == "mass":
                reduced = xr.Dataset({name: col_int(terms[name], dp)
                                      for name in terms.data_vars})
            else:
                mask = xr.where(terms["total"]["level"] <= pb, 1.0, np.nan
                                ).transpose(*terms["total"].dims)
                reduced = xr.Dataset({name: col_int_trapz(terms[name], mask)
                                      for name in terms.data_vars})
            check_terms = SIX_TERMS
            del terms, dp, weights
        gc.collect()

        # The pointwise terms sum to the total at every gridpoint, and the
        # column integral is linear, so the same holds for the maps.  Anything
        # beyond rounding here is a defect in the split, not in the physics.
        finite = reduced["total"].notnull()
        six = sum(reduced[name] for name in check_terms)
        scale = float(abs(reduced["total"]).where(finite).max())
        gap = float(abs(six - reduced["total"]).where(finite).max()) / scale
        worst_gap = max(worst_gap, gap)
        logging.info("band %3d:%3d  latitudes %6.1f to %6.1f  six-term gap "
                     "%.1e of the band peak  %.0f s", sl.start, sl.stop,
                     float(v["latitude"][sl.start]),
                     float(v["latitude"][sl.stop - 1]), gap,
                     _time.monotonic() - started)
        reduced_bands.append(reduced)
        zm_bands.append(zm)
    logging.info("worst six-term gap over all bands: %.1e of the band peak",
                 worst_gap)
    return (xr.concat(reduced_bands, "latitude"),
            xr.concat(zm_bands, "latitude"),
            worst_gap)


def attach_series_time_means(zonal_mean_fields, series, args):
    """The run's time mean of the assembly's zonal-mean net mass transport series.

    ``net_mass_daily_bar`` is the mean-circulation term's net mass transport
    under decision D6, the daily-mean requirement plus the corrected wind's
    vapor transport, time-averaged as the terms are; ``net_mass_corr_bar`` is
    the same for the corrected wind's own column mass transport, which must
    equal the level sum of ``m_bar_zm`` to rounding, an identity control the
    reduce step of ``code-review/checks/zonal_mean_weighting_era5.py`` prints.
    """
    kwargs = time_mean_kwargs(args)
    time_mean = TIME_MEANS[args.time_mean]
    lat = zonal_mean_fields["latitude"]
    for src, dest in (("net_mass_daily", "net_mass_daily_bar"),
                      ("net_mass_corr_zm", "net_mass_corr_bar")):
        arr = series[src].sel(latitude=lat).astype("float64")
        zonal_mean_fields[dest] = time_mean(arr, **kwargs)
    return zonal_mean_fields


# ------------------------------------------------------- the legacy split


def legacy_split(v, mse, p_sfc, args):
    """The published calculation, lazily on the whole domain, as before."""
    mask = sfc_pressure_mask(mse, p_sfc,
                             below_ground=np.nan if args.quadrature == "trapezoid"
                             else 0.0)
    terms = legacy_decompose(v, mse, mask, p_sfc,
                             temporal_resolution=args.temporal_resolution)
    terms["total"] = xr.where(mask == 1, v, np.nan) * xr.where(
        mask == 1, mse, np.nan)
    level = terms["total"]["level"]
    if args.quadrature == "mass":
        dp = dp_from_sfc_pressure(level, p_sfc)
        return xr.Dataset({name: col_int(terms[name], dp) for name in terms})
    return xr.Dataset({name: col_int_trapz(terms[name], mask) for name in terms})


# ---------------------------------------------------------------- driver


def main(argv=None):
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)

    if args.barotropic_correction and args.zonal_mean != "plain":
        raise SystemExit(
            "--barotropic-correction reproduces the published calculation, "
            "which takes an arithmetic zonal mean; it cannot be combined with "
            "--zonal-mean mass-weighted"
        )
    if args.barotropic_correction and args.time_mean != "boxcar":
        raise SystemExit(
            "--barotropic-correction reproduces the published calculation, "
            "which uses the 30-day centred rolling mean; it cannot be combined "
            f"with --time-mean {args.time_mean}"
        )
    if args.barotropic_correction and args.quadrature != "trapezoid":
        raise SystemExit(
            "--barotropic-correction reproduces the published calculation, "
            "which masks below-ground points to NaN and integrates with the "
            "trapezoid rule; under the mass-weighted quadrature the layer that "
            "straddles the surface is NaN and the column integral is undefined"
        )
    if args.barotropic_correction and args.split != "wind":
        raise SystemExit(
            "--barotropic-correction reproduces the published calculation, "
            "which splits the wind and the MSE; pass --split wind"
        )

    (v, mse, p_sfc, series), source = read_year(args)
    zonal_mean_fields = None
    worst_gap = None
    if args.barotropic_correction:
        reduced = legacy_split(v, mse, p_sfc, args)
    else:
        logging.info("streaming the fields one latitude band at a time: %s",
                     dict(v.sizes))
        reduced, zonal_mean_fields, worst_gap = exact_split_by_band(
            v, mse, p_sfc, args)
        if series is not None:
            zonal_mean_fields = attach_series_time_means(
                zonal_mean_fields, series, args)
        del v, mse
        gc.collect()

    # Drop the flanking months that only existed to fill the time-mean window.
    in_year = reduced["time"].dt.year == args.year
    reduced = reduced.isel(time=in_year)
    if zonal_mean_fields is not None:
        zonal_mean_fields = zonal_mean_fields.isel(
            time=zonal_mean_fields["time"].dt.year == args.year)

    out_dir = args.out or (paths.product_root() / config_tag(args))
    out_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "year": args.year,
        "quadrature": args.quadrature,
        "time_mean": args.time_mean,
        "lanczos_cutoff_days": args.lanczos_cutoff_days,
        "lanczos_lobes": args.lanczos_lobes,
        "flank_months": args.flank_months,
        "barotropic_correction": str(args.barotropic_correction),
        "zonal_mean": args.zonal_mean,
        "split": args.split,
        "lat_band": args.lat_band,
        "interfaces": args.interfaces,
        "fields_tag": fields_tag(args),
        "input_source": source,
        "boundary": " ".join(str(b) for b in args.boundary),
        "temporal_resolution_hours": args.temporal_resolution,
        "spatial_resolution_degrees": args.spatial_resolution,
        "written": datetime.datetime.now().astimezone().isoformat(
            timespec="seconds"),
    }
    if worst_gap is not None:
        run_config["six_term_gap_relative_to_band_peak"] = worst_gap

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

    if zonal_mean_fields is not None:
        out = out_dir / f"zonal_mean_fields_{args.year}.nc"
        if out.exists():
            out.unlink()
        long_names = {
            "v_bar_zm": ("zonal mean of the time-mean meridional wind", "m s-1"),
            "m_bar_zm": ("arithmetic zonal mean of the time-mean layer mass flux v dp/g",
                         "kg m-1 s-1"),
            "h_bar_zm": ("zonal mean of the time-mean moist static energy, weighted by "
                         "the time-mean layer thickness in the mass-flux split",
                         "J kg-1"),
            "dp_zm": ("arithmetic zonal mean of the layer thickness", "Pa"),
            "dp_bar_zm": ("arithmetic zonal mean of the time-mean layer thickness", "Pa"),
            "net_mass_daily_bar": ("time mean of the zonal-mean column mass transport "
                                   "with the dry part from the daily-mean requirement, "
                                   "the mean-circulation net mass transport of decision "
                                   "D6", "kg m-1 s-1"),
            "net_mass_corr_bar": ("time mean of the zonal-mean column mass transport of "
                                  "the corrected wind, from the assembly; equals the "
                                  "level sum of m_bar_zm to rounding", "kg m-1 s-1"),
        }
        for name in zonal_mean_fields.data_vars:
            long_name, units = long_names[name]
            zonal_mean_fields[name].attrs.update(long_name=long_name, units=units)
        zonal_mean_fields.assign_attrs(
            description=("the zonal-mean time-mean fields the mean-circulation "
                         "term is the product of, with the zonal-mean layer "
                         "thickness; the zonal means use the run's weighting"),
            **run_config,
        ).to_netcdf(out)
        logging.info("wrote %s", out)


if __name__ == "__main__":
    main()
