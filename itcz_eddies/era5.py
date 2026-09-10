"""Reading ERA5 files off disk, with the subsampling the scripts perform.

Haochang Luo's scripts carry 36 definitions of a function named ``read_data``
with seven distinct bodies, 18 of ``read_data2`` with three bodies, and 15
identical copies of ``read_data3``.  Comparing their abstract syntax trees
shows the seven ``read_data`` bodies differ in exactly five ways, each of which
is a keyword argument here:

1. whether the whole ``Dataset`` or a single named variable comes back
   (``var``);
2. whether the latitude coordinate is left ascending or flipped back to ERA5's
   own descending order before returning (``lat_descending``);
3. whether the domain is the whole globe or a latitude-longitude box
   (``boundary``);
4. whether the file is required to have a ``time`` coordinate, or the time
   subsampling is skipped when it has none (``require_time``);
5. whether the pressure levels are restricted to 100 to 1000 hPa
   (``level_range``).

Everything else that separates them is a ``print`` or a ``logging`` call.

Two properties of this reader are worth stating because the scripts' own
comment calls the step "slicing and regridding", which it is not.

**The spatial and temporal steps subsample, they do not regrid.**  The stride
``int(spatial_resolution / builtin_spatial_resolution)`` selects every nth
gridpoint of the stored field.  Going from ERA5's native 0.25 degrees to the
0.5 degrees the manuscript uses therefore throws away three of every four
gridpoints rather than averaging them.

**A stride below one raises.**  ``int(0.5 / 2.0)`` is 0, and xarray rejects a
zero step.  So no archive coarser than the requested ``spatial_resolution`` can
be read, which is the constraint that set the grid of the synthetic archive in
``code-review/synthetic/make_archive.py``.

Times are left encoded.  ``xr.open_mfdataset`` is called with
``decode_times=False``, matching the scripts, because the ERA5 pressure-level
files and the mass-adjustment files carry different time encodings and the
scripts align them by raw hour count.  Call ``xr.decode_cf(ds,
decode_times=True)`` on the result when real datetimes are wanted.
"""

from __future__ import annotations

import logging
import pathlib
from collections.abc import Sequence

import numpy as np
import xarray as xr
from xarray.coding.times import decode_cf_datetime

from .names import LAT_STR, LEV_STR, LON_STR, TIME_STR

__all__ = ["read_data", "read_forecast_data", "read_meanflux_by_valid_time", "stride"]


def stride(requested: float, builtin: float) -> int:
    """Subsampling step, with the failure mode named.

    ``int(requested / builtin)`` is what every one of Haochang Luo's readers
    computes inline.  It gives 0 whenever the stored spacing is coarser than
    the requested spacing, and xarray then raises ``ValueError: step cannot be
    zero`` from deep inside ``.sel``, which is hard to trace back to the cause.
    """
    step = int(requested / builtin)
    if step < 1:
        raise ValueError(
            f"requested spacing {requested} is finer than the stored spacing "
            f"{builtin}, giving a subsampling step of {step}; this reader can "
            "only subsample, never interpolate"
        )
    return step


def read_data(
    paths: str | pathlib.Path | Sequence[str | pathlib.Path],
    temporal_resolution: float,
    spatial_resolution: float,
    var: str | None = None,
    boundary: Sequence[float] | None = None,
    lat_descending: bool = False,
    require_time: bool = True,
    level_range: tuple[float, float] | None = None,
) -> xr.Dataset | xr.DataArray:
    """Open one or more ERA5 files and subsample them in time and space.

    Parameters
    ----------
    paths
        One path or a sequence of paths, concatenated along ``time`` in the
        order given.
    temporal_resolution
        Requested time step, in the same units as the file's own ``time``
        coordinate, which for ERA5 is hours.
    spatial_resolution
        Requested horizontal spacing in degrees.
    var
        Name of a single variable to return.  When ``None`` the whole
        ``Dataset`` comes back.
    boundary
        ``[lat0, lat1, lon0, lon1]`` in degrees.  When ``None`` the whole globe
        is kept, which is what the mass-adjustment scripts do.
    lat_descending
        When ``True`` the returned latitude coordinate is ERA5's own
        descending order.  The mass-adjustment scripts need this because
        ``windspharm`` requires latitudes running north to south.
    require_time
        When ``False``, a file with no ``time`` coordinate is accepted and only
        the horizontal subsampling is applied.
    level_range
        ``(p_bottom_of_range, p_top_of_range)`` in hPa, passed straight to
        ``.sel``.  Only the four climatology scripts use this, with
        ``(100, 1000)``.
    """
    ds = xr.open_mfdataset(
        paths, concat_dim=TIME_STR, combine="nested", decode_times=False
    )
    if len(ds[LAT_STR]) > 1 and ds[LAT_STR][0] > ds[LAT_STR][1]:
        ds = ds.reindex({LAT_STR: ds[LAT_STR][::-1]})

    builtin_spatial = float((ds[LON_STR][1] - ds[LON_STR][0]).values)
    space_step = stride(spatial_resolution, builtin_spatial)

    if boundary is None:
        lat_slice = slice(-90, 90, space_step)
        lon_slice = slice(0, 360, space_step)
    else:
        lat_slice = slice(min(boundary[0], boundary[1]),
                          max(boundary[0], boundary[1]), space_step)
        lon_slice = slice(min(boundary[2], boundary[3]),
                          max(boundary[2], boundary[3]), space_step)

    selection = {LAT_STR: lat_slice, LON_STR: lon_slice}
    if TIME_STR in ds.coords:
        times = ds[TIME_STR]
        builtin_temporal = (times[1] - times[0]).values
        selection[TIME_STR] = slice(
            times[0], times[-1], stride(temporal_resolution, builtin_temporal)
        )
    elif require_time:
        raise KeyError(f"no '{TIME_STR}' coordinate in {paths}")

    ds = ds.sel(selection)
    if level_range is not None:
        ds = ds.sel({LEV_STR: slice(*level_range)})

    logging.info("finished reading data")
    out = ds if var is None else ds[var]
    if lat_descending:
        out = out.reindex({LAT_STR: out[LAT_STR][::-1]})
    return out


def read_forecast_data(
    paths: Sequence[str | pathlib.Path],
    var: str,
    temporal_resolution: float,
    spatial_resolution: float,
    boundary: Sequence[float] | None = None,
    lat_descending: bool = False,
) -> xr.DataArray:
    """Read an ERA5 forecast mean-flux product and put it on a plain time axis.

    ERA5's forecast products are stored on two axes, ``forecast_initial_time``
    and ``forecast_hour``, rather than on a single time axis.  This is the
    consolidation of ``read_mean_flux`` in ``MSEadjust/main.py`` and the
    identical ``read_data3`` that appears in 15 of the analysis scripts.

    Both of those functions rebuild the time coordinate arithmetically, as
    ``forecast_initial_time[0] + temporal_resolution * arange(1, n+1)``, rather
    than by combining the two stored axes.  That is reproduced here, because
    the resulting values are what every downstream ``.sel(time=...)`` in the
    scripts matches against.  It assumes the file list is contiguous and in
    order, which is what ``paths.meanflux_files`` returns.
    """
    ds = xr.open_mfdataset(
        paths, concat_dim="forecast_initial_time", combine="nested",
        decode_times=False,
    )
    if len(ds[LAT_STR]) > 1 and ds[LAT_STR][0] > ds[LAT_STR][1]:
        ds = ds.reindex({LAT_STR: ds[LAT_STR][::-1]})

    builtin_spatial = float((ds[LON_STR][1] - ds[LON_STR][0]).values)
    space_step = stride(spatial_resolution, builtin_spatial)
    if boundary is None:
        lat_slice = slice(-90, 90, space_step)
        lon_slice = slice(0, 360, space_step)
    else:
        lat_slice = slice(min(boundary[0], boundary[1]),
                          max(boundary[0], boundary[1]), space_step)
        lon_slice = slice(min(boundary[2], boundary[3]),
                          max(boundary[2], boundary[3]), space_step)

    ds = ds.sel({
        "forecast_hour": slice(temporal_resolution, 12, temporal_resolution),
        LAT_STR: lat_slice,
        LON_STR: lon_slice,
    })
    stacked = ds.stack(time=("forecast_initial_time", "forecast_hour"))
    data = stacked[var].transpose(TIME_STR, LAT_STR, LON_STR)

    n_times = len(data[TIME_STR])
    times = (ds["forecast_initial_time"].values[0]
             + temporal_resolution * np.arange(1, 1 + n_times))
    out = xr.DataArray(
        data.values,
        dims=[TIME_STR, LAT_STR, LON_STR],
        coords={LAT_STR: data[LAT_STR], LON_STR: data[LON_STR], TIME_STR: times},
    )
    logging.info("finished reading data")
    if lat_descending:
        out = out.reindex({LAT_STR: out[LAT_STR][::-1]})
    return out


def read_meanflux_by_valid_time(
    paths: Sequence[str | pathlib.Path],
    var: str,
    spatial_resolution: float,
    times: Sequence[np.datetime64],
    lat_descending: bool = False,
) -> xr.DataArray:
    """An ERA5 mean-flux product at the given valid times, on a plain time axis.

    Each stored value is the mean rate over the hour ending at its valid
    time, ``forecast_initial_time + forecast_hour``.  The rate bracketing an
    analysis hour ``t`` is the mean of the values valid at ``t`` and at
    ``t + 1 h``: hours 6 and 7 of the run started six hours earlier for 00
    and 12 UTC, and hour 12 of one run with hour 1 of the next for 06 and 18
    UTC.  ``read_forecast_data`` rebuilds the time axis arithmetically the
    way Haochang Luo's scripts did, which serves only the 00 and 12 UTC case;
    this reader serves any analysis hour and reads only the requested slabs,
    which is what ``scripts/assemble_fields.py`` needs for the net energy
    input since 2026-09-09.

    ``times`` are the valid times wanted, as ``datetime64``; every one must
    be in the files, and the result is in their order.  The horizontal
    subsampling is the stride of ``read_data`` taken on the stored grid,
    which for ERA5 runs from the north pole.
    """
    wanted = np.asarray(times, dtype="datetime64[ns]")
    found: dict[np.datetime64, np.ndarray] = {}
    lat = lon = None
    for path in paths:
        with xr.open_dataset(path, decode_times=False) as ds:
            init = ds["forecast_initial_time"]
            units = init.attrs["units"]
            if not units.startswith("hours"):
                raise ValueError(f"forecast_initial_time in {path} is in {units!r}, not hours")
            fh = ds["forecast_hour"].values
            valid = decode_cf_datetime(
                (init.values[:, None] + fh[None, :]).ravel(), units,
                init.attrs.get("calendar", "standard"),
            ).astype("datetime64[ns]").reshape(init.size, fh.size)
            step = stride(spatial_resolution, float(ds[LON_STR][1] - ds[LON_STR][0]))
            if lat is None:
                lat = ds[LAT_STR].values[::step]
                lon = ds[LON_STR].values[::step]
            hit = np.isin(valid, wanted)
            if not hit.any():
                continue
            # One contiguous block of initializations, read at full resolution
            # and strided in memory: the files are chunked one initialization
            # deep and deflated, and a strided read through the netCDF library
            # costs ten times a full one (measured 2026-09-09 on the hourly
            # surface files, 3.2 s against 0.32 s a slab).
            rows = np.flatnonzero(hit.any(axis=1))
            block = ds[var].isel(forecast_initial_time=slice(int(rows[0]), int(rows[-1]) + 1)
                                 ).transpose("forecast_initial_time", "forecast_hour",
                                             LAT_STR, LON_STR).values[:, :, ::step, ::step]
            for i, j in zip(*np.nonzero(hit[rows[0]:rows[-1] + 1])):
                t = valid[rows[0] + i, j]
                if t not in found:
                    found[t] = np.array(block[i, j])
            del block
    missing = [t for t in wanted if t not in found]
    if missing:
        raise ValueError(f"{var}: {len(missing)} of the {wanted.size} valid times wanted "
                         f"are not in the files, the first {missing[0]}")
    out = xr.DataArray(
        np.stack([found[t] for t in wanted]), dims=(TIME_STR, LAT_STR, LON_STR),
        coords={TIME_STR: wanted, LAT_STR: lat, LON_STR: lon}, name=var,
    )
    if bool(lat[0] > lat[-1]) != lat_descending:
        out = out.isel({LAT_STR: slice(None, None, -1)})
    logging.info("read %s at %d valid times", var, wanted.size)
    return out
