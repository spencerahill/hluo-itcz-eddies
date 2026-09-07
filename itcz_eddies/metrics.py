"""ITCZ position metrics and flux conversions.

``precip_centroid`` is moved verbatim from
``main/adjusted_MMC_stationary_transient_calculation.py``, where it is one of
two bodies of that name across the scripts; the other differs only in a local
variable name and behaves identically, checked and recorded in the Resolved
section of ``code-review/FINDINGS.md``.

Three properties of the moved function are worth stating, since none is
visible at the call site.

It hardcodes the latitude band 20S to 20N, and it takes the *latitude* grid
spacing from the *longitude* coordinate, which is right only while the two
spacings are equal.  They are equal in ERA5 and in GPCP as this project reads
them.

Its assignment ``top_plus_bottom[:, 0] = 0`` addresses the second axis
positionally, so the input has to be two-dimensional with latitude second.

The centroid it returns is the grid latitude nearest the half-area point,
found with ``idxmin``, so it is quantized to the grid.  At half a degree that
quantization is 0.5 degrees, which is not negligible next to the seasonal
excursions the manuscript plots.  ``energy_flux_equator`` below has the same
property and offers an interpolated alternative.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from .names import LAT_STR, LON_STR

__all__ = [
    "RAD_EARTH",
    "precip_centroid",
    "energy_flux_equator",
    "flux_to_petawatts",
    "cross_equatorial_transport",
]

# Earth's radius as set in the analysis scripts.  puffins.constants.RAD_EARTH
# is the same value.
RAD_EARTH = 6.371e6


def precip_centroid(Data):
    P = Data.sel(latitude = slice(-20,20))
    spatial_resolution = abs(P.longitude[1] - P.longitude[0])
    P_area = P*np.cos(P.latitude/180*np.pi)
    top_plus_bottom = (P_area+P_area.shift(latitude = 1))
    top_plus_bottom[:,0] = 0
    P_cent = (top_plus_bottom/2*spatial_resolution).cumsum("latitude")
    median = 0.5*(top_plus_bottom/2*spatial_resolution).sum("latitude")
    lat = abs(P_cent-median).idxmin(dim = "latitude")
    return lat

def energy_flux_equator(flux, lat_str=LAT_STR, lon_str=LON_STR,
                        interpolate=False):
    """Latitude where the zonal-mean flux changes sign.

    The published definition, reproduced when ``interpolate`` is ``False``, is
    ``abs(flux.mean(longitude)).idxmin(latitude)``: the grid latitude at which
    the absolute value of the zonal-mean flux is smallest.  That is quantized
    to the grid, and it can land far from the zero crossing when the profile
    is flat near the equator or when it crosses zero more than once.

    With ``interpolate=True`` the zero crossing is found by linear
    interpolation between the two grid latitudes that bracket the sign change
    closest to the ``idxmin`` latitude.  The difference between the two is a
    measure of what the quantization costs.

    Parameters
    ----------
    flux
        Flux with a longitude dimension, or already zonally averaged.
    interpolate
        Whether to interpolate the crossing rather than returning a grid
        latitude.
    """
    zonal = flux.mean(lon_str) if lon_str in flux.dims else flux
    coarse = abs(zonal).idxmin(lat_str)
    if not interpolate:
        return coarse

    lat = zonal[lat_str]
    below = zonal.isel({lat_str: slice(None, -1)})
    above = zonal.isel({lat_str: slice(1, None)})
    lat_below = lat.isel({lat_str: slice(None, -1)}).values
    lat_above = lat.isel({lat_str: slice(1, None)}).values

    below_vals = below.values
    above_vals = above.values
    crosses = np.sign(below_vals) != np.sign(above_vals)
    with np.errstate(invalid="ignore", divide="ignore"):
        weight = below_vals / (below_vals - above_vals)
    crossing = lat_below + weight * (lat_above - lat_below)
    crossing = np.where(crosses, crossing, np.nan)

    axis = below.get_axis_num(lat_str)
    coarse_vals = np.expand_dims(coarse.values, axis)
    distance = np.abs(crossing - coarse_vals)
    distance = np.where(np.isnan(crossing), np.inf, distance)
    nearest = np.argmin(distance, axis=axis)
    picked = np.take_along_axis(crossing, np.expand_dims(nearest, axis),
                                axis=axis).squeeze(axis)
    return xr.DataArray(picked, dims=coarse.dims, coords=coarse.coords)


def flux_to_petawatts(flux_per_meter, lat_str=LAT_STR, radius=RAD_EARTH):
    """Convert a flux per unit length into a transport around a latitude circle.

    The scripts write this inline as ``Data * 1e-15 * 2 * pi * a *
    cos(latitude)``.  The input is W/m, the length of the latitude circle is
    ``2 * pi * a * cos(latitude)`` in meters, and the ``1e-15`` takes watts to
    petawatts.
    """
    circumference = (2 * np.pi * radius
                     * np.cos(flux_per_meter[lat_str] * np.pi / 180.0))
    return flux_per_meter * circumference * 1e-15


def cross_equatorial_transport(transport, lat_str=LAT_STR, lon_str=LON_STR,
                               band=(-5.0, 5.0)):
    """Transport averaged over an equatorial band, in whatever units it carries.

    ``CET.py`` averages the full-flux transport over 5S to 5N in latitude and
    over all longitudes.  Averaging rather than taking the value at the
    equator smooths the grid noise; it also means the quantity is not the
    transport across the equator itself, which is what the name suggests.
    """
    selected = transport.sel({lat_str: slice(*band)})
    dims = [lat_str] + ([lon_str] if lon_str in selected.dims else [])
    return selected.mean(dim=dims)
