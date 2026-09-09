r"""The column mass budget and the barotropic correction that closes it.

Reanalysis winds on pressure levels do not conserve column mass: the
analysis produces the wind and the surface pressure separately, and the
interpolation to pressure levels adds its own error (Trenberth 1991,
*J. Climate* 4, 707).  For an energy transport this matters at first order,
because the column-integrated MSE flux is nearly the column mass transport
times a column-mean MSE near 3.2e5 J/kg: a spurious net zonal-mean mass
transport equivalent to a barotropic wind of 1 cm/s carries 1.3 PW across
the equator (F17).  Measured on ERA5 for July 1997, the raw wind's
zonal-mean column mass transport is 1 to 2 cm/s in the deep tropics (F19).

The standard remedy is to correct the divergent part of the wind
barotropically so that the column mass budget closes.  The budget closed
here is the one for dry air (Trenberth 1991, his equation 3; Mayer, Mayer
and Haimberger 2021, *J. Climate* 34, 3955, their equation 4),

.. math::

    \frac{\partial}{\partial t}\frac{p_s - g W}{g}
      + \nabla\cdot\int (1 - q)\,\mathbf{v}\,\frac{dp}{g} = 0,

with :math:`W` the total column water vapor, because dry air has no sources
or sinks, so no forecast evaporation or precipitation enters and every term
is an analysis quantity.  The correction is the divergent barotropic wind
whose column integral removes the residual, found by a spherical-harmonic
inversion (``puffins.budget_adj.uv_col_budg_adj``), and it is applied
unchanged at every level.

Two things the pipeline recommendation of 2026-09-09 established about how
the tendency has to be formed.  The semidiurnal pressure tide has a period
of 12 hours, so at 00 and 12 UTC its column mass tendency is the same fixed
zonal-wavenumber-2 pattern, about 1.7e-3 kg/m2/s in the tropics, thirty
times the evaporation minus precipitation term; a tendency formed from
12-hourly (or 6-hourly) samples cannot see it, while the analyzed wind
divergence contains it, so the residual acquires a spurious wavenumber-2
pattern and the correction a spurious wavenumber-2 barotropic wind of order
0.5 m/s (Trenberth 1991 found the same in twice-daily ECMWF analyses).
The tendency therefore comes from the HOURLY surface pressure and column
vapor, centered on the analysis time, which ERA5 archives as 2-D fields.
And the zonal-mean column mass transport of the corrected wind equals the
physical requirement, the rate of change of the dry mass north of each
latitude, which is the polar-cap integral ``polar_cap_transport`` forms
from the same tendency.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from .columns import col_int
from .mse import adjusted_col_fluxes
from .names import LAT_STR, LEV_STR, LON_STR, TIME_STR

__all__ = [
    "RAD_EARTH",
    "centered_tendency",
    "dry_air_mass_correction",
    "mass_correction_from_columns",
    "polar_cap_transport",
]

RAD_EARTH = 6.371e6


def centered_tendency(hourly, times, half_window_hours=1, time_str=TIME_STR):
    """Centered time derivative of an hourly series at the given times, per second.

    ``(x(t + w) - x(t - w)) / (2 w)`` with ``w = half_window_hours``.  Both
    ``t + w`` and ``t - w`` must be present in ``hourly``.

    With ``half_window_hours=1`` this is the instantaneous tendency the mass
    correction is formed with, which sees the semidiurnal tide as the
    analyzed divergence does.  With ``half_window_hours=12`` the difference
    spans 24 hours, cancels every 24-hour and 12-hour periodic component
    exactly, and includes what the assimilation adds at its window
    boundaries: it is the daily-mean tendency from which decision D6 takes
    the mean-circulation term's net mass transport.
    """
    delta = np.timedelta64(half_window_hours, "h")
    later = hourly.sel({time_str: times + delta})
    earlier = hourly.sel({time_str: times - delta})
    out = (later.values - earlier.values) / (2.0 * half_window_hours * 3600.0)
    coords = {k: v for k, v in hourly.coords.items() if time_str not in v.dims}
    coords[time_str] = times
    return xr.DataArray(out, dims=hourly.dims, coords=coords)


def dry_air_mass_correction(u, v, sphum, dp, dry_mass_tendency,
                            lev_str=LEV_STR, lat_str=LAT_STR, lon_str=LON_STR,
                            time_str=TIME_STR):
    """Barotropic wind correction that closes the dry-air column mass budget.

    Parameters
    ----------
    u, v
        Horizontal wind on pressure levels, m/s, global, latitude descending
        from the north pole as ``windspharm`` requires.
    sphum
        Specific humidity on the same grid, kg/kg.
    dp
        Layer thickness in Pa from ``columns.dp_from_sfc_pressure``.
    dry_mass_tendency
        Rate of change of the dry column mass ``(p_s - g W) / g`` in
        kg/m2/s on (time, latitude, longitude), from ``centered_tendency``.

    Returns
    -------
    (du, dv)
        The correction in m/s on (time, latitude, longitude), to be added to
        the wind at every level, and the dry column mass it was formed
        against, kg/m2.
    """
    dry = 1.0 - sphum
    u_col = col_int(u * dry, dp, lev_str=lev_str)
    v_col = col_int(v * dry, dp, lev_str=lev_str)
    mass = col_int(dry, dp, lev_str=lev_str)
    du, dv = mass_correction_from_columns(u_col, v_col, mass, dry_mass_tendency,
                                          lat_str=lat_str, lon_str=lon_str,
                                          time_str=time_str)
    return du, dv, mass


def mass_correction_from_columns(u_col, v_col, mass, tendency, lat_str=LAT_STR,
                                 lon_str=LON_STR, time_str=TIME_STR):
    """The barotropic correction from column integrals already formed.

    ``u_col`` and ``v_col`` are the column mass fluxes whose divergence the
    budget constrains, ``mass`` the column mass they are divided by to give
    a wind, and ``tendency`` the storage term, all on (time, latitude,
    longitude).  ``dry_air_mass_correction`` forms the integrals from the
    fields and calls this; ``assembly.corrected_fields`` calls it with
    integrals accumulated one analysis time at a time, since a month of
    global three-dimensional fields does not fit in memory.  Returns
    ``(du, dv)`` in m/s.
    """
    u_adj, v_adj = adjusted_col_fluxes(u_col, v_col, tendency, xr.zeros_like(tendency),
                                       lat_str=lat_str, lon_str=lon_str,
                                       time_str=time_str)
    return (u_adj - u_col) / mass, (v_adj - v_col) / mass


def polar_cap_transport(zonal_mean_source, lat_str=LAT_STR, radius=RAD_EARTH):
    r"""Northward transport across each latitude implied by a zonal-mean source.

    .. math::

        T(\phi) = \frac{a}{\cos\phi}\int_{\phi}^{\pi/2} [S]\cos\phi'\,d\phi'

    for a source per unit area :math:`S`: the transport across :math:`\phi`
    that balances the integral of the source over the cap north of it.  For
    the mass budget :math:`S` is the column mass tendency minus the net
    surface source, and :math:`T` is the zonal-mean column mass transport
    the wind must carry, in kg per meter per second.  The latitude coordinate
    may run either way; the integral always starts at the north pole, and
    the value at the south pole is the global integral, which no divergent
    correction can remove.
    """
    lat = zonal_mean_source[lat_str].values
    descending = lat[0] > lat[-1]
    src = zonal_mean_source if descending else zonal_mean_source.isel({lat_str: slice(None, None, -1)})
    phi = np.deg2rad(src[lat_str].values)
    axis = src.get_axis_num(lat_str)
    integrand = np.moveaxis(np.asarray(src.values) * np.cos(phi).reshape(
        [-1 if i == axis else 1 for i in range(src.ndim)]), axis, 0)
    dphi = np.abs(np.diff(phi))
    steps = 0.5 * (integrand[1:] + integrand[:-1]) * dphi.reshape([-1] + [1] * (integrand.ndim - 1))
    cum = np.concatenate([np.zeros_like(integrand[:1]), np.cumsum(steps, axis=0)])
    with np.errstate(divide="ignore", invalid="ignore"):
        vals = radius * cum / np.cos(phi).reshape([-1] + [1] * (integrand.ndim - 1))
    out = src.copy(data=np.moveaxis(vals, 0, axis))
    return out if descending else out.isel({lat_str: slice(None, None, -1)})
