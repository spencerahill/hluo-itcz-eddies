"""Moist static energy, the column energy budget, and the mass adjustment.

The pieces here are the ones ``puffins`` already provides, plus the two steps
the scripts add on top of them.

``puffins.thermodynamics.moist_static_energy`` takes geopotential *height* and
multiplies by gravity, while ERA5 stores *geopotential* and the scripts add it
straight in.  The two agree exactly when the height handed to ``puffins`` is
the ERA5 geopotential divided by the same gravity, since the division and the
multiplication cancel, so the choice of gravity does not enter the MSE at all.
What does enter is the specific heat: the scripts use 1004.0 J/kg/K and
``puffins.constants.C_P`` is 1003.5, a difference of 0.05% of the dry static
energy.  Both are available here and the scripts' value is the default, so the
published numbers can be reproduced.

``puffins.budget_adj.uv_col_budg_adj`` performs the Trenberth-style adjustment
and returns the *adjusted column fluxes*.  ``MSEadjust/main.py`` instead
returns the adjustment converted into a per-unit-mass quantity, which the flux
decomposition then divides by the MSE to get a wind correction.  Both forms are
here, and ``tests/test_mse.py`` shows they are the same adjustment: the wind
correction, integrated over the column against the MSE, returns exactly the
column flux adjustment ``puffins`` subtracts.

The budget being closed is

    d<h>/dt + div(<u h>, <v h>) = F_net,

with angle brackets for the column integral and ``F_net`` the net energy input
to the column, which is the net downward flux at the top of the atmosphere
minus the net downward flux at the surface.  Exact closure is not the target
for this budget; the adjustment goes through a spherical-harmonic inversion
whose round-trip error depends on the residual's spatial spectrum, measured in
``code-review/checks/harmonic_roundtrip.py``.
"""

from __future__ import annotations

from puffins.budget_adj import resid_after_col_adj, uv_col_budg_adj
from puffins.constants import C_P as C_P_PUFFINS
from puffins.constants import GRAV_EARTH, L_V
from puffins.thermodynamics import moist_static_energy as _puffins_mse

from .derivatives import ddt
from .names import LAT_STR, LON_STR, TIME_STR

__all__ = [
    "C_P_SCRIPTS",
    "C_P_PUFFINS",
    "L_V",
    "moist_static_energy",
    "net_energy_input",
    "column_tendency",
    "adjusted_col_fluxes",
    "budget_residual",
    "flux_adjustment_per_unit_mass",
    "adjust_wind_by_mse_flux",
]

# The specific heat hardcoded in MSE_calculation.py and in the decomposition
# script.  puffins.constants.C_P is 1003.5.
C_P_SCRIPTS = 1004.0


def moist_static_energy(temp, geopot, sphum, c_p=C_P_SCRIPTS, l_v=L_V,
                        grav=GRAV_EARTH):
    """Moist static energy from ERA5 temperature, geopotential and humidity.

    Parameters
    ----------
    temp
        Temperature in K, ERA5 variable ``T``.
    geopot
        Geopotential in m2/s2, ERA5 variable ``Z``.  This is geopotential, not
        geopotential height; ERA5 stores the former.
    sphum
        Specific humidity in kg/kg, ERA5 variable ``Q``.
    c_p
        Specific heat of dry air at constant pressure, J/kg/K.  Default is the
        scripts' 1004.0; ``C_P_PUFFINS`` is 1003.5.
    l_v
        Latent heat of vaporization, J/kg.  The scripts and ``puffins`` agree
        on 2.5e6.
    grav
        Only used to convert the geopotential to a height for the call into
        ``puffins`` and then to convert it back, so it cancels exactly and no
        choice of value changes the result.

    Returns
    -------
    Moist static energy in J/kg.
    """
    return _puffins_mse(temp, geopot / grav, sphum, c_p=c_p, grav=grav, l_v=l_v)


def net_energy_input(sensible, latent, sfc_sw, sfc_lw, toa_sw, toa_lw):
    """Net energy input to the atmospheric column, W/m2.

    All six ERA5 mean-flux fields are stored positive downward, which is not
    stated in the file attributes and was settled from the data itself by
    ``code-review/checks/flux_sign_convention.py``.  The combination below is
    the one in ``MSEadjust/main.py``: the net downward flux at the top of the
    atmosphere minus the net downward flux at the surface.

    Parameters
    ----------
    sensible, latent
        ERA5 ``MSSHF`` and ``MSLHF``, surface turbulent fluxes.
    sfc_sw, sfc_lw
        ERA5 ``MSNSWRF`` and ``MSNLWRF``, surface net radiative fluxes.
    toa_sw, toa_lw
        ERA5 ``MTNSWRF`` and ``MTNLWRF``, top-of-atmosphere net radiative
        fluxes.
    """
    return toa_sw + toa_lw - sfc_sw - sfc_lw - latent - sensible


def column_tendency(col_energy, per_second=True):
    """Time derivative of a column-integrated quantity.

    ``col_energy`` must carry an undecoded ``time`` coordinate in hours; see
    ``derivatives.ddt``, which raises otherwise.
    """
    return ddt(col_energy, per_second=per_second)


def adjusted_col_fluxes(u_col, v_col, tendency, source,
                        lat_str=LAT_STR, lon_str=LON_STR, time_str=TIME_STR):
    """Column fluxes adjusted so the budget closes, from ``puffins``.

    Wraps ``puffins.budget_adj.uv_col_budg_adj``, passing this project's
    dimension names.  Returns the adjusted ``(u_col, v_col)``.

    ``windspharm``, which performs the spherical-harmonic transform, requires
    a global latitude grid that is either Gaussian or equally spaced with the
    poles included when the count is odd, and excluded when it is even.  ERA5
    at half-degree spacing has 361 latitudes from 90 to -90, which satisfies
    the odd case.
    """
    return uv_col_budg_adj(u_col, v_col, tendency, source,
                           lat_str=lat_str, lon_str=lon_str, time_str=time_str)


def budget_residual(u_col_adjusted, v_col_adjusted, tendency, source):
    """Column budget residual after adjustment, from ``puffins``."""
    return resid_after_col_adj(u_col_adjusted, v_col_adjusted, tendency, source)


def flux_adjustment_per_unit_mass(u_col_adjustment, v_col_adjustment, p_sfc,
                                  p_top, grav=GRAV_EARTH):
    """Divide a column flux adjustment by the column mass.

    This is the last three lines of ``adjust`` in ``MSEadjust/main.py``, which
    are what makes its output differ from ``puffins``'.  The column mass is
    ``(p_sfc - p_top) / g``, so dividing the column flux adjustment by it gives
    a flux per unit mass, in W m/kg.  The flux decomposition then divides that
    by the MSE to get a wind correction in m/s.

    Parameters
    ----------
    u_col_adjustment, v_col_adjustment
        The adjustment to the column fluxes, W/m.
    p_sfc, p_top
        Surface and top-of-column pressure in Pa.
    """
    column_mass = (p_sfc - p_top) / grav
    return u_col_adjustment / column_mass, v_col_adjustment / column_mass


def adjust_wind_by_mse_flux(wind, flux_adjustment_per_mass, mse):
    """Subtract the mass-adjustment wind correction from a wind field.

    ``v_adj = v - vMSE_adjust / MSE``, line 310 of the flux decomposition
    script and line 280 of ``MSEadjust/main_v.py``.

    The correction is inversely proportional to the MSE, which makes the
    column integral of the corrected MSE flux come out exactly right: with
    ``dv = V/((p_s - p_t)/g) / h``, the column integral of ``dv * h`` is ``V``
    at every gridpoint whatever the vertical structure of ``h``.
    ``tests/test_mse.py`` checks that identity.

    What it does not do is remove a column-mean mass flux.  The implied mass
    correction is the column integral of ``dv``, which is ``V`` times the
    mass-weighted mean of ``1/h`` and is not in general the mass adjustment
    the Trenberth method calls for.  Decision 1 in
    ``code-review/FINDINGS.md`` turns on this point.
    """
    return wind - flux_adjustment_per_mass / mse
