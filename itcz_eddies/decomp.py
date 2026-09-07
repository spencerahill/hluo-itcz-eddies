r"""The three-way split of the meridional MSE flux, and the terms it drops.

Notation, following ``code-review/time-mean-operator.pdf``: an overbar is the
time mean, a prime the departure from it, square brackets the zonal mean, an
asterisk the departure from the zonal mean.

**The exact split.**  For any time-mean operator whatever, the algebra of the
definitions gives

.. math::

    [vh] = [\bar v][\bar h] + [\bar v^* \bar h^*] + [v'h']
           + [\bar v h'] + [v' \bar h]

with no approximation: mean circulation, stationary eddy, transient eddy, and
two cross terms.  ``decompose`` returns all five.  The cross terms vanish only
when the time mean is idempotent, meaning that averaging an already-averaged
field returns it unchanged.  A centred rolling mean is not idempotent, a block
average is, and a Lanczos low-pass is nearly so.

**What the published code computes instead.**  Three differences, each of them
a decision recorded at the top of ``code-review/FINDINGS.md``.

1. It subtracts a barotropic correction ``B``, the column mean of the
   zonal-mean time-mean wind, from the mean-circulation wind alone.  That moves
   ``B[\bar h]`` out of the sum entirely and puts a surplus ``[B \bar h^*]``
   into the stationary term.
2. Its time mean is a 30-day centred rolling mean with ``min_periods=1``, which
   is not idempotent, so the two cross terms are dropped and the transient term
   keeps only ``[v'^* h'^*]``, dropping ``[v'][h']``.
3. It reduces with a ``skipna`` zonal mean and a trapezoid column integral
   taken to each longitude's own surface pressure, which is where the published
   38% gap comes from.  See ``columns`` for that half.

**The zonal mean has a weighting, and it decides whether five terms suffice.**
With an arithmetic zonal mean, the five-term split above closes exactly at each
pressure level, and stops closing once the terms are column-integrated to each
longitude's own surface pressure, because that integral weights longitudes
unequally.  Defining the zonal mean at each level as the average weighted by
that level's layer thickness restores the closure, since the departure from it
then has zero mass-weighted zonal integral by construction.  Measured in
``code-review/checks/zonal_mean_weighting.py``: 7.0e-16 of the peak with the
weighting against 1.0e-3 without it.  Pass ``weights=dp`` to ``decompose`` to
get it.  What changes is the definition of the stationary eddy, which becomes
the departure from a mass-weighted rather than an arithmetic zonal mean.

``legacy_decompose`` reproduces the published calculation, difference by
difference, so the current figures can be regenerated.  ``decompose`` computes
the exact split.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from .columns import nantrapz
from .names import LEV_STR, LON_STR, TIME_STR

__all__ = [
    "boxcar_time_mean",
    "block_time_mean",
    "lanczos_time_mean",
    "zonal_mean",
    "decompose",
    "legacy_decompose",
    "barotropic_correction",
]


def zonal_mean(arr, weights=None, lon_str=LON_STR):
    """Average around a latitude circle, optionally weighted by layer mass.

    With ``weights`` left as ``None`` this is the arithmetic mean, taken with
    ``skipna=False`` so a NaN anywhere on the circle propagates rather than
    changing the denominator.

    With ``weights`` set to the layer thicknesses from
    ``columns.dp_from_sfc_pressure``, it is the mass-weighted mean

        sum_lon (arr * dp) / sum_lon dp

    at each level.  The departure from that mean has zero mass-weighted zonal
    integral at every level by construction, which is what makes the five-term
    split survive a column integral taken to each longitude's own surface.
    """
    if weights is None:
        return arr.mean(lon_str, skipna=False)
    return ((arr * weights).sum(lon_str, skipna=False)
            / weights.sum(lon_str, skipna=False))


# ------------------------------------------------------------ time averaging


def boxcar_time_mean(arr, temporal_resolution=12, window_days=30,
                     time_str=TIME_STR):
    """Centred rolling mean, the published definition of the time mean.

    Reproduces ``running_mean``, which appears identically in 12 of the
    scripts: ``rolling(time=int(30*24/temporal_resolution), min_periods=1,
    center=True).mean()``.

    ``min_periods=1`` means the first and last half-window of the record are
    averaged over fewer points than the interior, so the operator is not
    translation invariant near the ends.  It is also not idempotent anywhere,
    which is what leaves the two cross terms of the exact split unassigned.
    """
    window = int(window_days * 24 / temporal_resolution)
    return arr.rolling({time_str: window}, min_periods=1, center=True).mean()


def block_time_mean(arr, temporal_resolution=12, window_days=30,
                    time_str=TIME_STR):
    """Non-overlapping block average, broadcast back onto the original times.

    Idempotent by construction, so both cross terms of the exact split vanish
    and the three-way decomposition closes on its own.  The cost is a time
    mean that steps rather than varying smoothly, and a partial final block
    whenever the record length is not a whole number of blocks.
    """
    window = int(window_days * 24 / temporal_resolution)
    n_times = arr.sizes[time_str]
    block_index = xr.DataArray(
        np.arange(n_times) // window,
        dims=[time_str],
        coords={time_str: arr[time_str]},
        name="block",
    )
    grouped = arr.groupby(block_index).mean(time_str)
    return grouped.sel(block=block_index).drop_vars("block")


def lanczos_weights(cutoff_days, temporal_resolution=12, lobes=60):
    """Lanczos low-pass weights, normalized to sum to one.

    Same construction as ``lanczos_bandpass_filter`` in the five EOF scripts:
    an ideal filter multiplied by the Lanczos window ``sinc(n / lobes)`` over
    ``2 * lobes + 1`` points.  The low-pass here keeps the long periods, so it
    is the band-pass with its lower cutoff set to zero.
    """
    steps_per_day = 24.0 / temporal_resolution
    cutoff = 1.0 / (cutoff_days * steps_per_day)  # cycles per sample
    n = np.arange(-lobes, lobes + 1)
    ideal = 2 * cutoff * np.sinc(2 * cutoff * n)
    weights = ideal * np.sinc(n / lobes)
    return weights / weights.sum()


def lanczos_time_mean(arr, temporal_resolution=12, cutoff_days=30, lobes=60,
                      time_str=TIME_STR):
    """Lanczos low-pass in time, the recommendation in decision 2 of F11.

    A convolution with the weights from ``lanczos_weights``, applied through
    ``rolling(...).construct(...)`` so it works on any dimensionality.  The
    ends of the record come back as NaN, over ``lobes`` steps at each end,
    because a low-pass has no honest value there.  That is a real difference
    from ``boxcar_time_mean``, which fills the ends with partial windows.
    """
    weights = xr.DataArray(
        lanczos_weights(cutoff_days, temporal_resolution, lobes),
        dims=["window"],
    )
    rolled = arr.rolling({time_str: weights.sizes["window"]}, center=True).construct(
        "window"
    )
    return rolled.dot(weights)


# ---------------------------------------------------------- the exact split


def decompose(v, mse, time_mean=boxcar_time_mean, lon_str=LON_STR,
              zonal_mean_terms=True, weights=None, **kwargs):
    r"""Split the meridional MSE flux into its exact terms.

    With ``zonal_mean_terms=True``, the default, every term is a zonal mean and the
    returned fields carry no longitude dimension.  The five terms then sum to
    :math:`[vh]` exactly.

    With ``zonal_mean_terms=False`` the terms are returned as four-dimensional
    fields, and a sixth term appears, ``zonal_cross``, equal to
    :math:`[\bar v]\bar h^* + \bar v^*[\bar h]`.  Its zonal mean is zero by
    construction, which is why the five-term form does not need it, and it is
    not zero pointwise.  The six four-dimensional terms sum to :math:`vh` at
    every gridpoint.

    That distinction is the whole of the reduction problem in F11.  Any
    reduction that weights longitudes equally annihilates ``zonal_cross`` and
    the five-term split survives it.  A reduction that weights longitudes
    unequally, which a column integral taken to each longitude's own surface
    pressure does, does not annihilate it, and the five terms then fail to add
    up by exactly the reduction of ``zonal_cross`` plus whatever the
    zonal-mean forms of the eddy terms lose.  ``tests/test_decomp.py`` checks
    both halves of that statement.

    The zonal means here use ``skipna=False``, so a NaN anywhere on a latitude
    circle propagates rather than changing that term's denominator.  Handle
    below-ground points with layer thicknesses of zero
    (``columns.dp_from_sfc_pressure``) rather than with NaN.

    Parameters
    ----------
    v
        Meridional wind, already mass-adjusted if it is going to be.
    mse
        Moist static energy on the same grid.
    time_mean
        Callable applied to ``v`` and to ``mse`` to define the overbar.
        Default is the published 30-day centred rolling mean.
    lon_str
        Name of the longitude dimension.
    zonal_mean_terms
        Whether to return the zonal-mean terms or the four-dimensional ones.
    weights
        Layer thicknesses, from ``columns.dp_from_sfc_pressure``.  When given,
        every zonal mean here is weighted by them.  That is what makes the
        five-term form close under a column integral taken to each longitude's
        own surface pressure; see the module docstring.
    **kwargs
        Passed through to ``time_mean``.

    Returns
    -------
    xarray.Dataset with the terms and the total:

    ``mmc``
        :math:`[\bar v][\bar h]`, the mean meridional circulation.
    ``stationary``
        :math:`\bar v^* \bar h^*`, the stationary eddies, zonally averaged
        when ``zonal_mean``.
    ``transient``
        :math:`v'h'`, the transient eddies, zonally averaged when
        ``zonal_mean``.  Includes the :math:`[v'][h']` part the published code
        drops.
    ``cross_mean_wind_eddy_mse``
        :math:`\bar v h'`, zero for an idempotent time mean.
    ``cross_eddy_wind_mean_mse``
        :math:`v' \bar h`, likewise.
    ``zonal_cross``
        Only when ``zonal_mean_terms=False``.  See above.
    ``total``
        :math:`vh`, computed directly rather than as the sum, so that the
        closure of the terms is a real check.
    """
    v_bar = time_mean(v, **kwargs)
    h_bar = time_mean(mse, **kwargs)
    v_prime = v - v_bar
    h_prime = mse - h_bar

    v_bar_zm = zonal_mean(v_bar, weights=weights, lon_str=lon_str)
    h_bar_zm = zonal_mean(h_bar, weights=weights, lon_str=lon_str)
    v_bar_star = v_bar - v_bar_zm
    h_bar_star = h_bar - h_bar_zm

    terms = {
        "mmc": v_bar_zm * h_bar_zm,
        "stationary": v_bar_star * h_bar_star,
        "transient": v_prime * h_prime,
        "cross_mean_wind_eddy_mse": v_bar * h_prime,
        "cross_eddy_wind_mean_mse": v_prime * h_bar,
        "total": v * mse,
    }
    if zonal_mean_terms:
        return xr.Dataset(
            {name: (arr if lon_str not in arr.dims
                    else zonal_mean(arr, weights=weights, lon_str=lon_str))
             for name, arr in terms.items()}
        )
    terms["zonal_cross"] = v_bar_zm * h_bar_star + v_bar_star * h_bar_zm
    return xr.Dataset(
        {name: arr.broadcast_like(terms["total"]) for name, arr in terms.items()}
    )


# --------------------------------------------------------- the legacy split


def barotropic_correction(v_zonal_mean, p_sfc, lev_str=LEV_STR):
    """The column mean the published code subtracts from the MMC wind.

    Line 324 of ``main/adjusted_MMC_stationary_transient_calculation.py``:
    ``v_mean - nantrapz(v_mean, level)/sp``.  The trapezoid integral of the
    zonal-mean wind, in hPa m/s, divided by the surface pressure in hPa, so
    the result is a wind in m/s.

    Note that this divides by the *full* surface pressure rather than by the
    pressure range the trapezoid actually spans, and that the integral is
    taken with the same NaN masking as the column integral, so it stops at
    the lowest level above the surface.  Both matter for what the subtraction
    means, and both are reproduced here.
    """
    return nantrapz(v_zonal_mean, v_zonal_mean[lev_str], dim=lev_str) / p_sfc


def legacy_decompose(v, mse, mask, p_sfc, temporal_resolution=12,
                     lon_str=LON_STR, lev_str=LEV_STR):
    """Reproduce the published three-way split, term by term.

    This is lines 318 to 339 of
    ``main/adjusted_MMC_stationary_transient_calculation.py``, with the same
    masking, the same ``skipna=True`` zonal means, and the same barotropic
    correction.  The three returned terms are four-dimensional, matching what
    the script column-integrates.

    Returns
    -------
    xarray.Dataset with ``mmc``, ``stationary`` and ``transient``.
    """
    v = xr.where(mask == 1, v, np.nan)
    mse = xr.where(mask == 1, mse, np.nan)

    v_time_mean = boxcar_time_mean(v, temporal_resolution=temporal_resolution)
    v_mean = v_time_mean.mean(lon_str, skipna=True)
    v_mean = xr.where(mask == 1, v_mean, np.nan)
    v_mean = v_mean - barotropic_correction(v_mean, p_sfc, lev_str=lev_str)

    mse_time_mean = boxcar_time_mean(mse, temporal_resolution=temporal_resolution)
    mse_mean = mse_time_mean.mean(lon_str, skipna=True)

    v_time_ano = v - v_time_mean
    v_ano = v_time_ano - v_time_ano.mean(lon_str, skipna=True)
    mse_time_ano = mse - mse_time_mean
    mse_ano = mse_time_ano - mse_time_ano.mean(lon_str, skipna=True)

    return xr.Dataset(
        {
            "mmc": (v_mean * mse_mean).broadcast_like(mse),
            "stationary": ((v_time_mean - v_mean) * (mse_time_mean - mse_mean)
                           ).mean(lon_str, skipna=True).broadcast_like(mse),
            "transient": v_ano * mse_ano,
        }
    )
