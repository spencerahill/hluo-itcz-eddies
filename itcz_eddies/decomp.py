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

**The lower boundary moves, and a split of the wind cannot close under it.**
F24 (2026-09-09) measured on ERA5 that the cross term :math:`[v'\bar h]`
survives every time filter, at 0.14 to 0.19 PW rms over the tropics, and
that 96 percent of it is the covariance of the transient wind with the
fluctuating thickness of the lowest layer: the transient eddy mass flux of
the surface layer times the MSE there.  The column integral makes the flux a
product of three fluctuating factors, wind, MSE and layer mass, and a
two-factor Reynolds split of wind and MSE leaves the third to a cross term
whatever the filter.  ``decompose_mass_flux`` splits the layer mass flux
``m = v dp / g`` and the MSE instead, so the layer mass is inside the
quantity being split and the cross terms vanish for a projection time mean
exactly as the algebra says.  It is the form the pipeline recommendation of
2026-09-09 adopts (``pipeline-2026-09-09/`` in the manuscript repository).

``legacy_decompose`` reproduces the published calculation, difference by
difference, so the current figures can be regenerated.  ``decompose`` computes
the exact split of the wind; ``decompose_mass_flux`` the split of the layer
mass flux.
"""

from __future__ import annotations

import numpy as np
import scipy.fft
import xarray as xr
from puffins.constants import GRAV_EARTH

from .columns import nantrapz
from .names import LEV_STR, LON_STR, TIME_STR

__all__ = [
    "boxcar_time_mean",
    "block_time_mean",
    "lanczos_time_mean",
    "ideal_time_mean",
    "zonal_mean",
    "decompose",
    "decompose_mass_flux",
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


def ideal_time_mean(arr, temporal_resolution=12, cutoff_days=30,
                    time_str=TIME_STR):
    """Ideal spectral low-pass in time: the Fourier coefficients above the
    cutoff frequency set to zero, everything below kept unchanged.

    The transfer function is exactly one or exactly zero at every frequency,
    so the operator is a projection: applying it twice gives what applying
    it once gave, and by the spectral identity in
    ``code-review/time-mean-operator.pdf`` (its equation for the cross terms
    as an integral of the cross-spectrum weighted by K(1-K)) both cross terms
    of the exact split vanish in the mean over the filtered record.  The
    Lanczos of ``lanczos_time_mean`` is a windowed approximation of this
    filter.

    Two properties to keep in mind.  The transform treats the record as
    periodic, so the ends of the record wrap onto each other, and a
    calculation that drops flanking months at each end should read enough
    of them to keep the wraparound out of the kept year.  And a sharp cutoff
    rings: a step in the data produces oscillations at the cutoff period on
    either side of it, which the Lanczos window is designed to suppress.
    Neither property produces NaN, so unlike the Lanczos every time step
    comes back with a value.

    The Nyquist coefficient, when the record length is even, sits at a
    period of two samples and is always removed for any cutoff longer than
    that.
    """
    axis = arr.get_axis_num(time_str)
    n_times = arr.sizes[time_str]
    step_days = temporal_resolution / 24.0
    freqs = scipy.fft.rfftfreq(n_times, d=step_days)      # cycles per day
    keep = freqs <= 1.0 / cutoff_days

    def _filter(values):
        spectrum = scipy.fft.rfft(values, axis=axis)
        shape = [1] * spectrum.ndim
        shape[axis] = keep.size
        spectrum = spectrum * keep.reshape(shape)
        return scipy.fft.irfft(spectrum, n=n_times, axis=axis)

    return arr.copy(data=_filter(np.asarray(arr.values, dtype="float64")))


# ---------------------------------------------------------- the exact split


def decompose(v, mse, time_mean=boxcar_time_mean, lon_str=LON_STR,
              zonal_mean_terms=True, weights=None, zonal_mean_fields=False,
              **kwargs):
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
    zonal_mean_fields
        When ``True`` the returned Dataset also carries ``v_bar_zm`` and
        ``h_bar_zm``, the zonal means of the time-mean wind and MSE that the
        ``mmc`` term is the product of, on their own ``(time, level,
        latitude)`` dimensions.  They are what a barotropic correction of
        the mean-circulation term needs: the column mean of ``v_bar_zm``
        times ``h_bar_zm``, integrated over the column, is the MSE flux of
        the zonal-mean wind's net mass transport, which decision 1 in
        ``code-review/FINDINGS.md`` is about.
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
        out = xr.Dataset(
            {name: (arr if lon_str not in arr.dims
                    else zonal_mean(arr, weights=weights, lon_str=lon_str))
             for name, arr in terms.items()}
        )
    else:
        terms["zonal_cross"] = v_bar_zm * h_bar_star + v_bar_star * h_bar_zm
        out = xr.Dataset(
            {name: arr.broadcast_like(terms["total"])
             for name, arr in terms.items()}
        )
    if zonal_mean_fields:
        out["v_bar_zm"] = v_bar_zm
        out["h_bar_zm"] = h_bar_zm
    return out


# ------------------------------------------- the split of the layer mass flux


def decompose_mass_flux(v, mse, dp, time_mean=boxcar_time_mean, lon_str=LON_STR,
                        zonal_mean_terms=True, zonal_mean_fields=False,
                        grav=GRAV_EARTH, **kwargs):
    r"""Split the meridional MSE flux into terms built from the layer mass flux.

    The flux through a layer of thickness ``dp`` is :math:`m h`, with
    :math:`m = v\,\delta p / g` the northward mass flux of the layer per unit
    length (kg per meter per second) and :math:`h` the MSE.  Writing an
    overbar for the time mean, a prime for the departure from it, square
    brackets for the arithmetic zonal mean and :math:`[\cdot]_w` for the
    zonal mean weighted by the time-mean layer thickness,

    .. math::

        [m h] = [\bar m][\bar h]_w + \bigl([\bar m \bar h] - [\bar m][\bar h]_w\bigr)
                + [m'h'] + [\bar m h'] + [m' \bar h]

    identically: mean circulation, stationary eddies, transient eddies and
    two cross terms.  The column integral of each term is its sum over
    levels, since the layer mass is inside :math:`m`, and it needs no
    weighting of its own.  The five pointwise terms sum to :math:`m h` at
    every gridpoint, and their zonal means sum to :math:`[m h]`, with no sixth
    term: the longitude dependence of the layer thickness, which breaks the
    five-term split of the wind (F15, F16), is inside :math:`m`.

    Three properties, each checked in ``tests/test_mass_flux_split.py``.

    The cross terms vanish in the mean over the record for a projection time
    mean even when the surface pressure fluctuates, because the covariance of
    the wind with the layer mass (F24) is inside :math:`m` and lands in the
    transient term where it belongs.  For the split of the wind it does not,
    whatever the filter.

    The net mass transport of the mean-circulation term,
    :math:`\sum_k [\bar m_k]`, equals the record mean of the zonal-mean
    column mass transport of the wind for a projection time mean, so it is
    whatever the wind's mass budget makes it: the physical vapor transport
    after a mass correction, and nothing else.  No barotropic correction is
    subtracted here; ``zonal_mean_fields`` returns what one needs.

    With a layer thickness that does not vary in time or longitude, every
    term equals the corresponding term of ``decompose`` times
    :math:`\delta p / g`.

    The mean-circulation term uses the mass-weighted zonal mean of the
    time-mean MSE, so it depends on the MSE only where there is mass, never
    on ERA5's extrapolated below-ground values.  The stationary term is
    defined as the remainder :math:`[\bar m \bar h] - [\bar m][\bar h]_w`,
    which equals :math:`[\bar m^* \bar h^*] + [\bar m][\bar h^*]` with the
    stars taken against those two means; the second piece is zero at levels
    no terrain cuts, and the remainder form keeps the below-ground MSE out of
    the split entirely.

    Parameters
    ----------
    v
        Meridional wind, mass-corrected or not; the term sizes say which.
    mse
        Moist static energy on the same grid.
    dp
        Layer thickness in Pa from ``columns.dp_from_sfc_pressure``, zero
        below ground and clipped at the surface, on the same grid as ``v``.
    time_mean
        Callable applied to ``m``, to ``mse`` and to ``dp``.
    zonal_mean_terms
        Whether to return zonal-mean terms or the pointwise ones.
    zonal_mean_fields
        When ``True`` the returned Dataset also carries ``m_bar_zm``,
        ``h_bar_zm`` and ``dp_bar_zm`` on (time, level, latitude), from which
        the net mass transport of the mean circulation and the column-mean
        MSE it would be assigned can be formed offline.
    grav
        Gravity, in ``m = v dp / grav``.
    **kwargs
        Passed through to ``time_mean``.

    Returns
    -------
    xarray.Dataset with ``mmc``, ``stationary``, ``transient``,
    ``cross_mean_wind_eddy_mse`` (:math:`\bar m h'`),
    ``cross_eddy_wind_mean_mse`` (:math:`m' \bar h`) and ``total``
    (:math:`m h`), each in W per meter per level.
    """
    m = v * dp / grav
    m_bar = time_mean(m, **kwargs)
    h_bar = time_mean(mse, **kwargs)
    w = time_mean(dp, **kwargs)
    m_prime = m - m_bar
    h_prime = mse - h_bar

    m_bar_zm = m_bar.mean(lon_str, skipna=False)
    w_sum = w.sum(lon_str, skipna=False)
    h_bar_zm = (h_bar * w).sum(lon_str, skipna=False) / w_sum.where(w_sum != 0)
    mmc = m_bar_zm * h_bar_zm

    terms = {
        "mmc": mmc,
        "stationary": m_bar * h_bar - mmc,
        "transient": m_prime * h_prime,
        "cross_mean_wind_eddy_mse": m_bar * h_prime,
        "cross_eddy_wind_mean_mse": m_prime * h_bar,
        "total": m * mse,
    }
    if zonal_mean_terms:
        out = xr.Dataset(
            {name: (arr if lon_str not in arr.dims
                    else arr.mean(lon_str, skipna=False))
             for name, arr in terms.items()}
        )
    else:
        out = xr.Dataset(
            {name: arr.broadcast_like(terms["total"])
             for name, arr in terms.items()}
        )
    if zonal_mean_fields:
        out["m_bar_zm"] = m_bar_zm
        out["h_bar_zm"] = h_bar_zm
        out["dp_bar_zm"] = w.mean(lon_str, skipna=False)
    return out


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
