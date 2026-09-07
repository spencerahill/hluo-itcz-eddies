"""Wheeler-Kiladis space-time spectral analysis.

Moved verbatim from ``main/myfun.py``, which adapted it from NCAR's own
implementation.  The bodies are unchanged, so this file can still be diffed
against that upstream; the only edits are the module docstring and the imports
below.

The one function here whose correctness a central claim of the manuscript
depends on is ``resolveWavesHayashi``, which separates eastward-propagating
from westward-propagating waves.  A sign error in its four reversed-stride
slice assignments would invert every propagation-direction reading in section 4
while leaving everything else looking normal.  It was checked on 2026-09-07 by
planting waves whose direction is known by construction and asking where the
power comes out; all four cases landed in the right cell, and the test carries
its own control on what "eastward" means in the planted field.  See
``code-review/checks/planted_wave_direction.py`` and the Resolved section of
``code-review/FINDINGS.md``.

Two functions in this file appear in ``myfun.py`` with more than one body.
``spacetime_power`` has five distinct bodies across the analysis scripts and
``plot_normalized_symmetric_spectrum`` has four; the copy moved here is
``main/myfun.py``'s, which is the one the manuscript's spectral figures use.
The others have not been diffed against it.
"""

from __future__ import annotations

import gc
import logging

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.signal import convolve2d, detrend

def  decompose2SymAsym(arr):
    """Mimic NCL function to decompose into symmetric and asymmetric parts.
    
    arr: xarra DataArray

    return: DataArray with symmetric in SH, asymmetric in NH

    Note:
        This function produces indistinguishable results from NCL version.
    """
    arr_flip = arr.assign_coords({"latitude":arr.latitude*-1})
    # flag to follow NCL convention and put symmetric component in SH 
    # & asymmetric in NH
    # method: use flip to reverse latitude, put in DataArray for coords, use loc/isel
    # to assign to negative/positive latitudes (exact equator is left alone)
   
    data_sym = 0.5*(arr + arr_flip)
    data_asy = 0.5*(arr - arr_flip)
    arr.loc[{'latitude':arr['latitude'][arr['latitude']<0]}] = data_sym.isel(latitude=data_sym.latitude<0)
    arr.loc[{'latitude':arr['latitude'][arr['latitude']>0]}] = data_asy.isel(latitude=data_asy.latitude>0)
    return arr

def rmvAnnualCycle(data, spd, fCrit):
    """remove frequencies less than fCrit from data.
    
    data: xarray DataArray
    spd: sampling frequency in samples-per-day
    fCrit: frequency threshold; remove frequencies < fCrit
    
    return: xarray DataArray, shape of data
    
    Note: fft/ifft preserves the mean because z = fft(x), z[0] is the mean.
          To keep the mean here, we need to keep the 0 frequency.
          
    Note: This function reproduces the results from the NCL version.

    Note: Two methods are available, one using fft/ifft and the other rfft/irfft.
          They both produce output that is indistinguishable from NCL's result.
    """
    n_time, n_lat, n_lon = data.shape
    time_ax = list(data.dims).index('time')
    # Method 1: Uses the complex FFT, returns the negative frequencies too, but they
    # should be redundant b/c they are conjugate of positive ones.
    cf = np.fft.fft(data, axis=time_ax)
    freq = np.fft.fftfreq(n_time, 1/spd)
    cf[(np.abs(freq) < fCrit), ...] = 0.0 
    z = np.fft.ifft(cf, n=n_time, axis=0)
    return xr.DataArray(z.real, dims=data.dims, coords=data.coords)

def convolvePosNeg(arr, k, dim, boundary_index):
    """Apply convolution of (arr, k) excluding data at boundary_index in dimension dim.
    
    arr: numpy ndarray of data
    k: numpy ndarray, same dimension as arr, this should be the kernel
    dim: integer indicating the axis of arr to split
    boundary_index: integer indicating the position to split dim
    
    Split array along dim at boundary_index;
    perform convolution on each sub-array;
    reconstruct output array from the two subarrays;
    the values of output at boundary_index of dim will be same as input.
    
    `convolve2d` is `scipy.signal.convolve2d()`
    """
    # arr: numpy ndarray
    # first pass is [0 : boundary_index)
    slc1 = [slice(None)] * arr.ndim
    slc1[dim] = slice(None, boundary_index)
    arr1 = arr[tuple(slc1)]
    ans1 = convolve2d(arr1, k, boundary='symm', mode='same')
    # second pass is [boundary_index+1, end]
    slc2 = [slice(None)] * arr.ndim
    slc2[dim] = slice(boundary_index+1,None)
    arr2 = arr[tuple(slc2)]
    ans2 = convolve2d(arr2, k, boundary='symm', mode='same')
    # fill in the output array
    arr[tuple(slc1)] = ans1
    arr[tuple(slc2)] = ans2
    return arr

def simple_smooth_kernel():
    """Provide a very simple smoothing kernel."""
    kern = np.array([[0, 1, 0],[1, 4, 1],[0, 1, 0]])#???????????????????????????????????
    return kern / kern.sum()

def smooth_wavefreq(data, kern=None, nsmooth=None, freq_ax=None, freq_name=None):
    """Apply a convolution of (data,kern) nsmooth times.
       The convolution is applied separately to the positive and negative frequencies.
       Either the name (freq_name: str) or axis index (freq_ax: int) of frequency is required, with the name preferred.
    """
    assert isinstance(data, xr.DataArray)
    if kern is None:
        kern = simple_smooth_kernel()
    if nsmooth is None:
        nsmooth = 20
    if freq_name is not None:
        axnum = list(data.dims).index(freq_name)
        nzero =  data.sizes[freq_name] // 2 # <-- THIS IS SUPPOSED TO BE THE INDEX AT FREQ==0.0
    elif freq_ax is not None:
        axnum = freq_ax
        nzero = data.shape[freq_ax] // 2
    else:
        raise ValueError("smooth_wavefreq needs to know how to find frequency dimension.")
    smth1pass = convolvePosNeg(data, kern, axnum, nzero) # this is a custom function to skip 0-frequency (mean)
    # note: the convolution is strictly 2D and the boundary condition is symmetric --> if kernel is normalized, preserves the sum.
    smth1pass = xr.DataArray(smth1pass, dims=data.dims, coords=data.coords) # ~copy_metadata
    # repeat smoothing many times:
    smthNpass = smth1pass.copy()
    del smth1pass
    gc.collect()
    for i in range(nsmooth):
        smthNpass = convolvePosNeg(smthNpass, kern, axnum, nzero)
    logging.info("Finishing smoothing...")
    return xr.DataArray(smthNpass, dims=data.dims, coords=data.coords)

def resolveWavesHayashi( varfft: xr.DataArray, nDayWin: int, spd: int ) -> xr.DataArray:  
    """This is a direct translation from the NCL routine to python/xarray.
    input:
        varfft : expected to have rightmost dimensions of wavenumber and frequency.
        varfft : expected to be an xarray DataArray with coordinate variables.
        nDayWin : integer that is the length of the segments in days.
        spd : the sampling frequency in `timesteps` per day (I think).

    returns:
        a DataArray that is reordered to have correct westward & eastward propagation.
    
    """
    #-------------------------------------------------------------
    # Special reordering to resolve the Progressive and Retrogressive waves 
    # Reference: Hayashi, Y. 
    #    A Generalized Method of Resolving Disturbances into 
    #    Progressive and Retrogressive Waves by Space and  
    #    Fourier and TimeCross Spectral Analysis
    #    J. Meteor. Soc. Japan, 1971, 49: 125-128.
    #-------------------------------------------------------------

    # in NCL varfft is dimensioned (2,mlon,nSampWin), but the first dim doesn't matter b/c python supports complex numbers.
    #
    # Create array PEE(NL+1,NT+1) which contains the (real) power spectrum.
    # all the following assume indexing starting with 0
    # In this array (PEE), the negative wavenumbers will be from pn=0 to NL/2-1 (left).
    # The positive wavenumbers will be for pn=NL/2+1 to NL (right).
    # Negative frequencies will be from pt=0 to NT/2-1 (left).
    # Positive frequencies will be from pt=NT/2+1 to NT  (right).
    # Information about zonal mean will be for pn=NL/2 (middle).
    # Information about time mean will be for pt=NT/2 (middle).
    # Information about the Nyquist Frequency is at pt=0 and pt=NT
    #

    # In PEE, define the 
    # WESTWARD waves to be either 
    #          positive frequency and negative wavenumber 
    #          OR 
    #          negative freq and positive wavenumber.
    # EASTWARD waves are either positive freq and positive wavenumber 
    #          OR negative freq and negative wavenumber.

    # Note that frequencies are returned from fftpack are ordered like so
    #    input_time_pos [ 0    1   2    3     4      5    6   7  ]
    #    ouput_fft_coef [mean 1/7 2/7  3/7 nyquist -3/7 -2/7 -1/7]  
    #                    mean,pos freq to nyq,neg freq hi to lo
    #
    # Rearrange the coef array to give you power array of freq and wave number east/west
    # Note east/west wave number *NOT* eq to fft wavenumber see Hayashi '71 
    # Hence, NCL's 'cfftf_frq_reorder' can *not* be used. 
    # BPM: This goes for np.fft.fftshift
    #
    # For ffts that return the coefficients as described above, here is the algorithm
    # coeff array varfft(2,n,t)   dimensioned (2,0:numlon-1,0:numtim-1)
    # new space/time pee(2,pn,pt) dimensioned (2,0:numlon  ,0:numtim  ) 
    #
    # Note: one larger in both freq/space dims
    # the initial index of 2 is for the real (indx 0) and imag (indx 1) parts of the array
    #
    #
    #    if  |  0 <= pn <= numlon/2-1    then    | numlon/2 <= n <= 1
    #        |  0 <= pt < numtim/2-1             | numtim/2 <= t <= numtim-1
    #
    #    if  |  0         <= pn <= numlon/2-1    then    | numlon/2 <= n <= 1
    #        |  numtime/2 <= pt <= numtim                | 0        <= t <= numtim/2
    #
    #    if  |  numlon/2  <= pn <= numlon    then    | 0  <= n <= numlon/2
    #        |  0         <= pt <= numtim/2          | numtim/2 <= t <= 0
    #
    #    if  |  numlon/2   <= pn <= numlon    then    | 0        <= n <= numlon/2
    #        |  numtim/2+1 <= pt <= numtim            | numtim-1 <= t <= numtim/2

    # local variables : dimvf, numlon, N, varspacetime, pee, wave, freq

    logging.debug(f"[Hayashi] nDayWin: {nDayWin}, spd: {spd}")
    dimnames = varfft.dims
    dimvf  = varfft.shape
    mlon   = len(varfft['wavenumber'])
    N      = dimvf[-1]
    logging.info(f"[Hayashi] input dims is {dimnames}, {dimvf}")
    logging.info(f"[Hayashi] input coords is {varfft.coords}")

    if len(dimnames) != len(varfft.coords):
        logging.error("The size of varfft.coords is incorrect.")
        raise ValueError("STOP")

    nshape = list(dimvf)
    nshape[-2] += 1
    nshape[-1] += 1
    logging.debug(f"[Hayashi] The nshape ends up being {nshape}")
    # this is a reordering, use Ellipsis to allow arbitrary number of dimensions,
    # but we insist that the wavenumber and frequency dims are rightmost.
    # we will fill the new array in increasing order (arbitrary choice)
    logging.debug("allocate the re-ordered array")
    #newempty = xr.DataArray(np.zeros(nshape),coords = varfft.coords,dims = varfft.dims)
    pee = np.full(nshape, np.nan,dtype = "complex")
    #varspacetime = xr.full_like(newempty, np.nan, dtype=type(varfft))
    # first two are the negative wavenumbers (westward), second two are the positive wavenumbers (eastward)
    
    logging.debug(f"[Hayashi] Assign values into array. Notable numbers: mlon//2={mlon//2}, N//2={N//2}")

    #  Create the real power spectrum pee = sqrt(real^2+imag^2)^2
    pee[..., 0:mlon//2, 0:N//2    ] = varfft[..., mlon//2:0:-1, N//2:] # neg.k, pos.w
    pee[..., 0:mlon//2, N//2:     ] = varfft[..., mlon//2:0:-1, 0:N//2+1]  # neg.k, 
    pee[..., mlon//2:   , 0:N//2+1] = varfft[..., 0:mlon//2+1,  N//2::-1]  # assign eastward & neg.freq.
    pee[..., mlon//2:   , N//2+1: ] = varfft[..., 0:mlon//2+1, -1:N//2-1:-1] # assign eastward & pos.freq.
    logging.debug("calculate power")

    logging.debug("put into DataArray")

    # add meta data for use upon return
    wave      = np.arange(-mlon // 2, (mlon // 2 )+ 1, 1, dtype=int)  
    freq      = np.linspace(-1*nDayWin*spd/2, nDayWin*spd/2, (nDayWin*spd)+1) / nDayWin

    odims = list(dimnames)
    odims[-2] = "wavenumber"
    odims[-1] = "frequency"
    ocoords = {}
    for c in varfft.coords:
        logging.debug(f"[hayashi] working on coordinate {c}")
        if (c != "wavenumber") and (c != "frequency"):
            ocoords[c] = varfft[c]
        elif c == "wavenumber":
            ocoords['wavenumber'] = wave
        elif c == "frequency":
            ocoords['frequency'] = freq
   
    pee = xr.DataArray(pee, dims=odims, coords=ocoords)

    return pee

def split_hann_taper(series_length, fraction):
    """Implements `split cosine bell` taper of length series_length where only fraction of points are tapered (combined on both ends).
    
    This returns a function that tapers to zero on the ends. To taper to the mean of a series X:
    XTAPER = (X - X.mean())*series_taper + X.mean()
    """
    npts = int(np.rint(fraction * series_length))  # total size of taper
    taper = np.hanning(npts)
    series_taper = np.ones(series_length)
    series_taper[0:npts//2+1] = taper[0:npts//2+1]
    series_taper[-npts//2+1:] = taper[npts//2+1:]

    return series_taper

def spacetime_power(data, segsize, noverlap, spd, dosymmetries=True, rmvLowFrq=True):
    """Perform space-time spectral decomposition and return power spectrum following Wheeler-Kiladis approach.
    
    data: an xarray DataArray to be analyzed; needs to have (time, latitude, lon) dimensions.
    segsize: integer denoting the size of time samples that will be decomposed (typically about 96)
    noverlap: integer denoting the number of days of overlap from one segment to the next
    spd: sampling rate, in "samples per day" (e.g. daily=1, 6-houry=4)
    
    latitude_bounds: a tuple of (southern_extent, northern_extent) to reduce data size.
    
    dosymmetries: if True, follow NCL convention of putting symmetric component in SH, antisymmetric in NH
                  If True, the function returns a DataArray with a `component` dimension.
                  
    rmvLowFrq: if True, remove frequencies < 1/segsize from data.
    
    Method
    ------
        1. Subsample in latitude if latitude_bounds is specified.
        2. Detrend the data (but keeps the mean value, as in NCL)
        3. High-pass filter if rmvLowFrq is True
        4. Construct symmetric/antisymmetric array if dosymmetries is True.
        5. Construct overlapping window view of data.
        6. Detrend the segments (strange enough, removing mean).
        7. Apply taper in time dimension of windows (aka segments).
        8. Fourier transform
        9. Apply Hayashi reordering to get propagation direction & convert to power.
       10. return DataArray with power.
       
    Notes
    -----
        Upon returning power, this should be comparable to "raw" spectra. 
        Next step would be be to smooth with `smooth_wavefreq`, 
        and divide raw spectra by smooth background to obtain "significant" spectral power.
        
    """

    segsize = spd*segsize
    noverlap = spd*noverlap
    assert segsize-noverlap > 0, f"Error, inconsistent specification of segsize and noverlap results in stride of {segsize-noverlap}, but must be > 0."

    slat = data['latitude'].min().item()
    nlat = data['latitude'].max().item()
        
    # "Remove dominant signals"
    # "detrend" the data, including removing the mean (uses scipy.signal.detrend):
    data = data.transpose('time', 'latitude', 'longitude')
    xdetr = detrend(data, axis=0, type='linear')
    xdetr = xr.DataArray(xdetr, dims=data.dims, coords=data.coords)
    
    # filter low-frequencies    
    if rmvLowFrq:
        data = rmvAnnualCycle(xdetr, spd, 1/(segsize/spd))
        logging.info("Annual cycle is removed")

    dimsizes = data.sizes  # dict
    lon_size = dimsizes['longitude']
    lat_size = dimsizes['latitude']
    lat_dim = data.dims.index('latitude')
    if dosymmetries:
        data = decompose2SymAsym(data)
        logging.info("Decomposited into symetric and asymetric")
    # testing: pass -- Gets the same result as NCL.

    # 2. Windowing with the xarray "rolling" operation, and then limit overlap with `construct` to produce a new dataArray.
    # WK99 recommend "2-month" overlap
    x_win = data.rolling(time=segsize, min_periods=segsize).construct("segments")  # WK99 use 96-day window
    x_win = x_win.isel(time=slice(segsize-1,None,segsize-noverlap))  

    logging.debug(f"[spacetime_power] x_win shape is {x_win.shape}")
    # Additional detrend for each segment: means??????????????????????????????????
    if  np.logical_not(np.any(np.isnan(x_win))):
        logging.info("No missing, so use simplest segment detrend.")
        x_win_detr = detrend(x_win, axis=-1, type='linear') #<-- missing data makes this not work
        x_win = xr.DataArray(x_win_detr, dims=x_win.dims, coords=x_win.coords)
    else:
        logging.warning("EXTREME WARNING -- This method to detrend with missing values present does not quite work, probably need to do interpolation instead.")
        logging.warning("There are missing data in x_win, so have to try to detrend around them.")
        x_win_cp = x_win.copy()
        logging.info(f"[spacetime_power] x_win_cp windowed data has shape {x_win_cp.shape} \n \t It is a numpy array, copied from x_win which has dims: {x_win.sizes} \n \t ** about to detrend this in the rightmost dimension.")
        x_win_cp[np.logical_not(np.isnan(x_win_cp))] = detrend(x_win_cp[np.logical_not(np.isnan(x_win_cp))])
        x_win = xr.DataArray(x_win_cp, dims=x_win.dims, coords=x_win.coords)
    
    # 3. Taper in time to make the signal periodic, as required for FFT.
    taper = split_hann_taper(segsize, 0.1)  
    x_wintap = x_win*taper 
    
    # Do the transform using 2D FFT
    #
    # z = np.fft.fft2(x_wintap, axes=(2,3)) / (lon_size * segsize)

    # Or do the transform with 2 steps
    z = np.fft.fft(x_wintap, axis=2) / lon_size  # note that np.fft.fft() produces same answers as NCL cfftf
    z = np.fft.fft(z, axis=3) / segsize 
    z = xr.DataArray(z, dims=("time","latitude","wavenumber","frequency"), 
                     coords={"time":x_wintap["time"], 
                             "latitude":x_wintap["latitude"],
                             "wavenumber":np.fft.fftfreq(lon_size, 1/lon_size),
                             "frequency":np.fft.fftfreq(segsize, 1/spd)})
    #
    # The FFT is returned following ``standard order`` which has negative frequencies in second half of array. 
    #
    # IMPORTANT: 
    # If this were typical 2D FFT, we would do the following to get the frequencies and reorder:
    #         z_k = np.fft.fftfreq(x_wintap.shape[-2], 1/lon_size)
    #         z_v = np.fft.fftfreq(x_wintap.shape[-1], 1)  # Assumes 1/(1-day) timestep
    # reshape to get the frequencies centered
    #         z_centered = np.fft.fftshift(z, axes=(2,3))
    #         z_k_c = np.fft.fftshift(z_k)
    #         z_v_c = np.fft.fftshift(z_v)
    # and convert to DataArray as this:
    #         d1 = list(x_win.dims)
    #         d1[-2] = "wavenumber"
    #         d1[-1] = "frequency"
    #         c1 = {}
    #         for d in d1:
    #             if d in x_win.coords:
    #                 c1[d] = x_win[d]
    #             elif d == "wavenumber":
    #                 c1[d] = z_k_c
    #             elif d == "frequency":
    #                 c1[d] = z_v_c
    #         z_centered = xr.DataArray(z_centered, dims=d1, coords=c1)
    # BUT THAT IS INCORRECT TO GET THE PROPAGATION DIRECTION OF ZONAL WAVES
    # (in testing, it seems to end up opposite in wavenumber)
    # Apply reordering per Hayashi to get correct wave propagation convention
    #     this function is customized to expect z to be a DataArray
    z_pee = resolveWavesHayashi(z, segsize//spd, spd)
    z_pee = abs(z_pee)**2
    # z_pee is spectral power already. 
    # z_pee is a DataArray w/ coordinate vars for wavenumber & frequency

    # average over all available segments and sum over latitude
    # OUTPUT DEPENDS ON SYMMETRIES
    if dosymmetries:
        # multipy by 2 b/c we only used one hemisphere
        z_symmetric = 2.0 * z_pee.isel(latitude=z_pee.latitude<0).mean(dim='time').sum(dim='latitude').squeeze()
        z_symmetric.name = "power"
        z_antisymmetric = 2.0 * z_pee.isel(latitude=z_pee.latitude>0).mean(dim='time').sum(dim='latitude').squeeze()
        z_antisymmetric.name = "power"
        z_final = xr.concat([z_symmetric, z_antisymmetric], "component")
        z_final = z_final.assign_coords({"component":["symmetric","antisymmetric"]})
    else:
        #latitude = z_pee['latitude']
        #lat_inds = np.argwhere(((latitude <= nlat)&(latitude >= slat))).squeeze()
        z_final = z_pee.mean(dim='time').sum(dim='latitude').squeeze()
    return z_final

def genDispersionCurves(nWaveType=6, nPlanetaryWave=50, rlat=0, Ahe=[50, 25, 12]):
    """
    Function to derive the shallow water dispersion curves. Closely follows NCL version.

    input:
        nWaveType : integer, number of wave types to do
        nPlanetaryWave: integer
        rlat: latitude in radians (just one latitude, usually 0.0)
        Ahe: [50.,25.,12.] equivalent depths
              ==> defines parameter: nEquivDepth ; integer, number of equivalent depths to do == len(Ahe)

    returns: tuple of size 2
        Afreq: Frequency, shape is (nWaveType, nEquivDepth, nPlanetaryWave)
        Apzwn: Zonal savenumber, shape is (nWaveType, nEquivDepth, nPlanetaryWave)
        
    notes:
        The outputs contain both symmetric and antisymmetric waves. In the case of 
        nWaveType == 6:
        0,1,2 are (ASYMMETRIC) "MRG", "IG", "EIG" (mixed rossby gravity, inertial gravity, equatorial inertial gravity)
        3,4,5 are (SYMMETRIC) "Kelvin", "ER", "IG" (Kelvin, equatorial rossby, inertial gravity)
    """
    nEquivDepth = len(Ahe) # this was an input originally, but I don't know why.
    pi     = np.pi
    radius = 6.37122e06    # [m]   average radius of earth
    g      = 9.80665        # [m/s] gravity at 45 deg latitude used by the WMO
    omega  = 7.292e-05      # [1/s] earth's angular vel
    ll    = 2.*pi*radius*np.cos(np.abs(rlat))
    Beta  = 2.*omega*np.cos(np.abs(rlat))/radius
    fillval = 1e20
    
    # Initialize the output arrays
    Afreq = np.empty((nWaveType, nEquivDepth, nPlanetaryWave))
    Apzwn = np.empty((nWaveType, nEquivDepth, nPlanetaryWave))

    for ww in range(1, nWaveType+1):
        for ed, he in enumerate(Ahe):
            # this loops through the specified equivalent depths
            # ed provides index to fill in output array, while
            # he is the current equivalent depth
            # T = 1./np.sqrt(Beta)*(g*he)**(0.25) This is close to pre-factor of the dispersion relation, but is not used.
            c = np.sqrt(g * he)  # phase speed   
            L = np.sqrt(c/Beta)  # was: (g*he)**(0.25)/np.sqrt(Beta), this is Rossby radius of deformation        

            for wn in range(1, nPlanetaryWave+1):
                s  = -20.*(wn-1)*2./(nPlanetaryWave-1) + 20.
                k  = 2.0 * pi * s / ll
                kn = k * L 

                # Anti-symmetric curves  
                if (ww == 1):       # MRG wave
                    if (k < 0):
                        dell  = np.sqrt(1.0 + (4.0 * Beta)/(k**2 * c))
                        deif = k * c * (0.5 - 0.5 * dell)
                    
                    if (k == 0):
                        deif = np.sqrt(c * Beta)
                    
                    if (k > 0):
                        deif = fillval
                    
                
                if (ww == 2):       # n=0 IG wave
                    if (k < 0):
                        deif = fillval
                    
                    if (k == 0):
                        deif = np.sqrt( c * Beta)
                    
                    if (k > 0):
                        dell  = np.sqrt(1.+(4.0*Beta)/(k**2 * c))
                        deif = k * c * (0.5 + 0.5 * dell)
                    
                
                if (ww == 3):       # n=2 IG wave
                    n=2.
                    dell  = (Beta*c)
                    deif = np.sqrt((2.*n+1.)*dell + (g*he) * k**2)
                    # do some corrections to the above calculated frequency.......
                    for i in range(1,5+1):
                        deif = np.sqrt((2.*n+1.)*dell + (g*he) * k**2 + g*he*Beta*k/deif)
                    
    
                # symmetric curves
                if (ww == 4):       # n=1 ER wave
                    n=1.
                    if (k < 0.0):
                        dell  = (Beta/c)*(2.*n+1.)
                        deif = -Beta*k/(k**2 + dell)
                    else:
                        deif = fillval
                    
                if (ww == 5):       # Kelvin wave
                    deif = k*c

                if (ww == 6):       # n=1 IG wave
                    n=1.
                    dell  = (Beta*c)
                    deif = np.sqrt((2. * n+1.) * dell + (g*he)*k**2)
                    # do some corrections to the above calculated frequency
                    for i in range(1,5+1):
                        deif = np.sqrt((2.*n+1.)*dell + (g*he)*k**2 + g*he*Beta*k/deif)
                
                eif  = deif  # + k*U since  U=0.0
                P    = 2.*pi/(eif*24.*60.*60.)  #  => PERIOD
                # dps  = deif/k  # Does not seem to be used.
                # R    = L #<-- this seemed unnecessary, I just changed R to L in Rdeg
                # Rdeg = (180.*L)/(pi*6.37e6) # And it doesn't get used.
            
                Apzwn[ww-1,ed-1,wn-1] = s
                if (deif != fillval):
                    # P = 2.*pi/(eif*24.*60.*60.) # not sure why we would re-calculate now
                    Afreq[ww-1,ed-1,wn-1] = 1./P
                else:
                    Afreq[ww-1,ed-1,wn-1] = fillval
    return  Afreq, Apzwn

def Crossspectra( data1: xr.DataArray, data2: xr.DataArray, nDayWin: int, spd: int ):

    varfft = abs(np.conjugate(data1)*data2)**2/((abs(data1)**2)*(abs(data2)**2))
    del data1, data2
    gc.collect()

    logging.debug(f"[Cross-spectra] nDayWin: {nDayWin}, spd: {spd}")
    dimnames = varfft.dims
    dimvf  = varfft.shape
    mlon   = len(varfft['wavenumber'])
    N      = dimvf[-1]
    logging.info(f"[Cross-spectra] input dims is {dimnames}, {dimvf}")
    logging.info(f"[Cross-spectra] input coords is {varfft.coords}")

    if len(dimnames) != len(varfft.coords):
        logging.error("The size of varfft.coords is incorrect.")
        raise ValueError("STOP")

    nshape = list(dimvf)
    nshape[-2] += 1
    nshape[-1] += 1
    logging.debug(f"[Cross-spectra] The nshape ends up being {nshape}")
    # this is a reordering, use Ellipsis to allow arbitrary number of dimensions,
    # but we insist that the wavenumber and frequency dims are rightmost.
    # we will fill the new array in increasing order (arbitrary choice)
    logging.debug("allocate the re-ordered array")
    #newempty = xr.DataArray(np.zeros(nshape),coords = varfft.coords,dims = varfft.dims)
    pee = np.full(nshape, np.nan, dtype=type(varfft))
    #varspacetime = xr.full_like(newempty, np.nan, dtype=type(varfft))
    # first two are the negative wavenumbers (westward), second two are the positive wavenumbers (eastward)
    
    logging.debug(f"[Cross-spectra] Assign values into array. Notable numbers: mlon//2={mlon//2}, N//2={N//2}")

    #  Create the real power spectrum pee = sqrt(real^2+imag^2)^2
    pee[..., 0:mlon//2, 0:N//2    ] = varfft[..., mlon//2:0:-1, N//2:] # neg.k, pos.w
    pee[..., 0:mlon//2, N//2:     ] = varfft[..., mlon//2:0:-1, 0:N//2+1]   # neg.k, 
    pee[..., mlon//2:   , 0:N//2+1] = varfft[..., 0:mlon//2+1,  N//2::-1]   # assign eastward & neg.freq.
    pee[..., mlon//2:   , N//2+1: ] = varfft[..., 0:mlon//2+1, -1:N//2-1:-1] # assign eastward & pos.freq.
    logging.debug("calculate power")

    logging.debug("put into DataArray")

    # add meta data for use upon return
    wave      = np.arange(-mlon // 2, (mlon // 2 )+ 1, 1, dtype=int)  
    freq      = np.linspace(-1*nDayWin*spd/2, nDayWin*spd/2, (nDayWin*spd)+1) / nDayWin

    odims = list(dimnames)
    odims[-2] = "wavenumber"
    odims[-1] = "frequency"
    ocoords = {}
    for c in varfft.coords:
        logging.debug(f"[hayashi] working on coordinate {c}")
        if (c != "wavenumber") and (c != "frequency"):
            ocoords[c] = varfft[c]
        elif c == "wavenumber":
            ocoords['wavenumber'] = wave
        elif c == "frequency":
            ocoords['frequency'] = freq
   
    pee = xr.DataArray(pee, dims=odims, coords=ocoords)

    return pee

def Cross_spacetime_power(data_pair, segsize, noverlap, spd, dosymmetries=True, rmvLowFrq=True):
    """Perform space-time cross spectral decomposition and return power spectrum following Wheeler-Kiladis approach.
    
    """

    segsize = spd*segsize
    noverlap = spd*noverlap
    assert segsize-noverlap > 0, f"Error, inconsistent specification of segsize and noverlap results in stride of {segsize-noverlap}, but must be > 0."

    zz = []
    for data in data_pair:
        slat = data['latitude'].min().item()
        nlat = data['latitude'].max().item()
        
        # "Remove dominant signals"
    
        # "detrend" the data, including removing the mean (uses scipy.signal.detrend):
        #  --> ncl version keeps the mean:
        xmean = data.mean(dim='time')
        xdetr = detrend(data, axis=0, type='linear')
        xdetr = xr.DataArray(xdetr, dims=data.dims, coords=data.coords)
        xdetr = xdetr+xmean # put the mean back in??????????????????????????????????????????
        # --> Tested and confirmed that this approach gives same answer as NCL
        
        # filter low-frequencies    
        if rmvLowFrq:
            data = rmvAnnualCycle(xdetr, spd, 1/segsize)
            logging.info("Annual cycle is removed")
        # --> Tested and confirmed that this function gives same answer as NCL

        dimsizes = data.sizes  # dict
        lon_size = dimsizes['longitude']
        lat_size = dimsizes['latitude']
        lat_dim = data.dims.index('latitude')
        if dosymmetries:
            data = decompose2SymAsym(data)
            logging.info("Decomposited into symetric and asymetric")
        # testing: pass -- Gets the same result as NCL.

        # 2. Windowing with the xarray "rolling" operation, and then limit overlap with `construct` to produce a new dataArray.
        # WK99 recommend "2-month" overlap
        x_win = data.rolling(time=segsize, min_periods=segsize).construct("segments")  # WK99 use 96-day window
        x_win = x_win.isel(time=slice(segsize-1,None,segsize-noverlap))  

        logging.debug(f"[spacetime_power] x_win shape is {x_win.shape}")
        # Additional detrend for each segment: means??????????????????????????????????
        if  np.logical_not(np.any(np.isnan(x_win))):
            logging.info("No missing, so use simplest segment detrend.")
            x_win_detr = detrend(x_win, axis=-1, type='linear') #<-- missing data makes this not work
            x_win = xr.DataArray(x_win_detr, dims=x_win.dims, coords=x_win.coords)
        else:
            logging.warning("EXTREME WARNING -- This method to detrend with missing values present does not quite work, probably need to do interpolation instead.")
            logging.warning("There are missing data in x_win, so have to try to detrend around them.")
            x_win_cp = x_win.copy()
            logging.info(f"[spacetime_power] x_win_cp windowed data has shape {x_win_cp.shape} \n \t It is a numpy array, copied from x_win which has dims: {x_win.sizes} \n \t ** about to detrend this in the rightmost dimension.")
            x_win_cp[np.logical_not(np.isnan(x_win_cp))] = detrend(x_win_cp[np.logical_not(np.isnan(x_win_cp))])
            x_win = xr.DataArray(x_win_cp, dims=x_win.dims, coords=x_win.coords)
    
        # 3. Taper in time to make the signal periodic, as required for FFT.
        # taper = np.hanning(segsize)  # WK seem to use some kind of stretched out hanning window; unclear if it matters
        taper = split_hann_taper(segsize, 0.1)  # try to replicate NCL's
        x_wintap = x_win*taper # would do XTAPER = (X - X.mean())*series_taper + X.mean()
                           # But since we have removed the mean, taper going to 0 is equivalent to taper going to the mean.
    
        # Do the transform using 2D FFT
        #
        # z = np.fft.fft2(x_wintap, axes=(2,3)) / (lon_size * segsize)
    
        # Or do the transform with 2 steps
        z = np.fft.fft(x_wintap, axis=2) / lon_size  # note that np.fft.fft() produces same answers as NCL cfftf
        z = np.fft.fft(z, axis=3) / segsize 
        z = xr.DataArray(z, dims=("time","latitude","wavenumber","frequency"), 
                         coords={"time":x_wintap["time"], 
                                 "latitude":x_wintap["latitude"],
                                 "wavenumber":np.fft.fftfreq(lon_size, 1/lon_size),
                                 "frequency":np.fft.fftfreq(segsize, 1/spd)})
        zz.append(z)
    # The FFT is returned following ``standard order`` which has negative frequencies in second half of array. 
    #
    # IMPORTANT: 
    # If this were typical 2D FFT, we would do the following to get the frequencies and reorder:
    #         z_k = np.fft.fftfreq(x_wintap.shape[-2], 1/lon_size)
    #         z_v = np.fft.fftfreq(x_wintap.shape[-1], 1)  # Assumes 1/(1-day) timestep
    # reshape to get the frequencies centered
    #         z_centered = np.fft.fftshift(z, axes=(2,3))
    #         z_k_c = np.fft.fftshift(z_k)
    #         z_v_c = np.fft.fftshift(z_v)
    # and convert to DataArray as this:
    #         d1 = list(x_win.dims)
    #         d1[-2] = "wavenumber"
    #         d1[-1] = "frequency"
    #         c1 = {}
    #         for d in d1:
    #             if d in x_win.coords:
    #                 c1[d] = x_win[d]
    #             elif d == "wavenumber":
    #                 c1[d] = z_k_c
    #             elif d == "frequency":
    #                 c1[d] = z_v_c
    #         z_centered = xr.DataArray(z_centered, dims=d1, coords=c1)
    # BUT THAT IS INCORRECT TO GET THE PROPAGATION DIRECTION OF ZONAL WAVES
    # (in testing, it seems to end up opposite in wavenumber)
    # Apply reordering per Hayashi to get correct wave propagation convention
    #     this function is customized to expect z to be a DataArray
    #z_pee = Crossspectra(zz[0],zz[1], segsize//spd, spd)
    z_pee1 = resolveWavesHayashi(zz[0], segsize//spd, spd)
    z_pee2 = resolveWavesHayashi(zz[1], segsize//spd, spd)

    # z_pee is cross spectral power already. 
    # z_pee is a DataArray w/ coordinate vars for wavenumber & frequency

    # average over all available segments and sum over latitude
    # OUTPUT DEPENDS ON SYMMETRIES
    if dosymmetries:
        # multipy by 2 b/c we only used one hemisphere
        z_pee3 = z_pee1.isel(latitude=z_pee1.latitude<0)
        z_pee4 = z_pee2.isel(latitude=z_pee2.latitude<0)
        CAB = (np.conjugate(z_pee3)*z_pee4).mean(dim=['time','latitude'])
        CAA = (np.conjugate(z_pee3)*z_pee3).mean(dim=['time','latitude'])
        CBB = (np.conjugate(z_pee4)*z_pee4).mean(dim=['time','latitude'])
        z_symmetric = (CAB*np.conjugate(CAB))/(CAA*CBB)
        z_symmetric.name = "power"

        z_pee3 = z_pee1.isel(latitude=z_pee1.latitude>0)
        z_pee4 = z_pee2.isel(latitude=z_pee2.latitude>0)
        CAB = (np.conjugate(z_pee3)*z_pee4).mean(dim=['time','latitude'])
        CAA = (np.conjugate(z_pee3)*z_pee3).mean(dim=['time','latitude'])
        CBB = (np.conjugate(z_pee4)*z_pee4).mean(dim=['time','latitude'])
        z_antisymmetric = (CAB*np.conjugate(CAB))/(CAA*CBB)
        z_antisymmetric.name = "power"
        z_final = xr.concat([z_symmetric, z_antisymmetric], "component")
        z_final = z_final.assign_coords({"component":["symmetric","antisymmetric"]})
    else:
        #latitude = z_pee['latitude']
        #logging.info(f"z_pee is {z_pee}")
        #lat_inds = np.argwhere(((latitude <= nlat)&(latitude >= slat))).squeeze()
        z_final = z_pee.mean(dim='time').sum(dim='latitude').squeeze()
    return z_final

def plot_normalized_symmetric_spectrum(s, cmap,levels=None, ofil=None):
    """Basic plot of normalized symmetric power spectrum with shallow water curves."""
    fb = [0, .8]  # frequency bounds for plot
    # get data for dispersion curves:
    swfreq,swwn = genDispersionCurves()
    # swfreq.shape # -->(6, 3, 50)
    swf = np.where(swfreq == 1e20, np.nan, swfreq)
    swk = np.where(swwn == 1e20, np.nan, swwn)

    fig, ax = plt.subplots()
    c = 'black' # COLOR FOR DISPERSION LINES/LABELS
    z = s.transpose().sel(frequency=slice(*fb), wavenumber=slice(-20,20))
    z.loc[{'frequency':0}] = np.nan
    kmesh0, vmesh0 = np.meshgrid(z['wavenumber'], z['frequency'])

    img = ax.contourf(kmesh0, vmesh0, z, 
                      levels=levels, 
                      cmap=cmap,  extend='both')
    
    for ii in range(3,6):
        ax.plot(swk[ii, 0,:], swf[ii,0,:], linestyle='dashed', color=c)
        ax.plot(swk[ii, 1,:], swf[ii,1,:], linestyle='dashed', color=c)
        ax.plot(swk[ii, 2,:], swf[ii,2,:], linestyle='dashed', color=c)
    ax.axvline(0, linestyle='dashed', color='lightgray')
    ax.set_xlim([-20,20])
    ax.set_ylim(fb)    
    #ax.set_title("Normalized Symmetric Component")
    ax.tick_params(direction='in', which='both')
    ax.set_xlabel(r"Wavenumber ($deg^{-1}$)")
    ax.set_ylabel(r"Frequency ($day^{-1}$)")
    fig.colorbar(img,
                 extendrect = True,
                 extendfrac = 'auto',
                 orientation='vertical',
                 shrink=0.8,
                 aspect = 30,
                 drawedges=False)
    if ofil is not None:
        fig.savefig(ofil, bbox_inches='tight')
    
    return

def plot_normalized_asymmetric_spectrum(s, cmap,levels = None,ofil=None):
    """Basic plot of normalized symmetric power spectrum with shallow water curves."""

    fb = [0, .8]  # frequency bounds for plot
    # get data for dispersion curves:
    swfreq,swwn = genDispersionCurves()
    # swfreq.shape # -->(6, 3, 50)
    swf = np.where(swfreq == 1e20, np.nan, swfreq)
    swk = np.where(swwn == 1e20, np.nan, swwn)

    fig, ax = plt.subplots()
    c = 'black' # COLOR FOR DISPERSION LINES/LABELS
    z = s.transpose().sel(frequency=slice(*fb), wavenumber=slice(-20,20))
    z.loc[{'frequency':0}] = np.nan
    kmesh0, vmesh0 = np.meshgrid(z['wavenumber'], z['frequency'])

    img = ax.contourf(kmesh0, vmesh0, z, 
                      levels=levels, 
                      cmap=cmap, extend='both')

    for ii in range(0,3):
        ax.plot(swk[ii, 0,:], swf[ii,0,:], linestyle='dashed', color=c)
        ax.plot(swk[ii, 1,:], swf[ii,1,:], linestyle='dashed', color=c)
        ax.plot(swk[ii, 2,:], swf[ii,2,:], linestyle='dashed', color=c)
    ax.axvline(0, linestyle='dashed', color='lightgray')
    ax.set_xlim([-20,20])
    ax.set_ylim(fb)
    #ax.set_title("Normalized Anti-symmetric Component")
    ax.tick_params(direction='in', which='both')
    ax.set_xlabel(r"Wavenumber ($deg^{-1}$)")
    ax.set_ylabel(r"Frequency ($day^{-1}$)")
    fig.colorbar(img,
                 extendrect = True,
                 extendfrac = 'auto',
                 orientation='vertical',
                 shrink=0.8,
                 aspect = 30,
                 drawedges=False)
    if ofil is not None:
        fig.savefig(ofil, bbox_inches='tight')

    return
