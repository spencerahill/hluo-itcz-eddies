"""Regression and significance testing.

Both functions here have two distinct bodies across the scripts, and the two
pairs differ in kind.

``regression`` differs between the two copies of ``myfun.py`` in whether the
standard deviation uses ``ddof=1``, which rescales every regression
coefficient.  ``sigtest`` differs more than that: the two bodies implement
different statistical procedures.  The copies moved here are
``main/myfun.py``'s, which is what the analysis scripts import.  Until real
data settles which is right, the other bodies are left where they are rather
than silently discarded; that is the third rule of PRIORITIES.md item 2.
"""

from __future__ import annotations

import logging

import dask.array as da
import numpy as np
import xarray as xr

def sigtest(idx, x, x_regress, adjusted=False, dim="time"):
    m1 = np.shape(x)
    
    # Convert idx and x to dask arrays for parallel processing (if they are not already dask arrays)
    if isinstance(idx, xr.DataArray):
        idx = idx.chunk({'time': -1})  # Automatically chunk by time dimension for Dask
    if isinstance(x, xr.DataArray):
        x = x.chunk({'time': -1})  # Automatically chunk by time dimension for Dask
    
    # Transpose arrays so 'time' is always the first dimension, done once before the loop
    for arr in [idx, x]:
        time_position = list(arr.dims).index(dim)
        if time_position != 0:
            arr = arr.transpose('time', *[dim for dim in arr.dims if dim != 'time'])
    
    # Determine the number of lags
    nlag = m1[0] - 1 if adjusted else int(m1[0] / 4)

    # Initialize arrays for correlation values
    Cxx = xr.DataArray(da.zeros(nlag), coords={"lag": np.arange(nlag)})
    Cyy = xr.DataArray(da.zeros((nlag, *x.shape[1:])), coords={"lag": np.arange(nlag), "latitude": x.latitude, "longitude": x.longitude})
    #Cyy = xr.DataArray(da.zeros(nlag), coords={"lag": np.arange(nlag)})

    # Precompute shifts for x and idx once before the loop
    idx_shifts = xr.DataArray([idx.shift(time=lag) for lag in range(nlag)],
                              coords = {"lag":range(nlag),"time":idx.time},dims = ["lag","time"])
    x_shifts = xr.DataArray([x.shift(time=lag) for lag in range(nlag)],
                            coords = {"lag":range(nlag),"time":x.time,"latitude":x.latitude,"longitude":x.longitude},
                            dims = ["lag","time","latitude","longitude"])

    # correlation computation over lag
    Cxx = xr.corr(idx,idx_shifts,dim=dim)
    Cyy = xr.corr(x,x_shifts,dim=dim)
    #for lag in range(nlag):
    #    Cxx[lag] = xr.corr(idx,idx_shifts[lag],dim=dim)
    #    Cyy[lag] = xr.corr(x,x_shifts[lag],dim=dim)
#

    # Compute Tau
    Tau = 2 * ((Cxx * Cyy).sum(dim="lag")) - 1.0

    # Calculate the maximum correlation coefficient threshold
    N_star = m1[0] / Tau
    rc = np.tanh(1.96 / np.sqrt(N_star - 3))
    rc = xr.where(N_star==3,1,rc)
    rc_max = rc.max(skipna = 'True')
    logging.info(f"maximum rc is {rc_max.values}")
    max_coord = rc.stack(z=("latitude", "longitude")).idxmax("z",skipna = 'True')
    logging.info(f"maximum rc is at {max_coord.values}")
    
    # Compute correlation for x and idx
    cc_var = xr.corr(idx, x, dim=dim)
    
    # Apply the correlation threshold to x_regress
    x_regress_new = x_regress.where((cc_var >= rc) | (cc_var <= -rc), np.nan)
    
    return x_regress_new.compute()

def regression(idx,x,test,*args, **kwargs):
    idx = (idx-idx.mean(dim = "time"))/idx.std(dim = "time",ddof = 1)
    dim = kwargs.get('dim', "time")
    m1 = len(idx)
    m2 = len(getattr(x,dim))
    if m1 == m2:
        x, idx = xr.align(x, idx,join = "override")
    else:
        print("The index must has the same length as the variable")
    VAR = xr.cov(idx, x, dim=dim)#.values #regression coefficient when IDX is standardized

    if test == True:
        rc = kwargs.get('rc', False)
        if rc:
            print("using rc from input...")
            cc_var = xr.corr(idx,x,dim = dim)
            VAR_new = VAR.where((cc_var >= rc) | (cc_var <= -rc),np.nan)
            return VAR_new
        else:
            print("calculating rc...")
            adjusted = kwargs.get('adjusted', False)
            VAR_test = sigtest(idx,x,VAR,adjusted,dim)
            return VAR_test
    else:
        print("no significance test")
        return VAR
