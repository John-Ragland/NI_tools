'''
utils.py - general utilities
'''

import xarray as xr
import numpy as np
from scipy import signal

def xr_filtfilt(ds, dim, b, a, compute=False):
    '''
    xr_filtfilt - zero phase linear filter of xarray object
        designed to be distributed
        ds must have only one chunk along filter dimension

    Parameters
    ----------
    ds : xr.Dataset
        dataset to filter
    dim : string
        dimension in ds along which to linearly filter
    b : list
        filter numerator coefficents
    a : list
        filter denominator coefficients
    compute : bool
        whether or not to return task map or computed result

    Returns
    -------
    ds_filter : xr.Dataset
        filtered dataset
    '''

    ds_filt = ds.map_blocks(__xr_filtfilt_chunk, args=(b,a, dim), template=ds)

    if compute:
        return ds_filt.compute()
    else:
        return ds_filt

def __xr_filtfilt_chunk(ds, b,a, dim):
    '''
    single chunk implentation of filtfilt

    Parameters
    ----------
    ds : xr.Dataset
    b : list
        filter coefs
    a : list
        filter coefs
    dim : string
        dimension to filter over
    '''

    dim_idx = list(ds.dims.keys()).index(dim)

    ds_filt = {}

    for var in list(ds.data_vars):
        ds_filt[var] = xr.DataArray(signal.filtfilt(b,a,ds[var].values, axis=dim_idx), dims=ds.dims, coords=ds.coords)

    ds_filtx = xr.Dataset(ds_filt)
    return ds_filtx