'''
inversion.py - toolset for estimating features from NCCFs, such as arrival time

John Ragland
Rebuilt from ground up 2023-06-05
'''

import numpy as np
import xarray as xr
import pandas as pd
from scipy import signal, interpolate
import scipy
from NI_tools.NI_tools import utils
import xrsignal

def estimate_at_Rxx(NCCFs, dim, peaks = None):
    '''
    estimate_at_Rxx - estimate arrival times using Auto-Correlation Method

    Parameters
    ----------
    NCCFs : xarray.DataArray
        NCCFs to calculate arrival times for
    dim : str
        Dimension to calculate arrival times along
    peaks : dict, optional
        dictory of peak names and slices of NCCF corresponding to peak
    '''
    
    if peaks is None:
        peaks = {
            'dA':slice(-2.5, -1.5),
            's1b0A':slice(-3.5, -2.5),
            's1b0B':slice(2.5, 3.5),
            's2b1A':slice(-5,-4),
            's2b1B':slice(4,5),
        }

    arrival_times = {}

    for peak in peaks.keys():

        pass
    return

def __estimate_at_Rxx_chunk(NCCFs, dim, peaks):
    '''
    __estimate_at_Rxx_chunk - estimate the arrival time using Auto-Correlation Method

    Parameters
    ----------
    NCCFs : xarray.DataArray
        NCCFs to calculate arrival times for
    dim : str
        Dimension to calculate arrival times along
    peaks : dict
        dictory of peak names and slices of NCCF corresponding to peak
    
    Returns
    -------
    at : xarray.DataArray
        Estimated arrival time
    '''

    dim_idx = NCCFs.get_axis_num(dim)

    data = NCCFs.values

    return 

def calculate_arrival_times(NCCFs, dim, b=None,a=None,  peaks=None, grid_tolerance=1e-13, second_whiten=False):
    '''
    calculate_arrival_times - calculate arrival times for given NCCF.
    Signal Processing Methods:
    - frequency whiten individual peak windows
    - take hilbert magnitude
    - argmax_interp using quadratic peak interpolation

    Parameters
    ----------
    NCCFs : xarray.DataArray
        NCCFs to calculate arrival times for
    dim : str
        Dimension to calculate arrival times along
    peaks : dict, optional
        Dictionary of peak windows. Keys will be passed to arrival_times
        and values should be slice (in coordinate units) of NCCF.
        If None, peaks is default to caldera peaks
    b : numpy.ndarray
        Numerator of filter
    a : numpy.ndarray   
        Denominator of filter
    grid_tolerance : float, optional
        Tolerance for determining if coordinates are uniform grid
    second_whiten : bool, optional
        If True, whiten again after hilbert magnitude

    Returns
    -------
    arrival_times : xarray.Dataset
        Dataset with arrival times for each peak
    '''

    if peaks is None:
        peaks = {
            'dA':slice(-2.5, -1.5),
            's1b0A':slice(-3.5, -2.5),
            's1b0B':slice(2.5, 3.5),
            's2b1A':slice(-5,-4),
            's2b1B':slice(4,5),
        }

    arrival_times = {}

    for peak in peaks.keys():
        data_slice = NCCFs.sel({dim:peaks[peak]})
        arrival_time = calculate_peak_arrival_time(data_slice, dim, b=b, a=a, grid_tolerance=grid_tolerance, second_whiten=second_whiten)

        # add to dataset
        arrival_times[peak] = arrival_time
    
    arrival_times_x = xr.Dataset(arrival_times)

    return arrival_times_x

def calculate_peak_arrival_time(peak, dim, b=None, a=None, grid_tolerance=1e-13, second_whiten=False):
    '''
    calculate_peak_arrival_time - calculate arrival time for data array that consists of a single peak

    Parameters
    ----------
    peak : xarray.DataArray
        DataArray containing a single peak
    dim : str
        Dimension to calculate arrival times along
    b : numpy.ndarray
        Numerator of filter
    a : numpy.ndarray
        Denominator of filter
    grid_tolerance : float, optional
        Tolerance for determining if coordinates are uniform grid
    second_whiten : bool, optional
        If True, whiten again after hilbert magnitude
    
    Returns
    -------
    arrival_time : xarray.DataArray
        DataArray with arrival time for peak
    '''


    # frequency whiten
    if second_whiten:
        data_w = utils.freq_whiten(peak, dim=dim, b=b, a=a)
    else:
        data_w = peak
    
    # hilbert magnitude
    data_c = xrsignal.hilbert_mag(data_w, dim=dim)

    # argmax_interp
    arrival_time = argmax_interp(data_c, dim=dim, grid_tolerance=grid_tolerance)

    return arrival_time

def argmax_interp(data, dim, grid_tolerance=1e-13):
    '''
    argmax_interp - find location of maximum along dimension 'dim' using
    argmax and quadratic peak interpolation

    maps function to __argmax_quadinterp_array if dataarray or dataset

    Parameters
    ----------
    data : xarray.DataArray or xarray.Dataset
        DataArray or Dataset to find argmax of
    dim : str
        Dimension to find argmax along
    grid_tolerance : float, optional
        Tolerance for determining if coordinates are uniform grid

    Returns
    -------
    argmax : xarray.DataArray or xarray.Dataset
        DataArray or Dataset with argmax along dim
    '''

    if isinstance(data, xr.DataArray):
        argmax =  __argmax_quadinterp_array(data, dim, grid_tolerance=grid_tolerance)
    elif isinstance(data, xr.Dataset):
        argmax = data.map(__argmax_quadinterp_array, dim=dim, grid_tolerance=grid_tolerance)
    else:
        raise ValueError('data must be xarray.DataArray or xarray.Dataset')
    
    return argmax
    
def __argmax_quadinterp_array(da, dim, grid_tolerance=1e-13):
    '''
    argmax_interp - find location of maximum along dimension 'dim' using
    argmax and quadratic peak interpolation
        
    all functionality is built on xarray and the coordinates of the
        DataArray are used to interpolate the maximum
    
    Parameters
    ----------
    da : xarray.DataArray
        DataArray to find argmax of
    dim : str
        Dimension to find argmax along
    grid_tolerance : float, optional
        Tolerance for determining if coordinates are uniform grid
    
    Returns
    -------
    argmax : xarray.DataArray
        DataArray with argmax along dim
    '''

    # Check if there are coordinates in the specified dimension
    if dim not in da.coords:
        raise ValueError('No coordinates in dimension: {}'.format(dim))

    # Check if coordinates in dimension dim are uniform grid (up to tolerance)
    is_uniform_grid = True
    diff = np.diff(da.coords[dim])
    if not np.allclose(diff, diff[0], atol=grid_tolerance):
        is_uniform_grid = False
        grid_diff = np.max(np.abs(diff - diff[0]))
    
    if not is_uniform_grid:
        raise ValueError('Coordinates in dimension {} are not uniform grid, tolerance of {}, difference of {}'.format(dim, grid_tolerance, grid_diff))

    other_dims = list(da.dims)
    other_dims.remove(dim)

    # round sampling period to nearest grid_tolerance
    Ts = np.round(da[dim].diff(dim=dim).values[0]/grid_tolerance)*grid_tolerance

    da_nonan = da
    for other_dim in other_dims:
        da_nonan = da_nonan.dropna(dim=other_dim)

    # Quadratic Peak Interpolation
    beta_idx = da_nonan.isel({dim:slice(1,-1)}).argmax(dim=dim) + 1
    alpha_idx = beta_idx-1
    gamma_idx = beta_idx+1

    alpha = da_nonan.isel({dim:alpha_idx})
    beta = da_nonan.isel({dim:beta_idx})
    gamma = da_nonan.isel({dim:gamma_idx})

    p = 0.5 * (alpha - gamma) / (alpha - 2 * beta + gamma)
    p_tot = beta_idx + p

    argmax = da_nonan[dim][0] + p_tot * Ts
    
    return argmax

