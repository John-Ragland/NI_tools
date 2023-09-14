'''
calculate.py - Tools for calculating NCCFs

contains combination of experimental and working code.
In general the code is desinged to work with dask arrays and distributed computing
'''

import dask
import scipy
import xarray as xr
import multiprocessing as mp
from scipy import signal
import numpy as np
import xrft

## Preprocessing functions
def preprocess(da, dim, W=30, Fs=200, fcs=[1, 90], include_coords=False):
    '''
    preprocess - takes time series and performs pre-processing steps for estimating cross-correlation

    Currently pre-processing is fixed to be
    - divide into 30 s segments
    - filter between 1 and 90 Hz
    - clip to 3 times std for every chunk (which is also 1 hour)
    - frequency whiten

    Parameters
    ----------
    da : xr.DataArray or xr.Dataset
        DataArray or Dataset of arbitrary dimensions
    dim : str
        dimension to apply pre-processing to (should be samples in time)
    W : float
        length of window in seconds
    Fs : float
        sampling rate in Hz 
    fcs : list
        cutoff frequencies for bandpass filter
    include_coords : bool
        whether to include coordinates in output DataArray

    Return
    ------
    data_whiten : np.array
        pre-procesesd data
    '''
    b, a = signal.butter(4, [fcs[0]/(Fs/2), fcs[1]/(Fs/2)], btype='bandpass')

    data_pp = da.map_blocks(__preprocess_chunk, kwargs=(
        {'dim': dim, 'b': b, 'a': a, 'W': W, 'Fs': Fs}), template=da)
    
    if include_coords:
        template = __preprocess_get_multiindex(da, dim=dim)
        data_pp = data_pp.reindex_like(template)

    return data_pp


def __preprocess_get_multiindex(da, dim='time', W=30, Fs=200):
    '''
    given a DataArray create multi-index for last dimension

    Parameters
    ----------
    da : xr.DataArray
        DataArray with last dimension as preprocessing dimension
    dim : str
        dimension to apply pre-processing to (should be samples in time)
    W : float
        length of window in seconds
    Fs : float
        sampling rate in Hz
    
    Returns
    -------
    mi : xr.DataArray
    '''

    # create template for map_blocks
    if isinstance(da, xr.DataArray):

        _, _, _, new_shape = __preprocess_get_size(da, dim='time', W=30, Fs=200)
        # build template
        dask_temp = dask.array.random.random(new_shape)
        da_new = xr.DataArray(dask_temp, dims=[f'long_{dim}', f'short_{dim}'], coords={f'long_{dim}':np.arange(new_shape[0]), f'short_{dim}':np.arange(new_shape[1])})
        template = da_new.stack({dim:[f'long_{dim}', f'short_{dim}']})
    
    elif isinstance(da, xr.Dataset):
        das = []
        for key in da.keys():           
            _, _, _, new_shape = __preprocess_get_size(da[key], dim='time', W=30, Fs=200)
            # build template
            dask_temp = dask.array.random.random(new_shape)
            da_new = xr.DataArray(dask_temp, dims=[f'long_{dim}', f'short_{dim}'], coords={f'long_{dim}':np.arange(new_shape[0]), f'short_{dim}':np.arange(new_shape[1])})
            da_stack = da_new.stack({dim:[f'long_{dim}', f'short_{dim}']})
            das.append(da_stack)
        
        template = xr.Dataset(dict(zip(da.keys(), das)))
    else:
        raise ValueError('da must be xr.DataArray or xr.Dataset')
    
    return template


def __preprocess_get_size(da, dim, W, Fs):
    '''
    __preprocess_get_size - get size of data to be preprocessed

    Parameters
    ----------
    da : xr.DataArray
        DataArray of arbitrary dimensions
    dim : str
        dimension to apply pre-processing to (should be samples in time)
    W : float
        length of window in seconds
    Fs : float
        sampling rate in Hz

    Returns
    -------
    old_dims : list
        list of dimensions in original DataArray
    old_shape : tuple
        shape of original DataArray
    new_dims : list
        list of dimensions in new DataArray. These are the stacked dimensions and should
        be the same as old dims expcept with dim last
    new_shape : tuple
        shape of new DataArray
    '''

    if isinstance(da, xr.DataArray):
        # transpose data to put dim last
        old_dims = list(da.dims)
        new_dims = old_dims.copy()
        new_dims.remove(dim)
        new_dims = new_dims + [dim]
        old_shape = da.shape
        new_shape = (old_shape[:-1] + (int(old_shape[-1]/(W*Fs)), W*Fs))

        return old_dims, old_shape, new_dims, new_shape,
    else:
        raise TypeError('da must be an xarray DataArray')


def __preprocess_chunk(data, dim, b,a,W=30, Fs=200):
    '''
    __preprocess_chunk_da - compute basic pre-processing for NCCF for 
        a DataArray of arbitrary dimension

    sends result to __preprocess_chunk_da

    Pre-processing is fixed to be
    - divide into 30 s segments
    - filter between 1 and 90 Hz
    - clip to 3 times std for every chunk (which is also 1 hour)
    - frequency whiten

    Parameters
    ----------
    da : xr.DataArray
        DataArray of arbitrary dimensions
    dim : str
        dimension to apply pre-processing to (should be samples in time)
    b : np.array
        numerator coefficients for filter
    a : np.array
        denominator coefficients for filter
    W : float
        length of window in seconds
    Fs : float
        sampling rate in Hz
    '''
    
    # check type
    if(isinstance(data, xr.DataArray)):
        return __preprocess_chunk_da(data, dim, b,a,W=30, Fs=200)
    elif(isinstance(data, xr.Dataset)):
        return data.map(__preprocess_chunk_da, dim=dim, b=b, a=a, W=W, Fs=Fs)
    else:
        raise TypeError('data must be an xarray DataArray or Dataset')


def __preprocess_chunk_da(da, dim, b, a, W=30, Fs=200):
    '''
    __preprocess_chunk - compute basic pre-processing for NCCF for 
        a DataArray of arbitrary dimension
    
    Pre-processing is fixed to be
    - divide into 30 s segments
    - filter between 1 and 90 Hz
    - clip to 3 times std for every chunk (which is also 1 hour)
    - frequency whiten

    Parameters
    ----------
    da : xr.DataArray
        DataArray of arbitrary dimensions
    dim : str
        dimension to apply pre-processing to (should be samples in time)
    b : np.array
        numerator coefficients for filter
    a : np.array
        denominator coefficients for filter
    W : float
        length of window in seconds
    Fs : float
        sampling rate in Hz
    '''
    # get new shape of data array
    dims, shape, new_dims_order, new_shape = __preprocess_get_size(da, dim, W, Fs)

    # reorder dimensions to put dim last
    da_t = da.transpose(*new_dims_order)

    # load single chunk into numpy array and reshape
    da_np = da_t.values
    da_rs = np.reshape(da_np, new_shape)

    # remove mean
    da_nm = da_rs - np.nanmean(da_rs, axis=-1, keepdims=True)

    # filter data
    da_filt = signal.filtfilt(b, a, da_nm, axis=-1)

    # clip data
    std = np.nanstd(da_filt, axis=-1)
    da_clip = np.clip(da_filt, -3*std[..., None], 3*std[..., None])

    # frequency whiten data
    da_whiten = freq_whiten(da_clip, b, a)
    
    da_stack = np.reshape(da_whiten, shape)

    # create stacked data array
    da_preprocess = xr.DataArray(da_stack, dims=new_dims_order)
    
    return da_preprocess.transpose(*dims)


def freq_whiten(data, b, a):
    '''
    freq_whiten - force magnitude of fft to be filter response magnitude
        (retains phase information)

    Parameters
    ----------
    data : np.array
        array of shape [... , segment, time] containing segments of time
        series data to individually whiten. can contain other dimensions prior
        to segment and time, but segment and time must be last two dimensions
    b : np.array
        numerator coefficients for filter
    a : np.array
        denominator coefficients for filter 

    Returns
    -------
    data_whiten : np.array
        whitened data of same shape as data
    '''
    # window data and compute unit pulse
    win = signal.windows.hann(data.shape[-1])
    pulse = signal.unit_impulse(data.shape[-1], idx='mid')
    for k in range(len(data.shape)-1):
        win = np.expand_dims(win, 0)
        pulse = np.expand_dims(pulse, 0)

    data_win = data * win

    # take fft
    data_f = scipy.fft.fft(data_win, axis=-1)
    data_phase = np.angle(data_f)

    H = np.abs(scipy.fft.fft(signal.filtfilt(b, a, pulse, axis=-1)))

    # construct whitened signal
    data_whiten_f = (np.exp(data_phase * 1j) * np.abs(H) **
                     2)  # H is squared because of filtfilt

    data_whiten = np.real(scipy.fft.ifft(data_whiten_f, axis=-1))

    return data_whiten


## Functions to Calculate NCCFs
def compute_NCCF_stack(ds, dim='time', W=30, Fs=200, fcs=[1,90], compute=True, stack=True):
    '''
    compute_NCCF_stack - takes dataset containing timeseries from two locations
        and calculates an NCCF for every chunk in the time dimensions.

    Parameters
    ----------
    ds : xr.Dataset
        dataset must have dimension time and two data variables
    W : float
        passed to __NCCF_chunk
    Fs : float
        passed to __NCCF_chunk
    fcs : list
        corner frequencies (in Hz) for processing. passed to preprocess
    compute : bool
        whether to return dask task map or computed NCCF stack
    stack : bool
        if true, then NCCF is stacked across chunks.
        if false, full NCCF stack is return (no averaging is done)
    '''
    if len(ds) != 2:
        raise Exception('dataset must only have 2 data variables')
    node1, node2 = list(ds.keys())

    chunk_size = ds.chunks['time'][0]
    # chunk sizes have to be the same for both data variables (i think this is required in xarray too)

    # create template dataarray
    # if stack is true, then linear stacking is computing for chunksize of ds
    if stack == True:
        dask_temp = dask.array.random.random(
            (int(ds[node1].shape[0]/chunk_size), 2*W*Fs-1))
        da_temp = xr.DataArray(dask_temp, dims=['time', 'delay'])
        # single value chunks in long time
        da_temp = da_temp.chunk({'delay': int(2*W*Fs-1), 'time': 1})

    else:
        dask_temp = dask.array.random.random(
            (int(ds[node1].shape[0]/(W*Fs)), 2*W*Fs-1))
        da_temp = xr.DataArray(dask_temp, dims=['time', 'delay'])
        # 1 hour chunks in long time
        da_temp = da_temp.chunk(
            {'delay': int(2*W*Fs-1), 'time': int(chunk_size/Fs/W)})

    NCCF_stack = ds.map_blocks(
        __NCCF_chunk, template=da_temp, kwargs={'dim': dim, 'stack': stack, 'fcs':fcs})
    NCCF_stack = NCCF_stack.assign_coords(
        {'delay': np.arange(-W+1/Fs, W, 1/Fs)})
    if compute:
        return NCCF_stack.compute()
    else:
        return NCCF_stack


def compute_MultiElement_NCCF(da, time_dim='time', element_dim='distance', W=30, Fs=200, ref_idx=0):
    '''
    compute_MultiElement_NCCF - takes dataarray containing timeseries with multiple elements
        and calculates an NCCF between each element and first element

    can only handle a single chunk in the element / distance dimension

    Parameters
    ----------
    da : xr.DataArray
        dataarray with dimensions ['time', 'element']. second dimension does not have to be named element
    time_dim : str
        name of time dimension
    element_dim : str
        name of element dimension
    W : float
        size of window in seconds
    Fs : float
        sampling rate in Hz
    ref_idx : int
        integer index in 'element' dimension that is correlated with every other element.
        Defaults to 0, where first element is correlated with every other element
    '''

    time_idx = da.dims.index(time_dim)

    # move delay dimension to last dimension
    dims = list(da.dims)

    # get chunk sizes
    chunk_sizes_all = da.chunksizes
    chunk_sizes = {}
    for dim in dims:
        chunk_sizes[dim] = chunk_sizes_all[dim][0]

    new_dims_order = dims.copy()
    new_dims_order.remove(time_dim)
    new_dims_order = new_dims_order + [time_dim]
    da_t = da.transpose(*new_dims_order)

    n_chunks = len(da.chunks[time_idx])

    dask_temp = dask.array.random.random((da_t.shape[0], n_chunks, 2*W*Fs-1))
    da_temp = xr.DataArray(dask_temp, dims=[
                           new_dims_order[0], time_dim, 'delay'], name='multi-element NCCF')
    
    # single value chunks in long time
    new_chunks = {
        time_dim:1,
        'delay':int(2*W*Fs-1),
        element_dim:chunk_sizes[element_dim]
    }
    da_temp = da_temp.chunk(new_chunks)

    NCCF_me = da.map_blocks(__compute_MultiElement_NCCF_chunk, template=da_temp, kwargs={
                            'time_dim': time_dim, 'W': W, 'Fs': Fs, 'ref_idx': ref_idx})
    return NCCF_me


def compute_NCCF_stack_auto(ds, dim='time', W=30, Fs=200, compute=True, stack=True):
    '''
    compute_NCCF_stack_auto - takes dataset containing timeseries data and calculates the 
        auto NCCF for every chunk in the time dimensions.

    Parameters
    ----------
    ds : xr.Dataset
        dataset must have dimension time and two data variables
    W : float
        passed to __NCCF_chunk
    Fs : float
        passed to __NCCF_chunk
    compute : bool
        whether to return dask task map or computed NCCF stack
    stack : bool
        if true, then NCCF is stacked across chunks.
        if false, full NCCF stack is return (no averaging is done)
    '''

    nodes = list(ds.keys())
    chunk_size = ds.chunks[dim][0]

    # chunk sizes have to be the same for both data variables (i think this is required in xarray too)

    # create template dataset
    # if stack is true, then linear stacking is computing for chunksize of ds
    if stack == True:
        dask_temp = dask.array.random.random(
            (int(ds[nodes[0]].shape[0]/chunk_size), 2*W*Fs-1))
        da_temp = xr.DataArray(dask_temp, dims=[dim, 'delay'])
        # single value chunks in long time
        da_temp = da_temp.chunk({'delay': int(2*W*Fs-1), dim: 1})
        #da_temp = xr.DataArray(np.ones((int(ds[node1].shape[0]/chunk_size), 2*W*Fs-1)), dims=['time','delay'])

    else:
        dask_temp = dask.array.random.random(
            (int(ds[nodes[0]].shape[0]/(W*Fs)), 2*W*Fs-1))
        da_temp = xr.DataArray(dask_temp, dims=[dim, 'delay'])
        # 1 hour chunks in long time
        da_temp = da_temp.chunk(
            {'delay': int(2*W*Fs-1), dim: int(chunk_size/Fs/W)})
        #da_temp = xr.DataArray(np.ones((int(ds[node1].shape[0]/(W*Fs)), 2*W*Fs-1)), dims=['time','delay'])

    # create template dataset from template data array
    ds_temp = xr.Dataset(dict(zip(nodes, [da_temp]*len(nodes))))

    NCCF_stack = ds.map_blocks(
        __NCCF_chunk_auto, template=ds_temp, kwargs={'W':W, 'Fs':Fs, 'stack_dim':'long_time', 'stack': stack, 'dim': dim}
    )

    NCCF_stack = NCCF_stack.assign_coords(
        {'delay': np.arange(-W+1/Fs, W, 1/Fs)})
    if compute:
        return NCCF_stack.compute()
    else:
        return NCCF_stack


def __NCCF_chunk(ds, dim, stack=True, fcs=[1, 90]):
    '''
    calculate NCCF for given dataset of time-series

    this function is defined to be used with .map_blocks() to calculate the
        NCCF stack, where each NCCF is calcualted from a single chunk

    Parameters
    ----------
    ds : xr.Dataset
        dataset with two data variables each consisting of a timeseries that
        NCCF is calculated from. each DA should have dimension time
        ds must only have two data dimensions
    stack : bool
        whether or not to return all small time correlations or stack then (default True)

    Returns
    -------
    NCCF : xr.DataArray
        data array with dimensions ['delay']
        - if stack is False, data will have dimensions ['delay', 'time']
    '''
    if len(ds) != 2:
        raise Exception('dataset must only have 2 data variabless')
    node1, node2 = list(ds.keys())

    ds_pp = preprocess(ds, dim=dim, fcs=fcs, include_coords=True).unstack()

    node1_pp = ds_pp[node1].values
    node2_pp = ds_pp[node2].values

    R_all = signal.fftconvolve(node1_pp, np.flip(
        node2_pp, axis=1), axes=1, mode='full')
   
    if stack:
        R = np.mean(R_all, axis=0)
        #tau = np.arange(-W+(1/Fs), W, 1/Fs)

        Rx = xr.DataArray(np.expand_dims(R, 0), dims=['time', 'delay'])
        return Rx

    else:
        Rallx = xr.DataArray(R_all, dims=['time', 'delay'])
        return Rallx


def __NCCF_chunk_auto(ds, dim, W, Fs, stack=True, stack_dim='long_time'):
    '''
    calculate NCCF autocorrelation for given dataset of time-series

    this function is defined to be used with .map_blocks() to calculate the
        NCCF stack, where each NCCF is calcualted from a single chunk

    Parameters
    ----------
    ds : xr.Dataset
        dataset of a timeseries.
    dim : str
        dimension to calculate cross-correlation in
    stack : bool
        whether or not to return all small time correlations or stack then (default True)

    Returns
    -------
    NCCF : xr.DataArray
        data array with dimensions ['delay']
        - if stack is False, data will have dimensions ['delay', 'time']
    '''

    nodes = list(ds.keys())

    ds_pp = preprocess(ds, dim=dim).unstack()

    R_all = ds_pp.map(__autocorrelate, dim='short_time')

    if stack:
        R = R_all.mean(stack_dim, keepdims=True)
        #R = R.assign_coords({'delay': np.arange(-W+(1/Fs), W, 1/Fs)})
        R = R.rename_dims({stack_dim:'time'})
        return R
    else:
        #R_all = R_all.assign_coords({'delay': np.arange(-W+(1/Fs), W, 1/Fs)})
        R_all = R_all.rename_dims({stack_dim:'time'})
        return R_all
    

def __autocorrelate(da, dim):
    '''
    autocorrelate - autocorrelate da along dimension dim
    
    Parameters
    ----------
    da : xr.DataArray
        data array to autocorrelate.
    dim : str
        dimension to autocorrelate along
        
    Returns
    -------
    R : xr.DataArray
        auto-correlation of da
    '''
    
    dim_idx = da.dims.index(dim)
    new_dims = list(da.dims)
    new_dims[dim_idx]='delay'
    
    
    R = signal.fftconvolve(da.values, np.flip(
        da.values, axis=dim_idx), axes=dim_idx, mode='full')
    
    Rx = xr.DataArray(R, dims=new_dims)
    return Rx


def __compute_MultiElement_NCCF_chunk(da, time_dim='time', W=30, Fs=200, ref_idx=0):
    '''
    compute_MultiElement_NCCF - takes dataset containing timeseries from two locations
        and calculates an NCCF for each element with the first element
    
    Not completely sure if this will cause downstream problems, but this function is seperated 
        from preprocessing
    
    Parameters
    ----------
    da : xr.DataArray
        dataarray with dimensions ['time', 'element']. second dimension does not have to be named element
    time_dim : str
        name of delay dimension
    W : float
        size of window in seconds
    Fs : float
        sampling rate in Hz
    ref_idx : int
        integer index in 'element' dimension that is correlated with every other element.
        Defaults to 0, where first element is correlated with every other element
    '''

    # move delay dimension to last dimension
    dims = list(da.dims)
    new_dims_order = dims.copy()
    new_dims_order.remove(time_dim)
    new_dims_order = new_dims_order + [time_dim]
    da_t = da.transpose(*new_dims_order)

    # load single chunk into numpy array
    da_np = da_t.values
    shape = da_np.shape

    # reshape into seperate segments of length W
    da_rs = np.reshape(da_np, (shape[:-1] + (int(shape[-1]/(W*Fs)), W*Fs)))

    # loop through all elements and compute NCCF
    for k in range(da_rs.shape[0]):
        R_all_single = np.expand_dims(signal.fftconvolve(da_rs[ref_idx, :, :], np.flip(
            da_rs[k, :, :], axis=-1), axes=-1, mode='full'), axis=0)
        if k == 0:
            R_all = R_all_single

        else:
            R_all = np.concatenate((R_all, R_all_single), axis=0)

    NCCF_chunk_unstacked = xr.DataArray(
        R_all, dims=[new_dims_order[0], 'samples', 'delay'])
    
    NCCF_chunk = NCCF_chunk_unstacked.mean(dim='samples')
    NCCF_chunk = NCCF_chunk.expand_dims(time_dim).transpose(
        new_dims_order[0], time_dim, 'delay'
    )


    return NCCF_chunk


## Selective Stacking Methods
def linear_stack(R, dim='time'):
    '''
    linear_stack - takes R and linearly stacks (average across time dimension)

    Parameters
    ----------
    R : xr.DataArrray
        un-averaged NCCF stack
    dim : str
        dimension to stack across
    '''
    return R.mean(dim)

# Could be broken
def selective_stack(R, time_select, delay_select):
    '''
    selective_stack - uses two seperable functions to select which parts of short
        and long time to stack
    
    Parameters
    ----------
    R : xr.DataArray
        un-averaged NCCF stack. should have dimensions ['time', 'delay']
    time_select : xr.DataArray
        vector that is multiplied by R to select in time (long time). elements should be [0,1]
        should have dimension time and have same shape and coordinates of R.time
    delay_select : xr.DataArray
        vector that is multiplied by R to select in delay (short time). elements should be [0,1]
        should have dimension delay and have same shape and coordinates of R.delay

    Returns
    -------
    R_stack : xr.DataArray
        stacked NCCF with dimensions ['delay']
    '''

    R_select = R*time_select*delay_select
    R_stack = linear_stack(R_select)

    return R_stack


def psd_time_sel(R, wband=(0.01, 0.05), pband=(50, 100)):
    '''
    PSD_percentile_stack - selectively stacks the NCCFs based on a percentile range for
        energy in a specific frequency band.
    
    The percentiles are calculated using all values of R dimension 'time'.

    Parameters
    ----------
    R : xr.DataArray
        un-averaged NCCF stack. should have dimensions ['time', 'delay']
    wband : tuple
        frequency band used for selective stacking in Hz. In radians/pi (i.e. 1 is nyquist rate)
    pband : tuple
        percentile band used for sective band (should be two numbers between 0 and 100)
    '''

    # calculate PSD of R
    nperseg = 1024
    _, R_psd = signal.welch(R.values, axis=R.dims.index(
        'delay'), nperseg=nperseg, noverlap=512)

    fidx1 = int((nperseg/2 + 1)*wband[0])
    fidx2 = int((nperseg/2 + 1)*wband[1])

    fband_energy = np.mean(R_psd[:, fidx1:fidx2], axis=1)
    percentiles = np.percentile(fband_energy, pband)

    percentile_mask = ((fband_energy > percentiles[0]) & (
        fband_energy < percentiles[1])).astype(int)
    time_select = xr.ones_like(
        R.isel({'delay': 0}).drop('delay'))*percentile_mask

    return time_select


def phase_weighted_stack(R):
    '''
    phase_weighted_stack - takes R and stacks using phase weighted stacking

    Parameters
    ----------
    R : xr.DataArrray
        un-averaged NCCF stack should have dimensions ['time', 'delay']
    '''
    Rf = scipy.fft.fft(R, axis=1)


## Archive 
def preprocess_archive(da, W=30, Fs=200, tide=False, tide_interp=None):
    '''
    preprocess - takes time series and performs pre-processing steps for estimating cross-correlation

    Currently pre-processing is fixed to be
    - divide into 30 s segments
    - filter between 1 and 90 Hz
    - clip to 3 times std for every chunk (which is also 1 hour)
    - frequency whiten

    Parameters
    ----------
    da : xr.DataArray
        should be 1D dataarray with dimension 'time'.
    W : float
        length of window in seconds
    Fs : float
        sampling rate in Hz 
    tide : bool
        if true, linear/geometric time warping is applied for s1b0 peak
    tide_interp : scipy.interpolate._interpolate.interp1d
        interpolates ns timestamp to change of tide in meters
        Currently there are not catches for the fact that default is None

    Return
    ------
    data_whiten : np.array
        pre-procesesd data
    '''
    # load single chunk into numpy array
    data = da.values

    # remove mean
    data_nm = data - np.nanmean(data)

    # reshape data to be segments of length W
    data_rs = np.reshape(data_nm, (int(len(data_nm)/(W*Fs)), int(W*Fs)))

    # set_nan = 0
    #nan_mask = np.isnan(data_rs)
    #data_rs[nan_mask] = 0

    if tide:
        D = 1523
        L = 3186
        c = 1481

        sample_time_coords = da.time[::W*Fs].values.astype(int)
        tidal_shift = tide_interp(sample_time_coords)
        time_shift = 2*np.sqrt(D**2 + (L/2)**2)/c - 2 * \
            np.sqrt((D-tidal_shift)**2 + (L/2)**2)/c
        timebin_shift = np.expand_dims(time_shift/0.005, 1)
        k = np.expand_dims(np.arange(0, W*Fs + 1), 0)
        phase_shift = np.exp(-1j*2*np.pi/(W*Fs)*k*timebin_shift)
        data_shift_f = scipy.fft.fft(
            np.hstack((data_rs, np.zeros((data_rs.shape[0], 1)))), axis=1) * phase_shift
        # force shifted signal to be real
        data_shift_f[:, int(data_shift_f.shape[1]/2+1):] = np.flip(
            np.conjugate(data_shift_f[:, 1:int(data_shift_f.shape[1]/2 + 1)]), axis=1)

        data_shift = np.real(scipy.fft.ifft(data_shift_f))[:, :-1]

    else:
        data_shift = data_rs

    # filter data
        # filter is 4th order butterwork [0.01, 0.9]
    b = [0.63496904, 0, - 2.53987615,  0,
         3.80981423, 0, - 2.53987615, 0, 0.63496904]
    a = [1, -0.73835614, -2.84105805, 1.53624064, 3.3497155, -
         1.14722815, -1.86018017, 0.29769033, 0.40318603]

    data_filt = signal.filtfilt(b, a, data_shift, axis=1)

    # clip data
    std = np.nanstd(data_filt)
    data_clip = np.clip(data_filt, -3*std, 3*std)

    # frequency whiten data
    data_whiten = freq_whiten(data_clip, b, a)

    return data_whiten

def compute_MultiElement_NCCF_PW_chunk(da, time_dim='time', W=30, Fs=200, eta=1, ref_idx=0):
    '''
    compute_MultiElement_PW_NCCF_chunk - takes dataset containing timeseries from two locations
        and calculates an NCCF phase weight for each element with the first element
       
    Parameters
    ----------
    da : xr.DataArray
        dataarray with dimensions ['time', 'element']. second dimension does not have to be named element
    time_dim : str
        name of delay dimension
    W : float
        size of window in seconds
    Fs : float
        sampling rate in Hz
    eta : float
        phase weight parameter
    ref_idx : int
        integer index in 'element' dimension that is correlated with every other element.
        Defaults to 0, where first element is correlated with every other element
    '''

    # move delay dimension to last dimension
    dims = list(da.dims)
    new_dims_order = dims.copy()
    new_dims_order.remove(time_dim)
    new_dims_order = new_dims_order + [time_dim]
    da_t = da.transpose(*new_dims_order)

    # load single chunk into numpy array
    da_np = da_t.values
    shape = da_np.shape

    # reshape into seperate segments of length W
    da_rs = np.reshape(da_np, (shape[:-1] + (int(shape[-1]/(W*Fs)), W*Fs)))

    # loop through all elements and compute NCCF
    for k in range(da_rs.shape[0]):
        R_all_single = np.expand_dims(signal.fftconvolve(da_rs[ref_idx, :, :], np.flip(
            da_rs[k, :, :], axis=-1), axes=-1, mode='full'), axis=0)
        if k == 0:
            R_all = R_all_single

        else:
            R_all = np.concatenate((R_all, R_all_single), axis=0)

    R_all_pw = np.exp(1j*np.angle(signal.hilbert(R_all, axis=-1)))

    NCCF_chunk_unstacked = xr.DataArray(
        R_all_pw, dims=[new_dims_order[0], 'samples', 'delay'])

    NCCF_chunk = NCCF_chunk_unstacked.mean(dim='samples')

    NCCF_chunk = NCCF_chunk.expand_dims(time_dim).transpose(
        new_dims_order[0], time_dim, 'delay'
    )

    return NCCF_chunk

def compute_MultiElement_NCCF_PhaseWeight(da, time_dim='time', element_dim='distance', W=30, Fs=200, ref_idx=0):
    '''
    compute_MultiElement_NCCF_PhaseWeight - takes dataarray containing timeseries with multiple elements
        and calculates an NCCF phase weight between each element and first element

    can only handle a single chunk in the element / distance dimension

    Parameters
    ----------
    da : xr.DataArray
        dataarray with dimensions ['time', 'element']. second dimension does not have to be named element
    time_dim : str
        name of time dimension
    element_dim : str
        name of element dimension
    W : float
        size of window in seconds
    Fs : float
        sampling rate in Hz
    '''

    time_idx = da.dims.index(time_dim)

    # move delay dimension to last dimension
    dims = list(da.dims)

    # get chunk sizes
    chunk_sizes_all = da.chunksizes
    chunk_sizes = {}
    for dim in dims:
        chunk_sizes[dim] = chunk_sizes_all[dim][0]

    new_dims_order = dims.copy()
    new_dims_order.remove(time_dim)
    new_dims_order = new_dims_order + [time_dim]
    da_t = da.transpose(*new_dims_order)

    n_chunks = len(da.chunks[time_idx])

    dask_temp = dask.array.random.random((da_t.shape[0], n_chunks, 2*W*Fs-1))
    da_temp = xr.DataArray(dask_temp, dims=[
                           new_dims_order[0], time_dim, 'delay'], name='multi-element NCCF')

    # single value chunks in long time
    new_chunks = {
        time_dim: 1,
        'delay': int(2*W*Fs-1),
        element_dim: chunk_sizes[element_dim]
    }
    da_temp = da_temp.chunk(new_chunks)

    NCCF_me = da.map_blocks(compute_MultiElement_NCCF_PW_chunk, template=da_temp, kwargs={
                            'time_dim': time_dim, 'W': W, 'Fs': Fs, 'ref_idx': ref_idx})
    return NCCF_me

