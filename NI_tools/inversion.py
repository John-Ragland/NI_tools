'''
Toolset for exploring measuring the propagation time of prop paths in TDGF
This code was copied from Noise_Interferometry. Not all aspects of code are tested to work
'''
import xarray as xr
import numpy as np
from tqdm import tqdm
import pickle
import os
from scipy import signal
import time as tm
import pandas as pd

#this is only used for frequency whitening code in calc_prop_times_whiten
#TODO get rid of this dependancy
from Noise_Interferometry.Modules import calculate

from matplotlib import pyplot as plt

# Do not print warnings to terminal
    # this could hurt debugging
import warnings
warnings.filterwarnings('ignore')

'''
Define Global Variables
setup slices and indices for different peaks, two slices are set up:
    - peak_slices : should contain the full peak (peak width is determine visually)
    - peak_slices_tight : only a couple of time bins wide. This is used for argmax stuff
'''
peak_names = ['dA', 's1b0A', 's2b1A', 's3b2A','s1b0B', 's2b1B', 's3b2B']
peak_names_short = ['dA', 's1b0A', 's1b0B', 's2b1A', 's2b1B']

# Peak windows manually defined
peak_locations = np.array([
    5575,
    5402,
    5068,
    4704,
    6597,
    6932,
    7294
])

window_widths_tight = np.array([10, 10, 10, 10, 10, 10, 10])
# I CHANGED DA WIDTH FROM 50 TO 100 AND I DONT KNOW WHAT THIS WILL CHANGE
window_widths = np.array([100, 100, 100, 100, 100, 100, 100])

peak_windows_tight = np.array([
    peak_locations - window_widths_tight,
    peak_locations + window_widths_tight
]).T
peak_windows = np.array([
    peak_locations - window_widths,
    peak_locations + window_widths
]).T

slices_tight = []
slices = []
for k in range(len(peak_windows)):
    slices.append(np.s_[peak_windows[k][0]:peak_windows[k][1]])
    slices_tight.append(np.s_[peak_windows_tight[k][0]:peak_windows_tight[k][1]])

peak_slices = dict(zip(peak_names,slices))
peak_slices_tight = dict(zip(peak_names, slices_tight))

def __hilbert_chunk(da, **kwargs):

    xc = np.abs(signal.hilbert(da.values, **kwargs))
    xcx = xr.DataArray(xc, dims=da.dims, coords=da.coords)

    return xcx

def hilbert_mag(da, dim, **kwargs):
    '''
    calculate the hilbert magnitude of da

    Parameters
    ----------
    da : xr.DataArray
    dim : str
        dimension over which to calculate hilbert transform
    '''

    dim_idx = list(da.dims).index(dim)
    kwargs['axis'] = dim_idx
    dac = da.map_blocks(__hilbert_chunk, kwargs=kwargs, template=da)
    
    return dac

def calc_prop_times(NCCFs, peaks=peak_names_short):
    '''
    calculate propagation times for all specified peaks.
    Calculations are distributed and require chunked data.
    Uses 3 point interpolation and argmax of specified window to get sub time bin arrival time.

    Parameters
    ----------
    NCCFs : xr.Dataset
        Dataset of multiple NCCFs stacks. Should have dimensions ['date','delay']
    peaks : list
        list of strings specifying peak names. should be contained in peak_names defined above

    Returns
    -------
    arrival_times : xr.Dataset
        estimated arrival times
    '''

    template = xr.ones_like(NCCFs).isel({'delay':0}).expand_dims({'peak': peaks}, axis=0).drop_vars('delay')
    prop_times = NCCFs.map_blocks(_calc_prop_times_chunk, args=[peaks], template=template)
    return prop_times

def _calc_prop_times_chunk(NCCFs, peak_names):
    '''
    takes xr.Dataset of multiple NCCF stacks and sends each to calc_prop_times which
        requires a single DataArray
    
    Parameters
    ----------
    NCCFs : xr.Dataset
        dataset of NCCF stacks. should have dimensions ['dates', 'delay']
    peak_name : string
        name of peak

    Returns
    -------
    prop_times : xr.Dataset
        xr.Dataset with dimensions ['dates']
    '''

    prop_times_d = {}
    for item in list(NCCFs.keys()):
        prop_time_peak_name = {}

        for peak_name in peak_names:
            prop_time_peak_name[peak_name] = calc_prop_time(NCCFs[item], peak_name)

        prop_times_d[item] = xr.concat(list(prop_time_peak_name.values()), dim='peak').assign_coords(
            {'peak': list(prop_time_peak_name.keys())})
          
    prop_times = xr.Dataset(prop_times_d)

    return prop_times

def calc_prop_time(NCCFs, peak_name, peak_slice=None, verbose=False, tight=True):
    '''
    calc_prop_time - calculates the propagation time for a given peak and NCCF stack.
    peak_slices_tight is used for the windowing of the peaks
    
    Parameters
    ----------
    NCCFs : xr.DataArray
        NCCF stack, stored as xr.DataArray. Dimension names should be
        delay and dates
    peak_name : str
        peak name that propagation time is to be calculated.
        if peak_name is custom, then peak_slice is used
    peak_slice : tuple
        start and end times in seconds of custom peak windows.
        only used if peak_name == 'custom'
        
        
    Returns
    -------
    prop_time : xarray.DataArray
        datarray containing propagation times that are calculated. X dimension is dates
        and corresponds to dates dimension of NCCFs
        
    '''
    if peak_name == 'custom':
        peak = NCCFs.loc[:,peak_slice[0]:peak_slice[1]].values
        # get index of max
        idx = NCCFs[:,peak_slice[0]:peak_slice[1]].argmax(dim='delay',skipna=False) + peak_slice[0]
    elif tight:
        peak = NCCFs[:,peak_slices_tight[peak_name]].values
        # get index of max
        idx = NCCFs[:,peak_slices_tight[peak_name]].argmax(dim='delay',skipna=False) + peak_slices_tight[peak_name].start
    else:
        peak = NCCFs[:,peak_slices[peak_name]].values
        # get index of max
        idx = NCCFs[:,peak_slices[peak_name]].argmax(dim='delay',skipna=False) + peak_slices[peak_name].start

    # create (52407, 3) array. --> [n-1, n, n+1]
    amplitudes = np.ones((NCCFs.shape[0],3))*np.nan
    indexes = np.ones((NCCFs.shape[0],3))*np.nan
    for k in tqdm(range(NCCFs.shape[0]), disable=(not verbose)):
        amplitudes[k,:] = NCCFs[k,(idx[k].values-1):(idx[k].values+2)].values
        indexes[k,:] = NCCFs.delay[(idx[k].values-1):(idx[k].values+2)].values

    #QURADRATIC PEAK INTERPOLATION
    p = 1/2*(amplitudes[:,0] - amplitudes[:,2])/(amplitudes[:,0] - 2*amplitudes[:,1] + amplitudes[:,2])
    # Throw out values where |p| > 0.5
    p[np.abs(p) > 0.5] = np.nan
    # Calculate interpolated index
    prop_time = indexes[:,1] + p*0.005 # in seconds
    
    prop_times_x = xr.DataArray(prop_time, dims=['dates'], coords={'dates':NCCFs.dates},  name = f'propagation times for {peak_name}')
    
    return prop_times_x

def argmax_interp(data, peak_slice, verbose=True):
    '''
    argmax_interp - calculates the peak location using argmax and quadratic peak interpolation.
    basically a simpler version of calc_prop_time, where it is more general and doesn't
    rely on xr.DataArray NCCF structure
    
    It would be nice if calc_prop_times uses this function. it doesn't
    
    Parameters
    ----------
    data : numpy array
        array of shape [M,N] where M is number of samples and N is number of delay points
    peak_slice : list
        list containing start in end bounds for window of interest
        
    Returns
    -------
    peak_locations : numpy array
        array of shape [M,] containing estimate peak locations (in time bins NOT seconds)
    '''
    M,N = data.shape
    peak = data[:, peak_slice[0]:peak_slice[1]]
    
    idx = np.argmax(peak, axis=1)
    amplitudes = np.ones((M,3))*np.nan
    
    for k in tqdm(range(M), disable=(not verbose)):
        if np.isnan(np.sum(peak[k,:])) | (idx[0]==0) | (idx[0]==19):
            amplitudes[k,:] = np.nan
        else:
            amplitudes[k,:] = peak[k, (idx[k]-1):(idx[k]+2)]
        
    # QUADRATIC PEAK INTERPOLATION
    p = 1/2*(amplitudes[:,0] - amplitudes[:,2])/(amplitudes[:,0] - 2*amplitudes[:,1] + amplitudes[:,2])
    # Throw out values where |p| > 0.5
    p[np.abs(p) > 0.5] = np.nan
    
    peak_locations = idx + p + peak_slice[0]
    
    return peak_locations
    
def match_filter_single_peak(NCCFs, peak, match_windows, verbose=True):
    '''
    match_filter - calculate match_filter for a given peak
    peak windows can be calculated with calculate_match_windows() and are
    saved as a dictionary in Noise_Interferometry.Modules.matched_peaks.pkl
    
    Parameters
    ----------
    NCCFs : xr.DataArray
        NCCF with dimensions ['dates','delay']
    peak : str
        peak name. valid options include any peak name in peak_names defined above
        exceptions to this are not programmed for
    match_windows : dict
        dictionary containing match-filter kernels for each peak. keys are peak names
        
    Returns
    -------
    NCCF_match : xr.DataArray
        NCCF * match_window where * is convolution. (match filtered NCCF)
    '''        
    kern = match_windows[peak]
    
    match_filtered_values = np.ones(NCCFs.shape)*np.nan
    for k in tqdm(range(NCCFs.shape[0]), disable=(not verbose)):
        match_filtered_values[k,:] = signal.correlate(NCCFs[k,:].values, kern, mode='same')
    NCCF_matched = xr.DataArray(
        match_filtered_values,
        dims=['dates','delay'],
        coords={'dates':NCCFs.dates, 'delay':NCCFs.delay},
        name=f'match filterred NCCF for peak {peak}')
    return NCCF_matched
    
def match_filter_yearly(NCCFs1, NCCFs201, fn_base):
    '''
    match_filter_year - calculates match-filtered NCCF for all 5 peaks over all 6 years.
    match-filter kernels change each calander year to be the normalized average peak
    window for that given calendar year.
    
    Parameters
    ----------
    NCCFs1 : xr.DataArray
        NCCF stack represented as xr.DataArray. 1 hour average. used for
        creating the yearly kernels
    NCCFs201 : xr.DataArray
        201 hour average NCCF. used for matched filtering
    fn_base : str
        file directory that the propagation times will be stored
    '''
    years = [2015, 2016, 2017, 2018, 2019, 2020]
    
    prop_times_all_years = []
    for count, year in enumerate(years):
        # get match-filter kernel for specific year
        print(f'Calculating kernels for {year}...')
        NCCF_avg = NCCFs1.loc[pd.Timestamp(f'{year}-01-01'):pd.Timestamp(f'{year+1}-01-01')].mean('dates')
        kerns = calculate_match_windows(NCCF_avg)
        
        print(f'Calculating {year} match-filerred NCCF')
        NCCF_slice = NCCFs201.loc[pd.Timestamp(f'{year}-01-01'):pd.Timestamp(f'{year+1}-01-01')]
        
        NCCF_match = {}
        for peak in peak_names:
            NCCF_match[peak] = match_filter_single_peak(NCCF_slice, peak, kerns)
            
        print(f'Calculating {year} propagation times...')
        prop_times = {}
        for peak in peak_names:
            prop_times[peak] = calc_prop_time(NCCF_match[peak], peak)
        prop_times_all_years.append(prop_times)
        
        
    prop_times_dA = []
    prop_times_s1b0A = []
    prop_times_s1b0B = []
    prop_times_s2b1A = []
    prop_times_s2b1B = []

    for k in range(6):
        prop_times_dA.append(prop_times_all_years[k]['dA'])
        prop_times_s1b0A.append(prop_times_all_years[k]['s1b0A'])
        prop_times_s1b0B.append(prop_times_all_years[k]['s1b0B'])
        prop_times_s2b1A.append(prop_times_all_years[k]['s2b1A'])
        prop_times_s2b1B.append(prop_times_all_years[k]['s2b1B'])

    dA_proptime = xr.concat(prop_times_dA, dim='dates')
    s1b0A_proptime = xr.concat(prop_times_s1b0A, dim='dates')
    s1b0B_proptime = xr.concat(prop_times_s1b0B, dim='dates')
    s2b1A_proptime = xr.concat(prop_times_s2b1A, dim='dates')
    s2b1B_proptime = xr.concat(prop_times_s2b1B, dim='dates')
    
    dA_proptime.to_netcdf(fn_base+'dA.nc')
    s1b0A_proptime.to_netcdf(fn_base+'s1b0A.nc')
    s1b0B_proptime.to_netcdf(fn_base+'s1b0B.nc')
    s2b1A_proptime.to_netcdf(fn_base+'s2b1A.nc')
    s2b1B_proptime.to_netcdf(fn_base+'s2b1B.nc')
    return prop_times_all_years

def match_filter_kernel_blocks(NCCFs1, NCCFs201, fn_base, block_length):
    '''
    match_filter_kernel_blocks - calculates the propagation time for all five peaks
        using the method of kernel blocks (Need to come up with a better name)... 
        Basically, the match filter kernel comes from the average NCCF over a given
        time block. For instance a block could be a calendar year, calendar month, or
        number of hours
        
        for blocks that are not evenly divisible time of NCCFs201, the excess is discarded.
    
    Parameters
    ----------
    NCCFs1 : xr.DataArray
        NCCF stack represented as xr.DataArray. 1 hour average. used for
        creating the yearly kernels
    NCCFs201 : xr.DataArray
        201 hour average NCCF. used for matched filtering
    fn_base : str
        file directory that the propagation times will be stored
    block_length : pd.timedelta
        block length in time. example: 730 hours
        
        block length in time. Can be pd.Timedelta or dateutil.relativedelta.relativedelta
        the latter is used for calandar months (not supported in pd.Timedelta)
    '''
    
    date_start = NCCFs1.dates[0].values
    date_end = NCCFs1.dates[-1].values

    prop_times_dA = []
    prop_times_s1b0A = []
    prop_times_s1b0B = []
    prop_times_s2b1A = []
    prop_times_s2b1B = []
    
    for k in tqdm(range(int((date_end - date_start)/block_length))):
        slice_start = date_start + k*block_length
        slice_end = date_start + (k+1)*block_length

        # calculate kernels for block slice
        kerns = calculate_match_windows(NCCFs1.loc[slice_start:slice_end].mean('dates'))
        
        # Calculate full match-filtered nccf
        dA_match = match_filter_single_peak(NCCFs201.loc[slice_start:slice_end], 'dA', kerns, verbose=False)
        s1b0A_match = match_filter_single_peak(NCCFs201.loc[slice_start:slice_end], 's1b0A', kerns, verbose=False)
        s1b0B_match = match_filter_single_peak(NCCFs201.loc[slice_start:slice_end], 's1b0B', kerns, verbose=False)
        s2b1A_match = match_filter_single_peak(NCCFs201.loc[slice_start:slice_end], 's2b1A', kerns, verbose=False)
        s2b1B_match = match_filter_single_peak(NCCFs201.loc[slice_start:slice_end], 's2b1B', kerns, verbose=False)
        
        # calculate prop_times
        prop_times_dA.append(calc_prop_time(dA_match, 'dA'))
        prop_times_s1b0A.append(calc_prop_time(s1b0A_match, 's1b0A'))
        prop_times_s1b0B.append(calc_prop_time(s1b0B_match, 's1b0B'))
        prop_times_s2b1A.append(calc_prop_time(s2b1A_match, 's2b1A'))
        prop_times_s2b1B.append(calc_prop_time(s2b1B_match, 's2b1B'))
        
    dA_proptime = xr.concat(prop_times_dA, dim='dates')
    s1b0A_proptime = xr.concat(prop_times_s1b0A, dim='dates')
    s1b0B_proptime = xr.concat(prop_times_s1b0B, dim='dates')
    s2b1A_proptime = xr.concat(prop_times_s2b1A, dim='dates')
    s2b1B_proptime = xr.concat(prop_times_s2b1B, dim='dates')
    
    #dA_proptime.to_netcdf(fn_base+'dA.nc')
    #s1b0A_proptime.to_netcdf(fn_base+'s1b0A.nc')
    #s1b0B_proptime.to_netcdf(fn_base+'s1b0B.nc')
    #s2b1A_proptime.to_netcdf(fn_base+'s2b1A.nc')
    #s2b1B_proptime.to_netcdf(fn_base+'s2b1B.nc')
    
    return dA_proptime, s1b0A_proptime, s1b0B_proptime, s2b1A_proptime, s2b1B_proptime
      
def calculate_match_windows(NCCF):
    '''
    calculate_match_windows - calculates the match-filter kernels for all peaks
    and windows specified at the beginning of this file
    
    Parameters
    ----------
    NCCF : xr.DataArray
        single NCCF with dimension ['delay']. This should be a sufficiently long
        NCCF (originally using 6-year)
    fn : str
        file name of saved pi
        
    Returns
    -------
    match_windows : dictionary
        dictionary keyed by peak names of each match filter kernel.
        This dictionary is also saved in ../Modules/matched_peaks.pkl
    
    '''
    
    match_windows = {}
    for peak in peak_names:
        match_windows[peak] = NCCF[peak_slices[peak]].values
        # normalize match windows
        match_windows[peak] = match_windows[peak] / np.max(np.abs(match_windows[peak]))
    fdir='../Modules/matched_peaks.pkl'
    
    with open(fdir, 'wb') as f:
        pickle.dump(match_windows, f)
    
    return match_windows
            
def get_slices(tight=False):
    if tight:
        return peak_slices_tight
    else:
        return peak_slices
      
def calculate_MatchFiltered_emergence(peaks=None):
    if peaks == None:
        peaks = peak_names
    
    for peak in peaks:
        print(f'Calculating Emergence for {peak} match filter:')
        print('    Loading File...')
        fn = f'/Volumes/Ocean_Acoustics/NCCFs/MJ03F-MJ03E/Inversion/1_hour_avg/{peak}_match_filter.nc'
        NCCFs = xr.open_dataarray(fn)
        
        # calculate noise bounds
        noise_bounds_s = [-16,-13]
        noise_bounds = []
        noise_bounds.append(np.argmin(np.abs(NCCFs.delay.values-noise_bounds_s[0])))
        noise_bounds.append(np.argmin(np.abs(NCCFs.delay.values-noise_bounds_s[1])))
        avg_time = 730
        
        SNR_emergences = []
        for temp, n in enumerate(range(0, 52560, avg_time)):
            start = n
            end = n + avg_time - 1

            print(f'\nMerging year hours {start} to {end}...\n')
            tm.sleep(0.3)
            single_peak_emergence = []
            single_peak_noise = []
            
            for count, k in enumerate(tqdm(range(start, end))):
                single_peak_emergence.append(
                    NCCFs[start:k+1,peak_slices[peak]].mean('dates').values
                )
                single_peak_noise.append(
                    NCCFs[start:k+1,noise_bounds[0]:noise_bounds[1]].mean('dates').values
                )
            emerge = np.array(single_peak_emergence)
            noise = np.array(single_peak_noise)
            SNR_emergence = np.max(emerge, axis=1)/(3*np.std(noise, axis=1))
            SNR_emergences.append(SNR_emergence)

        SNR_emergences = np.array(SNR_emergences)
        SNR_emergences_x = xr.DataArray(
            SNR_emergences, dims=['emerge', 'hour'], name=f'SNR emergences {peak}')
        
        fn = f'/Volumes/Ocean_Acoustics/NCCFs/MJ03F-MJ03E/emergence/match_filter/{peak}_emergence.nc'
        SNR_emergences_x.to_netcdf(fn)
        
def calc_prop_times_whiten(NCCFs, peak_name, verbose=False):
    '''
    calc_prop_times_whiten - calculate the propagation times using whitening method.
    - whiten 200 point peak window for given peak using butterworth magnitude
    - take magnitude of hilbert transform for peak window
    - find interpolated argmax (using calc_prop_time)
    
    This function is SUPER non-optimal and does ALOT of for-looping.
    
    Parameters
    ----------
    NCCFs : xr.DataArray
        NCCF stack of arbitrary size (delay dimension needs to be 11999 points)
    peak_name : string
        name of peak to calculate 
    verbose : bool
        should I talk to you?
        
    Returns
    -------
    prop_times : xr.DataArray
        calculated propagation times for given peak_name. inherits dates dimension from NCCFs
    '''
    
    prop_times_ls = []
    for k in tqdm(range(NCCFs.shape[0]), disable=(not verbose)):
        peak_window = np.expand_dims(NCCFs[k,peak_slices[peak_name]].values, axis=0)
        # whiten data with same function used in pre-processing
        peak_whitened = calculate.freq_whiten_b(peak_window, 200, np.array([1,90]))
        # take magnitude hilbert transform of peak window
        peak_whitened_c = signal.hilbert(peak_whitened)

        # get peak_slice_tight relative to regular slice
        start_idx = peak_slices_tight[peak_name].start - peak_slices[peak_name].start
        stop_idx = peak_slices[peak_name].stop - peak_slices_tight[peak_name].start

        # estimate arrival time
        prop_times_ls.append(argmax_interp(
            np.abs(peak_whitened_c), [start_idx, stop_idx], verbose=False)[0] \
                             + peak_slices[peak_name].start)

    # convert to delay time
    prop_times_tau = (np.array(prop_times_ls) - 5999)*0.005
    prop_times = xr.DataArray(prop_times_tau, dims=['dates'], coords={'dates':NCCFs.dates}, name='prop times with whitening')
    return prop_times
    