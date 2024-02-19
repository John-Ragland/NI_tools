'''
SNR.py - calculate SNR of various peaks. interger indices of peak windows
    are currently hode coded for caldera hydrophones, but it would be nice
    to make this more general in the future

This code was copied over from Noise_Interferometry below is old docs.
    analysisX - toolset for NCCF stacks using xarray
    mostly remaking useful functions from analysis.py for support
    with xarray datatype
'''

import xarray as xr
import numpy as np
from scipy import signal
from tqdm import tqdm

# Manually defined peak slices (in delay time dimension)
peak_windows = [
    [5535, 5635],
    [5303, 5503],
    [4969, 5169],
    [4622, 4782],
    [6367, 6467],
    [6498, 6698],
    [6833, 7033],
    [7242, 7402],
]

slices = []
for k in range(len(peak_windows)):
    slices.append(np.s_[peak_windows[k][0]:peak_windows[k][1]])
peak_names = ['dA', 's1b0A', 's2b1A', 's3b2A', 'dB', 's1b0B', 's2b1B', 's3b2B']
peak_slices = dict(zip(peak_names,slices))
peak_names_short = ['dA','s1b0A','s2b1A','s1b0B','s2b1B']

def snr_of_single_NCCF(NCCF, peak_id, noise_bounds=[-15,-10], peak_bounds=None, hann=True, hilbert=False):
    '''
    Calculate the SNR for a given peak for a single NCCF
    Parameters
    ----------
    NCCF : xarray.DataArray
        Data Array of single NCCF. Should be 1D 
        with dimension name, 'delay'.
    peak_id : str
        peak identifier string ('da','s1b0A','s2b1A','s1b0B','s2b1B')
        if peak_id = 'custom', then peak_bounds is used.
    noise_bounds : list
        list of length two defining the noise bounds that SNR
        is calculated from. default is [-15,-10]
    peak_bounds : tuple
        tuple of peak bounds in time bins. default is None and peak_id is used.
    hann : bool
        whether or not to divide by hann window for noise bounds
    hilbert : bool
        whether or not to take the hilbert transform of the NCCF

    Returns
    -------
    SNR : float
        SNR of the peak in dB
    '''
    
    if peak_id == 'custom':
        peak_slice = np.s_[peak_bounds[0]:peak_bounds[1]]
    else:
        peak_slice = peak_slices[peak_id]
        
    noise_idx = np.array([
        np.argmin(np.abs(NCCF.delay.values-noise_bounds[0])),
        np.argmin(np.abs(NCCF.delay.values-noise_bounds[1]))
    ])
    
    if hann:
        # Remove Hann window from NCCF for noise std
        hann_single = signal.windows.hann(int((len(NCCF)+1)/2))
        hann = signal.correlate(hann_single, hann_single, mode='full')
        hann = hann/np.max(hann)
        hann[:noise_idx[0]] = 1
        hann[noise_idx[1]:]=1
        hann_inv = 1/hann

        NCCF_nohan = np.abs(NCCF) * hann_inv
        noise_std = np.std(NCCF_nohan[noise_idx[0]:noise_idx[1]])
    else:
        noise_std = np.std(NCCF[noise_idx[0]:noise_idx[1]])

    if hilbert:
        NCCFc = np.abs(signal.hilbert(NCCF))
    else:
        NCCFc = NCCF
        
    # Calculate SNR
    peak_amp = np.max(NCCFc[peak_slice])
    SNR = 20*np.log10(peak_amp/(3*noise_std))
    
    return SNR

def snr_of_NCCFs(NCCFs, peak_id, noise_bounds=[-15,-10], peak_bounds=None, hann=True):
    '''
    Calculate the SNR for a given peak for a single NCCF
    Parameters
    ----------
    NCCF : xarray.DataArray
        Data Array of single NCCF. Should be 1D 
        with dimension name, 'delay'.
    peak_id : str
        peak identifier string ('da','s1b0A','s2b1A','s1b0B','s2b1B')
        if peak_id = 'custom', then peak_bounds is used.
    noise_bounds : list
        list of length two defining the noise bounds that SNR
        is calculated from. default is [-15,-10]
    peak_bounds : tuple
        tuple of peak bounds in time bins. default is None and peak_id is used.
    hann : bool
        whether or not to divide by hann window for noise bounds
    hilbert : bool
        whether or not to take the hilbert transform of the NCCF

    Returns
    -------
    SNR : float
        SNR of the peak in dB
    '''

    if peak_id == 'custom':
        peak_slice = np.s_[peak_bounds[0]:peak_bounds[1]]
    else:
        peak_slice = peak_slices[peak_id]

    noise_idx = np.array([
        np.argmin(np.abs(NCCFs.delay.values-noise_bounds[0])),
        np.argmin(np.abs(NCCFs.delay.values-noise_bounds[1]))
    ])

    if hann:
        hann_single = signal.windows.hann(int((NCCFs.sizes['delay']+1)/2))
        hann = signal.correlate(hann_single, hann_single, mode='full')
        hann = hann/np.max(hann)
        hann[:noise_idx[0]] = 1
        hann[noise_idx[1]:] = 1
        hann_inv = 1/hann

        NCCF_nohan = NCCFs*hann_inv
        noise_std = NCCF_nohan.isel({'delay':slice(noise_idx[0], noise_idx[1])}).std('delay')
        
    else:
        noise_std = NCCFs.isel({'delay':slice(noise_idx[0], noise_idx[1])}).std('delay')

    # Calculate SNR
    peak_amp = NCCFs.isel({'delay':peak_slice}).max('delay')
    SNR = 20*np.log10(peak_amp/(3*noise_std))

    return SNR

"""
def SNR_NCCFs(NCCFs, num_available=None, noise_bounds=[-15,-10], hilbert=False, peak_names_snr=None):
    '''
    Calculate SNR for all peaks for a NCCF stack
    Parameters
    ----------
    NCCF : xarray.DataArray
        Data Array of single NCCF. Should be 1D 
        with dimension name, 'delay'.
    num_available : xarray.DataArray
        Data array containing number of hours for each NCCF in
        NCCF. num_available and NCCF['delay'] should have same length
    noise_bounds : list
        list of length two defining the noise bounds that SNR
        is calculated from. default is [-15,-10]
    hilbert : bool
        passed to snr_of_single_NCCF
    peak_names_snr : list
        list of peak names to calculate SNR, if none, all peaks are calculated.

    Returns
    -------
    SNR : xarray.DataArray
        Data array of the SNR for every discernable peak
    '''
    
    if peak_names_snr == None:
        peak_names_snr = peak_names_short
    
    # build data array
    values = np.ones((5, NCCFs.shape[0]))*np.nan
    SNRs = xr.DataArray(values, dims=['peaks','dates'], coords={'peaks':peak_names_short, 'dates': NCCFs.dates}, name='SNRs')

    NCCFs.shape

    for k in tqdm(range(NCCFs.shape[0])):
        
        for peak in peak_names_snr:
            SNRs.loc[peak][k] = snr_of_single_NCCF(NCCFs[k,:], peak, noise_bounds)

    # Need at least 100 hours
    #SNRs[:,num_available < 100] = np.nan
    # Remove values under num available threshold
    return SNRs

"""