'''
utils.py - collection of functions used across NI_tools
'''
import numpy as np
from scipy import signal
import xarray as xr
import scipy
import xrsignal as xrs
import xrft

def freq_whiten(data, dim, b,a):
    '''
    freq_whiten - force magnitude of fft to be filter response magnitude
        (retains phase information) along a specified dimension of xarray.DataArray
    chunk in dimension 'dim', must be full dimension for data

    Parameters
    ----------
    data : xarray.DataArray
        DataArray containing the data to be whitened
    dim : str
        Name of the dimension along which to perform the whitening
    b : np.array
        numerator coefficients for filter
    a : np.array
        denominator coefficients for filter 

    Returns
    -------
    data_whiten : xarray.DataArray
        Whitened data with the same dimensions as the input data
    '''
    return data.map_blocks(__freq_whiten_chunk, kwargs={'dim':dim, 'b':b, 'a':a}, template=data)

def __freq_whiten_chunk(data, dim, b, a):
    '''
    __freq_whiten_chunk - force magnitude of fft to be filter response magnitude
        (retains phase information) along a specified dimension of xarray.DataArray
    single chunk operations are done in numpy

    Parameters
    ----------
    data : xarray.DataArray
        DataArray containing the data to be whitened
    dim : str
        Name of the dimension along which to perform the whitening
    b : np.array
        numerator coefficients for filter
    a : np.array
        denominator coefficients for filter 

    Returns
    -------
    data_whiten : xarray.DataArray
        Whitened data with the same dimensions as the input data
    '''
    # Convert the DataArray to a numpy array
    data_np = data.values

    # Get the index of the specified dimension
    dim_index = data.dims.index(dim)
    
    # Window data and compute unit pulse
    win = signal.windows.hann(data_np.shape[dim_index])
    pulse = signal.unit_impulse(data_np.shape[dim_index], idx='mid')
    
    for k in range(len(data_np.shape) - 1):
        win = np.expand_dims(win, 0)
        pulse = np.expand_dims(pulse, 0)

    data_win = data_np * win

    # Take fft
    data_f = scipy.fft.fft(data_win, axis=dim_index)
    data_phase = np.angle(data_f)

    H = np.abs(scipy.fft.fft(signal.filtfilt(b, a, pulse, axis=-1)))

    # Construct whitened signal
    data_whiten_f = (np.exp(data_phase * 1j) * np.abs(H) **
                     2)  # H is squared because of filtfilt

    data_whiten_np = np.real(scipy.fft.ifft(data_whiten_f, axis=dim_index))
    
    data_whiten_x = xr.DataArray(data_whiten_np, dims=data.dims, coords=data.coords)
    return data_whiten_x

def SNR(
        NCCFs: xr.DataArray,
        peaks: dict,
        dim: str='delay',
        remove_hann: bool=True,
        noise_window: slice=slice(-10,-5)
):
    '''
    compute SNR for dictionary of peaks.
    Method to calculate SNR:
    - calculate noise standard deviation (remove Hann winow from pre-processing)
    '''

    noise_slice = NCCFs.sel({dim:noise_window})
    if remove_hann:
        hann_single = signal.windows.hann(int((NCCFs.sizes[dim]+1)/2))
        hann = signal.correlate(hann_single, hann_single, mode='full')
        hann = xr.DataArray(hann/np.max(hann), dims=dim, coords = {dim:NCCFs.coords[dim].values})
        hann_inv = (1/hann**2)
    else:
        hann_inv = xr.DataArray(np.ones(NCCFs.sizes[dim]), dims=dim, coords = {dim:NCCFs.coords[dim].values})

    # remove hann window from noise sample
    noise_slice = 3*((noise_slice*hann_inv.sel({dim:noise_window})).std(dim=dim))

    snrs = {}
    # amplitude
    for peak in peaks:
        amp = np.abs(xrs.hilbert(NCCFs.sel({dim:peaks[peak]}), dim=dim)).max(dim=dim)
        snrs[peak] = 20*np.log10(amp/noise_slice)

    return xr.Dataset(snrs)


def SNRx(NCCFs : xr.DataArray, peaks : dict, noise : slice, dim : str='delay'):
    '''
    SNRx - compute SNR for dictionary of peaks.

    $\mathrm{SNR} = 20 \log_{10} \left( \frac{\max(|x|)}{\sigma_n} \right)$
    where $\sigma_n$ is the standard deviation of a slice with no peaks (specified by noise parameter)

    **notes on application**
    - this was created from code in OOI NI inversion/results.ipynb, and is the algorithm used for estimating
        SNR in *Ragland et. al. Using ocean ambient sound to measure local integrated deep ocean temperature, 2024*
    Parameters
    ----------
    NCCFs : xarray.DataArray
        NCCFs to compute SNR for. NCCF should be cross-correlation (**NOT** the derivative of the cross-correlation).
    peaks : dict
        dictionary of peaks. keys are peak names and values are slices (in dim cooridinates)
    noise : slice
        slice to use for noise standard deviation
    dim : str, optional
        dimension to compute SNR along, by default 'delay'
    '''
    dNCCFs = NCCFs.differentiate(dim)

    dNCCFc = xrs.hilbert(dNCCFs, dim=dim)

    amps = {}
    for peak in peaks:
        amps[peak] = np.abs(dNCCFc.sel({dim:peaks[peak]})).max(dim=dim)

    amps_x = xr.Dataset(amps)

    noise = np.sqrt(((np.abs(dNCCFc)**2).sel({dim:noise})).mean(dim=dim))

    snr = amps_x / noise

    return snr

def compute_fbarrms(NCCFs, peaks, dim='delay'):
    '''
    compute_fbarrms - compute the rms frequency bandwidth from NCCFs
    '''

    fbars = {}
    for peak in peaks:
        NCCFs_f = xrft.fft(NCCFs.sel({dim: peaks[peak]}), dim=dim)
        Sxx_f = (NCCFs_f * np.conjugate(NCCFs_f)).real.sel({f'freq_{dim}':slice(0, None)})

        fbar = (Sxx_f[f'freq_{dim}'] * (Sxx_f)).mean(f'freq_{dim}') / (Sxx_f).mean(f'freq_{dim}')
        f2bar = (Sxx_f[f'freq_{dim}']**2 * (Sxx_f)).mean(f'freq_{dim}') / (Sxx_f).mean(f'freq_{dim}')

        fbar_rms = np.sqrt(f2bar - fbar**2)

        fbars[peak] = fbar_rms
    
    return xr.Dataset(fbars)


def compute_sigma(NCCFs, peaks, dim='delay'):
    '''
    compute_sigma - compute the standard deviation of arrival time estimate

    Parameters
    ----------
    NCCFs : xarray.DataArray
        NCCFs to compute standard deviation for. NCCF should be cross-correlation (**NOT** the derivative of the cross-correlation).
    peaks : dict
        dictionary of peaks. keys are peak names and values are slices (in dim cooridinates)
    dim : str, optional
        delay dimension of noise cross-correlation functions (Default value = 'delay')
    '''

    dNCCFs = NCCFs.differentiate(dim)
    snrs = SNRx(NCCFs, peaks, slice(-1,1), dim=dim)
    fbars = compute_fbarrms(dNCCFs, peaks, dim=dim)

    sigmas = {}
    for peak in peaks:
        sigmas[peak] = 1/(2*np.pi*fbars[peak] * snrs[peak])
    
    return xr.Dataset(sigmas)