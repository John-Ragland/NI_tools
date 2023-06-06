'''
utils.py - collection of functions used across NI_tools
'''
import numpy as np
from scipy import signal
import xarray as xr
import scipy

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