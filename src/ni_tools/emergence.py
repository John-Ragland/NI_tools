'''
develop tools to study and compare the emergence of the EGF in the NCCF
'''
import xarray as xr
import numpy as np
from ni_tools import calculate
from tqdm import tqdm
import multiprocessing as mp

def calc_emergence_NCCF(R, W, Fs, avg_time_inc):
    '''
    calc_emergence_NCCF - takes a un-averaged NCCF stack (only averaged by W) and
        successively stacks (hopefully using different methods) to create emergence
        NCCF. This should have dimensions (avg_time x delay)
    
    This function will create an emergence for all of R. To slice and create different
        samples of emergence see calc_emergence_NCCF_samples()
    
    Parameters
    ----------
    R : xr.DataArray
        unstacked NCCF. Should have dimensions (delay x time). time is average time.
        Can be calculated with OOI_hydrophone_cloud.processing.compute_NCCF_stack()
    W : float
        window of R (in s)
    Fs : float
        sampling rate of R (in Hz)
    avg_time_inc : float
        resolution of emergence in seconds (how many seconds between each emergence NCCF)
        - time*W % avg_time_inc must be = 0
        - time*W must be >= avg_time_inc
    '''

    # test that avg_time_inc is valid given W and dimension of R.
    if R.sizes['time']*W % avg_time_inc != 0:
        raise Exception("R['time'] x W must be integer divisible by avg_time_inc")
    elif R.sizes['time']*W < avg_time_inc:
        raise Exception("R['time'] x W must be larger than avg_time_inc")

    # loop through and succesivly stack R (with multiprocessing)
    num_emerge = int(R.sizes['time']*W/avg_time_inc)

    # multiprocessing abandoned for now because it took longer
    
    #workers = mp.cpu_count()
    #with mp.Pool(workers) as pool:
    #    R_slices = [R.isel({'time':slice(0,int(k*avg_time_inc/W))}) for k in range(1,num_emerge)]
    #    NCCF_emerge = pool.map(calculate.linear_stack, R_slices)

    # get time_sel and delay_sel
    #TODO make this part not hard coded...
    delay_sel = xr.ones_like(R.isel({'time': 0}))
    time_sel = calculate.psd_time_sel(R, pband=(25,100))

    NCCF_emerge = []
    for k in range(1,num_emerge):
        R_slice = R.isel({'time':slice(0,int(k*avg_time_inc/W))})
        time_sel_slice = time_sel.isel({'time':slice(0,int(k*avg_time_inc/W))})
        NCCF_emerge.append(calculate.selective_stack(R_slice, time_sel_slice, delay_sel))

    NCCF_emerge_x = xr.concat(NCCF_emerge, dim='avg_time')
    NCCF_emerge_x = NCCF_emerge_x.assign_coords({'avg_time':np.arange(avg_time_inc/3600, R.sizes['time']*W/3600, avg_time_inc/3600)[:-1]})
    return NCCF_emerge_x
