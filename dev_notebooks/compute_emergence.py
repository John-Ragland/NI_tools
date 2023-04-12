# Compute Emergence - copied from emergence_comparison.ipynb

import xarray as xr
from OOI_hydrophone_cloud import utils
from OOI_hydrophone_cloud.processing import processing
import pandas as pd
import os
from NI_tools.NI_tools import emergence
from tqdm import tqdm

storage_options = {'account_name':'lfhydrophone', 'account_key':os.environ['AZURE_KEY_lfhydrophone']}
ds = xr.open_zarr('abfs://hydrophonedata/lf_hydrophone_data_test.zarr', storage_options=storage_options)

time_base = pd.Timestamp('2017-01-01')

avg_time_int = 60*10 # 10 minutes or 600 seconds
NCCF_emerge_int = 10 # hours

zarr_store = '/datadrive/NCCF_emergence/emergence_10hr_10min_2017_1-5Hz_PSD_selective.zarr'

for k in tqdm(range(int(365*24/NCCF_emerge_int))):
    start_time = time_base + k*pd.Timedelta(hours=NCCF_emerge_int)
    end_time = start_time + pd.Timedelta(hours=NCCF_emerge_int)

    ds_sliced = utils.slice_ds(ds, start_time, end_time, include_coord=False)[['AXCC1', 'AXEC2']]

    NCCF_full = processing.compute_NCCF_stack(ds_sliced, stack=False)
    NCCF_emerge = emergence.calc_emergence_NCCF(NCCF_full, 30, 200, 10*60)

    # add time dimension (which will be append dimension)
    NCCF_emerge = NCCF_emerge.expand_dims('time')
    NCCF_emerge = NCCF_emerge.assign_coords({'time':[start_time.value]})
    # chunk NCCF emerge
    NCCF_emerge = NCCF_emerge.chunk({'time':1, 'avg_time':59, 'delay':11999})

    # for now I'm just going to put the UNIX nanosecond string because I dont' understand the error
    if k == 0:
        xr.Dataset({'NCCF_emerge':NCCF_emerge}).to_zarr(zarr_store, mode='w-')
    else:
        xr.Dataset({'NCCF_emerge':NCCF_emerge}).to_zarr(zarr_store, append_dim='time')
