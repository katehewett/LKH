# testing script to plot requested data from OOI M2M 
# run after toy_extract_data.py

# imports
from lo_tools import Lfun

import os
import xarray as xr

#import matplotlib.pyplot as plt
#import matplotlib.dates as mdates

fn_p = '/Users/katehewett/Documents/LKH_output/OOI_data_exploration/testing'
fn = 'CE02SHSM_CTD_simple.nc'
#fn = 'CE02SHSM_RID27_telemetered_ctdbp_cdef_dcl_instrument.nc'
fnp = os.path.join(fn_p, fn)

ds = xr.open_dataset(fnp, decode_times=False)
ot = ds['ocean_time'].values
days = (ot - ot[0])/86400.
mdays = Lfun.modtime_to_mdate_vec(ot)
mdt = mdates.num2date(mdays) # list of datetimes of data

SP = ds['SP']

fs=16
plt.rc('font', size=fs)
fig = plt.figure(figsize=(18,10))

ax1.plot(ds.time, SP.values, color='tab:blue', linewidth=2, alpha=0.8) 

