'''
This code is to fix the times Cape Elizabeth 

Cape Elizabeth is a bit of a mess. 
the times aren't monotonically increasing so needs help 
before running spike tests 

'''

testing = True 

import os
import sys
import pandas as pd
import numpy as np
import xarray as xr
import posixpath
from datetime import timedelta

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
import matplotlib.pyplot as plt

from lo_tools import Lfun, zfun
    
Ldir = Lfun.Lstart()
    
# processed data location
source = 'ocnms'
otype = 'moor' 
in_dir = Ldir['parent'] / 'LKH_data' / 'sutton2019_surface_buoys' / 'py_files'
out_dir = in_dir 

if os.path.exists(out_dir)==True:
    Lfun.make_dir(out_dir, clean = False)
if os.path.exists(in_dir)==False:
    print('input path does not exist')
    sys.exit()

sn_name_dict = {
    'CAPEELIZABETH':'Cape Elizabeth'
} 
sn_list = list(sn_name_dict.keys())

sn = sn_list[0]
print(sn)
    
fn_in = posixpath.join(in_dir, (sn +'.nc'))
fn_out = posixpath.join(out_dir, (sn + '_sorted.nc'))

ds = xr.open_dataset(fn_in, decode_times=True)  

'''
Search Duplicates (we know there are dups...)
Remove before reordering 
'''
dt = pd.to_datetime(ds.time_utc,format = '%YYYY-%m-%d HH:MM',errors='coerce')
df = pd.DataFrame({'date':dt.normalize(),'time_utc':dt}) 

# fix duplicates < already tested after fixing time order 
duplicates = df.duplicated(subset='time_utc', keep=False)
duplicate_time_indices = df[df.duplicated(subset='time_utc', keep=False)].index
new_time = df.drop(duplicate_time_indices)
new_datetime = pd.to_datetime(new_time['time_utc'])   

# duplicates = df2.duplicated(subset='time_utc', keep=False)
dup_index1 = df[df.duplicated('time_utc', keep='first')].index # first
dup_index2 = df[df.duplicated('time_utc', keep='last')].index # last ;  keep=False would give the index for the pair of dups

new_time = df.drop(dup_index1)
new_time = new_time.reset_index(drop=True) # resets the index after the drop so can enter to a new ds below
new_datetime = pd.to_datetime(new_time['time_utc'])

# make a new dataset and cut out the duplicates
coords = {'time_utc':new_datetime}
ds_new = xr.Dataset(coords=coords, attrs=ds.attrs)

var_list = list(ds.keys())

for vn in var_list: 
    A = ds[str(vn)].values
    A1 = np.delete(A,dup_index1)
    A2 = np.delete(A,dup_index1)
    
    if np.all(np.isnan(A1))==False: # if all = True; vn.values = all nans
        L1 = np.sum(~np.isnan(A1)) # find which has more nonnan values
        L2 = np.sum(~np.isnan(A2))
        if L1>L2:
            ds_new[str(vn)]=xr.DataArray(A1, dims={'time_utc':new_datetime},attrs=ds[str(vn)].attrs)
        else:
            ds_new[str(vn)]=xr.DataArray(A2, dims={'time_utc':new_datetime},attrs=ds[str(vn)].attrs)
    else: 
        ds_new[str(vn)]=xr.DataArray(A1, dims={'time_utc':new_datetime},attrs=ds[str(vn)].attrs)
 
del new_time, new_datetime, coords, dup_index1, dup_index2, duplicates, duplicate_time_indices, A, A1, A2, L1, L2

'''
sort the times so monotonically increasing 
'''
dt2 = pd.to_datetime(ds_new.time_utc,format = '%YYYY-%m-%d HH:MM',errors='coerce')
df2 = pd.DataFrame({'time_utc':dt2}) 

if df2['time_utc'].is_monotonic_increasing == False:
    print('issue with time order')
    sorted_index = df2.sort_values(by='time_utc').index
    # sys.exit() # we know CE has an issue

new_time = df2['time_utc'][sorted_index]
new_time = new_time.reset_index(drop=True) # resets the index after the drop so can enter to a new ds below
new_datetime = pd.to_datetime(new_time)

# make a new dataset and cut out the duplicates
coords = {'time_utc':new_datetime}
ds_mono = xr.Dataset(coords=coords, attrs=ds.attrs)

var_list = list(ds_new.keys())

for vn in var_list: 
    A = ds_new[str(vn)].values
    A = A[sorted_index] # it's an array, don't need to drop index
    ds_mono[str(vn)]=xr.DataArray(A, dims={'time_utc':new_datetime},attrs=ds_new[str(vn)].attrs)

dt3 = pd.to_datetime(ds_mono.time_utc,format = '%YYYY-%m-%d HH:MM',errors='coerce')
df3 = pd.DataFrame({'time_utc':dt3})

if df3['time_utc'].is_monotonic_increasing == False:
    print('not fixed')
else:
    print('monotonic correction')
 
'''check results'''
for vn in var_list:
    fs=12
    plt.rc('font', size=fs)
    fig = plt.figure(figsize=(16,8))
    ax0 = plt.subplot2grid((3,1),(0,0),colspan=1,rowspan=1)
    ax0.set_ylabel(str(vn)+' '+str(ds_new[str(vn)].attrs['units']))
    ax0.set_title(str(sn)+': sutton et al. 2019 data (blue) sorted (orange)')
    plt.grid(True)
    
    if np.all(np.isnan(ds[str(vn)].values))==False:
        plt.plot(ds['time_utc'],ds[str(vn)],color = 'blue',marker='.',linestyle='-',linewidth=1,label = 'original') 
        plt.plot(ds_new['time_utc'],ds_new[str(vn)],color = 'black',marker='.',linestyle='-',linewidth=1,alpha=0.3, label ='drop dups')
        plt.plot(ds_mono['time_utc'],ds_mono[str(vn)],color = 'orange',marker='.',linestyle='-',linewidth=1,alpha=0.6,label ='fix time') 
    plt.legend(loc='best') 
    
    ax1 = plt.subplot2grid((3,1),(1,0),colspan=1,rowspan=1) 
    if np.all(np.isnan(ds[str(vn)].values))==False:
        plt.plot(ds_mono['time_utc'],ds_mono[str(vn)],color = 'orange',marker='.',linestyle='-',linewidth=1,alpha=1,label ='fix time') 
    
    plt.legend(loc='best') 

ds_mono.to_netcdf(fn_out, unlimited_dims='time')
