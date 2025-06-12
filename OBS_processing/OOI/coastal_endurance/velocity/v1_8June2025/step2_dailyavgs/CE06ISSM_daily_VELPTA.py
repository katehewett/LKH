'''
AFTER running the following code, 
CE06ISSM_organize_VELPTA.py

Calc daily avgs to compare with lowpass (and avg) mooring extractions.

'''

import os
import sys
import pandas as pd
import numpy as np
import xarray as xr
import posixpath
from datetime import timedelta
import pickle 

from lo_tools import Lfun

Ldir = Lfun.Lstart()

moor = 'CE06ISSM'
loco = 'surfacebuoy'
#loco = 'nsif'

# processed data location
otype = 'moor' 
moor = 'CE06ISSM'
in_dir = Ldir['parent'] / 'LKH_data' / 'OOI' / 'CE' / 'coastal_moorings' / moor / 'velocity' / loco
out_dir = in_dir / 'daily'

if loco == 'nsif':
    fn_in = posixpath.join(in_dir, (moor+'_nsif_VELPTA.nc')) 
    fn_out = posixpath.join(out_dir, (moor+'_nsif_VELPTA_DAILY.nc'))
elif loco == 'surfacebuoy':
    fn_in = posixpath.join(in_dir, (moor+'_surfacebuoy_VELPTA.nc')) 
    fn_out = posixpath.join(out_dir, (moor+'_surfacebuoy_VELPTA_DAILY.nc'))

fig_out_dir =  Ldir['parent'] / 'LKH_output' / 'OOI' / 'CE' / 'coastal_moorings' / moor / 'velocity' / loco / 'plots' / 'daily_avg_processing'

if os.path.exists(out_dir)==False:
    Lfun.make_dir(out_dir, clean = False)
if os.path.exists(fig_out_dir)==False:
    Lfun.make_dir(fig_out_dir, clean = False)
if os.path.exists(in_dir)==False:
    print('input path does not exist')
    sys.exit()

cdx = 1 
DTmax = 12

ds = xr.open_dataset(fn_in, decode_times=True)  
dt = pd.to_datetime(ds.ocean_time,format = '%YYYY-%m-%d HH:MM',errors='coerce')
df = pd.DataFrame({'date':dt.normalize(),'ocean_time':dt}) 
df['time_diff'] = df['ocean_time'].diff()
df['LT_DTmax'] = df['ocean_time'].diff()< pd.Timedelta(hours=DTmax)

if df['ocean_time'].is_monotonic_increasing == False:
    print('issue with times')
    sys.exit()
else: 
    print('times pass')

#daily averages 
df = pd.DataFrame({'date':dt.normalize(),'ocean_time':dt}) 
df['time_diff'] = df['ocean_time'].diff()
df['LT_DTmax'] = df['ocean_time'].diff()< pd.Timedelta(hours=DTmax)
df['u'] = ds.u.values
df['v'] = ds.v.values
df['w'] = ds.w.values

u_daily_stats = df.groupby('date').agg(udaily=('u', 'mean'), count=('u','count'), coverage = ('time_diff','sum'))
v_daily_stats = df.groupby('date').agg(vdaily=('v', 'mean'), count=('v','count'), coverage = ('time_diff','sum'))
w_daily_stats = df.groupby('date').agg(wdaily=('w', 'mean'), count=('w','count'), coverage = ('time_diff','sum'))

# replace values where there isn't more than 12 hours of coverage for a daily average
u_daily_stats['udaily'] = u_daily_stats['udaily'].where(u_daily_stats['coverage']>timedelta(hours=12),np.nan) 
v_daily_stats['vdaily'] = v_daily_stats['vdaily'].where(v_daily_stats['coverage']>timedelta(hours=12),np.nan) 
w_daily_stats['wdaily'] = w_daily_stats['wdaily'].where(w_daily_stats['coverage']>timedelta(hours=12),np.nan) 

# resample to fill "empty days" with NaNs
u_resampled = u_daily_stats.resample('D').asfreq()
v_resampled = v_daily_stats.resample('D').asfreq()
w_resampled = w_daily_stats.resample('D').asfreq()

# set up time and then pack for saving in LO friendly format
tdaily = u_resampled.index + timedelta(hours=12)

if loco == 'nsif':
    z = np.ones(np.shape(tdaily))*-7
elif loco == 'surfacebuoy':
    z = np.ones(np.shape(tdaily))*-1

VELPTA = xr.Dataset()
VELPTA['ocean_time'] = (('ocean_time'), tdaily, {'long_name':'daily timestamps, assume UTC'})
VELPTA['z'] = (('ocean_time'), z, {'units':'m', 'long_name':'altitude from OOI; depth below surface'})

VELPTA['u'] = (('ocean_time'), u_resampled['udaily'].values, {'units':'m.s-1', 'long_name': 'Daily avg OOI Eastward Sea Water Velocity','positive':'eastward',
                                      'metadata link':'https://mmisw.org/ont/cf/parameter/eastward_sea_water_velocity',
                                      'source_file': fn_in})
VELPTA['v'] = (('ocean_time'), v_resampled['vdaily'].values, {'units':'m.s-1', 'long_name': 'Daily avg OOI Northward Sea Water Velocity','positive':'northward',
                                      'metadata link':'http://mmisw.org/ont/cf/parameter/northward_sea_water_velocity',
                                      'source_file': fn_in})
VELPTA['w'] = (('ocean_time'), w_resampled['wdaily'].values, {'units':'m.s-1', 'long_name': 'Daily avg Upward Sea Water Velocity','positive':'upward',
                                      'metadata link':'http://mmisw.org/ont/cf/parameter/upward_sea_water_velocity',
                                      'source_file': fn_in})

if loco == 'nsif':
    VELPTA['u'].attrs['moored_location'] = 'nsif ~7m below surface'
    VELPTA['v'].attrs['moored_location'] = 'nsif ~7m below surface'
    VELPTA['w'].attrs['moored_location'] = 'nsif ~7m below surface'
elif loco == 'surfacebuoy':
    VELPTA['u'].attrs['moored_location'] = 'surface buoy ~1m below surface'
    VELPTA['v'].attrs['moored_location'] = 'surface buoy ~1m below surface'
    VELPTA['w'].attrs['moored_location'] = 'surface buoy ~1m below surface'

VELPTA.to_netcdf(fn_out, unlimited_dims='ocean_time')

print('saved file')
sys.exit()

# PLOTTING
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.pyplot as plt
import cmcrameri.cm as cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# map
plt.close('all')
fs=14
plt.rc('font', size=fs)
fig = plt.figure(figsize=(18,10))
fig.set_size_inches(18,10, forward=False)

axu = plt.subplot2grid((4,2), (0,0), colspan=2,rowspan=1)
cpu = axu.pcolormesh(ADCP['ocean_time'],ADCP['z'],ADCP['u'],vmin=-0.5,vmax=0.5,cmap=cm.roma_r)
fig.colorbar(cpu, ax=axu,label = 'daily mean u +east')
axu.set_ylabel('Z [m]')
axu.set_title(moor+ ' ADCP @ mfd')

axv = plt.subplot2grid((4,2), (1,0), colspan=2,rowspan=1)
cpv = axv.pcolormesh(ADCP['ocean_time'],ADCP['z'],ADCP['v'],vmin=-1,vmax=1,cmap=cm.roma_r)
fig.colorbar(cpv, ax=axv,label = 'daily mean v +north')
axv.set_ylabel('Z [m]')

axw = plt.subplot2grid((4,2), (2,0), colspan=2,rowspan=1)
cpw = axw.pcolormesh(ADCP['ocean_time'],ADCP['z'],ADCP['v'],vmin=-0.05,vmax=0.05,cmap=cm.roma_r)
fig.colorbar(cpw, ax = axw, label = 'w +up')
axw.set_ylabel('Z [m]')

axe = plt.subplot2grid((4,2), (3,0), colspan=2,rowspan=1)
cpe = axe.pcolormesh(ADCP['ocean_time'],ADCP['z'],ADCP['velprof'],vmin=-0.1,vmax=0.1,cmap=cm.roma_r)
fig.colorbar(cpe, ax=axe,label = 'daily mean velprof')
axw.set_ylabel('Z [m]')

fig.tight_layout()

'''
# PLOTTING
# map
plt.close('all')
fs=14
plt.rc('font', size=fs)
fig = plt.figure(figsize=(18,10))
fig.set_size_inches(18,10, forward=False)

axu = plt.subplot2grid((4,2), (0,0), colspan=2,rowspan=1)
cpu = axu.pcolormesh(tdaily,z,ub,vmin=-0.5,vmax=0.5,cmap=cm.roma_r)

axv = plt.subplot2grid((4,2), (1,0), colspan=2,rowspan=1)
cpv = axv.pcolormesh(tdaily,z,vb,vmin=-0.5,vmax=0.5,cmap=cm.roma_r)

axw = plt.subplot2grid((4,2), (2,0), colspan=2,rowspan=1)
cpw = axw.pcolormesh(tdaily,z,wb,vmin=-0.5,vmax=0.5,cmap=cm.roma_r)

axe = plt.subplot2grid((4,2), (3,0), colspan=2,rowspan=1)
cpe = axe.pcolormesh(tdaily,z,eb,vmin=-0.5,vmax=0.5,cmap=cm.roma_r)
'''