'''
This code is to run basic QC on NOAA data products for 3 moornings: 
    Chaba 
    Cape Elizabeth 
    Cape Arago 

(see Sutton et al 2014 + 2019) 

We first ran process_webdata.py which saves iput .nc files here: 
/Users/katehewett/Documents/LKH_data/sutton2019_surface_buoys/py_files/

This is just to look at time stamps for a meeting

'''

import sys
import os 
import pandas as pd
import numpy as np
import xarray as xr
import posixpath

import matplotlib.pyplot as plt

from lo_tools import Lfun

Ldir = Lfun.Lstart()
    
# processed data location
otype = 'moor' 
in_dir = Ldir['parent'] / 'LKH_data' / 'sutton2019_surface_buoys' / 'py_files'

out_dir = Ldir['parent'] / 'LKH_output' / 'sutton2019_surface_buoys' / 'march2025_discussion' / 'plots'

if os.path.exists(out_dir)==False:
    Lfun.make_dir(out_dir, clean = False)
if os.path.exists(in_dir)==False:
    print('input path does not exist')
    sys.exit()

# This is a little lazy, but you need to enter the station name below with the list here:
# update to keyboard enter sn_name_dict and DTmax
#sn_name_dict = {
#    'CHABA':'Chaba',
#    'CAPEELIZABETH':'Cape Elizabeth', # less varialbes
#    'CAPEARAGO':'Cape Arago',
#}

sn_name_dict = {
   'CAPEARAGO':'Cape Arago'
} 
sn_list = list(sn_name_dict.keys())

sn = sn_list[0]
print(sn)

if str(sn)=='CAPEELIZABETH':
    fn_in = posixpath.join(in_dir, (sn +'_sorted.nc')) # had to fix times and remove dups in CE 
else: 
    fn_in = posixpath.join(in_dir, (sn +'.nc'))

ds = xr.open_dataset(fn_in, decode_times=True)  

dt = pd.to_datetime(ds.time_utc,format = '%YYYY-%m-%d HH:MM',errors='coerce')
df = pd.DataFrame({'date':dt.normalize(),'time_utc':dt}) 
df['time_diff'] = df['time_utc'].diff()
df['time_diff_hours'] = df['time_diff'] / np.timedelta64(1, 'h')

plt.close('all')
fs=12
plt.rc('font', size=fs)
fig = plt.figure(figsize=(16,8))

ax = plt.subplot2grid((3,1),(0,0),colspan=1,rowspan=1)
plt.axhline(y=3, color='r', linestyle='-', linewidth=2)
plt.plot(df['time_utc'][1:-1],df['time_diff_hours'][1:-1],'b.')
ax.set_title(str(sn))
ax.set_ylabel('DT hours')
ax.set_ylim([0,15])
ax.set_yticks(np.arange(0,16,1), minor=True)
plt.grid(True)

ax1 = plt.subplot2grid((3,1),(1,0),colspan=1,rowspan=1)
plt.plot(ds.time_utc,ds.SA.values,'r.')
plt.plot(ds.time_utc,ds.CT.values,'b.')
ax1.set_ylabel('SA r; CT b')

ax2 = plt.subplot2grid((3,1),(2,0),colspan=1,rowspan=1)
plt.plot(ds.time_utc,ds['xCO2_air'].values,'k.')
plt.plot(ds.time_utc,ds['pCO2_air'].values,'b.')
ax2.set_ylabel('xCO2_air k; pCO2_air b')

plt.gcf().tight_layout()

figname = str(sn) + '.png'
fig.savefig(out_dir / figname)