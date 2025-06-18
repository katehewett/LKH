'''

timeseries plots 1/year
not 2017! - that has 2 model vals

'''

import os
import sys
import pandas as pd
import numpy as np
import xarray as xr
import posixpath
from datetime import datetime, timedelta
import matplotlib.dates as mdates
from datetime import datetime
import pickle 

import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.pyplot as plt
import cmcrameri.cm as cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from lo_tools import Lfun

Ldir = Lfun.Lstart()

# TODO put to args 
thisyr = 2024

#vel = 'u'
vel = 'v'

moor = 'CE09OSSM' 
grid = 'cas7_t0_x4b'
# grid = 'cas7_t1_x4a'#only 2017

mfd_in = Ldir['parent'] / 'LKH_data' / 'OOI'/ 'CE' / 'coastal_moorings' / moor / 'velocity' / 'mfd' / 'daily'
sb_in = Ldir['parent'] / 'LKH_data' / 'OOI'/ 'CE' / 'coastal_moorings' / moor / 'velocity' / 'surfacebuoy' / 'daily'
nsif_in = Ldir['parent'] / 'LKH_data' / 'OOI'/ 'CE' / 'coastal_moorings' / moor / 'velocity' / 'nsif' / 'daily'

in_dir = Ldir['parent'] / 'LO_output' / 'extract' / grid / 'moor' / 'OOI_WA_SM' 
fn = moor + '_' + str(thisyr) + '.01.01_' + str(thisyr) + '.12.31.nc'
LO_in = posixpath.join(in_dir,fn)
LO_ds = xr.open_dataset(LO_in, decode_times=True)  

LOot = pd.to_datetime(LO_ds.ocean_time)
#LOz1 = np.nanmean(LO_ds.z_rho.values[:,(27,28,29)])
if vel == 'u':
    #LO1m = np.nanmean(LO_ds.u.values[:,(27,28,29)],axis=1)
    LO1 = LO_ds.u.values[:,-1]
    LO7 = LO_ds.u.values[:,-4]
    LO20 = LO_ds.u.values[:,-7]
    LO80 = LO_ds.u.values[:,17]
    LO500 = LO_ds.u.values[:,2]

if vel == 'v':
    LO1 = np.nanmean(LO_ds.v.values[:,(27,28,29)],axis=1)
    LO7 = LO_ds.v.values[:,-4]
    LO20 = LO_ds.v.values[:,-7]
    LO80 = LO_ds.v.values[:,17]
    LO500 = LO_ds.v.values[:,2]

'''
fn_1 = moor + '_' + vel + 'monthly_surfacebuoy_VELPTA_'+str(thisyr)+'.pkl'
fn_7 = moor + '_' + vel + 'monthly_nsif_VELPTA_'+str(thisyr)+'.pkl'
fn_mfd = moor + '_' + vel + 'monthly_mfd_ADCP_'+str(thisyr)+'.pkl'
fn_nsif = moor + '_' + vel + 'monthly_nsif_ADCP_'+str(thisyr)+'.pkl'
model2 = moor + '_' + vel + 'monthly_'+str(thisyr)+'_cas7_t1_x4a.pkl'
model1 = moor + '_' + vel + 'monthly_'+str(thisyr)+'_cas7_t0_x4b.pkl'
'''
out_dir = Ldir['parent'] / 'LKH_output' / 'OOI'/ 'CE' / 'coastal_moorings' / moor / 'velocity' / 'timeseries_byDepth' / 'plots'
if os.path.exists(out_dir)==False:
    Lfun.make_dir(out_dir, clean = False)

################################################
### Create an array of datetime objects
start_date = datetime(thisyr, 1, 1)
end_date = datetime(thisyr+1, 1, 15)
interval = timedelta(days=15)
date_array = []
current_date = start_date
while current_date <= end_date:
    date_array.append(current_date)
    current_date += interval
################################################################################################
plt.close('all')
fs=12
plt.rc('font', size=fs)
fig = plt.figure(figsize=(14,8))
fig.set_size_inches(14,8, forward=False)

################################################
# plot surface buoy 
fn_in = posixpath.join(sb_in, (moor+'_surfacebuoy_VELPTA_DAILY.nc'))
ds = xr.open_dataset(fn_in, decode_times=True)  
dt = pd.to_datetime(ds.ocean_time,format = '%YYYY-%m-%d HH:MM',errors='coerce')
yrs = dt.year
mdx = np.where(yrs==thisyr)
ot = dt[mdx]
var = ds[vel].values[mdx]

ax1 = plt.subplot2grid((5,1), (0,0), colspan=1,rowspan=1)
ax1.axhline(y=0, color='k', linestyle=':',alpha=0.8)
ax1.plot(ot,var,color = 'DodgerBlue', marker='none', alpha=0.8, linestyle='-', linewidth = 2, label = 'sb -1m')
ax1.set_xlim(date_array[0], date_array[-1])
ax1.set_xticks(date_array)
ax1.set_ylim(-0.5,0.5)
date_format = mdates.DateFormatter('%b%d')
ax1.xaxis.set_major_formatter(date_format)
ax1.set_title(moor +': ' + str(thisyr))
ax1.set_ylabel(vel+' m/s')

labels = ax1.xaxis.get_ticklabels() # Get the x-axis labels
for i, label in enumerate(labels): # Iterate through labels and hide every other one
    if i % 2 != 0:
        label.set_visible(False)

ax1.minorticks_on()
ax1.grid(which='major')

ax1.plot(LOot,LO1,color = 'Crimson', marker='none', alpha=0.8, linestyle='-', linewidth = 2, label = grid)
#ax1.plot(LOot,LO1,color = 'pink', marker='none', alpha=0.8, linestyle='-', linewidth = 2, label = grid)

ax1.legend(fontsize=10,loc = 'lower left')
ax1.set_xticklabels([])

del ds, dt, yrs, mdx, ot, var
################################################
# plot nsif velpta
fn_in = posixpath.join(nsif_in, (moor+'_nsif_VELPTA_DAILY.nc'))
ds = xr.open_dataset(fn_in, decode_times=True)  
dt = pd.to_datetime(ds.ocean_time,format = '%YYYY-%m-%d HH:MM',errors='coerce')
yrs = dt.year
mdx = np.where(yrs==thisyr)
ot = dt[mdx]
var = ds[vel].values[mdx]

ax2 = plt.subplot2grid((5,1), (1,0), colspan=1,rowspan=1)
ax2.axhline(y=0, color='k', linestyle=':',alpha=0.8)
ax2.plot(ot,var,color = 'DodgerBlue', marker='none', alpha=0.8, linestyle='-', linewidth = 2, label = 'nsif -7m')
ax2.set_xlim(date_array[0], date_array[-1])
ax2.set_xticks(date_array)
ax2.set_ylim(-0.5,0.5)
date_format = mdates.DateFormatter('%b%d')
ax2.xaxis.set_major_formatter(date_format)
ax2.set_ylabel(vel+' m/s')

labels2 = ax2.xaxis.get_ticklabels() # Get the x-axis labels
for i, label in enumerate(labels2): # Iterate through labels and hide every other one
    if i % 2 != 0:
        label.set_visible(False)

ax2.minorticks_on()
ax2.grid(which='major')

ax2.plot(LOot,LO7,color = 'Crimson', marker='none', alpha=0.8, linestyle='-', linewidth = 2, label = grid)
#ax1.plot(LOot,LO1,color = 'pink', marker='none', alpha=0.8, linestyle='-', linewidth = 2, label = grid)

ax2.legend(fontsize=10,loc = 'lower left')
ax2.set_xticklabels([])

del ds, dt, yrs, mdx, ot, var
################################################
# plot nsif ADCP 20 
fn_in = posixpath.join(nsif_in, (moor+'_nsif_ADCP_DAILY.nc'))
ds = xr.open_dataset(fn_in, decode_times=True)  
dt = pd.to_datetime(ds.ocean_time,format = '%YYYY-%m-%d HH:MM',errors='coerce')
yrs = dt.year
mdx = np.where(yrs==thisyr)
ot = dt[mdx]
var20 = ds[vel].values[mdx,-2].squeeze() #vals at -20

ax3 = plt.subplot2grid((5,1), (2,0), colspan=1,rowspan=1)
ax3.axhline(y=0, color='k', linestyle=':',alpha=0.8)
ax3.plot(ot,var20,color = 'DodgerBlue', marker='none', alpha=0.8, linestyle='-', linewidth = 2, label = 'nsif -20m')
ax3.set_xlim(date_array[0], date_array[-1])
ax3.set_xticks(date_array)
ax3.set_ylim(-0.5,0.5)
date_format = mdates.DateFormatter('%b%d')
ax3.xaxis.set_major_formatter(date_format)
ax3.set_ylabel(vel+' m/s')

labels3 = ax3.xaxis.get_ticklabels() # Get the x-axis labels
for i, label in enumerate(labels3): # Iterate through labels and hide every other one
    if i % 2 != 0:
        label.set_visible(False)

ax3.minorticks_on()
ax3.grid(which='major')

ax3.plot(LOot,LO20,color = 'Crimson', marker='none', alpha=0.8, linestyle='-', linewidth = 2, label = grid)
#ax1.plot(LOot,LO1,color = 'pink', marker='none', alpha=0.8, linestyle='-', linewidth = 2, label = grid)

ax3.legend(fontsize=10,loc = 'lower left')
ax3.set_xticklabels([])

################################################
# plot nsif ADCP 80 nsif and mfd
var80n = ds[vel].values[mdx,3].squeeze() #vals at -80

fn_in = posixpath.join(mfd_in, (moor+'_mfd_ADCP_DAILY.nc'))
ds = xr.open_dataset(fn_in, decode_times=True)  
dt = pd.to_datetime(ds.ocean_time,format = '%YYYY-%m-%d HH:MM',errors='coerce')
yrs = dt.year
mdx = np.where(yrs==thisyr)
otm = dt[mdx]
var80m = ds[vel].values[mdx,-3].squeeze() #vals at -80

ax4 = plt.subplot2grid((5,1), (3,0), colspan=1,rowspan=1)
ax4.axhline(y=0, color='k', linestyle=':',alpha=0.8)
ax4.plot(ot,var80n,color = 'DodgerBlue', marker='none', alpha=0.8, linestyle='-', linewidth = 2, label = 'nsif -80m')
ax4.plot(otm,var80m,color = 'Green', marker='none', alpha=0.6, linestyle=':', linewidth = 2, label = 'mfd -80m')
ax4.set_xlim(date_array[0], date_array[-1])
ax4.set_xticks(date_array)
ax4.set_ylim(-0.5,0.5)
date_format = mdates.DateFormatter('%b%d')
ax4.xaxis.set_major_formatter(date_format)
ax4.set_ylabel(vel+' m/s')

labels4 = ax4.xaxis.get_ticklabels() # Get the x-axis labels
for i, label in enumerate(labels4): # Iterate through labels and hide every other one
    if i % 2 != 0:
        label.set_visible(False)

ax4.minorticks_on()
ax4.grid(which='major')

ax4.plot(LOot,LO80,color = 'Crimson', marker='none', alpha=0.8, linestyle='-', linewidth = 2, label = grid)
#ax1.plot(LOot,LO1,color = 'pink', marker='none', alpha=0.8, linestyle='-', linewidth = 2, label = grid)

ax4.legend(fontsize=10,loc = 'lower left')
ax4.set_xticklabels([])

################################################
# plot nsif ADCP 500 mfd
var500m = ds[vel].values[mdx,2].squeeze() #vals at -500

ax5 = plt.subplot2grid((5,1), (4,0), colspan=1,rowspan=1)
ax5.axhline(y=0, color='k', linestyle=':',alpha=0.8)
ax5.plot(otm,var500m,color = 'Green', marker='none', alpha=0.6, linestyle='-', linewidth = 2, label = 'mfd -500m')
ax5.set_xlim(date_array[0], date_array[-1])
ax5.set_xticks(date_array)
ax5.set_ylim(-0.15,0.15)
date_format = mdates.DateFormatter('%b%d')
ax5.xaxis.set_major_formatter(date_format)
ax5.set_ylabel(vel+' m/s')

labels5 = ax5.xaxis.get_ticklabels() # Get the x-axis labels
for i, label in enumerate(labels5): # Iterate through labels and hide every other one
    if i % 2 != 0:
        label.set_visible(False)

ax5.minorticks_on()
ax5.grid(which='major')

ax5.plot(LOot,LO500,color = 'Crimson', marker='none', alpha=0.8, linestyle='-', linewidth = 2, label = grid)
#ax1.plot(LOot,LO1,color = 'pink', marker='none', alpha=0.8, linestyle='-', linewidth = 2, label = grid)

ax5.legend(fontsize=10,loc = 'lower left')

fig.tight_layout()

figname = moor+'_'+vel+'_'+str(thisyr)+'_timeseries.png'
fig.savefig(out_dir / figname)
print('saved')