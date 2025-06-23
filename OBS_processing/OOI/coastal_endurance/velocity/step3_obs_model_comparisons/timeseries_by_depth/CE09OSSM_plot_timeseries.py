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
thisyr = 2014

#vel = 'u'
vel = 'v'

moor = 'CE09OSSM' 
grid = 'cas7_t0_x4b'
# grid = 'cas7_t1_x4a'#only 2017

#####################################################################################
out_dir = Ldir['parent'] / 'LKH_output' / 'OOI'/ 'CE' / 'coastal_moorings' / moor / 'velocity' / 'obs_model' / 'plots' / 'timeseries_by_depth'
if os.path.exists(out_dir)==False:
    Lfun.make_dir(out_dir, clean = False)

########## indirs ###################################################################### 
mfd_in = Ldir['parent'] / 'LKH_data' / 'OOI'/ 'CE' / 'coastal_moorings' / moor / 'velocity' / 'mfd' / 'daily'
sb_in = Ldir['parent'] / 'LKH_data' / 'OOI'/ 'CE' / 'coastal_moorings' / moor / 'velocity' / 'surfacebuoy' / 'daily'
nsif_in = Ldir['parent'] / 'LKH_data' / 'OOI'/ 'CE' / 'coastal_moorings' / moor / 'velocity' / 'nsif' / 'daily'

in_dir = Ldir['parent'] / 'LO_output' / 'extract' / grid / 'moor' / 'OOI_WA_SM' 
fn = moor + '_' + str(thisyr) + '.01.01_' + str(thisyr) + '.12.31.nc'
LO_in = posixpath.join(in_dir,fn)
LO_ds = xr.open_dataset(LO_in, decode_times=True)  

########## MODEL ###################################################################### 
LOot = pd.to_datetime(LO_ds.ocean_time)
LOz = np.nanmean(LO_ds.z_rho.values,axis=0)

LO1 = LO_ds[vel].values[:,28]
LO7 = LO_ds[vel].values[:,-4]
LO20 = LO_ds[vel].values[:,-7]
#LO65 = np.nanmean(LO_ds[vel].values[:,(18,19)],axis=1)
LO80 = np.nanmean(LO_ds[vel].values[:,(17,18)],axis=1)
LO200 = np.nanmean(LO_ds[vel].values[:,(11,12)],axis=1)
#LO300 = LO_ds[vel].values[:,8]
LO500 = LO_ds[vel].values[:,2]

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
# plot surface -1 and -7 
plt.close('all')
fs=12
plt.rc('font', size=fs)
fig1 = plt.figure(figsize=(14,8))
fig1.set_size_inches(14,8, forward=False)

# load and plot surface buoy OBS 
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
ax1.set_title(moor +': ' + str(thisyr), loc = 'left')
ax1.set_ylabel(vel+' m/s')

labels = ax1.xaxis.get_ticklabels() # Get the x-axis labels
for i, label in enumerate(labels): # Iterate through labels and hide every other one
    if i % 2 != 0:
        label.set_visible(False)

ax1.minorticks_on()
ax1.grid(which='major')

ax1.plot(LOot,LO1,color = 'Crimson', marker='none', alpha=0.8, linestyle='-', linewidth = 2, label = grid)

if vel == 'u':
    plt.text(datetime(thisyr,2,1), -0.4, grid, fontdict={'size': 10, 'weight': 'bold', 'color': 'Crimson'})
if vel == 'v':
    plt.text(datetime(thisyr,5,1), -0.4, grid, fontdict={'size': 10, 'weight': 'bold', 'color': 'Crimson'})

plt.text(datetime(thisyr,3,5), -0.4, 'OBS(-1m)', fontdict={'size': 10, 'weight': 'bold', 'color': 'DodgerBlue'})

#ax1.legend(fontsize=10,loc = 'lower left')
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

plt.text(datetime(thisyr,3,5), -0.4, 'OBS(-7m)', fontdict={'size': 10, 'weight': 'bold', 'color': 'DodgerBlue'})

#ax2.legend(fontsize=10,loc = 'lower left')
#ax2.set_xticklabels([])

figname1 = moor+'_'+vel+'_'+str(thisyr)+'_timeseries_surface.png'
fig1.savefig(out_dir / figname1)
print('saved surface plot')
del ds, dt, yrs, mdx, ot, var

################################################
# organize nsif ADCP
fn_in = posixpath.join(nsif_in, (moor+'_nsif_ADCP_DAILY.nc'))
dsn = xr.open_dataset(fn_in, decode_times=True)  
dt = pd.to_datetime(dsn.ocean_time,format = '%YYYY-%m-%d HH:MM',errors='coerce')
yrs = dt.year
mdx1 = np.where(yrs==thisyr)
otn = dt[mdx1]
nvar = dsn[vel].values[mdx1]
ZN = dsn.z.values[1,:]
nsif20 = nvar[:,np.where(ZN==-20)[0]].squeeze()
#nsif65 = nvar[:,4]
nsif80 = nvar[:,np.where(ZN==-80)[0]].squeeze()

# organize mfd ADCP
fn_in = posixpath.join(mfd_in, (moor+'_mfd_ADCP_DAILY.nc'))
dsm = xr.open_dataset(fn_in, decode_times=True)  
dt = pd.to_datetime(dsm.ocean_time,format = '%YYYY-%m-%d HH:MM',errors='coerce')
yrs = dt.year
mdx2 = np.where(yrs==thisyr)
otm = dt[mdx2]
mvar = dsm[vel].values[mdx2]
ZM = dsm.z.values[1,:]
#mfd65 = mvar[:,-1]
mfd80 = mvar[:,np.where(ZM==-80)[0]].squeeze()
mfd200 = mvar[:,np.where(ZM==-200)[0]].squeeze()
#mfd300 = np.nanmean(mvar[:,(13,14)],axis=1)
mfd500 = mvar[:,0].squeeze()

################################################
#plot midlayer -20 -80 -200 -500m
plt.close('all')
fs=12
plt.rc('font', size=fs)
fig2 = plt.figure(figsize=(14,8))
fig2.set_size_inches(14,8, forward=False)

ax1 = plt.subplot2grid((5,1), (0,0), colspan=1,rowspan=1)
ax1.axhline(y=0, color='LightGrey', linestyle='-',alpha=0.8)
ax1.plot(LOot,LO20,color = 'Crimson', marker='none', alpha=0.8, linestyle='-', linewidth = 2)
ax1.plot(otn,nsif20,color = 'DodgerBlue', marker='none', alpha=0.8, linestyle='-', linewidth = 2)
ax1.set_title(moor +': ' + str(thisyr), loc = 'left')

if vel == 'u':
    plt.text(datetime(thisyr,2,1), -0.4, grid, fontdict={'size': 10, 'weight': 'bold', 'color': 'Crimson'})
if vel == 'v':
    plt.text(datetime(thisyr,5,1), -0.4, grid, fontdict={'size': 10, 'weight': 'bold', 'color': 'Crimson'})

plt.text(datetime(thisyr,3,5), -0.35, 'OBS(-20m)', fontdict={'size': 11, 'weight': 'bold', 'color': 'DodgerBlue'})

ax2 = plt.subplot2grid((5,1), (1,0), colspan=1,rowspan=1)
ax2.axhline(y=0, color='LightGrey', linestyle='-',alpha=0.8)
ax2.plot(LOot,LO80,color = 'Crimson', marker='none', alpha=0.8, linestyle='-', linewidth = 2)
ax2.plot(otn,nsif80,color = 'DodgerBlue', marker='none', alpha=0.8, linestyle='-', linewidth = 2)
ax2.plot(otm,mfd80,color = 'Navy', marker='none', alpha=0.8, linestyle=':', linewidth = 2)

plt.text(datetime(thisyr,3,5), -0.35, 'NSIF (-80m)', fontdict={'size': 11, 'weight': 'bold', 'color': 'DodgerBlue'})
plt.text(datetime(thisyr,4,5), -0.35, 'MFD (-80m)', fontdict={'size': 11, 'weight': 'bold', 'color': 'Navy'})

ax3 = plt.subplot2grid((5,1), (2,0), colspan=1,rowspan=1)
ax3.axhline(y=0, color='LightGrey', linestyle='-',alpha=0.8)
ax3.plot(LOot,LO200,color = 'Crimson', marker='none', alpha=0.8, linestyle='-', linewidth = 2)
ax3.plot(otm,mfd200,color = 'Navy', marker='none', alpha=0.8, linestyle='-', linewidth = 2)

plt.text(datetime(thisyr,3,5), -0.35, 'MFD (-200m)', fontdict={'size': 11, 'weight': 'bold', 'color': 'Navy'})

ax4 = plt.subplot2grid((5,1), (3,0), colspan=1,rowspan=1)
ax4.axhline(y=0, color='LightGrey', linestyle='-',alpha=0.8)
ax4.plot(LOot,LO500,color = 'Crimson', marker='none', alpha=0.8, linestyle='-', linewidth = 2)
ax4.plot(otm,mfd500,color = 'Navy', marker='none', alpha=0.8, linestyle='-', linewidth = 2)

if vel == 'u':
    plt.text(datetime(thisyr,3,5), -0.08, 'OBS (-500m)', fontdict={'size': 11, 'weight': 'bold', 'color': 'Navy'})
if vel == 'v':
    plt.text(datetime(thisyr,3,5), -0.3, 'OBS (-500m)', fontdict={'size': 11, 'weight': 'bold', 'color': 'Navy'})


ax1.set_xlim(date_array[0], date_array[-1])
ax1.set_xticks(date_array)
ax1.set_ylim(-0.5,0.5)
ax1.set_yticks(np.arange(-0.5,0.6,0.25))
date_format = mdates.DateFormatter('%b%d')
ax1.xaxis.set_major_formatter(date_format)
ax1.set_ylabel(vel+' m/s')
ax1.minorticks_on()
ax1.grid(which='major')
ax1.set_xticklabels([])

ax2.set_xlim(date_array[0], date_array[-1])
ax2.set_xticks(date_array)
ax2.set_ylim(-0.5,0.5)
ax2.set_yticks(np.arange(-0.5,0.6,0.25))
date_format = mdates.DateFormatter('%b%d')
ax2.xaxis.set_major_formatter(date_format)
ax2.set_ylabel(vel+' m/s')
ax2.minorticks_on()
ax2.grid(which='major')
ax2.set_xticklabels([])

ax3.set_xlim(date_array[0], date_array[-1])
ax3.set_xticks(date_array)
ax3.set_ylim(-0.5,0.5)
ax3.set_yticks(np.arange(-0.5,0.6,0.25))
date_format = mdates.DateFormatter('%b%d')
ax3.xaxis.set_major_formatter(date_format)
ax3.set_ylabel(vel+' m/s')
ax3.minorticks_on()
ax3.grid(which='major')
ax3.set_xticklabels([])

ax4.set_xlim(date_array[0], date_array[-1])
ax4.set_xticks(date_array)
if vel == 'u':
    ax4.set_ylim(-0.1,0.1)
    ax4.set_yticks(np.arange(-0.1,0.11,0.05))
if vel == 'v':
    ax4.set_ylim(-0.5,0.5)
    ax4.set_yticks(np.arange(-0.5,0.6,0.25))
date_format = mdates.DateFormatter('%b%d')
ax4.xaxis.set_major_formatter(date_format)
ax4.set_ylabel(vel+' m/s')
ax4.minorticks_on()
ax4.grid(which='major')

labels4 = ax4.xaxis.get_ticklabels() # Get the x-axis labels
for i, label in enumerate(labels4): # Iterate through labels and hide every other one
    if i % 2 != 0:
        label.set_visible(False)

fig2.tight_layout()

figname2 = moor+'_'+vel+'_'+str(thisyr)+'_timeseries_ADCP.png'
fig2.savefig(out_dir / figname2)
print('saved ADCP plot')