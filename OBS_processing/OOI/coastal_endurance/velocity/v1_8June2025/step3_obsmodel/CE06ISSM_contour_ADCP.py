'''
contour plot for CE06ISSM

obs are packed in LO format and have similar timesteps, so can match and search

'''

import os
import sys
import pandas as pd
import numpy as np
import xarray as xr
import posixpath
from datetime import timedelta
import pickle 

import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.pyplot as plt
import cmcrameri.cm as cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from lo_tools import Lfun

Ldir = Lfun.Lstart()

# processed data location
model2_in =  Ldir['parent'] / 'LO_output' / 'extract' / 'cas7_t1_x4a' / 'moor' / 'OOI_WA_SM' / 'CE06ISSM_2017.01.01_2017.12.31.nc'
model1_in =  Ldir['parent'] / 'LO_output' / 'extract' / 'cas7_t0_x4b' / 'moor' / 'OOI_WA_SM' / 'CE06ISSM_2017.01.01_2017.12.31.nc'

ds1 = xr.open_dataset(model1_in)
ds2 = xr.open_dataset(model2_in)

obs_in = Ldir['parent'] / 'LKH_data' / 'OOI' / 'CE' / 'coastal_moorings' / 'CE06ISSM' / 'velocity' / 'mfd' / 'daily' / 'CE06ISSM_mfd_ADCP_DAILY.nc'
ds = xr.open_dataset(obs_in)

fig_out_dir =  Ldir['parent'] / 'LKH_output' / 'OOI' / 'CE' / 'coastal_moorings' / 'CE06ISSM' / 'velocity' / 'mfd' / 'plots' / 'obs_mod'
if os.path.exists(fig_out_dir)==False:
    Lfun.make_dir(fig_out_dir, clean = False)

# isolate the 2017 data from ds (obs)
dt = pd.to_datetime(ds.ocean_time,format = '%YYYY-%m-%d HH:MM',errors='coerce')
idx = np.where(dt.year==2017)[0]

ot = dt[idx]
U = ds.u.values[:,idx]
V = ds.v.values[:,idx]
W = ds.w.values[:,idx]

z = ds.z.values

# PLOTTING
Zm1 = np.nanmean(ds1.z_rho.values,axis=0)
Zm2 = np.nanmean(ds2.z_rho.values,axis=0)

ot1 = pd.to_datetime(ds1.ocean_time.values,format = '%YYYY-%m-%d HH:MM',errors='coerce')
ot2 = pd.to_datetime(ds2.ocean_time.values,format = '%YYYY-%m-%d HH:MM',errors='coerce')
################################################################################### plot U's 
plt.close('all')
fs=14
plt.rc('font', size=fs)
fig = plt.figure(figsize=(18,10))
fig.set_size_inches(18,10, forward=False)

ax = plt.subplot2grid((3,2), (0,0), colspan=2,rowspan=1)
cpu = ax.pcolormesh(ot,z,U,vmin=-0.5,vmax=0.5,cmap=cm.roma_r)
fig.colorbar(cpu, ax=ax,label = 'u +east')
ax.set_ylabel('OBS Z [m]')
ax.set_title('CE06ISSM daily avg (or lowpass cas7_t0_x4b) vars')
ax.set_ylim([-30, 0])

ax1 = plt.subplot2grid((3,2), (1,0), colspan=2,rowspan=1)
cpu = ax1.pcolormesh(ot1,Zm1,ds1.u.values.T,shading = 'nearest', vmin=-0.5,vmax=0.5,cmap=cm.roma_r)
fig.colorbar(cpu, ax=ax1,label = 'u +east')
ax1.set_ylabel('cas7_t0_x4b Z [m]')
ax1.set_ylim([-30, 0])
plt.axhline(y = -5, color = 'k', linestyle = '--', label = 'surface clip')

ax2 = plt.subplot2grid((3,2), (2,0), colspan=2,rowspan=1)
cpu = ax2.pcolormesh(ot2,Zm2,ds2.u.values.T,shading = 'nearest', vmin=-0.5,vmax=0.5,cmap=cm.roma_r)
fig.colorbar(cpu, ax=ax2,label = 'u +east')
ax2.set_ylabel('cas7_t1_x4a Z [m]')
ax2.set_ylim([-30, 0])
plt.axhline(y = -5, color = 'k', linestyle = '--', label = 'surface clip')

fig.tight_layout()

figname = 'CE06ISSM_2017_u_pcolor_obsmod.png'
fig.savefig(fig_out_dir / figname)

################################################################################### plot V's 
plt.close('all')
fs=14
plt.rc('font', size=fs)
fig = plt.figure(figsize=(18,10))
fig.set_size_inches(18,10, forward=False)

ax = plt.subplot2grid((3,2), (0,0), colspan=2,rowspan=1)
cpu = ax.pcolormesh(ot,z,V,vmin=-1,vmax=1,cmap=cm.roma_r)
fig.colorbar(cpu, ax=ax,label = 'v +north')
ax.set_ylabel('OBS Z [m]')
ax.set_title('CE06ISSM daily avg (or lowpass cas7_t0_x4b) vars')
ax.set_ylim([-30, 0])

ax1 = plt.subplot2grid((3,2), (1,0), colspan=2,rowspan=1)
cpu = ax1.pcolormesh(ot1,Zm1,ds1.v.values.T,shading = 'nearest', vmin=-1,vmax=1,cmap=cm.roma_r)
fig.colorbar(cpu, ax=ax1,label = 'v')
ax1.set_ylabel('cas7_t0_x4b Z [m]')
ax1.set_ylim([-30, 0])
plt.axhline(y = -5, color = 'k', linestyle = '--', label = 'surface clip')

ax2 = plt.subplot2grid((3,2), (2,0), colspan=2,rowspan=1)
cpu = ax2.pcolormesh(ot2,Zm2,ds2.v.values.T,shading = 'nearest', vmin=-1,vmax=1,cmap=cm.roma_r)
fig.colorbar(cpu, ax=ax2,label = 'v')
ax2.set_ylabel('cas7_t1_x4a Z [m]')
ax2.set_ylim([-30, 0])
plt.axhline(y = -5, color = 'k', linestyle = '--', label = 'surface clip')

fig.tight_layout()

figname = 'CE06ISSM_2017_v_pcolor_obsmod.png'
fig.savefig(fig_out_dir / figname)

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