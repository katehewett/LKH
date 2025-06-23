''' 
compare diff
'''

import os
import sys
import pandas as pd
import numpy as np
import xarray as xr
import posixpath
from datetime import datetime
from datetime import datetime
from matplotlib.colors import BoundaryNorm

import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

from lo_tools import Lfun

Ldir = Lfun.Lstart()

thisyr = 2017 # only 2017 
moor = 'CE09OSSM'
var = 'u'
#loco = 'mfd'
loco = 'nsif'

#####################################################################################
out_dir = Ldir['parent'] / 'LKH_output' / 'OOI'/ 'CE' / 'coastal_moorings' / moor / 'velocity' / 'obs_model' / 'plots' / 'pcolor_comparisons'
if os.path.exists(out_dir)==False:
    Lfun.make_dir(out_dir, clean = False)

########## MODEL ######################################################################
grid1 = 'cas7_t0_x4b'
in_dir = Ldir['parent'] / 'LO_output' / 'extract' / grid1 / 'moor' / 'OOI_WA_SM' 
fn1 = moor + '_' + str(thisyr) + '.01.01_' + str(thisyr) + '.12.31.nc'
LO_in = posixpath.join(in_dir,fn1)
LO_ds = xr.open_dataset(LO_in, decode_times=True)  

grid2 = 'cas7_t1_x4a'
in_dir2 = Ldir['parent'] / 'LO_output' / 'extract' / grid2 / 'moor' / 'OOI_WA_SM' 
fn2 = moor + '_' + str(thisyr) + '.01.01_' + str(thisyr) + '.12.31.nc'
LO2_in = posixpath.join(in_dir2,fn2)
LO2_ds = xr.open_dataset(LO2_in, decode_times=True)  

var1 = LO_ds[str(var)].values
var2 = LO2_ds[str(var)].values
NZ = np.shape(var1)[1]

z1 = LO_ds['z_rho'].values
z2 = LO2_ds['z_rho'].values

#d = var2 - var1 

if np.all(LO_ds.ocean_time==LO2_ds.ocean_time):
    o = pd.to_datetime(LO_ds.ocean_time)
else: print('mismatched time')
o = np.expand_dims(o,axis=1)
ot = np.tile(o,(1,NZ))

########## OBS ######################################################################
if loco == 'nsif':
    in_dir = Ldir['parent'] / 'LKH_data' / 'OOI' / 'CE' / 'coastal_moorings' / moor / 'velocity' / loco / 'daily'
    fn_in = posixpath.join(in_dir, (moor+'_nsif_ADCP_DAILY.nc')) 
elif loco == 'mfd':
    in_dir = Ldir['parent'] / 'LKH_data' / 'OOI' / 'CE' / 'coastal_moorings' / moor / 'velocity' / loco / 'daily'
    fn_in = posixpath.join(in_dir, (moor+'_mfd_ADCP_DAILY.nc')) 

ds = xr.open_dataset(fn_in, decode_times=True) 
o = pd.to_datetime(ds.ocean_time)
idx = np.where(o.year == thisyr)
T = o[idx]

varo = ds[str(var)].values[idx,:].squeeze()
zo = ds['z'].values[idx,:].squeeze()

NZo = np.shape(varo)[1]
TT = np.expand_dims(T,axis=1)
obs_time = np.tile(TT,(1,NZo))
del T, TT

#####################################################################################
plt.close('all')
fs=12
plt.rc('font', size=fs)
fig = plt.figure(figsize=(14,8))
fig.set_size_inches(14,8, forward=False)

levels = np.arange(-0.2,0.201,0.01)
cmap = plt.get_cmap('RdBu')
cmap.set_extremes(over = 'Navy',under='Maroon')
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=False)

ax1 = plt.subplot2grid((3,1), (0,0), colspan=1,rowspan=1)
pc1 = ax1.pcolor(ot,z1,var1, cmap=cmap, norm = norm)
#ax1.axhline(y=0, color='k', linestyle=':',alpha=0.8)
cbar1 = fig.colorbar(pc1, location='right', orientation='vertical',label = str(var)+' m/s',extend = 'both')
cbar1.set_ticks([-0.2,-0.1,0,0.1,0.2])

ax2 = plt.subplot2grid((3,1), (1,0), colspan=1,rowspan=1)
pc2 = ax2.pcolor(ot,z2,var2, cmap=cmap, norm = norm)
cbar2 = fig.colorbar(pc2, location='right', orientation='vertical',label = str(var)+' m/s',extend = 'both')
cbar2.set_ticks([-0.2,-0.1,0,0.1,0.2])

ax3 = plt.subplot2grid((3,1), (2,0), colspan=1,rowspan=1)
pc3 = ax3.pcolor(obs_time,zo,varo, cmap=cmap, norm = norm)
cbar3 = fig.colorbar(pc3, location='right', orientation='vertical',label = str(var)+' m/s',extend='both')
cbar3.set_ticks(np.arange(-0.2,0.201,0.1))

if loco == 'mfd':
    yy = np.arange(-500,-49,50)

if loco == 'nsif':
    yy = np.arange(-85,-14,5)

axes_list = fig.get_axes()
for idx in range(0,6,2):
    axes_list[idx].axvline(x = datetime(2017,2,1), color = 'k', label = 'axvline - full height')
    axes_list[idx].axvline(x = datetime(2017,3,1), color = 'k', label = 'axvline - full height')
    axes_list[idx].axvline(x = datetime(2017,4,1), color = 'k', label = 'axvline - full height')
    axes_list[idx].axvline(x = datetime(2017,5,1), color = 'k', label = 'axvline - full height')
    axes_list[idx].axvline(x = datetime(2017,6,1), color = 'k', label = 'axvline - full height')
    axes_list[idx].axvline(x = datetime(2017,7,1), color = 'k', label = 'axvline - full height')
    axes_list[idx].axvline(x = datetime(2017,8,1), color = 'k', label = 'axvline - full height')
    axes_list[idx].axvline(x = datetime(2017,9,1), color = 'k', label = 'axvline - full height')
    axes_list[idx].axvline(x = datetime(2017,10,1), color = 'k', label = 'axvline - full height')
    axes_list[idx].axvline(x = datetime(2017,11,1), color = 'k', label = 'axvline - full height')
    axes_list[idx].axvline(x = datetime(2017,12,1), color = 'k', label = 'axvline - full height')

    axes_list[idx].set_ylim(np.min(yy),np.max(yy))
    axes_list[idx].set_yticks(yy)

    axes_list[idx].set_ylabel('Z [m]')

ax1.set_xticklabels([])
ax2.set_xticklabels([])

ax1.set_title(moor + ': '+grid1,loc='left')
ax2.set_title(grid2,loc='left')
if loco == 'mfd':
    ax3.set_title('bottom ADCP, upward facing @ ' + loco,loc='left')
if loco == 'nsif':
    ax3.set_title('ADCP, downward facing @ ' + loco,loc='left')
fig.tight_layout()

figname = 'obs_model_pcolor_comparision_' + moor+'_'+var+'_'+str(thisyr)+'.png'
fig.savefig(out_dir / figname)
print('saved')

sys.exit()

plt.close('all')
fs=12
plt.rc('font', size=fs)
fig = plt.figure(figsize=(14,8))
fig.set_size_inches(14,8, forward=False)

ax3 = plt.subplot2grid((3,1), (2,0), colspan=1,rowspan=1)
ax3.plot(obs_time[~np.isnan(varo)],zo[~np.isnan(varo)], '.')