''' 
compare diff
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
import cmcrameri.cm as cmc
from matplotlib.colors import BoundaryNorm

import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.pyplot as plt
import cmcrameri.cm as cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from lo_tools import Lfun

Ldir = Lfun.Lstart()

thisyr = 2017 # only 2017 
moor = 'CE09OSSM'
var = 'u'

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

out_dir = Ldir['parent'] / 'LKH_output' / 'OOI'/ 'CE' / 'coastal_moorings' / moor / 'velocity' / 'LO_moor' / 'plots'
if os.path.exists(out_dir)==False:
    Lfun.make_dir(out_dir, clean = False)

v1 = LO_ds['v'].values
v2 = LO2_ds['v'].values
u1 = LO_ds['u'].values
u2 = LO2_ds['u'].values

NZ = np.shape(v1)[1]

z1 = np.nanmean(LO_ds['z_rho'].values,axis=0)
z2 = np.nanmean(LO2_ds['z_rho'].values,axis=0)

vbar1 = v1[:,-1]
vbar2 = v2[:,-1]
ubar1 = u1[:,-1]
ubar2 = u2[:,-1]

#d = var2 - var1 

if np.all(LO_ds.ocean_time==LO2_ds.ocean_time):
    o = pd.to_datetime(LO_ds.ocean_time)
else: print('mismatched time')
o = np.expand_dims(o,axis=1)
ot = np.tile(o,(1,NZ))

################################################
plt.close('all')
fs=12
plt.rc('font', size=fs)
fig = plt.figure(figsize=(14,8))
fig.set_size_inches(14,8, forward=False)

yy = np.arange(-0.6,0.7,0.2)
yy = np.append(yy,0,axis=None)

#xx = np.arange(-0.4,0.5,0.2)

ax1 = plt.subplot2grid((2,1), (0,0), colspan=1,rowspan=1)

ax1.axvline(x = datetime(2017,2,1), color = 'lightgrey', label = 'axvline - full height')
ax1.axvline(x = datetime(2017,3,1), color = 'lightgrey', label = 'axvline - full height')
ax1.axvline(x = datetime(2017,4,1), color = 'lightgrey', label = 'axvline - full height')
ax1.axvline(x = datetime(2017,5,1), color = 'lightgrey', label = 'axvline - full height')
ax1.axvline(x = datetime(2017,6,1), color = 'lightgrey', label = 'axvline - full height')
ax1.axvline(x = datetime(2017,7,1), color = 'lightgrey', label = 'axvline - full height')
ax1.axvline(x = datetime(2017,8,1), color = 'lightgrey', label = 'axvline - full height')
ax1.axvline(x = datetime(2017,9,1), color = 'lightgrey', label = 'axvline - full height')
ax1.axvline(x = datetime(2017,10,1), color = 'lightgrey', label = 'axvline - full height')
ax1.axvline(x = datetime(2017,11,1), color = 'lightgrey', label = 'axvline - full height')
ax1.axvline(x = datetime(2017,12,1), color = 'lightgrey', label = 'axvline - full height')
ax1.axhline(y = 0, color = 'k', linestyle = ':') 

ax1.plot(ot,vbar1,color = 'DodgerBlue', linestyle = '-',linewidth=2, alpha=0.5)
ax1.plot(ot,vbar2,color = 'Crimson', linestyle = '-',linewidth=2, alpha=0.5)

ax1.set_title(moor + ' velocity @ surface bin')
ax1.set_ylim(np.min(yy),np.max(yy))
ax1.set_yticks(yy)
ax1.set_ylabel('v m/s')

ax1.set_xlim(datetime(2017,1,1),datetime(2018,1,1))

##
ax2 = plt.subplot2grid((2,1), (1,0), colspan=1,rowspan=1)

ax2.axvline(x = datetime(2017,2,1), color = 'lightgrey', label = 'axvline - full height')
ax2.axvline(x = datetime(2017,3,1), color = 'lightgrey', label = 'axvline - full height')
ax2.axvline(x = datetime(2017,4,1), color = 'lightgrey', label = 'axvline - full height')
ax2.axvline(x = datetime(2017,5,1), color = 'lightgrey', label = 'axvline - full height')
ax2.axvline(x = datetime(2017,6,1), color = 'lightgrey', label = 'axvline - full height')
ax2.axvline(x = datetime(2017,7,1), color = 'lightgrey', label = 'axvline - full height')
ax2.axvline(x = datetime(2017,8,1), color = 'lightgrey', label = 'axvline - full height')
ax2.axvline(x = datetime(2017,9,1), color = 'lightgrey', label = 'axvline - full height')
ax2.axvline(x = datetime(2017,10,1), color = 'lightgrey', label = 'axvline - full height')
ax2.axvline(x = datetime(2017,11,1), color = 'lightgrey', label = 'axvline - full height')
ax2.axvline(x = datetime(2017,12,1), color = 'lightgrey', label = 'axvline - full height')
ax2.axhline(y = 0, color = 'k', linestyle = ':') 

ax2.plot(ot,ubar1,color = 'DodgerBlue', linestyle = '-',linewidth=2, alpha=0.5)
ax2.plot(ot,ubar2,color = 'Crimson', linestyle = '-',linewidth=2, alpha=0.5)

ax2.set_ylim(np.min(yy),np.max(yy))
ax2.set_yticks(yy)
ax2.set_ylabel('u m/s')

ax2.set_xlim(datetime(2017,1,1),datetime(2018,1,1))

figname = 'LO_surf_comparision_' + moor+'_'+str(thisyr)+'.png'
fig.savefig(out_dir / figname)
print('saved')