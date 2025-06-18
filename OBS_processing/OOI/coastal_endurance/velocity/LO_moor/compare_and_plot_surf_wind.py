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

v1 = LO_ds['v'].values[:,-1]
v2 = LO2_ds['v'].values[:,-1]
u1 = LO_ds['u'].values[:,-1]
u2 = LO2_ds['u'].values[:,-1]

z1 = np.nanmean(LO_ds['z_rho'].values,axis=0)
z2 = np.nanmean(LO2_ds['z_rho'].values,axis=0)

Uwind = LO_ds['Uwind'].values 
Vwind = LO_ds['Vwind'].values 
WIND = (Uwind**2 + Vwind**2)**(1/2)
wDIRradians = np.arctan2(Vwind, Uwind) #radiansr
wDIR = wDIRradians * (180/np.pi) #blowing towards

U1 = (u1**2 + v1**2)**(1/2)
DIRradians1 = np.arctan2(v1, u1) #radiansr
DIR1 = DIRradians1 * (180/np.pi) #blowing towards

U2 = (u2**2 + v2**2)**(1/2)
DIRradians2 = np.arctan2(v2, u2) #radiansr
DIR2 = DIRradians2 * (180/np.pi) #blowing towards / 
#d = var2 - var1 

if np.all(LO_ds.ocean_time==LO2_ds.ocean_time):
    ot = pd.to_datetime(LO_ds.ocean_time)
else: print('mismatched time')


################################################
plt.close('all')
fs=12
plt.rc('font', size=fs)
fig = plt.figure(figsize=(14,8))
fig.set_size_inches(14,8, forward=False)

yy = np.arange(0,0.7,0.1)
yw = np.arange(0,21,5)
#yy = np.append(yy,0,axis=None)

#xx = np.arange(-0.4,0.5,0.2)

ax1 = plt.subplot2grid((2,3), (0,0), colspan=2,rowspan=1)

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

plt.fill_between(ot,0,WIND*0.03,facecolor = 'Grey', edgecolor = 'None', alpha=0.5)
ax1.plot(ot,U1,color = 'DodgerBlue', linestyle = '-',linewidth=2, alpha=0.5)

ax1.set_title(moor + ' mag.vel @ surface bin + 3% wind speed')
ax1.set_ylim(np.min(yy),np.max(yy))
ax1.set_yticks(yy)
ax1.set_ylabel(grid1)

ax1.set_xlim(datetime(2017,1,1),datetime(2018,1,1))

##
ax2 = plt.subplot2grid((2,3), (1,0), colspan=2,rowspan=1)

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

plt.fill_between(ot,0,WIND*0.03,facecolor = 'Grey', edgecolor = 'None', alpha=0.5)
ax2.plot(ot,U2,color = 'Crimson', linestyle = '-',linewidth=2, alpha=0.5)

ax2.set_ylim(np.min(yy),np.max(yy))
ax2.set_yticks(yy)
ax2.set_ylabel(grid2)

ax2.set_xlim(datetime(2017,1,1),datetime(2018,1,1))

ax3 = plt.subplot2grid((2,3), (0,2), colspan=1,rowspan=1)
ax3.plot(WIND,U1,color = 'DodgerBlue', linestyle = 'none',marker = '.', alpha=0.5)
ax3.set_ylim(np.min(yy),np.max(yy))
ax3.set_yticks(yy)
ax3.set_xlim(np.min(yw),np.max(yw))
ax3.set_xticks(yw)


ax4 = plt.subplot2grid((2,3), (1,2), colspan=1,rowspan=1)
ax4.plot(WIND,U2,color = 'Crimson', linestyle = 'none',marker = '.', alpha=0.5)
ax4.set_ylim(np.min(yy),np.max(yy))
ax4.set_yticks(yy)
ax4.set_xlim(np.min(yw),np.max(yw))
ax4.set_xticks(yw)

figname = 'LO_surf_comparision_' + moor+'_'+str(thisyr)+'_SPEED.png'
fig.savefig(out_dir / figname)
print('saved')