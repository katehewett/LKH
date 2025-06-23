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
moor = 'CE06ISSM'
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

var1 = LO_ds[str(var)].values
var2 = LO2_ds[str(var)].values
NZ = np.shape(var1)[1]

z1 = np.nanmean(LO_ds['z_rho'].values,axis=0)
z2 = np.nanmean(LO2_ds['z_rho'].values,axis=0)

d = var2 - var1 

if np.all(LO_ds.ocean_time==LO2_ds.ocean_time):
    o = pd.to_datetime(LO_ds.ocean_time)
else: print('mismatched time')

o = np.expand_dims(o,axis=1)
ot = np.tile(o,(1,NZ))

zstring = np.arange(0,30,1)
df = pd.DataFrame({'ocean_time':pd.to_datetime(LO_ds.ocean_time)}) 
df[zstring] = d
df = df.set_index('ocean_time')
vmean = df.resample('ME').mean()
vmin = df.resample('ME').min()
vmax = df.resample('ME').max()

month_name = vmean.index.strftime("%B")

################################################
plt.close('all')
fs=12
plt.rc('font', size=fs)
fig = plt.figure(figsize=(14,8))
fig.set_size_inches(14,8, forward=False)

yy = np.arange(-29,1,5)
yy = np.append(yy,0,axis=None)

if var == 'u':
    xx = np.arange(-0.4,0.5,0.2)
if var == 'v': 
    xx = np.arange(-0.6,0.7,0.2)

for idx in range(0,12):
    if idx < 4: # 0 1 2 3 
        plt.subplot2grid((3,4), (0,idx), colspan=1,rowspan=1)
    elif idx < 8: # 4 5 6 7 
        plt.subplot2grid((3,4), (1,idx-4), colspan=1,rowspan=1)
    elif idx < 12: # 8 9 10 11 
        plt.subplot2grid((3,4), (2,idx-8), colspan=1,rowspan=1)

    plt.fill_betweenx(z1,vmin.iloc[idx],vmax.iloc[idx],facecolor = 'seagreen',edgecolor ='None', alpha=0.3)

    plt.plot(vmean.iloc[idx],z1,color = 'Green', linestyle = '-',linewidth=2, alpha=0.8) 

    plt.ylim(np.min(yy),np.max(yy))
    plt.yticks(yy)
    plt.xlim(np.min(xx),np.max(xx))
    plt.xticks(xx)

    if idx==0:
        plt.ylabel('Z [m]')
        if var == 'u':
            plt.text(-0.38, -4, moor, fontdict={'size': 10, 'weight': 'bold', 'color': 'black'})
            plt.text(-0.38, -7, str(thisyr), fontdict={'size': 10, 'weight': 'bold', 'color': 'black'})
            plt.text(-0.38, -25, grid1 + ' v1', fontdict={'size': 10, 'color': 'DodgerBlue'})
            plt.text(-0.38, -28, grid2+ ' v2', fontdict={'size': 10, 'color': 'Crimson'})
            plt.text(0.1, -28, 'diff = v2 - v1', fontdict={'size': 10, 'color': 'Green'})
        if var == 'v':
            plt.text(-0.58, -4, moor, fontdict={'size': 10, 'weight': 'bold', 'color': 'black'})
            plt.text(-0.58, -7, str(thisyr), fontdict={'size': 10, 'weight': 'bold', 'color': 'black'})
            plt.text(-0.58, -25, grid1 + ' v1', fontdict={'size': 10, 'color': 'DodgerBlue'})
            plt.text(-0.58, -28, grid2+ ' v2', fontdict={'size': 10, 'color': 'Crimson'})
            plt.text(0.15, -28, 'diff = v2 - v1', fontdict={'size': 10, 'color': 'Green'})

    if (idx==4) | (idx ==8) | (idx == 3) | (idx == 7) | (idx == 11) :
        plt.ylabel('Z [m]')

    if (idx == 3) | (idx == 7) | (idx == 11) | (idx == 2) | (idx == 6) | (idx == 10): 
        plt.gca().yaxis.set_label_position('right')
        plt.gca().yaxis.tick_right()

    if (idx == 3) | (idx == 7) | (idx == 11): 
        plt.gca().yaxis.tick_right()
        plt.gca().yaxis.set_label_position("right")

    if (idx == 1) | (idx == 5) | (idx == 9) | (idx == 2) | (idx == 6) | (idx == 10): 
        plt.gca().set_yticklabels([])
    
    if idx < 8: 
        plt.gca().set_xticklabels([])

    if idx > 7: 
        plt.xlabel(var + ' [m/s] min/mean/max ')

    plt.gca().axvline(x=0,color = 'Black',linestyle = ':')
    plt.title(month_name[idx])

fig.tight_layout()

figname = 'LO_profile_diffvel_' + moor+'_'+var+'_'+str(thisyr)+'.png'
fig.savefig(out_dir / figname)
print('saved')