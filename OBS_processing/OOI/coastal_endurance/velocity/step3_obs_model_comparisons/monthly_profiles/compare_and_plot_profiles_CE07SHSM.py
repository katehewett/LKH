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
moor = 'CE07SHSM'
loco = 'mfd'
#loco = 'nsif'

var = 'v'

#####################################################################################
out_dir = Ldir['parent'] / 'LKH_output' / 'OOI'/ 'CE' / 'coastal_moorings' / moor / 'velocity' / 'obs_model' / 'plots' / 'monthly_profiles'
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

z1 = np.nanmean(LO_ds['z_rho'].values,axis=0)
z2 = np.nanmean(LO2_ds['z_rho'].values,axis=0)

if np.all(LO_ds.ocean_time==LO2_ds.ocean_time):
    o = pd.to_datetime(LO_ds.ocean_time)
else: print('mismatched time')
o = np.expand_dims(o,axis=1)
ot = np.tile(o,(1,NZ))

zstring = np.arange(0,30,1)
df = pd.DataFrame({'ocean_time':pd.to_datetime(LO_ds.ocean_time)}) 
df[zstring] = var1
df = df.set_index('ocean_time')
vmean = df.resample('ME').mean()
vmin = df.resample('ME').min()
vmax = df.resample('ME').max()

df2 = pd.DataFrame({'ocean_time':pd.to_datetime(LO2_ds.ocean_time)}) 
df2[zstring] = var2
df2 = df2.set_index('ocean_time')
vmean2 = df2.resample('ME').mean()
vmin2 = df2.resample('ME').min()
vmax2 = df2.resample('ME').max()

month_name = vmean2.index.strftime("%B")
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

zstring2 = zo[1,:]
z3 = np.copy(zstring2)

df3 = pd.DataFrame({'ocean_time':T}) 
df3[zstring2] = varo
df3 = df3.set_index('ocean_time')
vmean3 = df3.resample('ME').mean()
vmin3 = df3.resample('ME').min()
vmax3 = df3.resample('ME').max()

################################################
plt.close('all')
fs=12
plt.rc('font', size=fs)
fig = plt.figure(figsize=(14,8))
fig.set_size_inches(14,8, forward=False)

yy = np.arange(-85,1,10)
yy = np.append(yy,0,axis=None)

if var == 'u':
    xx = np.arange(-0.4,0.5,0.2)
if var == 'v': 
    xx = np.arange(-0.8,0.9,0.4)

for idx in range(0,12):
    if idx < 4: # 0 1 2 3 
        plt.subplot2grid((3,4), (0,idx), colspan=1,rowspan=1)
    elif idx < 8: # 4 5 6 7 
        plt.subplot2grid((3,4), (1,idx-4), colspan=1,rowspan=1)
    elif idx < 12: # 8 9 10 11 
        plt.subplot2grid((3,4), (2,idx-8), colspan=1,rowspan=1)

    plt.fill_betweenx(z1,vmin.iloc[idx],vmax.iloc[idx],facecolor = 'DodgerBlue',edgecolor ='None', alpha=0.3)
    plt.fill_betweenx(z2,vmin2.iloc[idx],vmax2.iloc[idx],facecolor = 'Crimson',edgecolor ='None', alpha=0.3)
    plt.fill_betweenx(z3,vmin3.iloc[idx],vmax3.iloc[idx],facecolor = 'Green',edgecolor ='None', alpha=0.3)

    plt.plot(vmean.iloc[idx],z1,color = 'Navy', linestyle = '-',linewidth=2, alpha=0.8)
    plt.plot(vmean2.iloc[idx],z2,color = 'Crimson', linestyle = '-',linewidth=2, alpha=0.8)    
    plt.plot(vmean3.iloc[idx],z3,color = 'LightGreen', linestyle = '-',linewidth=2, alpha=0.8)  

    plt.ylim(np.min(yy),np.max(yy))
    plt.yticks(yy)
    plt.xlim(np.min(xx),np.max(xx))
    plt.xticks(xx)

    if idx==0:
        plt.ylabel('Z [m]')
        if var == 'u':
            plt.text(-0.38, -15, moor, fontdict={'size': 10, 'weight': 'bold', 'color': 'black'})
            plt.text(-0.38, -22, str(thisyr), fontdict={'size': 10, 'weight': 'bold', 'color': 'black'})
            plt.text(0.10, -55, grid1, fontdict={'size': 10, 'color': 'DodgerBlue'})
            plt.text(0.10, -65, grid2, fontdict={'size': 10, 'color': 'Crimson'})
            plt.text(0.1, -75, loco, fontdict={'size': 10, 'color': 'Green'})
        if var == 'v':
            plt.text(-0.78, -15, moor, fontdict={'size': 10, 'weight': 'bold', 'color': 'black'})
            plt.text(-0.78, -22, str(thisyr), fontdict={'size': 10, 'weight': 'bold', 'color': 'black'})
            plt.text(-0.78, -55, grid1, fontdict={'size': 10, 'color': 'DodgerBlue'})
            plt.text(-0.78, -65, grid2, fontdict={'size': 10, 'color': 'Crimson'})
            plt.text(0.4, -65, loco, fontdict={'size': 10, 'color': 'Green'})

    if idx == 8: 
        if var == 'v':
            plt.text(0.2, -80, '+north', fontdict={'size': 10, 'color': 'black'})
        if var == 'u':
            plt.text(0.2, -80, '+east', fontdict={'size': 10, 'color': 'black'})

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

    if var == 'v': 
        plt.gca().axvline(x=-0.4,color = 'LightGrey',linestyle = ':')
        plt.gca().axvline(x=0.4,color = 'LightGrey',linestyle = ':')

    plt.title(month_name[idx])

fig.tight_layout()

figname = 'monthly_profile_comparision_' + moor+'_'+var+'_'+loco+'_'+str(thisyr)+'.png'
fig.savefig(out_dir / figname)
print('saved')