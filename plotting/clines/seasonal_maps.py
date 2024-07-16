'''
plotting seasonal maps of 
N2max + depth 
sml 


'''

# imports
from lo_tools import Lfun, zfun
from lo_tools import plotting_functions as pfun
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import sys 
import xarray as xr
import netCDF4 as nc
from time import time
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from cmcrameri import cm as cm2
#import cmocean
import matplotlib.dates as mdates
from datetime import datetime
import pandas as pd


tt0 = time()

Ldir = Lfun.Lstart()

fn_o = Ldir['parent'] / 'plotting' / 'clines' 
fn_i = Ldir['LOo'] / 'extract' / 'cas7_t0_x4b' / 'clines' / 'shelf_box_2014.01.01_2014.12.31' 
fn_in = fn_i / 'shelf_box_pycnocline_2014.01.01_2014.12.31.nc'

ds = xr.open_dataset(fn_in, decode_times=True)     

ot = pd.to_datetime(ds.ocean_time.values)

xrho = ds['lon_rho'].values
yrho = ds['lat_rho'].values
mask_rho = ds['mask_rho'].values
h = ds['h'].values

var = ds['zSML'].values

mmonth = ot.month
myear = ot.year

wdx = np.where((mmonth>=1) & (mmonth<=3)); 
spdx = np.where((mmonth>=4) & (mmonth<=6)); 
sudx = np.where((mmonth>=7) & (mmonth<=9)); 
fdx = np.where((mmonth>=10) & (mmonth<=12)); 

winter = {}
spring = {}
summer = {}
fall = {}

owinter = var[wdx]                                   
ospring = var[spdx]
osummer = var[sudx]
ofall = var[fdx] 

masked_data = np.ma.masked_array(owinter, np.isnan(owinter)) 
OW = np.ma.average(masked_data, axis=0,keepdims=False)  
OW[mask_rho==0] = np.nan 

masked_data = np.ma.masked_array(ospring, np.isnan(ospring)) 
OSp = np.ma.average(masked_data, axis=0,keepdims=False)  
OSp[mask_rho==0] = np.nan 

masked_data = np.ma.masked_array(osummer, np.isnan(osummer)) 
OSu = np.ma.average(masked_data, axis=0,keepdims=False)  
OSu[mask_rho==0] = np.nan 

masked_data = np.ma.masked_array(ofall, np.isnan(ofall)) 
OF = np.ma.average(masked_data, axis=0,keepdims=False)  
OF[mask_rho==0] = np.nan 
 
winter['var_mean'] = OW
spring['var_mean'] = OSp
summer['var_mean'] = OSu
fall['var_mean'] = OF


plt.close('all')
fs=16
plt.rc('font', size=fs)
fig = plt.figure(figsize=(18,10))
fig.set_size_inches(18,10, forward=False)
# map

axw = plt.subplot2grid((3,4), (0,0), colspan=1,rowspan=3)
pfun.add_coast(axw)
pfun.dar(axw)
axw.axis([-128, -123, 42, 50])
axw.contour(xrho,yrho,h, [80],colors=['black'], linewidths=1, linestyles='solid',alpha=0.4)
axw.set_xticks([-128, -127, -126, -125, -124, -123])
#axw.grid(True)

axs = plt.subplot2grid((3,4), (0,1), colspan=1,rowspan=3)
pfun.add_coast(axs)
pfun.dar(axs)
axs.axis([-128, -123, 42, 50])
axs.contour(xrho,yrho,h, [80],colors=['black'], linewidths=1, linestyles='solid',alpha=0.4)
#axs.contour(xrho,yrho,h, [100, 200, 2000],
#colors=['grey','dimgrey','black'], linewidths=1, linestyles='solid')
axs.set_xticks([-128, -127, -126, -125, -124, -123])
#axs.grid(True)

axsu = plt.subplot2grid((3,4), (0,2), colspan=1,rowspan=3)
pfun.add_coast(axsu)
pfun.dar(axsu)
axsu.axis([-128, -123, 42, 50])
#axsu.contour(xrho,yrho,h, [40,80,130],
#colors=['dimgrey','black','dimgrey'], linewidths=0.5, linestyles='solid')
axsu.set_xticks([-128, -127, -126, -125, -124, -123])
#axsu.grid(True)

axf = plt.subplot2grid((3,4), (0,3), colspan=1,rowspan=3)
pfun.add_coast(axf)
pfun.dar(axf)
axf.axis([-128, -123, 42, 50])
axf.contour(xrho,yrho,h, [80],colors=['black'], linewidths=1, linestyles='solid',alpha=0.4)
#axf.contour(xrho,yrho,h, [100, 200, 2000],
#colors=['grey','dimgrey','black'], linewidths=1, linestyles='solid')
axf.set_xticks([-128, -127, -126, -125, -124, -123])
#axf.grid(True)

cpw = axw.pcolormesh(xrho, yrho, OW ,cmap=cm2.roma_r)
cps = axs.pcolormesh(xrho, yrho, OSp ,cmap=cm2.roma_r)
cpsu = axsu.pcolormesh(xrho, yrho, OSu ,cmap=cm2.roma_r)
cpf = axf.pcolormesh(xrho, yrho, OF ,cmap=cm2.roma_r)










