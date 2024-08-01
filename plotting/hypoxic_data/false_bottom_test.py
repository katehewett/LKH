'''
After running extract_hypoxic_volume_rev1.py 
we have 2 files: one with a false bottom of -200m 
and the other using the full wc. 
Running this test code to look at the difference quickly 

'''

# imports
from lo_tools import Lfun, zfun
from lo_tools import plotting_functions as pfun

import os 
import sys
import xarray as xr
import netCDF4 as nc
from time import time
import numpy as np
import pandas as pd
import posixpath

import matplotlib.pyplot as plt
from cmcrameri import cm as cmc
import matplotlib.dates as mdates
from datetime import datetime

Ldir = Lfun.Lstart()

fn_i =  '/Users/katehewett/Documents/LO_output/extract/cas7_t0_x4b/hypoxic_volume/LO_2014.01.01_2014.12.31'
fn_a = posixpath.join(fn_i,('LO_hypoxic_volume_lowpass_2014.01.01_2014.12.31.nc'))
fn_b = fn_i + ('/LO_hypoxic_volume_lowpass_2014.01.01_2014.12.31_false_bottom.nc')

ds = xr.open_dataset(fn_a, decode_times=True)
ds_fb = xr.open_dataset(fn_b, decode_times=True)

hyp = ds['hyp_dz'].values[2,:,:]
fhyp = ds_fb['hyp_dz'].values[2,:,:]
ot = pd.to_datetime(ds['ocean_time'].values[2])

lat = ds['lat_rho']
lon = ds['lon_rho']

h = ds['h']
mask_rho = ds['mask_rho']

hyp[mask_rho == 0] = np.nan
fhyp[mask_rho == 0] = np.nan

plt.close('all')
fs=11
plt.rc('font', size=fs)
height_of_image = 18 
width_of_image = 10 
fig1 = plt.figure(figsize=(height_of_image,width_of_image))
fig1.set_size_inches(height_of_image,width_of_image, forward=False)

ax0 = plt.subplot2grid((1,3), (0,0), colspan=1,rowspan=1)
ax1 = plt.subplot2grid((1,3), (0,1), colspan=1,rowspan=1)
ax2 = plt.subplot2grid((1,3), (0,2), colspan=1,rowspan=1)

clim = [go for go in range(0,200+1,10)] # colorbar limits 
clim[0]=0.2     
tlim = [go for go in range(0,200+1,50)] # colorbar limits 
tlim[0]=0.2
smap = cmc.roma.with_extremes(over='Navy')
        
## all depths 
pfun.add_coast(ax0)
pfun.dar(ax0)
ax0.axis([-126, -123.5, 45, 49])
ax0.contour(lon,lat,h.values, [200],colors=['black'], linewidths=1, linestyles='solid',alpha=0.8)
ax0.contour(lon,lat,h.values, [80],colors=['grey'], linewidths=0.5, linestyles='solid',alpha=1)
ax0.set_xticks([-126,-126.5,-125,-125.5,-124,-124.5,-123,-123.5])
ax0.set_title('All depths: ' + str(ot.year) + '.' + str(ot.month) + '.' + str(ot.day))
ax0.xaxis.set_ticklabels(['',-126.5,'',-125.5,'',-124.5,'',-123.5])
  
cpm0 = ax0.contourf(lon, lat, hyp, clim , cmap=smap, extend = "max")   
fig1.tight_layout()

tcb = plt.gcf().colorbar(cpm0, ax=ax0, ticks = tlim, shrink = 0.3, label='hyp dz [m]')
tcb.ax.yaxis.set_ticks_position('right')
  
## false bottom 
pfun.add_coast(ax1)
pfun.dar(ax1)
ax1.axis([-126, -123.5, 45, 49])
ax1.contour(lon,lat,h.values, [200],colors=['black'], linewidths=1, linestyles='solid',alpha=0.4)
ax1.contour(lon,lat,h.values, [80],colors=['grey'], linewidths=0.5, linestyles='solid',alpha=1)
ax1.set_xticks([-126,-126.5,-125,-125.5,-124,-124.5,-123,-123.5])
ax1.set_title('false bottom -200m')
ax1.xaxis.set_ticklabels(['',-126.5,'',-125.5,'',-124.5,'',-123.5])
  
cpm1 = ax1.contourf(lon, lat, fhyp, clim , cmap=smap, extend = "max")
tcb = plt.gcf().colorbar(cpm1, ax=ax1, ticks = tlim, shrink = 0.3, label='hyp dz [m]')
tcb.ax.yaxis.set_ticks_position('right')

## diff
pfun.add_coast(ax2)
pfun.dar(ax2)
ax2.axis([-126, -123.5, 45, 49])
ax2.contour(lon,lat,h.values, [200],colors=['black'], linewidths=1, linestyles='solid',alpha=0.4)
ax2.contour(lon,lat,h.values, [80],colors=['grey'], linewidths=0.5, linestyles='solid',alpha=1)
ax2.set_xticks([-126,-126.5,-125,-125.5,-124,-124.5,-123,-123.5])
ax2.set_title('diff')
ax2.xaxis.set_ticklabels(['',-126.5,'',-125.5,'',-124.5,'',-123.5])

A = hyp - fhyp
cpm = ax2.contourf(lon, lat, A) 

cpm2 = ax2.contourf(lon, lat, A, clim , cmap=smap, extend = "max")
tcb = plt.gcf().colorbar(cpm2, ax=ax2, ticks = tlim, shrink = 0.3, label='diff [m]')
tcb.ax.yaxis.set_ticks_position('right')


figname = 'J_hyp_false_bot_test.png'
figpath = '/Users/katehewett/Documents/LKH_output/plotting/hypoxic_data/'
a = posixpath.join(figpath,figname)
fig1.savefig(a)

