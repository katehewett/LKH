"""
Open percent time bottom water calculated using test_calc_bottom_O2_percentage.py
and mask shelf and plot 
"""

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
#import cmocean
import matplotlib.dates as mdates
from datetime import datetime
import pandas as pd

tt0 = time()

# 1 load datasets; assign values
fn1 = '/Users/katehewett/Documents/LO_roms/cas6_v0_live/f2022.08.08/ocean_his_0018.nc'
dsm1 = xr.open_dataset(fn1, decode_times=False)
mask_rho = dsm1.mask_rho.values.squeeze()      # 0 = land 1 = water
xrho = dsm1['lon_rho'].values
yrho = dsm1['lat_rho'].values
h = dsm1['h'].values
del fn1
del dsm1 

##Kate this is sloppy, you need to move this to LKH! 
dsm = xr.open_dataset('/Users/katehewett/Documents/LO_code/testing_plotting/clip_coords/shelf_mask_15_200m_VIclip.nc')
mask_shelf = dsm['mask_shelf'].values
del dsm

##Kate this is sloppy too 
ds = xr.open_dataset('/Users/katehewett/Documents/LKH_output/tests/percent_hypoxic/2017_2022_bottom_water_hypoxic_percentages.nc')

threshold = 'hyp'

winter = ds[threshold+'_winter'].values
spring = ds[threshold+'_spring'].values
summer = ds[threshold+'_summer'].values
fall = ds[threshold+'_fall'].values

winter[(mask_shelf==0) | (yrho < 42.75) | (yrho > 49.75)]=np.nan
spring[(mask_shelf==0) | (yrho < 42.75) | (yrho > 49.75)]=np.nan
summer[(mask_shelf==0) | (yrho < 42.75) | (yrho > 49.75)]=np.nan
fall[(mask_shelf==0) | (yrho < 42.75) | (yrho > 49.75)]=np.nan

threshold = 'severe'

winter2 = ds[threshold+'_winter'].values
spring2 = ds[threshold+'_spring'].values
summer2 = ds[threshold+'_summer'].values
fall2 = ds[threshold+'_fall'].values

winter2[(mask_shelf==0) | (yrho < 42.75) | (yrho > 49.75)]=np.nan
spring2[(mask_shelf==0) | (yrho < 42.75) | (yrho > 49.75)]=np.nan
summer2[(mask_shelf==0) | (yrho < 42.75) | (yrho > 49.75)]=np.nan
fall2[(mask_shelf==0) | (yrho < 42.75) | (yrho > 49.75)]=np.nan


print('Time to load = %0.2f sec' % (time()-tt0))

tt0 = time()
Rcolors = ['#73210D','#9C6B2F','#C5B563','#A9A9A9','#77BED0','#4078B3','#112F92'] #roma w grey not green #idx 3
    
# PLOTTING
#cmap = cmocean.cm.haline_r
#cmap = cm.jet_r

plt.close('all')
fs=16
plt.rc('font', size=fs)
fig = plt.figure(figsize=(18,10))

# map

axw = plt.subplot2grid((3,4), (0,0), colspan=1,rowspan=3)
pfun.add_coast(axw)
pfun.dar(axw)
axw.axis([-130, -122, 42, 52])
axw.contour(xrho,yrho,h, [100, 200, 2000],
colors=['grey','dimgrey','black'], linewidths=1, linestyles='solid')
axw.set_xticks([-130, -128, -126, -124, -122])
axw.grid(True)

axs = plt.subplot2grid((3,4), (0,1), colspan=1,rowspan=3)
pfun.add_coast(axs)
pfun.dar(axs)
axs.axis([-130, -122, 42, 52])
axs.contour(xrho,yrho,h, [100, 200, 2000],
colors=['grey','dimgrey','black'], linewidths=1, linestyles='solid')
axs.set_xticks([-130, -128, -126, -124, -122])
axs.grid(True)

axsu = plt.subplot2grid((3,4), (0,2), colspan=1,rowspan=3)
pfun.add_coast(axsu)
pfun.dar(axsu)
axsu.axis([-128, -122, 42, 50])
#axsu.contour(xrho,yrho,h, [40,80,130],
#colors=['dimgrey','black','dimgrey'], linewidths=0.5, linestyles='solid')
axsu.set_xticks([-128, -126, -124, -122])
#axsu.grid(True)

axf = plt.subplot2grid((3,4), (0,3), colspan=1,rowspan=3)
pfun.add_coast(axf)
pfun.dar(axf)
axf.axis([-130, -122, 42, 52])
axf.contour(xrho,yrho,h, [100, 200, 2000],
colors=['grey','dimgrey','black'], linewidths=1, linestyles='solid')
axf.set_xticks([-130, -128, -126, -124, -122])
axf.grid(True)
    
cpw = axw.pcolormesh(xrho, yrho, winter,vmin=0,vmax=6,cmap='jet')

cbaxes = inset_axes(axw, width="4%", height="40%", loc='lower left')
fig.colorbar(cpw, cax=cbaxes, orientation='vertical')
    
cps = axs.pcolormesh(xrho, yrho, spring,vmin=0,vmax=100,cmap='viridis')
cpsu = axsu.pcolormesh(xrho, yrho, summer,vmin=0,vmax=100,cmap='viridis')
cpf = axf.pcolormesh(xrho, yrho, fall,vmin=0,vmax=60,cmap='jet')

#ax.text(.33,.16,'100 m',color='grey',weight='bold',transform=ax.transAxes,ha='right')
#ax.text(.33,.13,'200 m',color='dimgrey',weight='bold',transform=ax.transAxes,ha='right')
#ax.text(.33,.1,'2000 m',color='black',weight='bold',transform=ax.transAxes,ha='right')
#ax.set_title('Area of Calculation')
axw.set_title('Winter')
axs.set_title('Spring')
axsu.set_title('Summer')
axf.set_title('Fall')

axw.set_ylabel('Latitude')

axs.set_yticklabels([])
axsu.set_yticklabels([])

axf.yaxis.set_label_position("right")
axf.yaxis.tick_right()
axf.set_ylabel('Latitude')

#axf.xaxis.tick_top()
#axs.xaxis.tick_top()

axw.set_xlabel('Longitude')
axsu.set_xlabel('Longitude')

axw.set_xticklabels(['-130',' ','-126',' ','-122'])
axs.set_xticklabels([' ','-128',' ','-124',' '])
#axsu.set_xticklabels(['-130',' ','-126',' ','-122'])
axf.set_xticklabels([' ','-128',' ','-124',' '])
   
axw.tick_params(axis = "x", labelsize = 14, labelrotation = 0)
axs.tick_params(axis = "x", labelsize = 14, labelrotation = 0)
axsu.tick_params(axis = "x", labelsize = 14, labelrotation = 0)
axf.tick_params(axis = "x", labelsize = 14, labelrotation = 0)


plt.colorbar(cpw,shrink=0.9,location='bottom')
plt.colorbar(cps,shrink=0.9,location='bottom')
plt.colorbar(cpsu,shrink=0.9,location='bottom')
plt.colorbar(cpf,shrink=0.9,location='bottom')

axsu.contour(xrho,yrho,summer2, [25, 75, 95],
colors=['white','black','red'], linewidths=1, linestyles='solid')   

#cp = ax.pcolormesh(xrho, yrho, summer,vmin=0,vmax=100,cmap='viridis')
#plt.colorbar(cp) # Add a colorbar to a plot

#ax.contour(xrho,yrho,SS, [20, 40, 60],
#colors=['black','black','black'], linewidths=1, linestyles='solid')

