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
from cmcrameri import cm as cm2
#import cmocean
import matplotlib.dates as mdates
from datetime import datetime
import pandas as pd

tt0 = time()
 
# 1 load datasets; assign values
dsm1 = xr.open_dataset('/Users/katehewett/Documents/LO_code/testing_plotting/clip_coords/shelf_mask_15_200m_coastal_clip.nc')
mask_rho = dsm1.mask_rho.values                # 0 = land 1 = water
mask_shelf = dsm1.mask_shelf.values            # 0 = nope 1 = shelf
xrho = dsm1['Lon'].values
yrho = dsm1['Lat'].values
h = dsm1['h'].values
del dsm1 

##Kate this is sloppy clean it up 
ds = xr.open_dataset('/Users/katehewett/Documents/LKH_output/tests/percent_hypoxic/2017_2022_bottom_water_hypoxic_percentages.nc')

threshold = 'hyp'    # choices mild corrosive severe 

winter = ds[threshold+'_winter'].values
spring = ds[threshold+'_spring'].values
summer = ds[threshold+'_summer'].values
fall = ds[threshold+'_fall'].values

winter[(mask_shelf==0)]=np.nan
spring[(mask_shelf==0)]=np.nan
summer[(mask_shelf==0)]=np.nan
fall[(mask_shelf==0)]=np.nan

threshold = 'severe'

winter2 = ds[threshold+'_winter'].values
spring2 = ds[threshold+'_spring'].values
summer2 = ds[threshold+'_summer'].values
fall2 = ds[threshold+'_fall'].values

winter2[(mask_shelf==0)]=np.nan
spring2[(mask_shelf==0)]=np.nan
summer2[(mask_shelf==0)]=np.nan
fall2[(mask_shelf==0)]=np.nan


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
fig.set_size_inches(18,10, forward=False)

# map
axsu = plt.subplot2grid((3,4), (0,2), colspan=1,rowspan=3)
pfun.add_coast(axsu)
pfun.dar(axsu)
axsu.axis([-128, -123, 42.75, 49.75])
axsu.set_xticks([-128, -127, -126, -125, -124, -123])

cpsu = axsu.pcolormesh(xrho, yrho, summer2,vmin=0,vmax=100,cmap=cm2.roma_r)
cbaxes = inset_axes(axsu, width="4%", height="70%", loc='lower left')
fig.colorbar(cpsu, cax=cbaxes, orientation='vertical')

axsu.contour(xrho,yrho,h, [80],colors=['black'], linewidths=1, linestyles='solid',alpha=0.4)

axsu.set_title('July - September')
axsu.set_ylabel('Latitude')
axsu.set_xlabel('Longitude')
axsu.set_xticklabels(['-128',' ','-126',' ','-124',' '])
axsu.tick_params(axis = "x", labelsize = 14, labelrotation = 0)

fig.savefig('/Users/katehewett/Documents/LKH_output/tests/percent_hypoxic/summer_percent_severehypoxic.png', dpi=720)


