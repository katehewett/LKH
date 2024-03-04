"""
Open file for Natalie and check a field 
make a plot for her reference of salt and save
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
import matplotlib.dates as mdates
from datetime import datetime
import pandas as pd

#tt0 = time()
 
# 1 load datasets; assign values
dsm1 = xr.open_dataset('/Users/katehewett/Documents/LO_output/extract/cas6_v0_live/box/Freeman_OCNMS_2017.01.01_2017.12.31_chunks/Freeman_OCNMS_2017.01.01_2017.12.31.nc')
mask_rho = dsm1.mask_rho.values                # 0 = land 1 = water
xrho = dsm1['lon_rho'].values
yrho = dsm1['lat_rho'].values
h = dsm1['h'].values
salt = dsm1['salt'].values
salt_s = salt[94,29,:,:] # april surface

#del dsm1 

#print('Time to load = %0.2f sec' % (time()-tt0))

#tt0 = time()
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
axsu.axis([-126, -123, 45, 49])
axsu.set_xticks([-126, -125, -124, -123, -122])
axsu.set_yticks([45, 46, 47, 48, 49])

cpsu = axsu.pcolormesh(xrho, yrho, salt_s,vmin=25,vmax=35,cmap=cm2.roma_r)
cbaxes = inset_axes(axsu, width="4%", height="70%", loc='lower left')
fig.colorbar(cpsu, cax=cbaxes, orientation='vertical')

axsu.contour(xrho,yrho,h, [80],colors=['black'], linewidths=1, linestyles='solid',alpha=0.4)

axsu.set_title('Surface Salt')
axsu.set_ylabel('Latitude')
axsu.set_xlabel('Longitude')
axsu.set_xticklabels(['-126',' ','-124',' ','-122',' '])
axsu.tick_params(axis = "x", labelsize = 14, labelrotation = 0)

#fig.savefig('/Users/katehewett/Documents/LKH_output/tests/percent_hypoxic/summer_percent_severehypoxic.png', dpi=720)


