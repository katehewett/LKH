"""
Plot the shelf mask for the whole LO domain and then clip to the 
job list of interest. For this work it is the OA_indicators job
listed under /LO_user/extract/corrosive_volume/job_list.py

We want to clip the old shelf mask (whole LO domain) 
to fit the new job list, s/t we can use it to mask our output
for the OA_indicators work. 

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
from xarray import Dataset

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from cmcrameri import cm as cm2
#import cmocean
import matplotlib.dates as mdates
from datetime import datetime
import pandas as pd

tt0 = time()
 
# 1 load datasets; assign values
dsm1 = xr.open_dataset('/Users/katehewett/Documents/LKH_data/shelf_masks/LO_domain/shelf_mask_15_200m_coastal_clip.nc')
mask_rho = dsm1.mask_rho.values                # 0 = land 1 = water
mask_shelf = dsm1.mask_shelf.values            # 0 = nope 1 = shelf
xrho = dsm1['Lon'].values
yrho = dsm1['Lat'].values
h = dsm1['h'].values
#del dsm1 

# open one file to get the lat lon of the extraction for 
# the OA_indicators job:
aa = [-125.5, -123.5, 42.75, 48.75]
dse = xr.open_dataset('/Users/katehewett/Documents/LO_output/extract/cas7_t0_x4b/corrosive_volume/OA_indicators_2013.01.01_2013.12.31/OA_indicators_corrosive_volume_2013.01.01_2013.12.31.nc')

Y = yrho[:,0]
X = xrho[0,:]

ilon0 = zfun.find_nearest_ind(X,dse['lon_rho'].values[0,0])
ilon1 = zfun.find_nearest_ind(X,dse['lon_rho'].values[0,-1])

ilat0 = zfun.find_nearest_ind(Y,dse['lat_rho'].values[0,0])
ilat1 = zfun.find_nearest_ind(Y,dse['lat_rho'].values[-1,0]) 

smolx = xrho[ilat0:ilat1+1,ilon0:ilon1+1]
smoly = yrho[ilat0:ilat1+1,ilon0:ilon1+1]
smask_shelf = mask_shelf[ilat0:ilat1+1,ilon0:ilon1+1]
h2 = h[ilat0:ilat1+1,ilon0:ilon1+1]
mask_rho2 = mask_rho[ilat0:ilat1+1,ilon0:ilon1+1]

#Rcolors = ['#73210D','#9C6B2F','#C5B563','#A9A9A9','#77BED0','#4078B3','#112F92'] #roma w grey not green #idx 3
    
plt.close('all')
fs=16
plt.rc('font', size=fs)
fig = plt.figure(figsize=(18,10))
fig.set_size_inches(18,10, forward=False)

axw = plt.subplot2grid((3,4), (0,0), colspan=1,rowspan=3)
pfun.add_coast(axw)
pfun.dar(axw)
axw.axis([-128, -123, 42.75, 49.75])
axw.plot(dse['lon_rho'],dse['lat_rho'],color='pink',marker='.', linestyle='none')
axw.plot(xrho[mask_shelf==1],yrho[mask_shelf==1],color='blue',marker='.', linestyle='none')
axw.plot(smolx[smask_shelf==1],smoly[smask_shelf==1],color='green',marker='.', linestyle='none')
axw.contour(xrho,yrho,h, [80,200],colors=['black'], linewidths=1, linestyles='solid',alpha=0.4)
axw.set_xticks([-128, -127, -126, -125, -124, -123])

axw.set_title('Area of Calculation')
axw.set_ylabel('Latitude')
axw.set_xlabel('Longitude')
axw.set_xticklabels(['-128',' ','-126',' ','-124',' '])
axw.tick_params(axis = "x", labelsize = 14, labelrotation = 0)

dsm2 = Dataset()

dsm2['lat_rho'] = (('eta_rho', 'xi_rho'),smoly,{'units':'degree_north'})
dsm2['lat_rho'].attrs['standard_name'] = 'grid_latitude_at_cell_center'
dsm2['lat_rho'].attrs['long_name'] = 'latitude of RHO-points'
dsm2['lat_rho'].attrs['field'] = 'lat_rho'

dsm2['lon_rho'] = (('eta_rho', 'xi_rho'),smolx,{'units': 'degree_east'})
dsm2['lon_rho'].attrs['standard_name'] = 'grid_longitude_at_cell_center'
dsm2['lon_rho'].attrs['long_name'] = 'longitude of RHO-points'
dsm2['lon_rho'].attrs['field'] = 'lon_rho'
 
dsm2['h'] = (('eta_rho', 'xi_rho'),h2,{'units': 'm'})
dsm2['h'].attrs['standard_name'] = 'sea_floor_depth'
dsm2['h'].attrs['long_name'] = 'time_independent bathymetry'
dsm2['h'].attrs['field'] = 'bathymetry'
dsm2['h'].attrs['grid'] =  'cas7_t0_x4b'

dsm2['mask_rho'] = (('eta_rho', 'xi_rho'),mask_rho2,{'units': 'm'})
dsm2['mask_rho'].attrs['standard_name'] = 'land_sea_mask_at_cell_center'
dsm2['mask_rho'].attrs['long_name'] = 'mask on RHO-points'
dsm2['mask_rho'].attrs['flag_values'] = np.array([0.,1.])
dsm2['mask_rho'].attrs['flag_meanings'] = 'land water'
dsm2['mask_rho'].attrs['grid'] =  'cas7_t0_x4b'

dsm2['mask_shelf'] = (('eta_rho', 'xi_rho'),smask_shelf,{'units': 'm'})
dsm2['mask_shelf'].attrs['standard_name'] = 'shelf_mask_at_cell_center'
dsm2['mask_shelf'].attrs['long_name'] = 'mask on RHO-points'
dsm2['mask_shelf'].attrs['flag_values'] = np.array([0.,1.])
dsm2['mask_shelf'].attrs['flag_meanings'] = 'notshelf shelf'
dsm2['mask_shelf'].attrs['grid'] =  'cas7_t0_x4b'

fn_o = '/Users/katehewett/Documents/LKH_data/shelf_masks/OA_indicators'
Lfun.make_dir(fn_o, clean=True)

fn_f = fn_o + '/OA_indicators_shelf_mask_15_200m_noEstuaries.nc'
dsm2.to_netcdf(fn_f)





