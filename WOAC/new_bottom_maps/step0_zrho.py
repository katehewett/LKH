'''
This is part of processing required to create time duration plot
of bottom water 

Step 0:
Run this code after running extract_box_chunks.py 
for job_list == OA_indicators. 

this code opens a history file and calculates the z_rhos 

'''

import sys
import argparse
from lo_tools import Lfun, zrfun, zfun
from subprocess import Popen as Po
from subprocess import PIPE as Pi
import os
import xarray as xr
from xarray import open_dataset, Dataset
import numpy as np
from time import time

import gsw

# one lowpassed file 
fn_name = '/Users/katehewett/Documents/LO_roms/cas7_t0_x4b/f2017.12.12/lowpassed.nc'
G, S, T = zrfun.get_basic_info(fn_name)
Lon = G['lon_rho'][0,:]
Lat = G['lat_rho'][:,0]
z_rho, z_w = zrfun.get_z(G['h'], 0*G['h'], S) # use 0 for SSH
z_rho = z_rho[0,:,:].squeeze()

aa = [-125.5, -123.5, 42.75, 48.75]
lon0, lon1, lat0, lat1 = aa

# OA_indicators lat/lon from jobs_list
def check_bounds(lon, lat):
    # error checking
    if (lon < Lon[0]) or (lon > Lon[-1]):
        print('ERROR: lon out of bounds ')
        sys.exit()
    if (lat < Lat[0]) or (lat > Lat[-1]):
        print('ERROR: lat out of bounds ')
        sys.exit()
    # get indices
    ilon = zfun.find_nearest_ind(Lon, lon)
    ilat = zfun.find_nearest_ind(Lat, lat)
    return ilon, ilat

ilon0, ilat0 = check_bounds(lon0, lat0)
ilon1, ilat1 = check_bounds(lon1, lat1)


"""
Saving variables that don't change over time - this is here to make the cat step faster
- h is bathymetric depth 
- ocean_time is a vector of time; [lowpass is converted to UTC and centered at 1200 UTC; daily is diff] 
- Lat and Lon on rho points 
"""

iz_rho = z_rho[ilat0:ilat1+1,ilon0:ilon1+1]
lon_rho = G['lon_rho'][ilat0:ilat1+1,ilon0:ilon1+1]
lat_rho = G['lat_rho'][ilat0:ilat1+1,ilon0:ilon1+1]
mask_rho = G['mask_rho'][ilat0:ilat1+1,ilon0:ilon1+1]
ih = G['h'][ilat0:ilat1+1,ilon0:ilon1+1]

ds1 = Dataset()

ds1['lat_rho'] = (('eta_rho', 'xi_rho'),lat_rho,{'units':'degree_north'})
ds1['lat_rho'].attrs['standard_name'] = 'grid_latitude_at_cell_center'
ds1['lat_rho'].attrs['long_name'] = 'latitude of RHO-points'
ds1['lat_rho'].attrs['field'] = 'lat_rho'

ds1['lon_rho'] = (('eta_rho', 'xi_rho'),lon_rho,{'units': 'degree_east'})
ds1['lon_rho'].attrs['standard_name'] = 'grid_longitude_at_cell_center'
ds1['lon_rho'].attrs['long_name'] = 'longitude of RHO-points'
ds1['lon_rho'].attrs['field'] = 'lon_rho'
 
ds1['h'] = (('eta_rho', 'xi_rho'),ih,{'units': 'm'})
ds1['h'].attrs['standard_name'] = 'sea_floor_depth'
ds1['h'].attrs['long_name'] = 'time_independent bathymetry'
ds1['h'].attrs['field'] = 'bathymetry'
ds1['h'].attrs['grid'] =  'cas7_t0_x4b'

ds1['mask_rho'] = (('eta_rho', 'xi_rho'),mask_rho,{'units': 'm'})
ds1['mask_rho'].attrs['standard_name'] = 'land_sea_mask_at_cell_center'
ds1['mask_rho'].attrs['long_name'] = 'mask on RHO-points'
ds1['mask_rho'].attrs['flag_values'] = np.array([0.,1.])
ds1['mask_rho'].attrs['flag_meanings'] = 'land water'
ds1['mask_rho'].attrs['grid'] =  'cas7_t0_x4b'

ds1['z_rho'] = (('eta_rho', 'xi_rho'),iz_rho,{'units': 'm'})
ds1['z_rho'].attrs['standard_name'] = 'z_rho'
ds1['z_rho'].attrs['long_name'] = 'vertical position on s_rho grid, positive up'
ds1['z_rho'].attrs['grid'] =  'cas7_t0_x4b'

sys.exit()
fn_final = '/Users/katehewett/Documents/LKH_output/WOAC/cas7_t0_x4b/bottom_maps/OA_indicators_z_rho.nc'
ds1.to_netcdf(fn_final)