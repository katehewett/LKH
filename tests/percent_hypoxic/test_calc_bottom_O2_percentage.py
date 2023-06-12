"""
Calculate percent time bottom water below a threshold for hypoxia 
takes about 2.5 minutes to run calculation for 6 years - one threshold 
~8-9 minutes to run all thresholds and 6 years 
"""

# imports
from lo_tools import Lfun, zfun
from lo_tools import plotting_functions as pfun

import sys 
import xarray as xr
import netCDF4 as nc
from time import time
import numpy as np

import matplotlib.dates as mdates
from datetime import datetime
import pandas as pd

tt0 = time()

# 1 load datasets; assign values
#fn1 = '/Users/katehewett/Documents/LO_roms/cas6_v0_live/f2022.08.08/ocean_his_0018.nc'
#dsm1 = xr.open_dataset(fn1, decode_times=False)
#mask_rho = dsm1.mask_rho.values.squeeze()      # 0 = land 1 = water
#del fn1
#del dsm1 

##Kate this is sloppy, you need to move this to LKH! 
#dsm = xr.open_dataset('/Users/katehewett/Documents/LO_code/testing_plotting/clip_coords/shelf_mask_15_200m_VIclip.nc')
#mask_shelf = dsm['mask_shelf'].values
#del dsm

Ldir = Lfun.Lstart()

fn_i = Ldir['LOo'] / 'extract' / 'cas6_v0_live' 
fn = fn_i / 'bottom_layer' / 'extractions_2017.01.01_2022.12.31' / 'Bottom_TSO2chem_2017.01.01_2022.12.31.nc'

ds = xr.open_dataset(fn, decode_times=False)     # vols

xrho = ds['Lon'].values
yrho = ds['Lat'].values
#h = ds['h'].values
oxygen = ds['oxygen'].values

ot = ds['ocean_time'].values
days = (ot - ot[0])/86400.
mdays = Lfun.modtime_to_mdate_vec(ot)
mdt = mdates.num2date(mdays) # list of datetimes of data

NT, NR, NC = np.shape(oxygen)              # vector of months and years for stats 
mmonth = np.nan * np.ones((NT))
myear = np.nan * np.ones((NT))
for jj in range(NT):
    mmonth[jj] = mdt[jj].month
    myear[jj] = mdt[jj].year

print('Time to load = %0.2f sec' % (time()-tt0))


tt0 = time()
# set values below hypoxic threshold to 1 and above to 0:
hmild = np.nan * np.ones([NT, NR, NC])
hhypoxic = np.nan * np.ones([NT, NR, NC])
hsevere = np.nan * np.ones([NT, NR, NC])
hanoxic = np.nan * np.ones([NT, NR, NC])
for t in range(NT):  
    print('Working on file %0.0f ...' % t)
    oxy = oxygen[t,:,:].squeeze()
    mi = np.ma.masked_where(oxy<=106.6,oxy)
    i = np.ma.masked_where(oxy<=60.9,oxy)
    s = np.ma.masked_where(oxy<=21.6,oxy)
    a = np.ma.masked_where(oxy<=0,oxy)
    rm = mi.mask.astype(int)
    ri = i.mask.astype(int)
    rs = s.mask.astype(int)
    ra = a.mask.astype(int)
    hmild[t,:,:] = rm
    hhypoxic[t,:,:] = ri
    hsevere[t,:,:] = rs
    hanoxic[t,:,:] = ra

# get season indicies
wdx = np.where((mmonth>=1) & (mmonth<=3)); wdx = np.squeeze(wdx)
spdx = np.where((mmonth>=4) & (mmonth<=6)); spdx = np.squeeze(spdx)
sudx = np.where((mmonth>=7) & (mmonth<=9)); sudx = np.squeeze(sudx)
fdx = np.where((mmonth>=10) & (mmonth<=12)); fdx = np.squeeze(fdx)

winter = {}
spring = {}
summer = {}
fall = {}

owinter = hmild[wdx,:,:]                                   # mild        
ospring = hmild[spdx,:,:]
osummer = hmild[sudx,:,:]
ofall = hmild[fdx,:,:] 

OW = np.nansum(owinter,axis=0)
OSp = np.nansum(ospring,axis=0)
OSu = np.nansum(osummer,axis=0)
OF = np.nansum(ofall,axis=0)
 
winter['mild'] = (OW/np.shape(owinter)[0])*100
spring['mild'] = (OSp/np.shape(ospring)[0])*100
summer['mild'] = (OSu/np.shape(osummer)[0])*100
fall['mild'] = (OF/np.shape(ofall)[0])*100

del OW, OSp, OSu, OF, owinter, ospring, osummer, ofall 

owinter = hhypoxic[wdx,:,:]                                   # intermediate        
ospring = hhypoxic[spdx,:,:]
osummer = hhypoxic[sudx,:,:]
ofall = hhypoxic[fdx,:,:] 

OW = np.nansum(owinter,axis=0)
OSp = np.nansum(ospring,axis=0)
OSu = np.nansum(osummer,axis=0)
OF = np.nansum(ofall,axis=0)
 
winter['hypoxic'] = (OW/np.shape(owinter)[0])*100
spring['hypoxic'] = (OSp/np.shape(ospring)[0])*100
summer['hypoxic'] = (OSu/np.shape(osummer)[0])*100
fall['hypoxic'] = (OF/np.shape(ofall)[0])*100

del OW, OSp, OSu, OF, owinter, ospring, osummer, ofall 

owinter = hsevere[wdx,:,:]                                   # severe        
ospring = hsevere[spdx,:,:]
osummer = hsevere[sudx,:,:]
ofall = hsevere[fdx,:,:] 

OW = np.nansum(owinter,axis=0)
OSp = np.nansum(ospring,axis=0)
OSu = np.nansum(osummer,axis=0)
OF = np.nansum(ofall,axis=0)
 
winter['severe'] = (OW/np.shape(owinter)[0])*100
spring['severe'] = (OSp/np.shape(ospring)[0])*100
summer['severe'] = (OSu/np.shape(osummer)[0])*100
fall['severe'] = (OF/np.shape(ofall)[0])*100

del OW, OSp, OSu, OF, owinter, ospring, osummer, ofall 

owinter = hanoxic[wdx,:,:]                                   # anoxic        
ospring = hanoxic[spdx,:,:]
osummer = hanoxic[sudx,:,:]
ofall = hanoxic[fdx,:,:] 

OW = np.nansum(owinter,axis=0)
OSp = np.nansum(ospring,axis=0)
OSu = np.nansum(osummer,axis=0)
OF = np.nansum(ofall,axis=0)
 
winter['anoxic'] = (OW/np.shape(owinter)[0])*100
spring['anoxic'] = (OSp/np.shape(ospring)[0])*100
summer['anoxic'] = (OSu/np.shape(osummer)[0])*100
fall['anoxic'] = (OF/np.shape(ofall)[0])*100

del OW, OSp, OSu, OF, owinter, ospring, osummer, ofall 

# put them in a dataset, ds1
NR, NC = winter['hypoxic'].shape        

ds1 = Dataset()

ds1['mild_winter'] = (('eta_rho', 'xi_rho'), winter['mild'], {'units':'%', 'long_name': 'percent time leq 106.6 millimole_oxygen m-3 JFM 2017-2022'})
ds1['mild_spring'] = (('eta_rho', 'xi_rho'), spring['mild'], {'units':'%', 'long_name': 'percent time leq 106.6 millimole_oxygen m-3 AMJ 2017-2022'})
ds1['mild_summer'] = (('eta_rho', 'xi_rho'), summer['mild'], {'units':'%', 'long_name': 'percent time leq 106.6 millimole_oxygen m-3 JuAS 2017-2022'})
ds1['mild_fall'] = (('eta_rho', 'xi_rho'), fall['mild'], {'units':'%', 'long_name': 'percent time leq 106.6 millimole_oxygen m-3 OND 2017-2022'})

ds1['hyp_winter'] = (('eta_rho', 'xi_rho'), winter['hypoxic'], {'units':'%', 'long_name': 'percent time leq 60.9 millimole_oxygen m-3 JFM 2017-2022'})
ds1['hyp_spring'] = (('eta_rho', 'xi_rho'), spring['hypoxic'], {'units':'%', 'long_name': 'percent time leq 60.9 millimole_oxygen m-3 AMJ 2017-2022'})
ds1['hyp_summer'] = (('eta_rho', 'xi_rho'), summer['hypoxic'], {'units':'%', 'long_name': 'percent time leq 60.9 millimole_oxygen m-3 JuAS 2017-2022'})
ds1['hyp_fall'] = (('eta_rho', 'xi_rho'), fall['hypoxic'], {'units':'%', 'long_name': 'percent time leq 60.9 millimole_oxygen m-3 OND 2017-2022'})

ds1['severe_winter'] = (('eta_rho', 'xi_rho'), winter['severe'], {'units':'%', 'long_name': 'percent time leq 21.6 millimole_oxygen m-3 JFM 2017-2022'})
ds1['severe_spring'] = (('eta_rho', 'xi_rho'), spring['severe'], {'units':'%', 'long_name': 'percent time leq 21.6 millimole_oxygen m-3 AMJ 2017-2022'})
ds1['severe_summer'] = (('eta_rho', 'xi_rho'), summer['severe'], {'units':'%', 'long_name': 'percent time leq 21.6 millimole_oxygen m-3 JuAS 2017-2022'})
ds1['severe_fall'] = (('eta_rho', 'xi_rho'), fall['severe'], {'units':'%', 'long_name': 'percent time leq 21.6 millimole_oxygen m-3 OND 2017-2022'})

ds1['anoxic_winter'] = (('eta_rho', 'xi_rho'), winter['anoxic'], {'units':'%', 'long_name': 'percent time zero oxygen JFM 2017-2022'})
ds1['anoxic_spring'] = (('eta_rho', 'xi_rho'), spring['anoxic'], {'units':'%', 'long_name': 'percent time zero oxygen AMJ 2017-2022'})
ds1['anoxic_summer'] = (('eta_rho', 'xi_rho'), summer['anoxic'], {'units':'%', 'long_name': 'percent time zero oxygen JuAS 2017-2022'})
ds1['anoxic_fall'] = (('eta_rho', 'xi_rho'), fall['anoxic'], {'units':'%', 'long_name': 'percent time zero oxygen OND 2017-2022'})

ds1.to_netcdf('/Users/katehewett/Documents/LKH_output/tests/percent_hypoxic/2017_2022_bottom_water_hypoxic_percentages.nc')

print('Time to calc seasonal percentage = %0.2f sec' % (time()-tt0))

#pwinter[(h>200) | (mask_shelf==0)]=np.nan
#pspring[(h>200) | (mask_shelf==0)]=np.nan
#psummer[(h>200) | (mask_shelf==0)]=np.nan
#pfall[(h>200) | (mask_shelf==0)]=np.nan






