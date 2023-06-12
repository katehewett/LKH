"""
Calculate percent time bottom water below a threshold for ARAG < = 1 
~9 minutes to run all thresholds and 6 years 
"""

# imports
from lo_tools import Lfun, zfun
from lo_tools import plotting_functions as pfun

import sys 
import xarray as xr
from xarray import open_dataset, Dataset
import netCDF4 as nc
from time import time
import numpy as np

import matplotlib.dates as mdates
from datetime import datetime
import pandas as pd

tt0 = time()

Ldir = Lfun.Lstart()

fn_i = Ldir['LOo'] / 'extract' / 'cas6_v0_live' 
fn = fn_i / 'bottom_layer' / 'extractions_2017.01.01_2022.12.31' / 'Bottom_TSO2chem_2017.01.01_2022.12.31.nc'

ds = xr.open_dataset(fn, decode_times=False)     # vols

xrho = ds['Lon'].values
yrho = ds['Lat'].values
#h = ds['h'].values
ARAG = ds['ARAG'].values

ot = ds['ocean_time'].values
days = (ot - ot[0])/86400.
mdays = Lfun.modtime_to_mdate_vec(ot)
mdt = mdates.num2date(mdays) # list of datetimes of data

NT, NR, NC = np.shape(ARAG)              # vector of months and years for stats 
mmonth = np.nan * np.ones((NT))
myear = np.nan * np.ones((NT))
for jj in range(NT):
    mmonth[jj] = mdt[jj].month
    myear[jj] = mdt[jj].year

print('Time to load = %0.2f sec' % (time()-tt0))


tt0 = time()
# set values below corrosive threshold to 1 and above to 0:
hmild = np.nan * np.ones([NT, NR, NC])
hcorr = np.nan * np.ones([NT, NR, NC])
hsevere = np.nan * np.ones([NT, NR, NC])

for t in range(NT):  
    print('Working on file %0.0f ...' % t)
    A = ARAG[t,:,:].squeeze()
    mi = np.ma.masked_where(A<=2,A)
    i = np.ma.masked_where(A<=1,A)
    s = np.ma.masked_where(A<=0.8,A)

    rm = mi.mask.astype(int)
    ri = i.mask.astype(int)
    rs = s.mask.astype(int)

    hmild[t,:,:] = rm
    hcorr[t,:,:] = ri
    hsevere[t,:,:] = rs

# get season indicies
wdx = np.where((mmonth>=1) & (mmonth<=3)); wdx = np.squeeze(wdx)
spdx = np.where((mmonth>=4) & (mmonth<=6)); spdx = np.squeeze(spdx)
sudx = np.where((mmonth>=7) & (mmonth<=9)); sudx = np.squeeze(sudx)
fdx = np.where((mmonth>=10) & (mmonth<=12)); fdx = np.squeeze(fdx)

winter = {}
spring = {}
summer = {}
fall = {}

owinter = hmild[wdx,:,:]                                   # mild ARAG <= 2   
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

owinter = hcorr[wdx,:,:]                                   # intermediate    ARAG<=1    
ospring = hcorr[spdx,:,:]
osummer = hcorr[sudx,:,:]
ofall = hcorr[fdx,:,:] 

OW = np.nansum(owinter,axis=0)
OSp = np.nansum(ospring,axis=0)
OSu = np.nansum(osummer,axis=0)
OF = np.nansum(ofall,axis=0)
 
winter['corrosive'] = (OW/np.shape(owinter)[0])*100
spring['corrosive'] = (OSp/np.shape(ospring)[0])*100
summer['corrosive'] = (OSu/np.shape(osummer)[0])*100
fall['corrosive'] = (OF/np.shape(ofall)[0])*100

del OW, OSp, OSu, OF, owinter, ospring, osummer, ofall 

owinter = hsevere[wdx,:,:]                                   # severe ARAG <= 0.5         
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


# put them in a dataset, ds1
NR, NC = winter['mild'].shape        

ds1 = Dataset()

ds1['mild_winter'] = (('eta_rho', 'xi_rho'), winter['mild'], {'units':'%', 'long_name': 'percent time ARAG leq 2 JFM 2017-2022'})
ds1['mild_spring'] = (('eta_rho', 'xi_rho'), spring['mild'], {'units':'%', 'long_name': 'percent time ARAG leq 2 AMJ 2017-2022'})
ds1['mild_summer'] = (('eta_rho', 'xi_rho'), summer['mild'], {'units':'%', 'long_name': 'percent time ARAG leq 2 JuAS 2017-2022'})
ds1['mild_fall'] = (('eta_rho', 'xi_rho'), fall['mild'], {'units':'%', 'long_name': 'percent time ARAG leq 2 OND 2017-2022'})

ds1['corrosive_winter'] = (('eta_rho', 'xi_rho'), winter['corrosive'], {'units':'%', 'long_name': 'percent time ARAG leq 1 JFM 2017-2022'})
ds1['corrosive_spring'] = (('eta_rho', 'xi_rho'), spring['corrosive'], {'units':'%', 'long_name': 'percent time ARAG leq 1 AMJ 2017-2022'})
ds1['corrosive_summer'] = (('eta_rho', 'xi_rho'), summer['corrosive'], {'units':'%', 'long_name': 'percent time ARAG leq 1 JuAS 2017-2022'})
ds1['corrosive_fall'] = (('eta_rho', 'xi_rho'), fall['corrosive'], {'units':'%', 'long_name': 'percent time ARAG leq 1 OND 2017-2022'})

ds1['severe_winter'] = (('eta_rho', 'xi_rho'), winter['severe'], {'units':'%', 'long_name': 'percent time ARAG leq 0.8 JFM 2017-2022'})
ds1['severe_spring'] = (('eta_rho', 'xi_rho'), spring['severe'], {'units':'%', 'long_name': 'percent time ARAG leq 0.8 AMJ 2017-2022'})
ds1['severe_summer'] = (('eta_rho', 'xi_rho'), summer['severe'], {'units':'%', 'long_name': 'percent time ARAG leq 0.8 JuAS 2017-2022'})
ds1['severe_fall'] = (('eta_rho', 'xi_rho'), fall['severe'], {'units':'%', 'long_name': 'percent time ARAG leq 0.8 OND 2017-2022'})

ds1.to_netcdf('/Users/katehewett/Documents/LKH_output/tests/percent_corrosive/2017_2022_bottom_water_corrosive_percentages.nc')

print('Time to calc seasonal percentage = %0.2f sec' % (time()-tt0))

#pwinter[(h>200) | (mask_shelf==0)]=np.nan
#pspring[(h>200) | (mask_shelf==0)]=np.nan
#psummer[(h>200) | (mask_shelf==0)]=np.nan
#pfall[(h>200) | (mask_shelf==0)]=np.nan






