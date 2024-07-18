'''
This code was written to extract the location of 
a mooring location from a box extraction
mooring loc = CE02SHSM 
data extracted from LO, job = NHL_transect 
for cas7_t0_x4b years 2014 - 2019 
(2019 = last full year processed when extracted 
box data; 2013 skipped for now)

and gsw was already run on the files in LKH_output SA/CT

'''

import xarray as xr
from xarray import open_dataset, Dataset
import numpy as np
import gsw
from lo_tools import Lfun, zfun, zrfun
from time import time
import sys
import gsw
import math 

#plotting things
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
import cmcrameri.cm as cmc

# when dealing with dates
from datetime import date, timedelta
import pandas as pd

Ldir = Lfun.Lstart()

fnn = 'NHL_transect_2014.01.01_2019.12.31'

in_dir = Ldir['parent'] / 'LKH_output'/ 'NHL' / 'explore_stratification' / fnn
fn_in = in_dir / (fnn + '_phys.nc')

testing = False

# 1. Find the index and simplify
tt0 = time()
ds = xr.open_dataset(fn_in, decode_times=True)

mlon = -124.304         # CE02SHSM location ~80m mooring
mlat = 44.6393

lat = ds.lat_rho
lon = ds.lon_rho

ilon = zfun.find_nearest_ind(lon, mlon)
ilat = zfun.find_nearest_ind(lat, mlat)

Lon = lon[ilat,ilon]
Lat = lat[ilat,ilon]

h = ds.h[ilat,ilon]
SA = ds.SA[:,:,ilat,ilon]
CT = ds.CT[:,:,ilat,ilon]
SIG0 = ds.SIG0[:,:,ilat,ilon]
SPICE = ds.SPICE0[:,:,ilat,ilon]
z_rho  = ds.z_rho[:,:,ilat,ilon]
z_w  = ds.z_w[:,:,ilat,ilon]

NT,NZ = SA.shape
NW = z_w.shape[1]

ot = pd.to_datetime(ds.ocean_time)    


plt.close('all')
fs=12
plt.rc('font', size=fs)
fig = plt.figure(figsize=(16,8))

ax0 = plt.subplot2grid((4,1),(0,0),colspan=1,rowspan=1)
ax1 = plt.subplot2grid((4,1),(1,0),colspan=1,rowspan=1)
ax2 = plt.subplot2grid((4,1),(2,0),colspan=1,rowspan=1)
ax3 = plt.subplot2grid((4,1),(3,0),colspan=1,rowspan=1)

X = np.expand_dims(ot,axis=1)
XX = np.tile(X,NZ)
del X

plt.tight_layout()

# plot temperature
axnum = 0
tmin = math.floor(np.min(CT.values))
tmax = math.ceil(np.max(CT.values))
tlevels = np.arange(tmin,tmax,0.1)

iCT = plt.gcf().axes[axnum].contourf(XX,z_rho,CT,tlevels,cmap=cmc.roma_r)
plt.gcf().axes[axnum].set_title('LO/NHL ~80m CE02SHSM')

tcb = plt.gcf().colorbar(iCT, ticks = [6,8,10,12,14,16,17], location='right', pad = 0.05, label='CT deg C')
ax0.contour(XX,z_rho,CT,[12.5], colors=['white'], linewidths=1, linestyles=':',alpha=1)

# plot salinity
axnum = 1
smin = math.floor(np.min(SA.values))
smax = math.ceil(np.max(SA.values))
slevels = np.arange(31.5,smax,0.1)
smap=cmc.roma_r.with_extremes(under='Navy')

iSA = plt.gcf().axes[axnum].contourf(XX,z_rho,SA,slevels,cmap=smap,extend="min")
ax1.contour(XX,z_rho,SA,[30], colors=['magenta'], linewidths=1, linestyles='solid',alpha=1)
ax1.contour(XX,z_rho,SA,[27], colors=['white'], linewidths=1, linestyles='solid',alpha=1)

scb = plt.gcf().colorbar(iSA, ticks = [31.5,32,33,34,35], location='right', pad = 0.05, label='SA')

# plot sigma
axnum = 2
SIG0min = math.floor(np.min(SIG0.values))
SIG0max = math.ceil(np.max(SIG0.values))
SIG0levels = np.arange(SIG0min,SIG0max,0.1)
SIG0map=cmc.roma_r.with_extremes(under='Navy')

iSIG0 = plt.gcf().axes[axnum].contourf(XX,z_rho,SIG0,SIG0levels,cmap=smap) #,extend="min")
ax2.contour(XX,z_rho,SIG0,[26.5], colors=['black'], linewidths=1, linestyles='solid',alpha=1)

SIG0cb = plt.gcf().colorbar(iSIG0, location='right', pad = 0.05, label='SIG0')

# plot spice
axnum = 3
SPICEmin = math.floor(np.min(SPICE.values))
SPICEmax = math.ceil(np.max(SPICE.values))
SPICElevels = np.arange(SPICEmin,SPICEmax,0.1)
SPICEmap=cmc.roma_r.with_extremes(under='Navy')

iSPICE = plt.gcf().axes[axnum].contourf(XX,z_rho,SPICE,SPICElevels,cmap=smap) #,extend="min")
#ax2.contour(XX,z_rho,SIG0,[26.5], colors=['black'], linewidths=1, linestyles='solid',alpha=1)

SPICEcb = plt.gcf().colorbar(iSPICE, location='right', pad = 0.05, label='SPICE')










