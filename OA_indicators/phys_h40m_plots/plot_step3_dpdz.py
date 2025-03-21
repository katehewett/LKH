"""
Plots ~h40m data mask from step2
across all years 
months are grouped and each month end label is the monthly average for that month 

run plot_step3_dpdz 

"""
# imports
from lo_tools import Lfun, zfun
from lo_tools import plotting_functions as pfun

import os 
import sys 
import argparse
import xarray as xr
import netCDF4 as nc
from time import time
import numpy as np
import pandas as pd
import pickle 
import matplotlib.pyplot as plt
from cmcrameri import cm
#import matplotlib.cm as cm
#import cmocean
import matplotlib.dates as mdates
from datetime import datetime
from matplotlib.colors import BoundaryNorm
import matplotlib.ticker as ticker
    
Ldir = Lfun.Lstart()

# load the shelf mask / assign vars 
fn_mask =  Ldir['parent'] / 'LKH_data' / 'shelf_masks' / 'OA_indicators' / 'OA_indicators_shelf_mask_15_200m_noEstuaries.nc'
dmask = xr.open_dataset(fn_mask) 
mask_shelf = dmask.mask_shelf 
h = dmask.h
xrho = dmask.lon_rho
yrho = dmask.lat_rho

# load the h mask to flag data between 35<h<45m
fn_hmask =  Ldir['parent'] / 'LKH_data' / 'shelf_masks' / 'OA_indicators' / 'OA_indicators_h40m_mask.nc'
hmask = xr.open_dataset(fn_hmask) 
mask_h40m = hmask.mask_h40m.values    # 0 outside; 1 h=35-45m 

# input location for pickled files 
fn_i = Ldir['parent'] / 'LKH_output' / 'OA_indicators' / 'cas7_t0_x4b' / 'rho_h40m_plots' 
fn_p = fn_i / 'dpdz_h40m' 

yr_list = [year for year in range(2013,2024)]
numyrs = len(yr_list)

plt.close('all')
fs=11
plt.rc('font', size=fs)
height_of_image = 15 
width_of_image = 8 
fig1 = plt.figure(figsize=(height_of_image,width_of_image))
fig1.set_size_inches(height_of_image,width_of_image, forward=False)
frac=0.047 * (height_of_image / (width_of_image/13))

# map
ax = plt.subplot2grid((2,3), (0,0), colspan=1, rowspan=1)
pfun.add_coast(ax)
pfun.dar(ax)
ax.axis([-125.5, -123.5, 42.75, 48.75])

ax.contour(xrho,yrho,h, [40],
colors=['white'], linewidths=1, linestyles='solid',alpha=0.4)
ax.contour(xrho,yrho,h, [80],
colors=['black'], linewidths=1, linestyles='solid',alpha=0.4)
ax.contour(xrho,yrho,h, [200],
colors=['black'], linewidths=1, linestyles='solid',alpha=0.6)
ax.contour(xrho,yrho,h, [1000],
colors=['black'], linewidths=1, linestyles='solid')
    
ax.set_title('h40m')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_xticks([-125.5, -124.5, -123.5])
ax.set_yticks([42.75,43,44,45,46,47,48,48.75])
ax.set_yticklabels(['42.75','43','44','45','46','47','48','48.75'])
ax.xaxis.set_ticklabels([-125.5, ' ' , -123.5])
ax.grid(False)

pts2 = ax.scatter(xrho*mask_h40m, yrho*mask_h40m, s=1, c = 'mediumseagreen')
plt.axhline(y = 48.4, color = 'Purple', linestyle = '-')

# Load the lat lon for one file 
pn = 'simple_dpdz_h40m_mean_2013_2023.pkl'
picklepath = fn_p/pn

if os.path.isfile(picklepath)==False:
    print('no file named: ' + pn)
    sys.exit()

with open(picklepath, 'rb') as fp3:
    A = pickle.load(fp3)
    print('loaded file')

ll = A['lat_rho'][0,:]
lat_rho = np.expand_dims(ll,axis=0) # just need the first row

y = A['lat_rho']
x = A['year_day']
dpdz = A['m_dpdz']

# bigger scale 
axp = plt.subplot2grid((2,3), (0,1), colspan=2) # surface
fig1.tight_layout()
#cmap = plt.get_cmap('RdBu')
cmap1=cm.batlow
levels = [0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4]
cmap1.set_extremes(over = 'Crimson',under='black')
norm1 = BoundaryNorm(levels, ncolors=cmap1.N, clip=False)
pcm = axp.pcolormesh(x, y, dpdz, cmap=cmap1, norm=norm1)

axp.set_yticklabels(['42.75','43','44','45','46','47','48','48.75'])
axp.set_xlim([1, 365])

plt.axhline(y = 48.75, color = 'k', label = 'axvline - full height')

plt.axvline(x = 1, color = 'k', label = 'axvline - full height')
plt.axvline(x = 32, color = 'k', label = 'axvline - full height')
plt.axvline(x = 61, color = 'k', label = 'axvline - full height')
plt.axvline(x = 92, color = 'k', label = 'axvline - full height')
plt.axvline(x = 122, color = 'k', label = 'axvline - full height')
plt.axvline(x = 153, color = 'k', label = 'axvline - full height')
plt.axvline(x = 183, color = 'k', label = 'axvline - full height')
plt.axvline(x = 214, color = 'k', label = 'axvline - full height')
plt.axvline(x = 245, color = 'k', label = 'axvline - full height')
plt.axvline(x = 275, color = 'k', label = 'axvline - full height')
plt.axvline(x = 306, color = 'k', label = 'axvline - full height')
plt.axvline(x = 336, color = 'k', label = 'axvline - full height')
plt.axvline(x = 365, color = 'k', label = 'axvline - full height')

plt.axhline(y = 48.375, color = 'Purple', linestyle = '-')

axp.set_xticks([1,32,61,92,122,153,183,214,245,275,306,336,365])

axp.grid(True)
cbar1 = fig1.colorbar(pcm, extend = 'both')




# clipped scale 
axp2 = plt.subplot2grid((2,3), (1,1), colspan=2) # surface
#cmap = plt.get_cmap('RdBu')
cmap2=cm.batlow
levels2 = [0,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.2]
cmap2.set_extremes(over = '#FBCDFA', under = '#06184F')
norm2 = BoundaryNorm(levels2, ncolors=cmap2.N, clip=False)
pcm2 = axp2.pcolormesh(x, y, dpdz, cmap=cmap2, norm=norm2)

axp2.set_yticklabels(['42.75','43','44','45','46','47','48','48.75'])
axp2.set_xlim([1, 365])

plt.axhline(y = 48.75, color = 'k', label = 'axvline - full height')

plt.axvline(x = 1, color = 'k', label = 'axvline - full height')
plt.axvline(x = 32, color = 'k', label = 'axvline - full height')
plt.axvline(x = 61, color = 'k', label = 'axvline - full height')
plt.axvline(x = 92, color = 'k', label = 'axvline - full height')
plt.axvline(x = 122, color = 'k', label = 'axvline - full height')
plt.axvline(x = 153, color = 'k', label = 'axvline - full height')
plt.axvline(x = 183, color = 'k', label = 'axvline - full height')
plt.axvline(x = 214, color = 'k', label = 'axvline - full height')
plt.axvline(x = 245, color = 'k', label = 'axvline - full height')
plt.axvline(x = 275, color = 'k', label = 'axvline - full height')
plt.axvline(x = 306, color = 'k', label = 'axvline - full height')
plt.axvline(x = 336, color = 'k', label = 'axvline - full height')
plt.axvline(x = 365, color = 'k', label = 'axvline - full height')

plt.axhline(y = 48.375, color = 'Purple', linestyle = '-')

axp2.set_xticks([1,32,61,92,122,153,183,214,245,275,306,336,365])

axp2.grid(True)
cbar2 = fig1.colorbar(pcm2, extend = 'both')


fig1.tight_layout()