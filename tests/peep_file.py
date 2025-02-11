"""
just looking at the values 
"""
# imports
from lo_tools import Lfun
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
import matplotlib.cm as cm
#import cmocean
import matplotlib.dates as mdates
from datetime import datetime
        
Ldir = Lfun.Lstart()

# organize and set paths before summing volumes 
#yr_list = [year for year in range(int(args.ys0), int(args.ys1)+1)]
#numyrs = len(yr_list)

# load the shelf mask / assign vars 
fn =  '/Users/katehewett/Documents/LO_output/extract/cas7_t0_x4b/box/aakritiv_surf_2020.01.01_2021.12.31_chunks/aakritiv_surf_2020.01.01_2021.12.31.nc'
ds = xr.open_dataset(fn) 
xrho = ds.lon_rho.values
yrho = ds.lat_rho.values
N = ds.NO3.values[1,:,:].squeeze()

plt.close('all')
fs=11
plt.rc('font', size=fs)
height_of_image = 15 
width_of_image = 8 
fig1 = plt.figure(figsize=(height_of_image,width_of_image))
fig1.set_size_inches(height_of_image, width_of_image, forward=False)
frac=0.047 * (height_of_image / (width_of_image/13))

plt.pcolor(xrho,yrho,N)

sys.exit()

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
ax.set_xticks([1,32,61,92,122,153,183,214,245,275,306,336,365])

plt.axhline(y = 10, color = 'grey')

ax.set_ylabel('shelf percent volume <1') 
ax.set_title('rough avg of shelf percent volume with \u03A9 ag < 1')

# cumulative volume
ax2 = plt.subplot2grid((2,3), (1,0), colspan=1, rowspan=1)
VFrac = {}
cumsum = {}
Vend = {}

for mm in range(0,len(yr_list)):
    VFrac[yr_list[mm]] = Vpercent_yr[yr_list[mm]]/100
    cumsum[yr_list[mm]] = np.cumsum(VFrac[yr_list[mm]])

    Vend[mm] = cumsum[yr_list[mm]][-1]

    ax2.plot(y,cumsum[yr_list[mm]],linewidth=2, alpha=0.6,label=str(yr_list[mm]))

cumsum_avg = sum(cumsum.values())/len(yr_list)
ax2.plot(y,cumsum_avg,c = 'k',linewidth=3, alpha=0.8, label = 'avg')

ax2.set_ylabel('cumsum Oag<1') 
ax2.set_ylim([0,220])
#leg2 = ax2.legend(bbox_to_anchor=(1,1),loc='upper left', frameon=False)

# references
ax3 = plt.subplot2grid((2,3), (1,1), colspan=1, rowspan=1)
values = np.arange(0.1,1.01,0.05)
#values = np.arange(0.1,0.61,0.05)

p = {}
cumend = {}

for mm in range(0,len(values)):
    n = np.ones(365)*values[mm]
    m = np.cumsum(n)
    ax3.plot(y,m,linewidth=2, alpha=0.6,label=str("%.2f" % values[mm]))
    p[mm] = round(values[mm],-2)
    cumend[mm] = round(m[-1],0)

    

ax3.plot(y,cumsum_avg,c = 'k',linewidth=3, alpha=0.8, label = 'avg')
ax3.set_ylim([0,220])
leg3 = ax3.legend(bbox_to_anchor=(1,1),loc='upper left', frameon=False)


'''
# make a table for Hana 
tab = pd.DataFrame.from_dict({'CCVI':Vend})
tab['year']=yr_list
'''

