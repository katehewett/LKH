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
fn_mask =  Ldir['parent'] / 'LKH_data' / 'shelf_masks' / 'OA_indicators' / 'OA_indicators_shelf_mask_15_200m_noEstuaries.nc'
dmask = xr.open_dataset(fn_mask) 
mask_shelf = dmask.mask_shelf 
h = dmask.h
xrho = dmask.lon_rho
yrho = dmask.lat_rho

# input location for pickled files 
fn_i = Ldir['parent'] / 'LKH_output' / 'OA_indicators' / 'cas7_t0_x4b' / 'volumes_by_threshold' / 'byregion' / 'arag1'

#plotting details
mask_dict = {}
mask_dict[48] = (yrho >=47.75) & (yrho < 48.75) & (mask_shelf == 1)
mask_dict[47] = (yrho >=46.75) & (yrho < 47.75) & (mask_shelf == 1)
mask_dict[46] = (yrho >=45.75) & (yrho < 46.75) & (mask_shelf == 1)
mask_dict[45] = (yrho >=44.75) & (yrho < 45.75) & (mask_shelf == 1)
mask_dict[44] = (yrho >=43.75) & (yrho < 44.75) & (mask_shelf == 1)
mask_dict[43] = (yrho >42.75) & (yrho < 43.75) & (mask_shelf == 1)
NMASK = len(mask_dict)

yr_list = [year for year in range(2013,2024)]
numyrs = len(yr_list)

lat_list = [48, 47, 46, 45, 44, 43]
# Rcolors = ['#73210D','#9C6B2F','#C5B563','#A9A9A9','#77BED0','#4078B3','#112F92'] #roma w grey not green #idx 3
Rcolors = ['#73210D','#9C6B2F','#C5B563','#77BED0','#4078B3','#112F92'] #no grey 

Vpercent_yr = {}
ot = {}

for ydx in range(0,numyrs): 
    pn = pn = 'byregion_arag1_regional_volumes_'+str(yr_list[ydx])+'.pkl'
    picklepath = fn_i / pn  
    
    if os.path.isfile(picklepath)==False:
        print('no file named: ' + pn)
        sys.exit()
    
    # Load the dictionary from the file
    with open(picklepath, 'rb') as fp:
        svol = pickle.load(fp)
        print('loaded'+str(yr_list[ydx]))

        #svol['Vtotal_corr'] = V_corrosive_region
        #svol['Vtotal_shelf'] = V_shelf_region
        #svol['Vcumsum'] = VR_cumsum
        #svol['ocean_time'] = ot
        #svol['Oag_threshold'] = '<'+str(threshold)
        #svol['group'] = args.group
        #svol['calc_region'] = args.job_type + ': regions'
        #svol['vol_units'] = 'm^3'
        
    Vregion = svol['Vtotal_corr']
    Vtotal = svol['Vtotal_shelf']
    ot[yr_list[ydx]] = svol['ocean_time']
    
    a = sum(Vregion.values())/sum(Vtotal.values())
    Vpercent_yr[yr_list[ydx]] = a[0:365]*100

plt.close('all')
fs=11
plt.rc('font', size=fs)
height_of_image = 15 
width_of_image = 8 
fig1 = plt.figure(figsize=(height_of_image,width_of_image))
fig1.set_size_inches(height_of_image, width_of_image, forward=False)
frac=0.047 * (height_of_image / (width_of_image/13))

# rough average 
ax = plt.subplot2grid((2,3), (0,0), colspan=3, rowspan=1)
y = np.arange(1,366)
Vavg = sum(Vpercent_yr.values())/len(yr_list)

VPavg_year = {}

for mm in range(0,len(yr_list)):
    ax.plot(y,Vpercent_yr[yr_list[mm]],linewidth=2, alpha=0.6,label=str(yr_list[mm]))
    
    VPavg_year[mm] = np.round(np.nanmean(Vpercent_yr[yr_list[mm]]),0)


ax.plot(y,Vavg,c = 'k',linewidth=3, alpha=0.8, label = 'avg')
leg = ax.legend(bbox_to_anchor=(1,1),loc='upper left', frameon=False)


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