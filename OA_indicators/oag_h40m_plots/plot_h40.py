"""
Plots shelf regions corrosive volume 
but show "good" vol ABOVE 1 for each region 

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
fn_ib = Ldir['parent'] / 'LKH_output' / 'OA_indicators' / 'cas7_t0_x4b' / 'oag_h40m' / 'bot_h40m' 
fn_is = Ldir['parent'] / 'LKH_output' / 'OA_indicators' / 'cas7_t0_x4b' / 'oag_h40m' / 'surf_h40m' 

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

plt.close('all')
fs=11
plt.rc('font', size=fs)
height_of_image = 18 
width_of_image = 10 
fig1 = plt.figure(figsize=(height_of_image,width_of_image))
fig1.set_size_inches(height_of_image,width_of_image, forward=False)
frac=0.047 * (height_of_image / (width_of_image/13))

ax1 = plt.subplot2grid((3,3), (0,1), colspan=2) 
ax2 = plt.subplot2grid((3,3), (1,1), colspan=2) 
fig1.tight_layout()

for ydx in range(0,numyrs): 
    pnb = pn = 'OA_indicators_Oag_h40m_'+str(yr_list[ydx])+'.pkl' 
    bpicklepath = fn_i / pnb 
    
    pns = pn = 'OA_indicators_Oag_h40m_'+str(yr_list[ydx])+'.pkl' 
    spicklepath = fn_i / pns
    
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
        
    Vregion = svol['VR_cumsum']
    Vtotal = svol['Vtotal_shelf']
    ot = svol['ocean_time']
    
    #if yr_list[ydx] < 2019: 
    #    axf = ax1 
    #else: 
    #    axf = ax2
        
    Vabove = {}
    Vpercent = {}
    
    ii = 0
    for mm in lat_list:
        Vabove[mm] = Vtotal[mm] - Vregion[mm]
        Vpercent[mm] = (Vabove[mm]/Vtotal[mm])*100 

        if ii < 3:
            axf = ax1
        else: 
            axf = ax2 
            
        axf.plot(ot,Vpercent[mm],c = Rcolors[ii])
        axf.grid(True)
        axf.set_ylabel('Regional shelf percent volume with Oag > 1')
        
        ii = ii + 1


fig1.savefig('/Users/katehewett/Documents/LKH_output/OA_indicators/cas7_t0_x4b/volumes_by_threshold/byregion/arag1/plots/Oag_regional_above1_NEW.png')




