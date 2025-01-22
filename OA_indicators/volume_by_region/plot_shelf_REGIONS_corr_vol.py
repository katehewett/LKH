"""
Plots shelf regions corrosive volume 
but show "good" vol ABOVE 1 for each region 


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

plt.close('all')
fs=11
plt.rc('font', size=fs)
height_of_image = 15 
width_of_image = 8 
fig1 = plt.figure(figsize=(height_of_image,width_of_image))
fig1.set_size_inches(height_of_image,width_of_image, forward=False)
frac=0.047 * (height_of_image / (width_of_image/13))

# map
ax = plt.subplot2grid((2,3), (0,0), colspan=1, rowspan=2)
#ax = plt.subplot2grid((2,10), (0,0), colspan=1,rowspan=10)
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

ax.text(0.95,.10,'40 m',color='lightgrey',weight='bold',transform=ax.transAxes,ha='right')
ax.text(0.95,.07,'80 m',color='black',weight='bold',transform=ax.transAxes,ha='right')
ax.text(0.95,.04,'*200 m',color='black',weight='bold',fontstyle = 'italic',transform=ax.transAxes,ha='right')
ax.text(0.95,.01,'1000 m',color='black',weight='bold',transform=ax.transAxes,ha='right')
      
ax.set_title('Area of Calculation')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_xticks([-125.5, -124.5, -123.5])
ax.set_yticks([42.75,43,44,45,46,47,48,48.75])
ax.set_yticklabels(['42.75','43','44','45','46','47','48','48.75'])
ax.xaxis.set_ticklabels([-125.5, -124.5 , -123.5])
ax.xaxis.set_ticklabels([-125.5, -124.5 , -123.5])
ax.grid(False)

# shade regions
ii = 0
for mm in lat_list:
    pts2 = ax.scatter(xrho*mask_dict[mm], yrho*mask_dict[mm], s=2, c = Rcolors[ii])
    ii = ii+1

ax1 = plt.subplot2grid((2,3), (0,1), colspan=2) # 2013 - 2017
ax2 = plt.subplot2grid((2,3), (1,1), colspan=2) # 2018 - 2023
fig1.tight_layout()

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
            axf.set_title('Shelf percent volume by region with \u03A9 ag > 1')
        else: 
            axf = ax2 
            
        axf.plot(ot,Vpercent[mm],c = Rcolors[ii],linewidth=2, alpha=0.8)
        axf.grid(True)
        axf.set_ylabel('shelf percent volume')
        
        axf.set_xlim(datetime(2013,1,1), datetime(2024,1,1))
        axf.set_ylim(0,100)
        
        axf.set_xticks([datetime(2013,1,1),datetime(2013,7,1),datetime(2014,1,1), datetime(2014,7,1),
        datetime(2015,1,1),datetime(2015,7,1),datetime(2016,1,1), datetime(2016,7,1),
        datetime(2017,1,1),datetime(2017,7,1),datetime(2018,1,1), datetime(2018,7,1),
        datetime(2019,1,1),datetime(2019,7,1),datetime(2020,1,1),datetime(2020,7,1),
        datetime(2021,1,1),datetime(2021,7,1),datetime(2022,1,1),datetime(2022,7,1),
        datetime(2023,1,1),datetime(2023,7,1),datetime(2023,12,31)])
    
        axf.set_xticklabels(['Jan13','Jul','Jan14', 'Jul',
        'Jan15','Jul','Jan16', 'Jul',
        'Jan17','Jul','Jan18', 'Jul',
        'Jan19','Jul','Jan20','Jul',
        'Jan21','Jul','Jan22','Jul',
        'Jan23','Jul','Jan24'])

        
        ii = ii + 1


fig1.savefig('/Users/katehewett/Documents/LKH_output/OA_indicators/cas7_t0_x4b/volumes_by_threshold/byregion/arag1/plots/Oag_regional_above1_NEW.png')




