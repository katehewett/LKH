"""
Plots cumulative shelf volume for WA and OR
at Oag<1 and <0.5 

Sum the cumulative percent volume / state

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
import seaborn as sns

# add a var type so can call arg threshold in command line
parser = argparse.ArgumentParser()
parser.add_argument('-mvar', '--variable', type=str) # select variable 
args = parser.parse_args()

Ldir = Lfun.Lstart()

# load the shelf mask / assign vars 
fn_mask =  Ldir['parent'] / 'LKH_data' / 'shelf_masks' / 'OA_indicators' / 'OA_indicators_shelf_mask_15_200m_noEstuaries.nc'
dmask = xr.open_dataset(fn_mask) 
mask_shelf = dmask.mask_shelf 
h = dmask.h
xrho = dmask.lon_rho
yrho = dmask.lat_rho

# input location for pickled files 
fn_i = Ldir['parent'] / 'LKH_output' / 'OA_indicators' / 'cas7_t0_x4b' / 'volumes_by_threshold' / 'bystate' / args.variable

#plotting details
mask_dict = {}
mask_dict['WA'] = (yrho > 46.25) & (mask_shelf == 1)
mask_dict['OR'] = (yrho <= 46.25) & (mask_shelf == 1)
NMASK = len(mask_dict)

yr_list = [year for year in range(2014,2024)]
numyrs = len(yr_list)

Rcolors = sns.color_palette()

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
ax.scatter(xrho*mask_dict['WA'], yrho*mask_dict['WA'], s=2, c = 'mediumseagreen')
ax.scatter(xrho*mask_dict['OR'], yrho*mask_dict['OR'], s=2, c = 'darkgreen')

ax1 = plt.subplot2grid((2,3), (0,1), colspan=1, rowspan = 1)  # WA
ax2 = plt.subplot2grid((2,3), (1,1), colspan=1, rowspan = 1)  # OR

fig1.tight_layout()

for ydx in range(0,numyrs): 
    pn_o = 'bystate_OR_'+args.variable+'_volumes_'+str(yr_list[ydx])+'.pkl'
    opicklepath = fn_i / pn_o 
    
    pn_w = 'bystate_WA_'+args.variable+'_volumes_'+str(yr_list[ydx])+'.pkl'
    wpicklepath = fn_i / pn_w
        
    if os.path.isfile(opicklepath)==False or os.path.isfile(wpicklepath)==False:
        print('no file named: ' + pn)
        sys.exit()
    
    # Load the dictionary from the file
    with open(wpicklepath, 'rb') as fp:
        wvol = pickle.load(fp)
        print('loaded'+str(yr_list[ydx]))

    with open(opicklepath, 'rb') as fp:
        ovol = pickle.load(fp)
        print('loaded'+str(yr_list[ydx]))
    
    Vwa_corr = wvol['Vtotal_corr'] 
    Vwa_shelf = wvol['Vtotal_shelf']
    
    WP = (Vwa_corr/Vwa_shelf) 
    
    Vor_corr = ovol['Vtotal_corr'] 
    Vor_shelf = ovol['Vtotal_shelf']
    
    OP = (Vor_corr/Vor_shelf)      

    WA_cumsum = np.cumsum(WP)           # cumulative volumes 
    OR_cumsum = np.cumsum(OP)
    
    ot = wvol['ocean_time']
    yd = np.arange(1,len(ot)+1,1) # lazy yearday

    if args.variable == 'arag1':
        ax1.set_title('$\mathregular{Washington^{+}}$: \u03A9 ag < 1')
        ax2.set_title('Oregon')
    elif args.variable == 'arag05':
        ax1.set_title('$\mathregular{Washington^{+}}$: \u03A9 ag < 0.5')
        ax2.set_title('Oregon')
            
    ax1.plot(yd,WA_cumsum,c = Rcolors[ydx],linewidth=3, alpha=0.8,label = str(yr_list[ydx]))
    ax1.set_ylabel('cumulative (corrosive volume / vol WA shelf)')
    
    ax2.plot(yd,OR_cumsum,c = Rcolors[ydx],linewidth=3, alpha=0.8,label = str(yr_list[ydx]))
    ax2.set_ylabel('cumulative (corrosive volume / vol OR shelf)')
    
    ax1.set_xlim(1,366)
    ax2.set_xlim(1,366)
    
    ax1.set_xticks([1,32,60,91,121,152,182,213,244,274,305,335,365])
    ax2.set_xticks([1,32,60,91,121,152,182,213,244,274,305,335,365])
    
    ax2.set_xlabel('year day')
    
    ax1.legend(frameon=False)
    ax2.legend(frameon=False)
    
    ax1.grid(True)
    ax2.grid(True)
    
    if args.variable == 'arag1':    
        ax1.set_ylim(0,250)
        ax2.set_ylim(0,250)        
    elif args.variable == 'arag05':  
        ax1.set_ylim(0,60)
        ax2.set_ylim(0,60) 

if args.variable == 'arag1':    
    fig1.savefig('/Users/katehewett/Documents/LKH_output/OA_indicators/cas7_t0_x4b/volumes_by_threshold/bystate/arag1/plots/PERCENT_Oag_state_above1_NEW.png')
elif args.variable == 'arag05':    
    fig1.savefig('/Users/katehewett/Documents/LKH_output/OA_indicators/cas7_t0_x4b/volumes_by_threshold/bystate/arag05/plots/PERCENT_Oag_state_above05_NEW.png')
    
