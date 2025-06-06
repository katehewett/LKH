"""
Plots cumulative shelf volume for WA and OR
at Oag<1 and <0.5 

Sum the cumulative percent volume / state

plot_cumulative_pvolume_R1 -mvar ARAG1

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
fn_i = Ldir['parent'] / 'LKH_output' / 'OA_indicators' / 'cas7_t0_x4b' / 'volumes_by_threshold' / 'byregion' / args.variable

if args.variable == 'ARAG1':
    fn_o = '/Users/katehewett/Documents/LKH_output/OA_indicators/cas7_t0_x4b/volumes_by_threshold/byregion/arag1/plots'
elif args.variable == 'ARAG05':
    fn_o = '/Users/katehewett/Documents/LKH_output/OA_indicators/cas7_t0_x4b/volumes_by_threshold/byregion/arag05/plots'

Lfun.make_dir(fn_o, clean=False)

#plotting details
mask_dict = {}
mask_dict[48] = (yrho >=47.75) & (yrho < 48.75) & (mask_shelf == 1)
mask_dict[47] = (yrho >=46.75) & (yrho < 47.75) & (mask_shelf == 1)
mask_dict[46] = (yrho >=45.75) & (yrho < 46.75) & (mask_shelf == 1)
mask_dict[45] = (yrho >=44.75) & (yrho < 45.75) & (mask_shelf == 1)
mask_dict[44] = (yrho >=43.75) & (yrho < 44.75) & (mask_shelf == 1)
mask_dict[43] = (yrho >42.75) & (yrho < 43.75) & (mask_shelf == 1)
NMASK = len(mask_dict)

yr_list = [year for year in range(2014,2024)]
numyrs = len(yr_list)

lat_list = [48, 47, 46, 45, 44, 43]

Rcolors = sns.color_palette()

plt.close('all')
fs=11
plt.rc('font', size=fs)
height_of_image = 15 
width_of_image = 8 
fig1 = plt.figure(figsize=(height_of_image,width_of_image))
fig1.set_size_inches(height_of_image,width_of_image, forward=False)
frac=0.047 * (height_of_image / (width_of_image/13))

ax1 = plt.subplot2grid((2,3), (0,1), colspan=1, rowspan = 1)  


fig1.tight_layout()

#initialize for entering the 365th day cumsum value + year 
R1 = np.full(numyrs, np.nan)
R2 = np.full(numyrs, np.nan)
R3 = np.full(numyrs, np.nan)
R4 = np.full(numyrs, np.nan)
R5 = np.full(numyrs, np.nan)
R6 = np.full(numyrs, np.nan)
RYEARS = np.full(numyrs, np.nan)

for ydx in range(0,numyrs): 
    pn = 'byregion_'+args.variable+'_regional_volumes_'+str(yr_list[ydx])+'.pkl'
    picklepath = fn_i / pn 
        
    if os.path.isfile(picklepath)==False:
        print('no file named: ' + pn)
        sys.exit()
    
    # Load the dictionary from the file
    with open(picklepath, 'rb') as fp:
        vol = pickle.load(fp)
        print('loaded'+str(yr_list[ydx]))
            
    V_corr = vol['Vtotal_corr'] 
    V_shelf = vol['Vtotal_shelf']
    
    P = {}
    region_cumsum = {}
    for mm in lat_list:
        P[mm] = (V_corr[mm]/V_shelf[mm])
        region_cumsum[mm] = np.cumsum(P[mm])
    
    '''
    R1[ydx] = region_cumsum[48][364] # always take the 365th so leap years don't have another day of volume 
    R2[ydx] = region_cumsum[47][364]
    R3[ydx] = region_cumsum[46][364]
    R4[ydx] = region_cumsum[45][364]
    R5[ydx] = region_cumsum[44][364]
    R6[ydx] = region_cumsum[43][364]
    RYEARS[ydx] = yr_list[ydx]
    '''

    ot = vol['ocean_time']
    yd = np.arange(1,len(ot)+1,1) # lazy yearday

    if args.variable == 'ARAG1':
        ax1.set_title('R1 \u03A9 $\\leq$ 1')
    elif args.variable == 'ARAG05':
        ax1.set_title('R1 \u03A9 $\\leq$ 0.5')
    
    ax1.plot(yd,region_cumsum[48],c = Rcolors[ydx],linewidth=3, alpha=0.8,label = str(yr_list[ydx]))
    ax1.set_ylabel('cumulative (corrosive volume / vol R1 shelf)')
    
    ax1.set_xlim(1,366)
    
    ax1.set_xticks([1,32,60,91,121,152,182,213,244,274,305,335,365])
    
    ax1.set_xlabel('year day')
    
    ax1.legend(frameon=False)
    ax1.legend(facecolor='white')
    
    ax1.grid(True)
    
    if args.variable == 'ARAG1':    
        ax1.set_ylim(0,250)    
    elif args.variable == 'ARAG05':  
        ax1.set_ylim(0,60)

if args.variable == 'ARAG1':    
    fig1.savefig(fn_o+'/cumsum_Oag_state_above1_R1.png')
elif args.variable == 'ARAG05':    
    fig1.savefig(fn_o+'/cumsum_Oag_state_above05_R1.png')
    
