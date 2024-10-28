"""
Calculates cumulative shelf volume for each region 
at Oag<1 and <0.5 
ARAG1 | ARAG1 

Sum the cumulative percent volume / region
grab the last value and put in a stacked bar graph

example calls:  run calc_cumulative_pvolume_byRegion -mvar ARAG1
                run calc_cumulative_pvolume_byRegion -mvar ARAG05
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
fn_o = Ldir['parent'] / 'LKH_output' / 'OA_indicators' / 'cas7_t0_x4b' / 'volumes_by_threshold' / 'byregion' / args.variable / 'cumulative_sum'

if os.path.exists(fn_o)==False:
    Lfun.make_dir(fn_o, clean = False)
    
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

Rcolors = ['#73210D','#9C6B2F','#C5B563','#77BED0','#4078B3','#112F92'] #no grey 
lat_list = [48, 47, 46, 45, 44, 43]

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
    
    R1[ydx] = region_cumsum[48][364] # always take the 365th so leap years don't have another day of volume 
    R2[ydx] = region_cumsum[47][364]
    R3[ydx] = region_cumsum[46][364]
    R4[ydx] = region_cumsum[45][364]
    R5[ydx] = region_cumsum[44][364]
    R6[ydx] = region_cumsum[43][364]
    RYEARS[ydx] = yr_list[ydx]

pn = 'byregion_'+args.variable+'_365_cumsum_'+str(yr_list[0])+'_'+str(yr_list[-1])+'.pkl'
picklepath = fn_o/pn

regional_cumsum = {}
regional_cumsum['R1'] = R1
regional_cumsum['R2'] = R2
regional_cumsum['R3'] = R3
regional_cumsum['R4'] = R4
regional_cumsum['R5'] = R5
regional_cumsum['R6'] = R6
regional_cumsum['RYEARS'] = RYEARS
regional_cumsum['lat_list'] = [48, 47, 46, 45, 44, 43]
regional_cumsum['notes'] = 'cumulative fraction: vol corrosive RX / vol total RX'
regional_cumsum['units'] = 'none fraction cumsum day on 365'

with open(picklepath, 'wb') as fm:
    pickle.dump(regional_cumsum, fm)
    print('svol dict saved successfully to file')
