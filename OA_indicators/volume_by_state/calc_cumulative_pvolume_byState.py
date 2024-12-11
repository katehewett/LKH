"""
Goes and grabs the cumulative shelf volume for each State 
on day 365 (already calculated) for 
Oag<1 and <0.5 
ARAG1 | ARAG0.5

The volumes are the 
Sum the cumulative percent volume / state
And we're doing this because we want to
grab the last value and put in a stacked bar graph

example calls:  
run calc_cumulative_pvolume_byState -mvar ARAG1
run calc_cumulative_pvolume_byState -mvar ARAG05
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
fn_o = Ldir['parent'] / 'LKH_output' / 'OA_indicators' / 'cas7_t0_x4b' / 'volumes_by_threshold' / 'bystate' / args.variable / 'cumulative_sum'

if os.path.exists(fn_o)==False:
    Lfun.make_dir(fn_o, clean = False)
    
#plotting details
mask_dict = {}
mask_dict['WA'] = (yrho > 46.25) & (mask_shelf == 1)
mask_dict['OR'] = (yrho <= 46.25) & (mask_shelf == 1)
NMASK = len(mask_dict)

yr_list = [year for year in range(2013,2024)]
numyrs = len(yr_list)

#initialize for entering the 365th day cumsum value + year 
RWA = np.full(numyrs, np.nan)
ROR = np.full(numyrs, np.nan)
RYEARS = np.full(numyrs, np.nan)

for ydx in range(0,numyrs): 
    pn_WA_in = 'bystate_WA_'+args.variable+'_volumes_'+str(yr_list[ydx])+'.pkl'
    pn_OR_in = 'bystate_OR_'+args.variable+'_volumes_'+str(yr_list[ydx])+'.pkl'

    picklepath_WA = fn_i / pn_WA_in
    picklepath_OR = fn_i / pn_OR_in
    #pn = 'bystate_'+args.variable+'_state_volumes_'+str(yr_list[ydx])+'.pkl'
    #picklepath_in = fn_i / pn 
        
    if os.path.isfile(picklepath_WA)==False:
        print('no file named: ' + pn_WA_in)
        sys.exit()

    if os.path.isfile(picklepath_OR)==False:
        print('no file named: ' + pn_OR_in)
        sys.exit()
    
    # Load the dictionary from the file
    with open(picklepath_WA, 'rb') as fp:
        WAvol = pickle.load(fp)
        print('loaded WA '+str(yr_list[ydx]))
    
    with open(picklepath_OR, 'rb') as fp2:
        ORvol = pickle.load(fp2)
        print('loaded OR '+str(yr_list[ydx]))
            
    WA_V_corr = WAvol['Vtotal_corr'] 
    WA_V_shelf = WAvol['Vtotal_shelf']

    OR_V_corr = ORvol['Vtotal_corr'] 
    OR_V_shelf = ORvol['Vtotal_shelf']
    
    WA_P = (WA_V_corr/WA_V_shelf)
    WA_cumsum = np.cumsum(WA_P)

    OR_P = (OR_V_corr/OR_V_shelf)
    OR_cumsum = np.cumsum(OR_P)

    RWA[ydx] = WA_cumsum[364] # always take the 365th so leap years don't have another day of volume 
    ROR[ydx] = OR_cumsum[364]
    RYEARS[ydx] = yr_list[ydx]

pn = 'bystate_'+args.variable+'_365_cumsum_'+str(yr_list[0])+'_'+str(yr_list[-1])+'.pkl'
picklepath = fn_o/pn

state_cumsum = {}
state_cumsum['WA'] = RWA
state_cumsum['OR'] = ROR
state_cumsum['RYEARS'] = RYEARS
state_cumsum['dividing_lat'] = [46.25]
state_cumsum['notes'] = 'cumulative fraction: vol corrosive RX / vol total RX'
state_cumsum['units'] = 'none fraction cumsum day on 365'

with open(picklepath, 'wb') as fm:
    pickle.dump(state_cumsum, fm)
    print('svol dict saved successfully to file')
