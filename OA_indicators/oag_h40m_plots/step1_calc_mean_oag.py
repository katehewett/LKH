"""
After Running, step0_calc_Oag.py
Use this code to find the average across each day of the year for each grid cell 

run step1_calc_mean_oag -bot True < plots bottom 
run step1_calc_mean_oag -surf True < plots surface 

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
from matplotlib.colors import BoundaryNorm

# command line arugments
parser = argparse.ArgumentParser()
# these flags get only surface or bottom fields if True
# - cannot have both True - It plots one or the other to avoid a 2x loop 
parser.add_argument('-surf', default=False, type=Lfun.boolean_string)
parser.add_argument('-bot', default=False, type=Lfun.boolean_string)
# get the args and put into Ldir
args = parser.parse_args()

# check for input conflicts:
if args.surf==True and args.bot==True:
    print('Error: cannot have surf and bot both True.')
    sys.exit()

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

# load the h mask to flag data between 35<h<45m
fn_hmask =  Ldir['parent'] / 'LKH_data' / 'shelf_masks' / 'OA_indicators' / 'OA_indicators_h40m_mask.nc'
hmask = xr.open_dataset(fn_hmask) 
mask_h40m = hmask.mask_h40m.values    # 0 outside; 1 h=35-45m 

# input location for pickled files 
fn_i = Ldir['parent'] / 'LKH_output' / 'OA_indicators' / 'cas7_t0_x4b' / 'oag_h40m_plots' 

if args.surf==True: 
    fn_s = fn_i / 'surf_h40m'
elif args.bot==True: 
    fn_b = fn_i / 'bot_h40m'

yr_list = [year for year in range(2013,2024)]
numyrs = len(yr_list)

NT = 11
NR = 366
NC = 941 

amat = np.full([NT,NR,NC],np.nan)
        
for ydx in range(0,numyrs): 
    
    pn = 'OA_indicators_Oag_h40m_'+str(yr_list[ydx])+'.pkl'
    if args.surf==True: 
        picklepath = fn_s/pn
    elif args.bot==True: 
        picklepath = fn_b/pn
    
    if os.path.isfile(picklepath)==False:
        print('no file named: ' + pn)
        sys.exit()
    
    # Load the dictionary from the file
    with open(picklepath, 'rb') as fp3:
        A = pickle.load(fp3)
        print('loaded'+str(yr_list[ydx]))

    # It's packed weird becuase we averaged across the 35-45m depth range (EW)... 
    # np.shape(ARAG) = (365, 941) = each column corresponds to a position unique position N-S; each row a unique yearday 
    ARAG = A['ARAG']   
    if np.shape(ARAG)[0] == 365:
        amat[ydx,0:365,:] = ARAG
    else: 
        amat[ydx,:,:] = ARAG

    if ydx == numyrs: 
        y = A['lat_rho']
        x = A['ocean_time']



'''
if args.surf==True: 
    fig1.savefig('/Users/katehewett/Documents/LKH_output/OA_indicators/cas7_t0_x4b/oag_h40m_plots/plot_output/Ysurf_map_Oag_NEW.png')
elif args.bot==True: 
    fig1.savefig('/Users/katehewett/Documents/LKH_output/OA_indicators/cas7_t0_x4b/oag_h40m_plots/plot_output/Ybot_map_Oag_NEW.png')

'''



