'''
This is part of processing required to create time duration plot
from surface and bottom data along the ~40m isobath

Step 1: 
Run this code after running extract_box_chunks.py 
for job_list == OA_indicators. 

!! code assumes extracted full years were saved XXXX.01.01 - XXXX.12.31 !! 
!! code needs shelf mask to run; mine is under ../LKH_data/shelf_masks/OA_indicators !!

General workflow: 
- Opens the .nc file with temp, salt, TIC, alkalinity saved and applies a shelf mask
- grabs data between h = 39 : h = 41 (for either suface -surf True; or bottom -bot True)
- calc's Oag at 3 thresholds for that mask 

Examples of how to call the script:
On personal computer with python open:
To grab all of the shelf for the OA_indicators job list:
run step1_extract_h40m_data -gtx cas7_t0_x4b -y0 2017 -y1 2017 -job OA_indicators -surf True -test True

'''

# imports
from lo_tools import Lfun, zfun
from lo_tools import plotting_functions as pfun

import os
import sys
import argparse
import pickle
import xarray as xr
import netCDF4 as nc
from time import time
import numpy as np
import math
import pandas as pd

import gsw
import PyCO2SYS as pyco2

import matplotlib.pyplot as plt
import matplotlib.cm as cm
#import cmocean
import matplotlib.dates as mdates
from datetime import datetime


tt0 = time()

# command line arugments
parser = argparse.ArgumentParser()
# which run was used:
parser.add_argument('-gtx', '--gtagex', type=str)   # e.g. cas7_t0_x4b
#parser.add_argument('-ro', '--roms_out_num', type=int) # 2 = Ldir['roms_out2'], etc.
# select years 
parser.add_argument('-y0', '--ys0', type=str) # e.g. 2014
parser.add_argument('-y1', '--ys1', type=str) # e.g. 2015
parser.add_argument('-job', '--job_type', type=str) # job 
# these flags get only surface or bottom fields if True
# - cannot have both True -
parser.add_argument('-surf', default=False, type=Lfun.boolean_string)
parser.add_argument('-bot', default=False, type=Lfun.boolean_string)
# Optional: set max number of subprocesses to run at any time
parser.add_argument('-Nproc', type=int, default=10)
# Optional: for testing
parser.add_argument('-test', '--testing', default=False, type=Lfun.boolean_string)
# get the args and put into Ldir
args = parser.parse_args()
# test that main required arguments were provided
argsd = args.__dict__
for a in ['gtagex']:
    if argsd[a] == None:
        print('*** Missing required argument: ' + a)
        sys.exit()
gridname, tag, ex_name = args.gtagex.split('_')
# get the dict Ldir
Ldir = Lfun.Lstart(gridname=gridname, tag=tag, ex_name=ex_name)
# add more entries to Ldir
for a in argsd.keys():
    if a not in Ldir.keys():
        Ldir[a] = argsd[a]
        
Ldir = Lfun.Lstart()

# check for input conflicts:
if args.surf==True and args.bot==True:
    print('Error: cannot have surf and bot both True.')
    sys.exit()
    
# organize and set paths before masking data based on h
yr_list = [year for year in range(int(args.ys0), int(args.ys1)+1)]
numyrs = len(yr_list)

# load the shelf mask 
fn_mask =  Ldir['parent'] / 'LKH_data' / 'shelf_masks' / args.job_type / 'OA_indicators_shelf_mask_15_200m_noEstuaries.nc'
dmask = xr.open_dataset(fn_mask) 
mask_shelf = dmask.mask_shelf.values # 0 not-shelf; 1 shelf 
mask_rho = dmask.mask_rho.values     # 0 land; 1 water

# load the h mask to flag data between 35<h<45m
fn_hmask =  Ldir['parent'] / 'LKH_data' / 'shelf_masks' / args.job_type / 'OA_indicators_h40m_mask.nc'
hmask = xr.open_dataset(fn_hmask) 
mask_h40m = hmask.mask_h40m.values    # 0 outside; 1 h=35-45m 

# name the output file where files will be dumped
if args.surf==True:
    fn_o = Ldir['parent'] / 'LKH_output' / 'OA_indicators' / args.gtagex / 'oag_h40m_plots' / 'surf_h40m' 
    Lfun.make_dir(fn_o, clean=False)
elif args.bot==True:
    fn_o = Ldir['parent'] / 'LKH_output' / 'OA_indicators' / args.gtagex / 'oag_h40m_plots' / 'bot_h40m' 
    Lfun.make_dir(fn_o, clean=False)
    
# run thru loop to load each year's .nc file 
for ydx in range(0,numyrs): 
    tt1 = time()
    # 1. input filename and locations
    if args.surf==True:
        fna = args.job_type+'_surf_'+str(yr_list[ydx])+'.01.01_'+str(yr_list[ydx])+'.12.31_chunks' 
        fnb = args.job_type+'_surf_'+str(yr_list[ydx])+'.01.01_'+str(yr_list[ydx])+'.12.31'+'.nc'
    elif args.bot==True:
        fna = args.job_type+'_bot_'+str(yr_list[ydx])+'.01.01_'+str(yr_list[ydx])+'.12.31_chunks'
        fnb = args.job_type+'_bot_'+str(yr_list[ydx])+'.01.01_'+str(yr_list[ydx])+'.12.31'+'.nc'
        
    fn_i = Ldir['LOo'] / 'extract' / args.gtagex / 'box' / fna
    fn_in = fn_i / fnb
    
    if os.path.isfile(fn_in)==False:
        print('no file named: '+fnb)
        sys.exit()
       
    ds = xr.open_dataset(fn_in, decode_times=True)     
    ot = pd.to_datetime(ds.ocean_time.values)
    lon = ds['lon_rho'].values
    lat = ds['lat_rho'].values
    h_all = ds['h'].values
    h = np.copy(h_all)
    
    plt.close('all')
    fs=16
    plt.rc('font', size=fs)
    fig = plt.figure(figsize=(18,10))

    # map
    ax = fig.add_subplot(131)
    pfun.add_coast(ax)
    pfun.dar(ax)
    ax.axis([-130, -122, 42, 52])
    
    SP = ds.salt.values               # practical salinity
    SP[:,mask_shelf==0] = np.nan        
    h[mask_shelf==0] = np.nan   
    
    Zvar = SP[100,:,:].squeeze()
    ax.pcolormesh(lon,lat,Zvar)
    contours = ax.contour(lon,lat,h, [40], colors=['red'], linewidths=1, linestyles='solid',alpha=0.8)
    
    # Find the index of the contour line corresponding to the level
    # where we want to extract data 
    level = 40 
    cidx = np.where(contours.levels == level)[0][0] 
    
    # Extract the contour line data
    contour_line = contours.collections[cidx].get_paths()[0].vertices
    x_data = contour_line[:,0]
    y_data = contour_line[:,1]
    
    ax.plot(x_data,y_data,'b.',markersize = 1)
    
    from scipy.interpolate import interp1d
    # Interpolate Z values along the contour line
    z_interp = interp1d(np.linspace(0, 1, len(x_data)), Zvar[np.round(y_data).astype(int), np.round(x_data).astype(int)])
    
    
    
    sys.exit()
    
    # 2. Prepare for carbon variable calculations
    # prep values and apply mask 
    SP = ds.salt.values          # practical salinity
    SP[SP<0] = 0                 # Make sure salinity is not negative. Could be a problem for pyco2.
    PT = ds.temp.values          # potential temperature [degC] 
    ALK = ds.alkalinity.values   # alkalinity [milli equivalents m-3 = micro equivalents L-1]
    TIC = ds.TIC.values          # TIC [millimol C m-3 = micromol C L-1]
    
    aS = SP[:,[mask_rho==1]].squeeze()
    SP[:,mask_h40m==0] = np.nan  # set everything outside the "main" 35-45m isobar == nan 
    PT[:,mask_h40m==0] = np.nan
    ALK[:,mask_h40m==0] = np.nan
    TIC[:,mask_h40m==0] = np.nan
    h = np.copy(ha)
    h[:,mask_h40m==0] = np.nan
    
    # note: surf and bot in extract_box_chunks doesn't save z_rhos. But using 0, 0.5, 1m for calcs all give values that are ~ the same to 5 decimal spaces: EG: z = 0, 0.5, 1m --> 
    # SP = 35 (z = 0) = 35.16591826174167
    # SP = 35 (z = -1) = 35.165925947997316
    # SP = 35 (z = -35) >> SA = 35.16634984172463
    # SP = 35 (z = -45) >> SA = 35.166586650966224
    # round Salt and Temp np.round(XX,2)
    if args.surf == True:
        pp = gsw.p_from_z(-0.5, ds.lat_rho) # pressure [dbar]
    elif args.bot == True:
        pp = gsw.p_from_z(h, ds.lat_rho) # pressure [dbar]   
    p = np.expand_dims(pp,axis=0)
    
    masked_data = np.ma.masked_array(V, np.isnan(V)) 
    vm = np.nanmean(masked_data,axis=0,keepdims=False)
    


    
    print('Time to load and prepare carbon fields = %0.2f sec' % (time()-tt1))
    sys.stdout.flush()
    
    
    
    
    
    
    