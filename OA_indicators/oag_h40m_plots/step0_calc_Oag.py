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
- calc's Oag and saves

Examples of how to call the script:
On personal computer with python open:
To grab all of the shelf for the OA_indicators job list:
run step1_extract_h40m_data -gtx cas7_t0_x4b -y0 2017 -y1 2017 -job OA_indicators -surf True -test True

Notes on testing: at first we tried to loop thru per day slices to calc ARAG (because the 3D array gets 
bogged down in CO2SYS. For those carbon calcs if mask_rho==1 it took ~8 seconds/day to calc ARAG
and 1 second/day if we mask_h40m==1. We just have to prep the carbon fields 1x
which takes ~30seconds. It was taking ~3 hours to calc one year. 
But, if we mask T,S,ALK,TIC,p data where h is between 30 - 50m; and 
take the mean of those vars along each row/lat_rho (axis = 2). Then we get one 2D array per year with a 
shape of [NT,NY]. To run the averaged 2D array, we can skip the loop, 
which takes 43 seconds to process one year:
    Time to load and prepare carbon fields = 17.57 sec
    Time to calculate ARAG for one year = 24.90 sec
    Total processing time for one year = 42.46 sec
    Total processing time all years = 42.48 sec [note, we just ran 2017 as a test]

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

import warnings

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
    h = ds['h'].values
    
    # 2. Prepare for variables for carbon calculations 
    SP = ds.salt.values          # practical salinity
    SP[SP<0] = 0                 # Make sure salinity is not negative. Could be a problem for pyco2.
    SP[:,mask_h40m==0] = np.nan
    
    PT = ds.temp.values          # potential temperature [degC] 
    PT[:,mask_h40m==0] = np.nan
    
    ALK = ds.alkalinity.values   # alkalinity [milli equivalents m-3 = micro equivalents L-1]
    ALK[:,mask_h40m==0] = np.nan
    
    TIC = ds.TIC.values          # TIC [millimol C m-3 = micromol C L-1]
    TIC[:,mask_h40m==0] = np.nan
    # note: surf and bot in extract_box_chunks doesn't save z_rhos. But using 0, -0.5, -1m [or -35; -45m] for SA calcs all give values that are ~ the same to ~3-4 decimal spaces: EG: z = 0, -1m --> 
    # SP = 35 (z = 0) = 35.16591826174167
    # SP = 35 (z = -1) = 35.165925947997316
    # SP = 35 (z = -35) >> SA = 35.16634984172463
    # SP = 35 (z = -45) >> SA = 35.166586650966224
    if args.surf == True: # just set to zero
        pp = gsw.p_from_z(0, ds.lat_rho) # pressure [dbar]
    elif args.bot == True: # and use h as depth
        pp = gsw.p_from_z(-h, ds.lat_rho) # pressure [dbar]   
    p = np.expand_dims(pp,axis=0)
    p[:,mask_h40m==0] = np.nan
          
    # we expect to see RuntimeWarnings in this block because we take mean of whole nan slices
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mSP = np.nanmean(SP,axis=2,keepdims=False)
        mPT = np.nanmean(PT,axis=2,keepdims=False)
        mALK = np.nanmean(ALK,axis=2,keepdims=False)
        mTIC = np.nanmean(TIC,axis=2,keepdims=False)
        mPres = np.nanmean(p,axis=2,keepdims=False)
    
    mlon = np.nanmean(lon) # mean of lon
    mlat = lat[:,0]        # take first column, all rows
        
    # We then use gsw routines to calculate in situ density and temperature.
    mSA = gsw.SA_from_SP(mSP, mPres, mlon, mlat) # absolute salinity [g kg-1]
    mCT = gsw.CT_from_pt(mSA, mPT) # conservative temperature [degC]
    mrho = gsw.rho(mSA, mCT, mPres) # in situ density [kg m-3]
    mti = gsw.t_from_CT(mSA, mCT, mPres) # in situ temperature [degC]
    # Convert from micromol/L to micromol/kg using in situ dentity because these are the
    # units expected by pyco2.
    mALK1 = 1000 * mALK / mrho
    mTIC1 = 1000 * mTIC / mrho
    # I'm not sure if this is needed. In the past a few small values of these variables had
    # caused big slowdowns in the MATLAB version of CO2SYS.
    mALK1[mALK1 < 100] = 100
    mTIC1[mTIC1 < 100] = 100
    
    print('Time to load and prepare carbon fields = %0.2f sec' % (time()-tt1))
    sys.stdout.flush()
    
    # 3. Calculate aragonite saturation 
    # If we feed CO2SYS the whole 3-D arrays at once, it gets bogged down, but can calc a 2D 
    # array quickly. And doing each day with a loop takes a long time to process. This is why which is why
    # we took averages of each value across the y-axis (axis = 2).
    # By taking the mean/row we go from NT,NY,NX = SA.shape = [365,941,234] 
    # to mSA.shape = NT,NY = [365,941], where each row reps the avg of the data between h=30-50m 
    # for that lat_rho. 
    
    tt2 = time()
    CO2dict = pyco2.sys(par1=mALK1, par1_type=1, par2=mTIC1, par2_type=2,
        salinity=mSP, temperature=mti, pressure=mPres,
        total_silicate=50, total_phosphate=2, opt_k_carbonic=10, opt_buffers_mode=0)
    ARAG = CO2dict['saturation_aragonite']
    
    print('Time to calculate ARAG for one year = %0.2f sec' % (time()-tt2))
    sys.stdout.flush()
    print('Total processing time for one year = %0.2f sec' % (time()-tt1))
    sys.stdout.flush()
    
    # expand time and mlat dims to match ARAG
    NT,NY =  np.shape(ARAG)

    ote = np.expand_dims(ot,axis=1)
    arr_ot = np.tile(ote,(1,NY))
    arr_mlat = np.tile(mlat,(NT,1))

    ARAG = {}
    ARAG['ocean_time'] = arr_ot
    ARAG['lat_rho'] = arr_ot
    ARAG['ARAG'] = ARAG
    ARAG['level'] = args.surf
    ARAG['source_file'] = str(fn_in)
    ARAG['calc_region'] = args.job_type + str(': 30m<h<50m')

    pn = args.job_type+'_Oag_h40m_'+str(yr_list[ydx])+'.pkl'
    picklepath = fn_o/pn
        
    with open(picklepath, 'wb') as fm:
        pickle.dump(ARAG, fm)
        print('Pickled year %0.0f' % yr_list[ydx])
        sys.stdout.flush()
       
print('Total processing time all years = %0.2f sec' % (time()-tt0))
sys.stdout.flush()



