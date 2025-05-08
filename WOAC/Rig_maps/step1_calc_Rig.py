'''
This is part of processing required to create Rig plots

Step 1: 
Run this code after running extract_box_chunks.py 
for job_list == OA_indicators. 
*AND AFTER RUNNING* step0_zrho.py 

!! code assumes extracted full years were saved XXXX.01.01 - XXXX.12.31 !! 
!! code needs shelf mask to run; mine is under ../LKH_data/shelf_masks/OA_indicators !!

General workflow: 
- Opens a file that has the z_rhos calc'd from step0_zrho.py
- Opens a .nc file with temp, salt, TIC, alkalinity saved and applies a shelf mask
- applies mask 
- calc's Oag and saves

Examples of how to call the script:
On personal computer with python open:
run step1_calc_Rig -gtx cas7_t0_x4b -gtx cas7_t0_x4b -y0 2017 -y1 2017 -job OA_indicators_phys

apogee: 
python step1_calc_Rig -gtx cas7_t0_x4b -gtx cas7_t0_x4b -y0 2017 -y1 2017 -job OA_indicators_phys > t01.log &

TODO: fix Rig!!!! after looking at velocities from OOI line

'''

# imports
from lo_tools import Lfun

import os
import sys
import argparse
import pickle
import xarray as xr
from time import time
import numpy as np
import pandas as pd

import gsw
import PyCO2SYS as pyco2

#import cmocean

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
    
# organize and set paths before masking data based on h
yr_list = [year for year in range(int(args.ys0), int(args.ys1)+1)]
numyrs = len(yr_list)
po = 1024    # backgroud rho
g = 9.81

# load the shelf mask 
fn_mask =  Ldir['parent'] / 'LKH_data' / 'shelf_masks' / 'OA_indicators' / 'OA_indicators_shelf_mask_15_200m_noEstuaries.nc'
dmask = xr.open_dataset(fn_mask) 
mask_shelf = dmask.mask_shelf.values # 0 not-shelf; 1 shelf 
mask_rho = dmask.mask_rho.values     # 0 land; 1 water

# name the output file where files will be dumped
fn_ho = Ldir['parent'] / 'LKH_output' / 'WOAC' / args.gtagex / 'Rig_maps' 
fn_o = Ldir['parent'] / 'LKH_output' / 'WOAC' / args.gtagex / 'Rig_maps' / 'pickled_values'
if os.path.exists(fn_o)==False:
    Lfun.make_dir(fn_o, clean=False)

# open zrho file
z_rho_file = fn_ho/'OA_indicators_z_rho.nc'
ds1 = xr.open_dataset(z_rho_file)  
h = np.expand_dims(ds1.h,axis=0)

f_in =  Ldir['LOo'] / 'extract' / 'cas7_t0_x4b' / 'box' 

# run thru loop to load each year's .nc file 
for ydx in range(0,numyrs): 
    tt1 = time()

    print('working on '+str(yr_list[ydx])+' setup...')

    # 1. input velocity data 
    #surface
    fna = args.job_type+'_surf_'+str(yr_list[ydx])+'.01.01_'+str(yr_list[ydx])+'.12.31_chunks' 
    fnb = args.job_type+'_surf_'+str(yr_list[ydx])+'.01.01_'+str(yr_list[ydx])+'.12.31'+'.nc'
    #bottom
    fnc = args.job_type+'_bot_'+str(yr_list[ydx])+'.01.01_'+str(yr_list[ydx])+'.12.31_chunks'
    fnd = args.job_type+'_bot_'+str(yr_list[ydx])+'.01.01_'+str(yr_list[ydx])+'.12.31'+'.nc'
  
    fn_i_surf = Ldir['LOo'] / 'extract' / args.gtagex / 'box' / fna / fnb
    fn_i_bot = Ldir['LOo'] / 'extract' / args.gtagex / 'box' / fnc / fnd
    
    if os.path.isfile(fn_i_surf)==False:
        print('no file named: '+fnb)
        sys.exit()

    if os.path.isfile(fn_i_bot)==False:
        print('no file named: '+fnd)
        sys.exit()

    ds_s = xr.open_dataset(fn_i_surf, decode_times=True)  
    ds_b = xr.open_dataset(fn_i_bot, decode_times=True)  

    if (np.all(ds_b.lon_rho.values==ds1.lon_rho.values)) and (np.all(ds1.lon_rho.values==ds_s.lon_rho.values)) and (np.all(ds_b.lat_rho.values==ds1.lat_rho.values)) and (np.all(ds1.lat_rho.values==ds_s.lat_rho.values)):
        print('lat lon passed')
        z_rho = ds1.z_rho.values
    else:
        sys.exit()

    Usurf = ((ds_s.u.values**2)+(ds_s.v.values**2))**(1/2)
    Ubot = ((ds_b.u.values**2)+(ds_b.v.values**2))**(1/2)
    dU2 = np.abs(Ubot-Usurf)**2
    dU2[:,mask_rho==0] = np.nan

    S2 = ((np.abs(ds_b.v.values-ds_s.v.values)/h)**2)+((np.abs(ds_b.u.values-ds_s.u.values)/h)**2)

    del ds_b, ds_s, fna, fnb, fnc, fnd, fn_i_surf, fn_i_bot

    print('Time to calc del. velocity = %0.2f sec' % (time()-tt1))
    tt1 = time()

    # 1. input OA_indicators job data (calc sigma)
    #surface
    fna = 'OA_indicators_surf_'+str(yr_list[ydx])+'.01.01_'+str(yr_list[ydx])+'.12.31_chunks' 
    fnb = 'OA_indicators_surf_'+str(yr_list[ydx])+'.01.01_'+str(yr_list[ydx])+'.12.31'+'.nc'
    #bottom
    fnc = 'OA_indicators_bot_'+str(yr_list[ydx])+'.01.01_'+str(yr_list[ydx])+'.12.31_chunks'
    fnd = 'OA_indicators_bot_'+str(yr_list[ydx])+'.01.01_'+str(yr_list[ydx])+'.12.31'+'.nc'
  
    fn_i_surf = Ldir['LOo'] / 'extract' / args.gtagex / 'box' / fna / fnb
    fn_i_bot = Ldir['LOo'] / 'extract' / args.gtagex / 'box' / fnc / fnd
    
    if os.path.isfile(fn_i_surf)==False:
        print('no file named: '+fnb)
        sys.exit()

    if os.path.isfile(fn_i_bot)==False:
        print('no file named: '+fnd)
        sys.exit()

    ds_s = xr.open_dataset(fn_i_surf, decode_times=True)  
    ds_b = xr.open_dataset(fn_i_bot, decode_times=True)  

    ot = pd.to_datetime(ds_s.ocean_time.values)
    lon = ds_b['lon_rho'].values
    lat = ds_b['lat_rho'].values

    if (np.all(ds_b.lon_rho.values==ds1.lon_rho.values)) and (np.all(ds1.lon_rho.values==ds_s.lon_rho.values)) and (np.all(ds_b.lat_rho.values==ds1.lat_rho.values)) and (np.all(ds1.lat_rho.values==ds_s.lat_rho.values)):
        print('lat lon passed')
    else:
        sys.exit()

    # 2. Calc sig for top/bottom 
    ### BOTTOM 
    SP = ds_b.salt.values          # practical salinity
    SP[SP<0] = 0                 # Make sure salinity is not negative. Could be a problem for pyco2.
    SP[:,mask_rho==0] = np.nan   # land
    SP[:,mask_shelf==0] = np.nan # not shelf

    PT = ds_b.temp.values          # potential temperature [degC] 
    PT[:,mask_rho==0] = np.nan   # land
    PT[:,mask_shelf==0] = np.nan # not shelf

    pp = gsw.p_from_z(z_rho, lat) # pressure [dbar]
    p = np.expand_dims(pp,axis=0)
    p[:,mask_rho==0] = np.nan   # land
    p[:,mask_shelf==0] = np.nan # not shelf
        
    SA = gsw.SA_from_SP(SP, p, lon, lat) # absolute salinity [g kg-1]
    CT = gsw.CT_from_pt(SA, PT) # conservative temperature [degC]
    rho_bot = gsw.rho(SA, CT, p) # in situ density [kg m-3]

    del SP, PT, SA, CT 

    ### SURFACE
    SP = ds_s.salt.values          # practical salinity
    SP[SP<0] = 0                 # Make sure salinity is not negative. Could be a problem for pyco2.
    SP[:,mask_rho==0] = np.nan   # land
    SP[:,mask_shelf==0] = np.nan # not shelf

    PT = ds_s.temp.values          # potential temperature [degC] 
    PT[:,mask_rho==0] = np.nan   # land
    PT[:,mask_shelf==0] = np.nan # not shelf

    SA = gsw.SA_from_SP(SP, 0, lon, lat) # absolute salinity [g kg-1]
    CT = gsw.CT_from_pt(SA, PT) # conservative temperature [degC]
    rho_surf = gsw.rho(SA, CT, p) # in situ density [kg m-3]

    drho = rho_bot-rho_surf
    Rig1 = (g*drho*h) / (po*dU2)
    N2 = (g/po) * (drho/h)
    Rig = N2 / S2

    print('Time to calc rho and Rig = %0.2f sec' % (time()-tt1))

    tt1 = time()

    # 3.Save somethings
    Rdict = {}
    Rdict['ocean_time'] = ds_s['ocean_time']
    Rdict['lat_rho'] = ds_s['lat_rho']
    Rdict['lon_rho'] = ds_s['lon_rho']
    Rdict['N2'] = N2
    Rdict['masks'] = dmask
    Rdict['notes'] = 'bulk_surf_bot'
    Rdict['job'] = 'OA_indicators + _phys'

    pn = args.job_type+'_N2_'+str(yr_list[ydx])+'.pkl'
    picklepath = fn_o/pn
        
    with open(picklepath, 'wb') as fm:
        pickle.dump(Rdict, fm)
        print('Pickled year %0.0f' % yr_list[ydx])
        sys.stdout.flush()

    del picklepath
    del pn