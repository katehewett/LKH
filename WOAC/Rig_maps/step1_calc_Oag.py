'''
This is part of processing required to create time duration plot
of bottom water 

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
To grab all of the shelf for the OA_indicators job list:
run step1_calc_Oag -gtx cas7_t0_x4b -y0 2013 -y1 2023 -job OA_indicators -bot True -test True

apogee: 
python step0_calc_Oag.py -gtx cas7_t0_x4b -y0 2013 -y1 2023 -job OA_indicators -bot True > bot.log&
python step1_calc_Oag.py -gtx cas7_t0_x4b -y0 2013 -y1 2023 -job OA_indicators -bot True > bot.log&

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
dsb = xr.open_dataset('/Users/katehewett/Documents/LO_output/extract/cas7_t0_x4b/box/OA_indicators_phys_bot_2017.01.01_2017.12.31_chunks/OA_indicators_phys_bot_2017.01.01_2017.12.31.nc', decode_times=True)
dss = xr.open_dataset('/Users/katehewett/Documents/LO_output/extract/cas7_t0_x4b/box/OA_indicators_phys_surf_2017.01.01_2017.12.31_chunks/OA_indicators_phys_surf_2017.01.01_2017.12.31.nc', decode_times=True)

sys.exit()


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

# name the output file where files will be dumped
if args.surf==True:
    fn_o = Ldir['parent'] / 'LKH_output' / 'WOAC' / args.gtagex / 'surface_maps' 
    Lfun.make_dir(fn_o, clean=False)
elif args.bot==True:
    fn_o = Ldir['parent'] / 'LKH_output' / 'WOAC' / args.gtagex / 'bottom_maps' 
    #Lfun.make_dir(fn_o, clean=False)
    z_rho_file = fn_o/'OA_indicators_z_rho.nc'
    ds1 = xr.open_dataset(z_rho_file)  

# run thru loop to load each year's .nc file 
for ydx in range(0,numyrs): 
    tt1 = time()

    print('working on '+str(yr_list[ydx])+' setup...')

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

    if (np.all(lon==ds1.lon_rho.values)) and (np.all(lat==ds1.lat_rho.values)):
        print('lat lon passed')
        z_rho = ds1.z_rho.values
    else: 
        sys.exit()

    # 2. Prepare for variables for carbon calculations 
    SP = ds.salt.values          # practical salinity
    SP[SP<0] = 0                 # Make sure salinity is not negative. Could be a problem for pyco2.
    SP[:,mask_rho==0] = np.nan   # land
    SP[:,mask_shelf==0] = np.nan # not shelf

    PT = ds.temp.values          # potential temperature [degC] 
    PT[:,mask_rho==0] = np.nan   # land
    PT[:,mask_shelf==0] = np.nan # not shelf
    
    ALK = ds.alkalinity.values   # alkalinity [milli equivalents m-3 = micro equivalents L-1]
    ALK[:,mask_rho==0] = np.nan   # land
    ALK[:,mask_shelf==0] = np.nan # not shelf
    
    TIC = ds.TIC.values          # TIC [millimol C m-3 = micromol C L-1]
    TIC[:,mask_rho==0] = np.nan   # land
    TIC[:,mask_shelf==0] = np.nan # not shelf

    pp = gsw.p_from_z(z_rho, ds.lat_rho) # pressure [dbar]
    p = np.expand_dims(pp,axis=0)
    p[:,mask_rho==0] = np.nan   # land
    p[:,mask_shelf==0] = np.nan # not shelf
        
    # We then use gsw routines to calculate in situ density and temperature.
    SA = gsw.SA_from_SP(SP, p, lon, lat) # absolute salinity [g kg-1]
    CT = gsw.CT_from_pt(SA, PT) # conservative temperature [degC]
    rho = gsw.rho(SA, CT, p) # in situ density [kg m-3]
    ti = gsw.t_from_CT(SA, CT, p) # in situ temperature [degC]
    # Convert from micromol/L to micromol/kg using in situ dentity because these are the
    # units expected by pyco2.
    ALK1 = 1000 * ALK / rho
    TIC1 = 1000 * TIC / rho
    # I'm not sure if this is needed. In the past a few small values of these variables had
    # caused big slowdowns in the MATLAB version of CO2SYS.
    ALK1[ALK1 < 100] = 100
    TIC1[TIC1 < 100] = 100
    
    print('Time to load and prepare carbon fields = %0.2f sec' % (time()-tt1))
    sys.stdout.flush()

    print('working on '+str(yr_list[ydx])+' co2sys calcs...')

    # 3. Calculate aragonite saturation and a map of corrosive thickness.
    # We do this one day at a time because pyco2 got bogged down if we fed it
    # the whole 3-D arrays at once. 
    tt0 = time()
    all_ARAG = np.nan * np.ones(SP.shape) # Initialize array (t,y,z) to hold results.
    nt, nr, nc = SP.shape # handy dimension sizes
    amat = np.nan * np.ones((nr,nc)) # Initialize array (y,x) for single layer.

    nday = np.shape(SP)[0]
    for ii in range(nday):
        tt00 = time()
        print('layer day: '+str(ii))
        # Note that by using the [mask_rho==1] operator on each of the layers
        # we go from a 2-D array with nan's to a 1-D vector with no nan's. We
        # do this to speed up the calculation. Note that the cas6 grid is about half
        # landmask. Even better mask_shelf==1 
        aALK = ALK1[ii,:,:].squeeze()[mask_shelf==1]
        aTIC = TIC1[ii,:,:].squeeze()[mask_shelf==1]
        aTemp = ti[ii,:,:].squeeze()[mask_shelf==1]
        aPres = p.squeeze()[mask_shelf==1]
        aSalt = SP[ii,:,:].squeeze()[mask_shelf==1]
        # Note: here is where to get info on the inputs, outputs, and units:
        # https://pyco2sys.readthedocs.io/en/latest/co2sys_nd/
        CO2dict = pyco2.sys(par1=aALK, par1_type=1, par2=aTIC, par2_type=2,
            salinity=aSalt, temperature=aTemp, pressure=aPres,
            total_silicate=50, total_phosphate=2, opt_k_carbonic=10, opt_buffers_mode=0)
        aARAG = CO2dict['saturation_aragonite']
    
        # Then write the arag field into the ARAG array, indexing with [mask_shelf==1].
        aamat = amat.copy()
        aamat[mask_shelf==1] = aARAG
        all_ARAG[ii,:,:] = aamat 
        sys.stdout.flush()

    print('Time to calculate ARAG for all layers = %0.2f sec' % (time()-tt0))

    ARAG_dict = {}
    ARAG_dict['ocean_time'] = ds['ocean_time']
    ARAG_dict['lat_rho'] = ds['lat_rho']
    ARAG_dict['lon_rho'] = ds['lon_rho']
    ARAG_dict['ARAG'] = all_ARAG
    if args.surf == True:
        ARAG_dict['level'] = 'surf'   
    elif args.bot ==True:
        ARAG_dict['level'] = 'bot'  
        ARAG_dict['calc_region'] = args.job_type + str(': bottom')
        pn = args.job_type+'_Oag_bottom_'+str(yr_list[ydx])+'.pkl'
    ARAG_dict['source_file'] = str(fn_in)

    picklepath = fn_o/pn
        
    with open(picklepath, 'wb') as fm:
        pickle.dump(ARAG_dict, fm)
        print('Pickled year %0.0f' % yr_list[ydx])
        sys.stdout.flush()

    del picklepath
    del pn

