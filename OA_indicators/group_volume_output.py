'''
Run this code first before plotting 
data produced by extract/corrosive_volume
This code will sum corrosive volumes across the shelf
mask for OA indicators

Right now this code produces 1 pickled file per year  
that will save time with a corrosive volume and a cumulative 
corrosive volume for the OA_indicators shelf
then does that again for Oregon / Washington 
(we used 46.25N ~mouth of the CR as a divider, near Cape Dissapointment WA)

!! code assumes extracted full years were saved XXXX.01.01 - XXXX.12.31 !! 

Examples of how to call the script:
On personal computer with python open:
To grab all of the shelf for the OA_indicators job list:
run group_volume_output -gtx cas7_t0_x4b -y0 2021 -y1 2021 -mvar arag1 -group shelf -job OA_indicators

Or repeat for all years (2013 - 2023):
run group_volume_output -gtx cas7_t0_x4b -y0 2013 -y1 2023 -mvar arag1 -group shelf -job OA_indicators

To seperate WA and Oregon and grab volumes: 
run group_volume_output -gtx cas7_t0_x4b -y0 2021 -y1 2021 -mvar arag1 -group bystate -job OA_indicators

To run on apogee, will first need to save the mask: 'OA_indicators_shelf_mask_15_200m_noEstuaries.nc'
under LKH_data, then can run:
python group_volume_output.py -gtx cas7_t0_x4b -y0 2021 -y1 2021 -mvar arag1 -group shelf -job OA_indicators > OA_arag1.log &

Timing::
it takes ~8-10 seconds/year to calculate and save data 
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
parser.add_argument('-mvar', '--variable', type=str) # select variable  
# select job name used
parser.add_argument('-group', type=str) # job name: whole shelf = shelf; WA/OR = bystate; regions = byregion
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

# organize and set paths before summing volumes 
yr_list = [year for year in range(int(args.ys0), int(args.ys1)+1)]
numyrs = len(yr_list)

# load the shelf mask 
fn_mask =  Ldir['parent'] / 'LKH_data' / 'shelf_masks' / args.job_type / 'OA_indicators_shelf_mask_15_200m_noEstuaries.nc'
dmask = xr.open_dataset(fn_mask) 
mask_shelf = dmask.mask_shelf 
#del dmask

# name the output file where files will be dumped
fn_o = Ldir['parent'] / 'LKH_output' / 'OA_indicators' / args.gtagex / 'OA_indicators' / args.group / args.variable
Lfun.make_dir(fn_o, clean=False)

for ydx in range(0,numyrs): 
    # inputs 
    fna = args.job_type+'_'+str(yr_list[ydx])+'.01.01_'+str(yr_list[ydx])+'.12.31' 
    fn_i = Ldir['LOo'] / 'extract' / args.gtagex / 'corrosive_volume' / fna
    fnb = 'OA_indicators_corrosive_volume_'+str(yr_list[ydx])+'.01.01_'+str(yr_list[ydx])+'.12.31'+'.nc'
    fn_in = fn_i / fnb
    if os.path.isfile(fn_in)==False:
        print('no file named: '+fnb)
        sys.exit()
       
    ds = xr.open_dataset(fn_in, decode_times=True)     
    ot = pd.to_datetime(ds.ocean_time.values)
    lon = ds['lon_rho'].values
    lat = ds['lat_rho'].values
    mask_rho = ds['mask_rho'].values
    h = ds['h'].values
    DA = ds['DA'].values
    
    if args.variable == 'arag1':
        var1 = ds['corrosive_int_dz'].values
        threshold = 1
    elif args.variable == 'arag17':
        var1 = ds['corrosive_mild_dz'].values
        threshold = 1.7
    elif args.variable == 'arag05':
        var1 = ds['corrosive_severe_dz'].values   
        threshold = 0.5    
    
    Vshelf = DA * h         # the total volume of water on rho points
    V = DA * var1           # the corrosive volume on each rho point
    
    #just focus on OA indicators job area bounds:
    # set to zero for summation; nan for plotting 
    V[:,mask_shelf==0] = 0  
    V[:,mask_rho==0] = 0
    Vshelf[mask_shelf==0] = 0
    Vshelf[mask_rho==0] = 0
    
    Vtotal_corr = np.sum(V, axis = (1,2)) # the total volume of corrosive water across the shelf
    Vtotal_shelf = np.sum(Vshelf)         # the total OA_indicators job shelf volume 
    
    Vcumsum = np.cumsum(Vtotal_corr) 
    
    if args.group == 'shelf':
        pn = args.group+'_'+args.variable+'_volumes_'+str(yr_list[ydx])+'.pkl'
        picklepath = fn_o/pn
        
        svol = {}
        svol['Vtotal_corr'] = Vtotal_corr
        svol['Vtotal_shelf'] = Vtotal_shelf
        svol['Vcumsum'] = Vcumsum
        svol['ocean_time'] = ot
        svol['Oag_threshold'] = '<'+str(threshold)
        svol['group'] = args.group
        svol['calc_region'] = args.job_type
        svol['vol_units'] = 'm^3'
        
        with open(picklepath, 'wb') as fm:
            pickle.dump(svol, fm)
            print('svol dict saved successfully to file')
    
    elif args.group == 'bystate':         #Seperate WA and Oregon        
        pnw = args.group+'_WA_'+args.variable+'_volumes_'+str(yr_list[ydx])+'.pkl'
        w_picklepath = fn_o/pnw
        
        pno = args.group+'_OR_'+args.variable+'_volumes_'+str(yr_list[ydx])+'.pkl'
        o_picklepath = fn_o/pno
    
        Vor_shelf = np.copy(Vshelf)
        Vwa_shelf = np.copy(Vshelf)
    
        V_or = np.copy(V)
        V_wa = np.copy(V)
    
        Vor_shelf[lat>46.25] = 0 # Oregon 
        V_or[lat>46.25] = 0
    
        Vwa_shelf[lat<=46.25] = 0  # Washington+
        V_wa[lat<=46.25] = 0

        Vwa_corr = np.sum(V_wa, axis = (1,2)) # the total volume of corrosive water across the shelf
        Vwa_shelf = np.sum(V_wa)              # the total OA_indicators job shelf volume 
    
        Vwa_cumsum = np.cumsum(Vwa_corr)
        Vwa_cumsum = np.cumsum(Vwa_corr)
        
        # Washington+ shelf 
        svol = {}
        svol['Vtotal_corr'] = Vwa_corr
        svol['Vtotal_shelf'] = Vwa_shelf
        svol['Vcumsum'] = Vwa_cumsum
        svol['ocean_time'] = ot
        svol['Oag_threshold'] = '<'+str(threshold)
        svol['group'] = args.group
        svol['calc_region'] = args.job_type + '_Washington'
        svol['vol_units'] = 'm^3'
        
        with open(w_picklepath, 'wb') as fm:
            pickle.dump(svol, fm)
            print('svol/WA dict saved successfully to file')

        # Oregon shelf 
        svol = {}
        svol['Vtotal_corr'] = Vor_corr
        svol['Vtotal_shelf'] = Vor_shelf
        svol['Vcumsum'] = Vor_cumsum
        svol['ocean_time'] = ot
        svol['threshold'] = '<'+str(threshold)
        svol['group'] = args.group
        svol['calc_region'] = args.job_type + '_Oregon'
        svol['vol_units'] = 'm^3'
        
        with open(o_picklepath, 'wb') as fm:
            pickle.dump(svol, fm)
            print('svol/OR dict saved successfully to file')
        
print('Total processing time = %0.2f sec' % (time()-tt0))