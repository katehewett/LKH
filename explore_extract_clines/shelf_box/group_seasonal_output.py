'''
Run this code first, before plotting seasonal maps. 
This code makes seasonal groupings <e.g. montly avgs> 
of data produced from extract_clines.py 

Right now this code will group data by month, 
and perform stat_type that we input: 
average, p50, p25, p75

It will save a smaller pickled file for that variable

!! code assumes extracted full years were saved XXXX.01.01 - XXXX.12.31 !! 

run group_seasonal_output -gtx cas7_t0_x4b -y0 2014 -y1 2017 -st average -mvar zDGmax -stat average -job shelf_box -test True

'''

# imports
from lo_tools import Lfun, zfun
from lo_tools import plotting_functions as pfun

import os
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
parser.add_argument('-ro', '--roms_out_num', type=int) # 2 = Ldir['roms_out2'], etc.
# select years 
parser.add_argument('-y0', '--ys0', type=str) # e.g. 2014
parser.add_argument('-y1', '--ys1', type=str) # e.g. 2015
parser.add_argument('-stat', '--stat_type', type=str) # stat type: average
parser.add_argument('-mvar', '--variable', type=str) # select variable  
# select job name used
parser.add_argument('-job', type=str) # job name: shelf_box
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

# organize and set paths before taking <average>
yr_list = [year for year in range(int(args.ys0), int(args.ys1)+1)]
numyrs = len(yr_list)

# name the output file where  seasonal dicts will be dumped
#dd_str = args.ys0 + '_' + Ldir['ys1']
#bb_str = '_'     # leaving this so can update to have other types of clines later in gtags
fn_o = Ldir['parent'] / 'LKH_output' / 'explore_extract_clines' / args.gtagex / args.job / 'seasonal'

for ydx in range(0,numyrs): 
    # inputs 
    fna = args.job+'_'+str(yr_list[ydx])+'.01.01_'+str(yr_list[ydx])+'.12.31' 
    fn_i = Ldir['LOo'] / 'extract' / args.gtagex / 'clines' / fna
    fnb = 'shelf_box_pycnocline_'+str(yr_list[ydx])+'.01.01_'+str(yr_list[ydx])+'.12.31'+'.nc'
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

    if args.variable == 'zDGmax':
        var1 = ds['zDGmax']
        var = np.abs(var1.values)
        
    NT,NR,NC = np.shape(var)
    
    del ds
    
    mmonth = ot.month
    myear = ot.year
    
    if args.stat_type == 'average':
        pn_o = args.job+'_monthly_'+args.stat_type+'_'+str(yr_list[ydx])+'.txt'
        picklepath = fn_o/pn_o
        
        # if monthly & stats = avg, initialize and take avgs
        varidx = {}            # flag index where time is in month ii 
        mvar = {}              # pull those flagged vars
        varmean = {}           # holds averages of monthly vars per grid cell 
        for ii in range(1,13):
            varidx[ii] = np.where(mmonth==ii)   
            mvar[ii] = var[varidx[ii]]          
            # extra steps to take avg so don't get error code when sum across an all nan layer 
            masked_data = np.ma.masked_array(mvar[ii], np.isnan(mvar[ii]))     
            vm = np.ma.average(masked_data, axis=0,keepdims=False)  
            vm[mask_rho==0] = np.nan
            varmean[ii] = vm
            del masked_data
    
        varmean['Lat'] = lat
        varmean['Lon'] = lon
        varmean['myear'] = myear[0]
    
        try: 
            var_file = open(picklepath, 'wt') 
            var_file.write(str(varmean)) 
            var_file.close() 
            print('saving file: ' + pn_o)
  
        except: 
            print('Unable to write to file named: ' + fnt)
        
    
print('Total processing time = %0.2f sec' % (time()-tt0))