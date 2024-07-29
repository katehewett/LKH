'''
Option to run this code first before plotting <monthly/seasonal> maps
of data produced by extract/box or clines.
This code will make seasonal groupings <e.g. montly avgs> 
of data produced from extract_hypoxic_volume.py 

Right now this code will group data by month, 
and perform stat_type that we input: 
basic = mean "average" std and var

It will save a smaller pickled file for that variable

!! code assumes extracted full years were saved XXXX.01.01 - XXXX.12.31 !! 

personal computer // python 
run group_monthly_output -gtx cas7_t0_x4b -y0 2014 -y1 2021 -mvar hyp_dz -lt lowpass -stat basic 

apogee: 
python group_monthly_output.py -gtx cas7_t0_x4b -y0 2014 -y1 2021 -mvar hyp_dz -stat basic > DGmax.log &
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
parser.add_argument('-lt', '--list_type', type=str) # stat type: average
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
fn_o = Ldir['parent'] / 'LKH_output' / 'hypoxic_volume' / args.gtagex / 'monthly' / args.variable
Lfun.make_dir(fn_o, clean=False)

for ydx in range(0,numyrs): 
    # inputs 
    fna = 'LO_'+str(yr_list[ydx])+'.01.01_'+str(yr_list[ydx])+'.12.31' 
    fn_i = Ldir['LOo'] / 'extract' / args.gtagex / 'hypoxic_volume' / fna
    fnb = 'LO_hypoxic_volume_'+str(args.list_type)+'_'+str(yr_list[ydx])+'.01.01_'+str(yr_list[ydx])+'.12.31'+'.nc'
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

    # TODO Remove abs values and rerun - was confusing when looking at data earlier. 
    if args.variable == 'hyp_dz':
        var1 = ds['hyp_dz']
        var = np.abs(var1.values)
    elif args.variable == 'mild_dz':
        var1 = ds['mild_dz']
        var = np.abs(var1.values)
    elif args.variable == 'severe_dz':
        var1 = ds['severe_dz']
        var = np.abs(var1.values)
    elif args.variable == 'anoxic_dz':
        var1 = ds['anoxic_dz']
        var = np.abs(var1.values)
        
    NT,NR,NC = np.shape(var)
    
    del ds
    
    mmonth = ot.month
    myear = ot.year
    
    if args.stat_type == 'basic':
        # set output path picklepath and pickled filename pn_o
        pn_m = args.variable+'_monthly_average_'+str(yr_list[ydx])+'.pkl'
        pn_s = args.variable+'_monthly_std_'+str(yr_list[ydx])+'.pkl'
        pn_v = args.variable+'_monthly_var_'+str(yr_list[ydx])+'.pkl'
        mpicklepath = fn_o/pn_m
        spicklepath = fn_o/pn_s
        vpicklepath = fn_o/pn_v
    
        varidx = {}            # flag index where time is in month ii 
        mvar = {}              # pull those flagged vars
        vmean = {}             # holds averages of monthly vars per grid cell
        vstd = {}              #       stdev 
        vvar = {}              #       variances 
                
        for ii in range(1,13):
            vbool = (mmonth == ii)*1            # *1 so 1/0 not T/F
            V = var[vbool==1,:,:]
            # fof all-NaN slices, a RuntimeWarning is raised. To avoid, mask data 1st:
            masked_data = np.ma.masked_array(V, np.isnan(V)) 
            vm = np.nanmean(masked_data,axis=0,keepdims=False)
            vs = np.nanstd(masked_data,axis=0,keepdims=False)
            vv = np.nanvar(masked_data,axis=0,keepdims=False)
            
            vm[mask_rho==0] = np.nan
            vs[mask_rho==0] = np.nan
            vv[mask_rho==0] = np.nan
            
            vmean[ii] = vm
            vstd[ii] = vs
            vvar[ii] = vv
            
            del masked_data 
    
        vmean['Lat'] = lat
        vmean['Lon'] = lon
        vmean['myear'] = myear[0]
        
        vstd['Lat'] = lat
        vstd['Lon'] = lon
        vstd['myear'] = myear[0]
        
        vvar['Lat'] = lat
        vvar['Lon'] = lon
        vvar['myear'] = myear[0]
        
        # average
        with open(mpicklepath, 'wb') as fm:
            pickle.dump(vmean, fm)
            print('vmean dict saved successfully to file')
        
        # stdev     
        with open(spicklepath, 'wb') as fs:
            pickle.dump(vstd, fs)
            print('vstd dict saved successfully to file')
        
        # var
        with open(vpicklepath, 'wb') as fv:
            pickle.dump(vvar, fv)
            print('vvar dict saved successfully to file')
        
    
print('Total processing time = %0.2f sec' % (time()-tt0))