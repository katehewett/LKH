'''
grabs h and mask_rho for pickle files 
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
# select years 
parser.add_argument('-y0', '--ys0', type=str) # e.g. 2014
parser.add_argument('-y1', '--ys1', type=str) # e.g. 2014
# select job name used
parser.add_argument('-job', type=str) # job name: shelf_box
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

# name the output file where  seasonal dicts will be dumped
fn_o = Ldir['parent'] / 'LKH_output' / 'explore_extract_clines' / args.gtagex / args.job / 'monthly' 
Lfun.make_dir(fn_o, clean=True)

fna = args.job+'_'+str(yr_list[ydx])+'.01.01_'+str(yr_list[ydx])+'.12.31' 
fn_i = Ldir['LOo'] / 'extract' / args.gtagex / 'clines' / fna
fnb = 'shelf_box_pycnocline_'+str(yr_list[ydx])+'.01.01_'+str(yr_list[ydx])+'.12.31'+'.nc'
fn_in = fn_i / fnb

ds2 = dataset('')
ds = xr.open_dataset(fn_in, decode_times=True)   
h = ds.h
mask_rho = ds.mask_rho

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
    elif args.variable == 'DGmax':
        var1 = ds['DGmax']
        var = np.abs(var1.values)
#    elif args.variable == 'SA_DGmax':
#        var1 = ds.SA[:,2,:,:]       # grab DGmax zone = 2
#        var = np.abs(var1.values)
    elif args.variable == 'zSML':
        var1 = ds['zSML']
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
        vmean = {}           # holds averages of monthly vars per grid cell
        vstd = {}            #       stdev 
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