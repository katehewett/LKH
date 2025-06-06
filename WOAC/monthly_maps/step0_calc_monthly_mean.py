'''
This is part of processing required to create monthly means
of Oag at different thresholds (lt 0.5 or 1)

!! code assumes extracted full years were saved XXXX.01.01 - XXXX.12.31 !! 
!! code needs shelf mask to run; mine is under ../LKH_data/shelf_masks/OA_indicators !!

Examples of how to call the script:
On personal computer with python open:
To grab all of the shelf for the OA_indicators job list:
run step0_calc_monthly_mean -gtx cas7_t0_x4b -y0 2020 -job OA_indicators -var arag1 

apogee: 
python step0_calc_monthly_mean.py -gtx cas7_t0_x4b -y0 2020 -job OA_indicators -var arag1 > test.log&

'''

# imports
from lo_tools import Lfun

import os
import sys
import argparse
import pickle
import xarray as xr
import numpy as np
import pandas as pd

# command line arugments
parser = argparse.ArgumentParser()
# which run was used:
parser.add_argument('-gtx', '--gtagex', type=str)   # e.g. cas7_t0_x4b
#parser.add_argument('-ro', '--roms_out_num', type=int) # 2 = Ldir['roms_out2'], etc.
# select years 
parser.add_argument('-y0', '--ys0', type=str) # e.g. 2014
parser.add_argument('-job', '--job_type', type=str) # job 
# enter variable, options: arag1 arag17 arag05:: Arag numbers: 1 = 1; 17 = 1.7; 05 = 0.5
parser.add_argument('-var', '--variable', type=str)
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

# load the shelf mask 
fn_mask =  Ldir['parent'] / 'LKH_data' / 'shelf_masks' / args.job_type / 'OA_indicators_shelf_mask_15_200m_noEstuaries.nc'
dmask = xr.open_dataset(fn_mask) 
mask_shelf = dmask.mask_shelf.values # 0 not-shelf; 1 shelf 
mask_rho = dmask.mask_rho.values     # 0 land; 1 water

# name the output file where files will be dumped
if (args.variable == 'arag17') | (args.variable == 'arag1') | (args.variable == 'arag05'):
    in_dir = Ldir['LOo'] / 'extract' / args.gtagex / 'corrosive_volume' / str('OA_indicators_' + args.ys0 + '.01.01_' + args.ys0 + '.12.31') 
    fn = 'OA_indicators_corrosive_volume_' + args.ys0 + '.01.01_' + args.ys0 + '.12.31.nc'
    fn_in = in_dir / fn
    out_dir = Ldir['parent'] / 'LKH_output' / 'WOAC' / args.gtagex / 'maps_monthly_mean' / 'Oag' / args.variable
else: 
    print('no variable named' + args.variable)
    sys.exit()

if os.path.exists(out_dir)==False:
    Lfun.make_dir(out_dir, clean = False)

if os.path.isfile(fn_in)==False:
    print('no file named: '+fn_in)
    sys.exit()

ds = xr.open_dataset(fn_in, decode_times=True)  

if args.variable == 'arag1':
    dzcorr = ds['corrosive_int_dz'].values
    threshold = 1
elif args.variable == 'arag17':
    dzcorr = ds['corrosive_mild_dz'].values
    threshold = 1.7
elif args.variable == 'arag05':
    dzcorr = ds['corrosive_severe_dz'].values   
    threshold = 0.5  

ot = pd.to_datetime(ds.ocean_time.values)
h = ds['h'].values
D = dzcorr/h

#just focus on OA indicators job area bounds:
# set to zero for summation
D[:,mask_shelf==0] = 0  
D[:,mask_rho==0] = 0

# grab seasonal indicies to sum over 
df = pd.DataFrame({'date': ot})
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.month        # Extract the month from the datetime column

Dmean = {}
for idx in range(1,13):
    print('working on month: ' + str(idx))
    wdx = np.where((df['month']==idx))[0]
    w = D[wdx,:,:]
    wmean = np.nanmean(w,axis=0)
    Dmean[idx] = wmean

# pickle data
Dmean['year'] = args.ys0
Dmean['months'] = 'Jan - Dec'
Dmean['Oag_threshold'] = '<'+str(threshold)
Dmean['calc_region'] = args.job_type
Dmean['units'] = 'mean decimal percent corrosive per month'
Dmean['description'] = '(sum watercolumn dz corrosive)/h'
Dmean['lat_rho'] = ds.lat_rho.values
Dmean['lon_rho'] = ds.lon_rho.values
Dmean['mask_shelf'] = dmask.mask_shelf
Dmean['mask_rho'] = dmask.mask_rho

pn = 'shelf_percent_corrosive_'+args.variable+'_'+args.ys0+'.pkl'
picklepath = out_dir / pn

with open(picklepath, 'wb') as fm:
    pickle.dump(Dmean, fm)
    print('Pickled shelf')