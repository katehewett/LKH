'''
test file from step2b


run step2_calc_N2_seasonal.py -gtx cas7_t0_x4b -y0 2017 -y1 2017

''' 
# imports and set up command line arugments
import os
import sys 
import argparse
import pandas as pd
import numpy as np
import pickle 

from lo_tools import Lfun

parser = argparse.ArgumentParser()
# these flags get only surface or bottom fields if True
# - cannot have both True - It plots one or the other to avoid a 2x loop 
parser.add_argument('-gtx', '--gtagex', type=str)   # e.g. cas7_t0_x4b
# select years 
parser.add_argument('-y0', '--ys0', type=str) # e.g. 2014
parser.add_argument('-y1', '--ys1', type=str) # e.g. 2015
# get the args and put into Ldir
args = parser.parse_args()

Ldir = Lfun.Lstart()

# add this in as a command line argument 
threshold = 1

# name the input and output location where files will be dumped
fn_i = Ldir['parent'] / 'LKH_output' / 'WOAC' / args.gtagex / 'Rig_maps' / 'pickled_values'  
fn_o = Ldir['parent'] / 'LKH_output' / 'WOAC' / args.gtagex / 'Rig_maps' / 'N2_seasonal_pickled_values'

if os.path.exists(fn_o)==False:
    Lfun.make_dir(fn_o, clean=False)

# add to loop 
ys0 = args.ys0 
ys1 = args.ys1 #2024
yr_list = [year for year in range(int(ys0), int(ys1)+1)]
numyrs = len(yr_list)

# 1. concat all the years for ARAG and otime 
for ydx in range(0,numyrs): 

    pn = 'OA_indicators_phys_N2_'+str(yr_list[ydx])+'.pkl'
    picklepath = fn_i/pn

    # Load the dictionary from the file
    with open(picklepath, 'rb') as fp:
        A = pickle.load(fp)
        print('loaded'+str(yr_list[ydx]))

    a = A['N2']
    ot = pd.to_datetime(A['ocean_time'].values)

    if ydx == 0:
        N2 = a
        otime = ot
    else: 
        otime = np.concatenate([otime,ot])
        N2 = np.concatenate([N2,a])

lat_rho=A['lat_rho'].values
lon_rho=A['lon_rho'].values

# grab seasonal indicies to sum over 
df = pd.DataFrame({'date': otime})
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.month        # Extract the month from the datetime column

wdx = np.where((df['month']>=1) & (df['month']<=3)); wdx = np.squeeze(wdx)
spdx = np.where((df['month']>=4) & (df['month']<=6)); spdx = np.squeeze(spdx)
sudx = np.where((df['month']>=7) & (df['month']<=9)); sudx = np.squeeze(sudx)
fdx = np.where((df['month']>=10) & (df['month']<=12)); fdx = np.squeeze(fdx)

winter = N2[wdx,:,:]
N2_winter = {}
#N2_winter['mean'] = np.nanmean(winter,axis=0)
N2_winter['p50'] = np.percentile(winter,50,axis=0)
N2_winter['p25'] = np.percentile(winter,25,axis=0)
N2_winter['p75'] = np.percentile(winter,75,axis=0)
N2_winter['winter_months'] = [1,2,3]
N2_winter['years'] = [args.ys0,args.ys1]
N2_winter['lat_rho'] = lat_rho
N2_winter['lon_rho'] = lon_rho

spring = N2[spdx,:,:]
N2_spring = {}
#N2_spring['mean'] = np.nanmean(spring,axis=0)
N2_spring['p50'] = np.percentile(spring,50,axis=0)
N2_spring['p25'] = np.percentile(spring,25,axis=0)
N2_spring['p75'] = np.percentile(spring,75,axis=0)
N2_spring['spring_months'] = [4,5,6]
N2_spring['years'] = [args.ys0,args.ys1]
N2_spring['lat_rho'] = lat_rho
N2_spring['lon_rho'] = lon_rho

summer = N2[sudx,:,:]
N2_summer = {}
#N2_summer['mean'] = np.nanmean(summer,axis=0)
N2_summer['p50'] = np.percentile(summer,50,axis=0)
N2_summer['p25'] = np.percentile(summer,25,axis=0)
N2_summer['p75'] = np.percentile(summer,75,axis=0)
N2_summer['summer_months'] = [7,8,9]
N2_summer['years'] = [args.ys0,args.ys1]
N2_summer['lat_rho'] = lat_rho
N2_summer['lon_rho'] = lon_rho

fall = N2[fdx,:,:]
N2_fall = {}
#N2_fall['mean'] = np.nanmean(fall,axis=0)
N2_fall['p50'] = np.percentile(fall,50,axis=0)
N2_fall['p25'] = np.percentile(fall,25,axis=0)
N2_fall['p75'] = np.percentile(fall,75,axis=0)
N2_fall['fall_months'] = [10,11,12]
N2_fall['years'] = [args.ys0,args.ys1]
N2_fall['lat_rho'] = lat_rho
N2_fall['lon_rho'] = lon_rho

del picklepath 
del pn 

pw = 'OA_indicators_N2_winter_'+str(args.ys0)+'_'+str(args.ys1)+'.pkl'
psp = 'OA_indicators_N2_spring_'+str(args.ys0)+'_'+str(args.ys1)+'.pkl'
psu = 'OA_indicators_N2_summer_'+str(args.ys0)+'_'+str(args.ys1)+'.pkl'
pf = 'OA_indicators_N2_fall_'+str(args.ys0)+'_'+str(args.ys1)+'.pkl'

picklepath = fn_o/pw
with open(picklepath, 'wb') as fm:
    pickle.dump(N2_winter, fm)
    print('Pickled winter file')
    sys.stdout.flush()

picklepath = fn_o/psp
with open(picklepath, 'wb') as fm:
    pickle.dump(N2_spring, fm)
    print('Pickled spring file')
    sys.stdout.flush()

picklepath = fn_o/psu
with open(picklepath, 'wb') as fm:
    pickle.dump(N2_summer, fm)
    print('Pickled summer file')
    sys.stdout.flush()

picklepath = fn_o/pf
with open(picklepath, 'wb') as fm:
    pickle.dump(N2_fall, fm)
    print('Pickled fall file')
    sys.stdout.flush()
