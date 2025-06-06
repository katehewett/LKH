'''
calc monthly means of N2


run step2_calc_N2_monthly -gtx cas7_t0_x4b -y0 2017 -y1 2017

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
fn_o = Ldir['parent'] / 'LKH_output' / 'WOAC' / args.gtagex / 'Rig_maps' / 'N2_monthly_means'

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

N2mean = {}
for idx in range(1,13):
    print('working on month: ' + str(idx))
    wdx = np.where((df['month']==idx))[0]
    w = N2[wdx,:,:]
    wmean = np.nanmean(w,axis=0)
    N2mean[idx] = wmean

N2mean['lat_rho'] = lat_rho
N2mean['lon_rho'] = lon_rho

if args.ys0==args.ys1:
    pn = 'shelf_monthly_mean_N2_' + args.ys0 + '.pkl'
else: 
    pn = 'shelf_monthly_means_N2_' + args.ys0 + '_' + args.ys1 + '.pkl'

picklepath = fn_o/pn
with open(picklepath, 'wb') as fm:
    pickle.dump(N2mean, fm)
    print('Pickled file')
    sys.stdout.flush()