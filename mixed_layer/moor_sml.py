'''

This script estimates 
- the surface mixed layer using a threshold of 0.03 
- the location of the max drho/dz 
for a mooring extraction 
 
And then saves the data as a pickled file. With an option to make plots

This toy script assumes:
(1) Your input naming structure assumes you saved moorname_YEAR.01.01_YEAR.12.31 
(2) I made an example for a mooring extraction that I already did for you to play with it - ex call below
(3) This works on my computer and apogee, but you'll need to update the output path 'LKH_output' to match your directory 
fn_o = Ldir['parent'] / 'YOURFOLDERNAME' / 'mixed_layer' / 'extract' / args.gtagex / 'moor' / args.job_type

example call:
run moor_mixed_layer -gtx cas7_t0_x4b -y0 2017 -job OCNMS_CE_moorings -moor CE042 -plot True 


set plot to False to save time 
'''

import sys
import os
import argparse
from lo_tools import Lfun

import xarray as xr
import pandas as pd
import numpy as np
from datetime import datetime
import pickle 

import matplotlib.pyplot as plt
import gsw

# command line arugments
parser = argparse.ArgumentParser()
# which run to use
parser.add_argument('-gtx', '--gtagex', type=str)   # e.g. cas7_v0_x4b
# select time period and frequency
parser.add_argument('-y0', '--ys0', type=str) # e.g. 2014
#parser.add_argument('-y1', '--ys1', type=str) # e.g. 2015
#parser.add_argument('-lt', '--list_type', type=str) # list type: hourly, daily, weekly, lowpass
# select job name
parser.add_argument('-job', '--job_type', type=str) # job 
parser.add_argument('-moor', '--mname', type=str) # moor name 
parser.add_argument('-plot', '--mplot', default=False, type=Lfun.boolean_string) # make plot?? 
# for grabbing pycnocline
#parser.add_argument('-pycno', '--pycnocline', default=True, type=Lfun.boolean_string)
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

thisYR = int(args.ys0)

# name the input+output paths where files will be dumped
in_dir = Ldir['LOo'] / 'extract' / args.gtagex / 'moor' / args.job_type 
fn_name = args.mname+'_'+str(thisYR)+'.01.01_'+str(thisYR)+'.12.31'+'.nc'
fn_in = in_dir / fn_name

out_dir = Ldir['parent'] / 'LKH_output' / 'mixed_layer' / 'extract' / args.gtagex / 'moor' / args.job_type
out_dir_fig = out_dir / 'plots'

if os.path.exists(out_dir)==False:
    Lfun.make_dir(out_dir, clean = False)
    Lfun.make_dir(out_dir_fig, clean = False)
if os.path.exists(in_dir)==False:
    print('input path does not exist')
    sys.exit()

#############################for sml bml calc, could make this an argument
threshold = 0.03
#############################

# open and calc mixed layers (surface and bottom)
# already have z rho and w's
ds = xr.open_dataset(fn_in, decode_times=True) 

NT,NZ = np.shape(ds.salt.values)     
NW = np.shape(ds.z_w.values)[1]

z_rho = ds.z_rho.values 
z_w = ds.z_w.values 
lat = ds.lat_rho.values
lon = ds.lon_rho.values

ot = ds.ocean_time.values
df = pd.DataFrame({'datetime':ot}) 
df['date'] = df['datetime'].dt.date

# 1. Calc SA and SIG0: careful!! there's a holdup with sending gsw LARGE files - 
# it can accomodate 4D variables, but has an upper limit 
# (don't worry here, it's chill because we're inputting small files)
tempC = ds.temp.values 
SP = ds.salt.values

P = gsw.p_from_z(z_rho,lat)
SA = gsw.SA_from_SP(SP, P, lon, lat)
CT = gsw.CT_from_pt(SA, tempC)
SIG0 = gsw.sigma0(SA,CT)   
po = np.round(np.nanmean(SIG0+1000),0)
print('po [kg/m3] = '+str(po))

# 2. Estimate SML using the threshold method 
Dsurf = SIG0[:,-1] 
sthresh = np.abs(SIG0 - Dsurf[:,None]) 
 
# flip along z axis so that ocean packed surface:bottom
# find where the first value passes the threshold ("B")
# you don't need to flip, could prob save time if you didn't
fsthresh = np.flip(sthresh, axis=1)
fz_rho = np.flip(z_rho, axis=1)
fz_w = np.flip(z_w, axis=1)

# the flag returned by argmin acts on SIG0, which is on 
# z_rho. The base of the sml, with our thresh, occurs somewhere 
# between those two z_rho points at B and B-1. 
# Assign the midpoint fz_w[:,B] as the base. 
# Note: if sml result is zero, the depth is saved as 
# the first z_w point in the flipped "fz_w" array. 
# So when we calc the thickness, zeta - SML base, 
# we get zero for the SML thickness.
B = np.argmin(fsthresh<threshold,axis=1,keepdims=True)
SML_z = np.take_along_axis(fz_w,B,axis=1)

# 3. find loco of max drho/dz and grab value
dp1 = np.diff(SIG0,axis=1)
dz1 = np.diff(z_rho,axis=1)
dpdz = dp1/dz1

D = np.argmax(np.abs(dpdz),axis=1,keepdims=True)
Dz = np.take_along_axis(z_w,D+1,axis=1)
Dval = np.take_along_axis(dpdz,D,axis=1)

# 4. save files 
mixed_layer = {}
mixed_layer['SML_z [m]'] = SML_z
mixed_layer['dpdz_max_z [m]'] = Dz
mixed_layer['dpdz_value [kg/m3]'] = Dval
mixed_layer['datetimes'] = ot

pn = args.mname + '_SML_'+str(thisYR)+'.01.01_'+str(thisYR)+'.12.31'+'.pkl'
picklepath = out_dir / pn

with open(picklepath, 'wb') as fm:
    pickle.dump(mixed_layer, fm)
    print('mixed layer pickled')

# 4. Plotting - optional 
if args.mplot == True:
    plt.close('all')
    fs=12
    plt.rc('font', size=fs)
    fig = plt.figure(figsize=(16,8))

    ax = plt.subplot2grid((3,1),(0,0),colspan=1,rowspan=1)
    plt.plot(ot,Dz,'b',linewidth=1,alpha=0.8, label = 'max dpdz')
    plt.plot(ot,SML_z,'r',linewidth=1,alpha=0.8, label = 'SML')
    ax.set_title(args.mname)
    ax.set_ylabel('depth [m]')
    plt.legend(loc='lower right')

    fig_name = args.mname + '_SML_'+str(thisYR)+'.01.01_'+str(thisYR)+'.12.31'+'.png'
    fig.savefig(out_dir_fig / fig_name)

if args.mplot == True: 

    ymax = 2
    ymin = np.round(np.nanmin(ds.z_w)) - 1

    for ydx in range(0,NT):
        print('plotting ' + str(df['date'][ydx]))

        plt.close('all')
        fs=12
        plt.rc('font', size=fs)
        fig2 = plt.figure(figsize=(8,8))

        xmin = np.floor(np.nanmin(SIG0[ydx,:])) - 1
        xmax = np.round(np.nanmax(SIG0[ydx,:])) + 1

        ax = plt.subplot2grid((3,3),(0,0),colspan=2,rowspan=3)
        plt.plot(SIG0[ydx,:],z_rho[ydx,:],'k.-', label='SIG0')
        plt.axhline(y=SML_z[ydx,:], color='b', linestyle='--', label='SML threshold')
        plt.axhline(y=Dz[ydx,:], color='r', linestyle='--', label='max drho/dz')
        ax.set_title(str(args.mname + ': ' + str(df['date'][ydx])),loc = 'left')
        
        ax.xaxis.tick_top()
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])

        ax.set_xlabel('SIG0 [kg/m3]')
        ax.set_ylabel('depth [m]')
        plt.legend(loc='lower left')

        ax1 = plt.subplot2grid((3,3),(0,2),colspan=1,rowspan=3)
        plt.plot(-dpdz[ydx,:],z_w[ydx,1:-1],'k.-', label='dpdz')
        ax1.set_xlabel('drho/dz')
        plt.axhline(y=Dz[ydx,:], color='r', linestyle='--', label='max drho/dz')
        plt.axhline(y=SML_z[ydx,:], color='b', linestyle='--', label='SML threshold')
        ax1.set_ylim([ymin, ymax])

        fig2_name = 'sml'+str(ydx+1)+'.png' 
        fig2.savefig(out_dir_fig / fig2_name)
    





