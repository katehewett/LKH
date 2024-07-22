'''
After running group_monthly_output <or equivalent> can 
run this code to produce maps of the data across each 
month and over years 2014 - 20(21)

21 July 2024: 
plotting 2014 - 2017 = Fig 1 
         2018 - 2021 = Fig 2 
And as get more years will change orientation

run plot_monthly_maps -gtx cas7_t0_x4b -y0 2014 -y1 2017 -mvar zDGmax -stat vavg -job shelf_box -vmin 0 -vmax 150
run plot_monthly_maps -gtx cas7_t0_x4b -y0 2014 -y1 2017 -mvar zDGmax -stat vavg -job shelf_box -vmin 0 -vmax 50

run plot_monthly_maps -gtx cas7_t0_x4b -y0 2014 -y1 2017 -mvar DGmax -stat vavg -job shelf_box 


'''

# imports
from lo_tools import Lfun, zfun
from lo_tools import plotting_functions as pfun

import os 
import sys
import xarray as xr
import netCDF4 as nc
from time import time
import numpy as np
import argparse
import pandas as pd
import pickle 

import matplotlib.pyplot as plt
from cmcrameri import cm as cmc
import matplotlib.dates as mdates
from datetime import datetime

# command line arugments
parser = argparse.ArgumentParser()
# which run was used:
parser.add_argument('-gtx', '--gtagex', type=str)   # e.g. cas7_t0_x4b
# select years 
parser.add_argument('-y0', '--ys0', type=str) # e.g. 2014
parser.add_argument('-y1', '--ys1', type=str) # e.g. 2015
parser.add_argument('-stat', '--stat_type', type=str) # stat type: vavg, vstdev, vvar
parser.add_argument('-mvar', '--variable', type=str) # select variable  
# select job name used
parser.add_argument('-job', type=str) # job name: shelf_box
# Optional: select min/max scale on colorbar
parser.add_argument('-vmin','--min_val', type=str) 
parser.add_argument('-vmax','--max_val', type=str) 
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

tt0 = time()

Ldir = Lfun.Lstart()

# set up some things for plotting and define input path 
fig_dict = {
    '2014':[1,0,0], # YEAR: fig num, subplot axrow if split, numerical order
    '2015':[1,1,1],
    '2016':[1,2,2],
    '2017':[1,3,3],
    '2018':[2,0,4],
    '2019':[2,1,5],
    '2020':[2,2,6],
    '2021':[2,3,7]
}

yr_list = [year for year in range(int(args.ys0), int(args.ys1)+1)]
numyrs = len(yr_list)
mo_list = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sept','Oct','Nov','Dec']  

## want h and mask_rho (didn't save with pickle files)
fna = args.job+'_'+str(yr_list[0])+'.01.01_'+str(yr_list[0])+'.12.31' 
fn_i = Ldir['LOo'] / 'extract' / args.gtagex / 'clines' / fna
fnb = 'shelf_box_pycnocline_'+str(yr_list[0])+'.01.01_'+str(yr_list[0])+'.12.31'+'.nc'
fn_in = fn_i / fnb
ds = xr.open_dataset(fn_in, decode_times=True)
h = ds['h']
mask_rho = ds['mask_rho']
del ds, fn_i, fnb, fn_in

fn_i = Ldir['parent'] / 'LKH_output' / 'explore_extract_clines' / args.gtagex / args.job / 'monthly' / args.variable 
fn_o = Ldir['parent'] / 'LKH_output' / 'explore_extract_clines' / args.gtagex / args.job / 'monthly' / args.variable / 'plotting'
Lfun.make_dir(fn_o, clean=False)

plt.close('all')
fs=11
plt.rc('font', size=fs)
height_of_image = 18 
width_of_image = 10 
fig1 = plt.figure(figsize=(height_of_image,width_of_image))
fig1.set_size_inches(height_of_image,width_of_image, forward=False)
frac=0.047 * (height_of_image / (width_of_image/13))
          
# open pickle files and plot all years       
for ydx in range(0,numyrs): 
    
    # load pickled data 
    if args.stat_type == 'vavg':
        pn = args.variable+'_monthly_average_'+str(yr_list[ydx])+'.pkl' 
    elif args.stat_type == 'vstdev':
        pn = args.variable+'_monthly_std_'+str(yr_list[ydx])+'.pkl' 
    elif args.stat_type == 'vvar':
        pn = args.variable+'_monthly_var_'+str(yr_list[ydx])+'.pkl' 
    
    picklepath = fn_i/pn  
    
    if os.path.isfile(picklepath)==False:
        print('no file named: ' + pn)
        sys.exit()
    
    # Load the dictionary from the file
    with open(picklepath, 'rb') as fp:
        data1 = pickle.load(fp)
        print('loaded'+str(yr_list[ydx]))
    
    #if yr_list[ydx] == 2014:
    #    plt.rc('font', size=fs)
    #    fig1 = plt.figure(figsize=(18,10))
    #    fig1.set_size_inches(18,10, forward=False)
    #elif yr_list[ydx] == 2018:
    #    plt.rc('font', size=fs)
    #    fig2 = plt.figure(figsize=(18,10))
    #    fig2.set_size_inches(18,10, forward=False)
        
    # plot each months map, 13th column is saved for colorbar and text 
    for ii in range(1,13): 
        axnum = ii-1 

        ax = plt.subplot2grid((4,13), (fig_dict[str(yr_list[ydx])][1],axnum), colspan=1,rowspan=1)
        pfun.add_coast(ax)
        pfun.dar(ax)
        ax.axis([-125, -123.5, 46, 49])
        ax.contour(data1['Lon'],data1['Lat'],h.values, [200],colors=['black'], linewidths=1, linestyles='solid',alpha=0.4)
        ax.set_xticks([-125.5,-125,-124.5,-124,-123.5])
        ax.set_title(mo_list[axnum]+' '+str(data1['myear']))
        #axw.grid(True)
        
        if ii<12 and ii!=1: 
            ax.yaxis.set_ticklabels([])
        if ii == 12:
            ax.yaxis.tick_right()
        
        ax.xaxis.set_ticklabels([])
            
        if int(yr_list[ydx]) == 2017 or int(yr_list[ydx]) == 2021:
            ax.xaxis.set_ticklabels([-125.5,'','','',-123.5])
        
        if args.variable == 'zDGmax':  
            if args.stat_type == 'vavg':
                smap=cmc.roma.with_extremes(under='Maroon',over='Navy')
                if args.max_val == None: smax = 150 #math.ceil(np.max(var))
                else: smax = int(args.max_val)
                if args.min_val == None: smin = 0  #math.floor(np.min(var))
                else: smin = int(args.min_val)
            elif args.stat_type == 'vstdev':
                smap=cmc.roma_r.with_extremes(under='Navy',over='Maroon')
                if args.max_val == None: smax = 60 #math.ceil(np.max(var))
                else: smax = int(args.max_val)
                if args.min_val == None: smin = 0  #math.floor(np.min(var))
                else: smin = int(args.min_val)
        
            slevels = np.arange(smin,smax+0.5,0.5)

        cpm = ax.contourf(data1['Lon'], data1['Lat'], data1[ii].data,slevels,cmap=smap,extend = "both")      
        fig1.tight_layout()
        
axc = plt.subplot2grid((4,13), (0,12), colspan=1,rowspan=4)
cpm2 = axc.contourf(data1['Lon'], data1['Lat'], data1[ii].data,slevels,cmap=smap,extend = "both")
if args.variable == 'zDGmax' and args.stat_type == 'vavg':
    ff = np.floor((smax+1-smin)/10)
    ll = [go for go in range(smin,smax+1,10)]
    tcb = plt.gcf().colorbar(cpm2, ticks = ll, location='right',pad = 0.05, fraction = frac, label='avg zDGmax [m]')
    #tcb.ax.yaxis.set_ticks_position('left')
    #tcb.ax.yaxis.set_label_position('left')
    tcb.ax.invert_yaxis()
elif args.variable == 'zDGmax' and args.stat_type == 'vstdev':
    ff = np.floor((smax+1-smin)/10)
    ll = [go for go in range(smin,smax+1,10)]
    tcb = plt.gcf().colorbar(cpm2, ticks = ll, location='right',pad = 0.05, fraction = frac, label='stdev zDGmax')
    
axc.remove()

figname = 'NEW_WA_monthly_' + args.stat_type + '_' + args.variable + '_' + args.ys0 + '_' + args.ys1 + '.png'
fig1.savefig(fn_o / figname)
