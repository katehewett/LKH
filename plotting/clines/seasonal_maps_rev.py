'''
After running group_monthly_output <or equivalent> can 
run this code to produce maps of the data across each 
month and over years 2014 - 20(21)

21 July 2024: 
plotting 2014 - 2017 = Fig 1 
         2018 - 2021 = Fig 2 
And as get more years will change orientation

'''

# imports
from lo_tools import Lfun, zfun
from lo_tools import plotting_functions as pfun

import os 
import xarray as xr
import netCDF4 as nc
from time import time
import numpy as np
import argparse
import pandas as pd
import pickle 

import matplotlib.pyplot as plt
from cmcrameri import cm as cm2
import matplotlib.dates as mdates
from datetime import datetime

# command line arugments
parser = argparse.ArgumentParser()
# which run was used:
parser.add_argument('-gtx', '--gtagex', type=str)   # e.g. cas7_t0_x4b
# select years 
parser.add_argument('-y0', '--ys0', type=str) # e.g. 2014
parser.add_argument('-y1', '--ys1', type=str) # e.g. 2015
parser.add_argument('-stat', '--stat_type', type=str) # stat type: mean
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

tt0 = time()

Ldir = Lfun.Lstart()

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

mo_list = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sept','Oct','Nov','Dec']  
yr_list = list(fig_dict.keys())
numyrs = len(yr_list)

fn_i = Ldir['parent'] / 'LKH_output' / 'explore_extract_clines' / args.gtagex / args.job / 'monthly' / args.variable / 'test'

    
plt.close('all')
fs=11
        
for ydx in range(0,numyrs): 
    
    # load pickled data 
    if args.stat_type == 'mean':
        pn_m = args.variable+'_monthly_average_'+str(yr_list[ydx])+'.txt' 
    elif args.stat_type == 'stdev':
        pn = args.variable+'_monthly_std_'+str(yr_list[ydx])+'.p' 
    elif args.stat_type == 'var':
        pn = args.variable+'_monthly_var_'+str(yr_list[ydx])+'.txt' 
    
    picklepath = fn_i/pn  
    
    if os.path.isfile(picklepath)==False:
        print('no file named: ' + picklepath)
        sys.exit()
    
    # Load the dictionary from the file
    with open(picklepath, 'rb') as fp:
        data1 = pickle.load(fp)
        print('loaded'+str(yr_list[ydx]))

    
