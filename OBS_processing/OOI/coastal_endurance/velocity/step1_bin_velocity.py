'''
Step 2: 
This code is to process data downloaded from OOI for the WA(and OR) surface mooring velocity data 

(1) opens a LO mooring extraction to grab bin edges 
(2) opens associated velocity data and  bins accordingly
(3) organizes and plots data to see what is there

need to enter the info you entered for a grid extraction (just one - to grab the bin edges for the mooring):
For example (I have already extracted): -gtx cas7_t0_x4b -ro 2 -0 2017.01.01 -1 2017.12.31 -lt lowpass -job OOI_WA_SM

-gtx cas7_t0_x4b (this is the grid I compared)
-y0 2017 (this is the year moors were extracted)
-job OOI_WA_SM (OOI_OR_SM)

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