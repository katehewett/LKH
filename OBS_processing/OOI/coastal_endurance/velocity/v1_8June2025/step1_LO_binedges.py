'''
Step 1: 
This code opens associated LO mooring extractions 
and grabs z_rho and z_w and pickles for later processing steps 

Enter the info you used for a grid extraction 
For example (I have already extracted):
-gtx cas7_t0_x4b -ro 2 -0 2017.01.01 -1 2017.12.31 -lt lowpass -job OOI_WA_SM
-gtx cas7_t0_x4b -ro 2 -0 2017.01.01 -1 2017.12.31 -lt lowpass -job OOI_OR_SM

example call:
run step1_LO_binedges -gtx cas7_t0_x4b -ro 2 -0 2017.01.01 -1 2017.12.31 -lt lowpass -job OOI_WA_SM
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
# which run to use
parser.add_argument('-gtx', '--gtagex', type=str)   # e.g. cas6_v3_l08b
parser.add_argument('-ro', '--roms_out_num', type=int) # 1 = Ldir['roms_out1'], etc.
# select time period and frequency
parser.add_argument('-0', '--ds0', type=str) # e.g. 2019.07.04
parser.add_argument('-1', '--ds1', type=str) # e.g. 2019.07.06
parser.add_argument('-lt', '--list_type', type=str) # list type: hourly, daily, or lowpass
parser.add_argument('-job', type=str) # job name 
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

#input path + list of files in path
in_dir = Ldir['parent'] / 'LO_output' / 'extract' / args.gtagex / 'moor' / args.job 

if os.path.exists(in_dir)==True:
    fn_list = os.listdir(in_dir)
if os.path.exists(in_dir)==False:
    print('input path does not exist')
    sys.exit()

numfiles = len(fn_list)

for nf in fn_list:
    fn = nf
    print('working on file: ' + fn)
    fn_in = in_dir / fn
    ds = xr.open_dataset(fn_in)
    sys.exit()

sys.exit()
fn_list = Lfun.get_fn_list(Ldir['list_type'], Ldir, Ldir['ds0'], Ldir['ds1'])
