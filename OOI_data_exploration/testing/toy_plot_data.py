# testing script to plot requested data from OOI M2M 
# run after toy_extract_data.py

# imports
from lo_tools import Lfun, zfun
from lo_tools import plotting_functions as pfun

import sys
import os
import xarray as xr
import netCDF4 as nc
from time import time
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import matplotlib.dates as mdates
from datetime import datetime
import pandas as pd

fn_p = '/Users/katehewett/Documents/LKH_output/OOI_data_exploration/testing'
fn = 'CE02SHSM_RID27_telemetered_ctdbp_cdef_dcl_instrument.nc'
fnp = os.path.join(fn_p, fn)

ds = xr.open_dataset(fnp, decode_times=False)
ot = ds['ocean_time'].values
days = (ot - ot[0])/86400.
mdays = Lfun.modtime_to_mdate_vec(ot)
mdt = mdates.num2date(mdays) # list of datetimes of data

SP = ds['SP']

fs=16
plt.rc('font', size=fs)
fig = plt.figure(figsize=(18,10))

plot(mdt, SP, color='#73210D', linewidth=2, alpha=0.8)

