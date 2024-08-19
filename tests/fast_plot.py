'''
checking lat lon

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

Ldir = Lfun.Lstart()

## want h and mask_rho (didn't save with pickle files)
fna = 'LO_2014.01.01_2014.12.31' 
fn_i = Ldir['LOo'] / 'extract' / 'cas7_t0_x4b' / 'hypoxic_volume' / fna
fnb = 'LO_hypoxic_volume_lowpass_2014.01.01_2014.12.31'+'.nc'
fn_in = fn_i / fnb
ds = xr.open_dataset(fn_in, decode_times=True)
h = ds['h']
mask_rho = ds['mask_rho']

plt.close('all')
fs=11
plt.rc('font', size=fs)
height_of_image = 18 
width_of_image = 10 
fig1 = plt.figure(figsize=(height_of_image,width_of_image))
fig1.set_size_inches(height_of_image,width_of_image, forward=False)
frac=0.047 * (height_of_image / (width_of_image/13))

ax = plt.subplot2grid((1,3), (0,1), colspan=1,rowspan=1)
pfun.add_coast(ax)
pfun.dar(ax)
ax.axis([-125.5, -123.5, 42, 48.75])
ax.contour(ds.lon_rho.values,ds.lat_rho.values,h.values, [200],colors=['black'], linewidths=1, linestyles='solid',alpha=0.4)


		