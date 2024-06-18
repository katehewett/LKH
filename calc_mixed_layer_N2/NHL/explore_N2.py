"""
Code calculates SA and SIG0 + N2 and dT/dz
Locates position of N2 max and dT/dz max

Note: this code was writted to work on nc-files extracted using 
LO/extract/box/extract_box_chunks.py
which have grid info saved. If re-write can get grid info using
a history file and G, S, T = zrfun.get_basic_info(), which can then 
calc z_rho and z_w 

We are exploring the NHL_transect job here 
for cas7_t0_x4b years 2014 - 2019 
(2019 = last full year processed; 2013 skipped for now)

All years 
Time to load data = 0.09 sec
Time to extract data = 0.10 sec
Time to calc gsw vars = 72.00 sec
Time to calc N2 = 4.59 sec

"""

import xarray as xr
import pandas as pd
from xarray import open_dataset, Dataset
import numpy as np
import gsw
from lo_tools import Lfun, zrfun
from time import time
import sys
import gsw

#plotting things
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
import cmcrameri.cm as cmc

Ldir = Lfun.Lstart()

fnn = 'NHL_transect_2014.01.01_2019.12.31'
fn_in = Ldir['LOo'] / 'extract' / 'cas7_t0_x4b' / 'box' / (fnn + '_chunks') / (fnn + '.nc')

in_dir = Ldir['parent'] / 'LKH_output'/ 'calc_mixed_layer_N2' / 'NHL' / fnn
fn_s = fnn + '_N2_S2.nc'
fn_in = in_dir / (fn_s)

ds = xr.open_dataset(fn_in, decode_times=False)

NT, NZ, NETA, NXI = ds.CT.shape
NW = ds.z_w.shape[1]

df = pd.DataFrame({'times':ds['ocean_time']})



