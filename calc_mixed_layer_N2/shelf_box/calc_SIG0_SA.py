"""
Code calculates SA and SIG0 for large nc files that want to use 
GSW, which crashes w

Note: this code was writted to work on nc-files extracted using 
LO/extract/box/extract_box_chunks.py
which have grid info saved. If re-write can get grid info using
a history file and G, S, T = zrfun.get_basic_info(), which can then 
calc z_rho and z_w 

We are exploring the shelf_box job here 
Want to calculate the SML BML and location of max N2 
for cas7_t0_x4b years 2014 - 2019 
(2019 = last full year processed; 2013 skipped for now)

Todos: write flags so can enter 'box' or 'moor' and the filename etc 
so can use command line 

"""

import xarray as xr
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

fnn = 'shelf_box_2014.01.01_2015.12.31'
#fnn = 'shelf_box_2016.01.01_2017.12.31'
#fnn = 'shelf_box_2018.01.01_2019.12.31'
fn_in = Ldir['LOo'] / 'extract' / 'cas7_t0_x4b' / 'box' / (fnn + '_chunks') / (fnn + '.nc')

testing = True

# 1. Get some grid information and calc SA + SIG0
tt0 = time()
ds = xr.open_dataset(fn_in, decode_times=False)
print('Time to load data = %0.2f sec' % (time()-tt0))

lat = ds.lat_rho
lon = ds.lon_rho
h = ds.h.values 
mask_rho = ds.mask_rho

print('testing loop')

z_w = ds.z_w
z_rho = ds.z_rho
tempC = ds.temp
SP = ds.salt

NT, NZ, NETA, NXI = tempC.shape

# GSW calcs can handle a 4D value P = gsw.p_from_z(z_rho,lat)
# but gsw scripts are VERY slow or crash when NT is large 
# these lines of script are to speed up the process - should swap 
# to using something like chunk
SA = np.nan * np.ones(SP.shape)           # initialize to hold results (time, z, x, y)
CT = np.nan * np.ones(SP.shape)
SIG0 = np.nan * np.ones(SP.shape)
P = np.nan * np.ones(SP.shape)

for ii in range(5):
    P[ii,:,:,:] = gsw.p_from_z(z_rho[ii,:,:,:],lat) 
    SA[ii,:,:,:] = gsw.SA_from_SP(SP[ii,:,:,:], P[ii,:,:,:], lon, lat)
    CT[ii,:,:,:] = gsw.CT_from_pt(SA[ii,:,:,:], tempC[ii,:,:,:])
    SIG0[ii,:,:,:] = gsw.sigma0(SA[ii,:,:,:],CT[ii,:,:,:])

print('Time to extract data = %0.2f sec' % (time()-tt0))
sys.stdout.flush()
