"""
Code calculates SML and BML using threshold method 
Flag first instance where:
del T(z_rho) = T(z_rho) - T0 > -0.05 
where T0 is either the surface(bottom)-most temperature value

We mark the base(top) of the SML(BML) as the z_w coordinate 
at the bottom(top) of the flagged rho-point

Note: this code was writted to work on nc-files extracted using 
LO/extract/box/extract_box_chunks.py
which have grid info saved. If re-write can get grid info using
a history file and G, S, T = zrfun.get_basic_info(), which can then 
calc z_rho and z_w 

We are exploring the NHL line in these test scripts. 
Want to calculate the SML BML and locaiton of max N2 
for cas7_t0_x4b years 2014 - 2019 
(2019 = last full year processed; 2013 skipped for now)
Especially interested in blob years and high CR river year 2017

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

fnn = 'NHL_transect_2014.01.01_2019.12.31'
fn_in = Ldir['LOo'] / 'extract' / 'cas7_t0_x4b' / 'box' / (fnn + '_chunks') / (fnn + '.nc')

# 1. Get some grid information 
tt0 = time()
ds = xr.open_dataset(fn_in, decode_times=False)
zsurf = ds.z_w.values[:,-1,:,:]
zsurf = ds.z_w.values[:,-1,0,0].squeeze()

plt.close('all')
fs=12
plt.rc('font', size=fs)
fig = plt.figure(figsize=(16,8))

ax1 = plt.subplot2grid((4,4),(0,0),colspan=1,rowspan=1)
axnum = 0 
zplot = plt.gcf().axes[axnum].plot(ds.ocean_time,zsurf,color='blue',marker='none') 





