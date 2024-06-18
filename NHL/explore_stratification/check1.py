"""
steb1b (intermediate)
Code to check/plot the SA CT and SIG0 from step1

"""

import xarray as xr
from xarray import open_dataset, Dataset
import numpy as np
import gsw
from lo_tools import Lfun, zrfun
from time import time
import sys
import gsw

import statistics as stat

#plotting things
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
import cmcrameri.cm as cmc

Ldir = Lfun.Lstart()

fnn = 'NHL_transect_2014.01.01_2019.12.31'

fn_in = Ldir['LOo'] / 'extract' / 'cas7_t0_x4b' / 'box' / (fnn + '_chunks') / (fnn + '.nc')
fn_in2 = Ldir['parent'] / 'LKH_output'/ 'NHL' / 'explore_stratification' / fnn / (fnn + '_phys.nc')

idx = 1     # we took data from the lat nearest 44.65, but keep dimensions

ds = xr.open_dataset(fn_in, decode_times=False)
ds2 = xr.open_dataset(fn_in2, decode_times=False)

salt = ds['salt'][200,-1,idx,:].squeeze()
temp = ds['temp'][200,-1,idx,:].squeeze()

salt2 = ds2['SA'][200,-1,:,:].squeeze()
temp2 = ds2['CT'][200,-1,:,:].squeeze()
SIG0 = ds2['SIG0'][200,-1,:,:].squeeze()

ot = ds['ocean_time'].values[:] # this is years since day 1 of extraction ot[0] = 0 and is 2014 1 1 at 12noon 

fig = plt.figure(figsize=(16,8))
ax1 = fig.add_subplot(111)

ax1.scatter(salt,temp,s=10,c='blue',marker = 'o',label='ds')
ax1.scatter(salt2,temp2,s=10,c='red',marker = '.',label='ds2')

fig2 = plt.figure(figsize=(16,8))
ax2 = fig2.add_subplot(111)
plt.plot(SIG0)


# SIG0  pcolormesh
X = ds2.lon_rho.values
Y = ds2['z_rho'][200,:,:,:].squeeze()
C = ds2['SIG0'][200,:,:,:].squeeze()
fig3,ax3 = plt.subplots()
ax3.pcolormesh(X,Y,C,vmin = 17.5, vmax = 27)
