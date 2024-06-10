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
ds1 = xr.open_dataset(fn_in2, decode_times=False)

plt.close('all')
fs=12
plt.rc('font', size=fs)
fig = plt.figure(figsize=(16,8))

salt = ds['salt'][200,:,idx,:]
temp = ds['temp'][200,:,idx,:] 

salt2 = ds2['SA'][200,:,idx,:]
temp2 = ds2['CT'][200,:,idx,:] 

plt.plot(salt,temp,'o')
plt.plot(salt2,temp2,'x')

plt.show()