"""
Plots shelf region hypoxia 
"""
# imports
from lo_tools import Lfun, zfun
from lo_tools import plotting_functions as pfun

import sys 
import xarray as xr
import netCDF4 as nc
from time import time
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm
#import cmocean
import matplotlib.dates as mdates
from datetime import datetime
import pandas as pd

plot_regions = True 

Ldir = Lfun.Lstart()
fn_i = Ldir['LOo'] / 'extract' / 'cas6_v0_live' 
fn = fn_i / 'corrosive_volume' / 'combined_2017_2022' / 'corrosive_volumes_2017_2022_withgrid.nc'

ds = xr.open_dataset(fn, decode_times=False)     # vols
ot = ds['ocean_time'].values
days = (ot - ot[0])/86400.
mdays = Lfun.modtime_to_mdate_vec(ot)
mdt = mdates.num2date(mdays) # list of datetimes of data

ds1 = xr.open_dataset('/Users/katehewett/Documents/LKH_output/tests/percent_corrosive/2017_2022_corrosiveVol_percentages.nc')
ds2 = xr.open_dataset('/Users/katehewett/Documents/LKH_output/tests/percent_hypoxic/2017_2022_hypoxicVol_percentages_2.nc')

hyp = ds2.frac_hypV.values;          # 3 x 2191 x 7 < 3 zones, time, 7 regions
severe = ds2.frac_severeV.values;
anoxic = ds2.frac_anoxicV.values;
corr = ds1.frac_corrosive.values;     

NZ, NT, NR = np.shape(hyp)

mmonth = np.nan * np.ones((NT))
myear = np.nan * np.ones((NT))
for jj in range(NT):
    mmonth[jj] = mdt[jj].month
    myear[jj] = mdt[jj].year
    
Rcolors = ['#73210D','#9C6B2F','#C5B563','#A9A9A9','#77BED0','#4078B3','#112F92'] #roma w grey 

plt.close('all')

shelf = 1 # mid 

fs=16
plt.rc('font', size=fs)
fig = plt.figure(figsize=(18,10))
    
ax1 = plt.subplot2grid((2,3), (0,1), colspan=2)
ax2 = plt.subplot2grid((2,3), (1,1), colspan=2)
plt.subplots_adjust(wspace=0, hspace=0)
    
ZZ = corr[shelf,:,3]*0
ax1.fill_between(mdt,ZZ,corr[shelf,:,3], color=Rcolors[3], alpha=0.6)
ax2.fill_between(mdt,ZZ,corr[shelf,:,3], color=Rcolors[3], alpha=0.6)
    
ax1.plot(mdt, corr[shelf,:,0], color=Rcolors[0], linewidth=2, alpha=0.8)
ax1.plot(mdt, corr[shelf,:,1], color=Rcolors[1], linewidth=2, alpha=0.8)
ax1.plot(mdt, corr[shelf,:,2], color=Rcolors[2], linewidth=2, alpha=0.8)
    
ax2.plot(mdt, corr[shelf,:,4], color=Rcolors[4], linewidth=2, alpha=0.8)
ax2.plot(mdt, corr[shelf,:,5], color=Rcolors[5], linewidth=2, alpha=0.8)
ax2.plot(mdt, corr[shelf,:,6], color=Rcolors[6], linewidth=2, alpha=0.8)
          
ax1.set_title('Shelf water corrosive volume, $\u03A9$ $\leq $ 1')
#ax2.set_title('percent of total shelf')
    
ax1.yaxis.set_label_position("right")
ax1.yaxis.tick_right()
ax1.set_ylabel('percent corrosive')
#ax1.set_ylabel('percent total shelf')
    
ax2.yaxis.set_label_position("right")
ax2.yaxis.tick_right()
ax2.set_ylabel('percent corrosive')
#ax2.set_ylabel('percent total shelf')
    
#ax1.legend(['volume', 'percentage'])
    
ax1.grid(True)
ax2.grid(True)
    
ax1.set_xticks([datetime(2017,1,1),datetime(2017,7,1),datetime(2018,1,1), datetime(2018,7,1),
datetime(2019,1,1),datetime(2019,7,1), datetime(2020,1,1),datetime(2020,7,1),
datetime(2021,1,1),datetime(2021,7,1),datetime(2022,1,1),datetime(2022,7,1),datetime(2022,12,31)])
 
ax1.set_xticklabels([ ])   
ax1.tick_params(axis = "x", labelsize = 14, labelrotation = 0)

ax1.set_xlim(mdt[0], mdt[-1])
ax2.set_xlim(mdt[0], mdt[-1])

ax1.set_ylim(0,100)
ax2.set_ylim(0,100)

ax2.set_xticks([datetime(2017,1,1),datetime(2017,7,1),datetime(2018,1,1), datetime(2018,7,1),
datetime(2019,1,1),datetime(2019,7,1), datetime(2020,1,1),datetime(2020,7,1),
datetime(2021,1,1),datetime(2021,7,1),datetime(2022,1,1),datetime(2022,7,1),datetime(2022,12,31)],)
    
ax2.set_xticklabels(['Jan17','Jul','Jan18','Jul',
'Jan18','Jul','Jan19','Jul',
'Jan20','Jul','Jan21','Jul','Jan22'])
    
ax2.tick_params(axis = "x", labelsize = 14, labelrotation = 0)
    
ax1.set_axisbelow(True)
ax2.set_axisbelow(True)

fig.savefig('/Users/katehewett/Documents/LKH_output/tests/fig_wcoa/mid_Oag.png')


    