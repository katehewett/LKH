'''
plotting seasonal maps of 
sml 
N2max + depth 



'''

# imports
from lo_tools import Lfun, zfun
from lo_tools import plotting_functions as pfun
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import sys 
import xarray as xr
import netCDF4 as nc
from time import time
import numpy as np
import math

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cmcrameri.cm as cmc
#import cmocean
import matplotlib.dates as mdates
from datetime import datetime
import pandas as pd


tt0 = time()

Ldir = Lfun.Lstart()

fig_dict = {
    '2014':[1,0,0], # YEAR: fig num, subplot axrow if split, numerical order
    '2015':[1,1,1],
    '2016':[1,2,2],
    '2017':[1,3,3],
    '2018':[2,0,4],
    '2019':[2,1,5],
    '2020':[2,2,6],
    '2021':[2,3,7]
}

mo_list = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sept','Oct','Nov','Dec']

plt.close('all')
fs=11
plt.rc('font', size=fs)
fig1 = plt.figure(figsize=(18,10))
fig1.set_size_inches(18,10, forward=False)
    
fig2 = plt.figure(figsize=(18,10))
fig2.set_size_inches(18,10, forward=False)
    
yr_list = list(fig_dict.keys())
numyrs = len(yr_list)

fn_o = Ldir['parent'] / 'plotting' / 'clines' 

for ydx in range(0,numyrs): 
    # loop thru and load files 
    fna = 'shelf_box_'+yr_list[ydx]+'.01.01_'+yr_list[ydx]+'.12.31'
    fn_i = Ldir['LOo'] / 'extract' / 'cas7_t0_x4b' / 'clines' / fna
    fnb = 'shelf_box_pycnocline_'+yr_list[ydx]+'.01.01_'+yr_list[ydx]+'.12.31'+'.nc'
    fn_in = fn_i / fnb

    ds = xr.open_dataset(fn_in, decode_times=True)     

    ot = pd.to_datetime(ds.ocean_time.values)

    xrho = ds['lon_rho'].values
    yrho = ds['lat_rho'].values
    mask_rho = ds['mask_rho'].values
    h = ds['h'].values

    #var = ds['zSML'].values
    var1 = ds['zDGmax']
    var = np.abs(var1.values)
    NT,NR,NC = np.shape(var)
    
    del ds
    
    mmonth = ot.month
    myear = ot.year
    
    # if monthly & stats = avg, initialize and take avgs
    varidx = {}            # flag index where time is in month ii 
    mvar = {}              # pull those flagged vars
    varmean = {}           # holds averages of monthly vars per grid cell 
    for ii in range(1,13):
        varidx[ii] = np.where(mmonth==ii)   
        mvar[ii] = var[varidx[ii]]          
        # extra steps to take avg so don't get error code when sum across an all nan layer 
        masked_data = np.ma.masked_array(mvar[ii], np.isnan(mvar[ii]))     
        vm = np.ma.average(masked_data, axis=0,keepdims=False)  
        vm[mask_rho==0] = np.nan
        varmean[ii] = vm
        del masked_data
        
    if fig_dict[yr_list[ydx]][0]==1:
        fig1
    elif fig_dict[yr_list[ydx]][0]==2:
        fig2
    
    # plot each months map, 13th column is saved for colorbar and text 
    for ii in range(1,13): 
        axnum = ii-1 

        ax = plt.subplot2grid((4,13), (fig_dict[yr_list[ydx]][1],axnum), colspan=1,rowspan=1)
        pfun.add_coast(ax)
        pfun.dar(ax)
        ax.axis([-127.5, -123.5, 43, 50])
        ax.contour(xrho,yrho,h, [200],colors=['black'], linewidths=1, linestyles='solid',alpha=0.4)
        ax.set_xticks([-127.5, -125, -123.5])
        ax.set_title(mo_list[axnum]+' '+yr_list[ydx])
        #axw.grid(True)
        
        if ii<12 and ii!=1: 
            ax.yaxis.set_ticklabels([])
        if ii == 12:
            ax.yaxis.tick_right()
        if int(yr_list[ydx]) != 2017 | int(yr_list[ydx]) != 2021:
            ax.xaxis.set_ticklabels([])

        smin = 10 #math.floor(np.min(var))
        smax = 150 #math.ceil(np.max(var))
        slevels = np.arange(smin,smax,0.5)
        smap=cmc.roma_r.with_extremes(under='Navy',over='Maroon')
        
        cpm = ax.contourf(xrho, yrho, varmean[ii],slevels,cmap=smap,extend = "both") 
        
        
        plt.gcf().tight_layout()




