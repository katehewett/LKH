# this is a script to plot processed OCNMS data from LO/obs 
# This script will generate a data availabilty plot for 10 sites 

# imports
from lo_tools import Lfun, zfun, zrfun
from lo_tools import plotting_functions as pfun

#from time import time
#import sys 
import xarray as xr
#import netCDF4 as nc

import matplotlib.pyplot as plt
import numpy as np
#import matplotlib.dates as mdates
#from datetime import datetime
#import pandas as pd

Ldir = Lfun.Lstart()

# processed data location
source = 'ocnms'
otype = 'moor' 
data_dir = Ldir['LOo'] / 'obs' / source / otype

# named in this order for plotting 
sn_name_dict = {
    'MB042':'Makah Bay 42m',
    'MB015':'Makah Bay 15m',
    'CA042':'Cape Alava 42m',
    'CA015':'Cape Alava 15m',
    'TH042':'Teahwhit Head 42m',
    'TH015':'Teahwhit Head 15m',
    'KL027':'Kalaloch 27m',
    'KL015':'Kalaloch 15m',
    'CE042':'Cape Elizabeth 42m',
    'CE015':'Cape Elizabeth 15m'  
}

# PLOTTING
plt.close('all')
fs=12
plt.rc('font', size=fs)
fig = plt.figure(figsize=(18,10))

# add mooring locations and labels to map 
axmap = plt.subplot2grid((5,3),(0,2),colspan=1,rowspan=5)
pfun.add_coast(axmap,color='grey')
pfun.dar(axmap)
axmap.axis([-125, -123, 47, 49])

axmap.text(-124.64,48.3,'Makah Bay',color='Black',weight='bold',alpha=0.8)
axmap.text(-124.67,48.13,'Cape Alava',color='Black',weight='bold',alpha=0.8)
axmap.text(-124.52,47.85,'Teahwhit Head',color='Black',weight='bold',alpha=0.8)
axmap.text(-124.35,47.57,'Kalaloch',color='Black',weight='bold',alpha=0.8)
axmap.text(-124.25,47.32,'Cape Elizabeth',color='Black',weight='bold',alpha=0.8)

axmap.set_xticks([-125,-124.5,-124,-123.5,-123])
axmap.set_xticklabels([-125,'',-124,'',-123])
axmap.set_yticks([47, 47.5, 48, 48.5, 49])
axmap.set_yticklabels([47, '', 48, '', 49])
axmap.set_xlabel('Longitude')
axmap.set_ylabel('Latitude')
axmap.set_title('OCNMS moorings')
                  
# plot data available for recent processed data 
sn_list = list(sn_name_dict.keys())
#sn_list = ['CE042']

ax1 = plt.subplot2grid((5,3),(0,0),colspan=1,rowspan=1)
ax2 = plt.subplot2grid((5,3),(0,1),colspan=1,rowspan=1)
ax3 = plt.subplot2grid((5,3),(1,0),colspan=1,rowspan=1)
ax4 = plt.subplot2grid((5,3),(1,1),colspan=1,rowspan=1)
ax5 = plt.subplot2grid((5,3),(2,0),colspan=1,rowspan=1)
ax6 = plt.subplot2grid((5,3),(2,1),colspan=1,rowspan=1)
ax7 = plt.subplot2grid((5,3),(3,0),colspan=1,rowspan=1)
ax8 = plt.subplot2grid((5,3),(3,1),colspan=1,rowspan=1)
ax9 = plt.subplot2grid((5,3),(4,0),colspan=1,rowspan=1)
ax10 = plt.subplot2grid((5,3),(4,1),colspan=1,rowspan=1)
 

ii = 1   
for sn in sn_list:
    print(sn)
    in_fn = data_dir / (sn + '_2011_2023_hourly.nc')
    
    ds = xr.open_dataset(in_fn, decode_times=True)
    IT = ds['IT'].values
    SAL = ds['SP'].values
    OXY = ds['DO (uM)'].values

    # axes[0] is the sitemap 
    if ~np.isnan(OXY).all():
        plt.gcf().axes[0].plot(ds.lon,ds.lat, marker = 'o',color = '#e41a1c',markersize=5)
        o = OXY
        o[~np.isnan(OXY)]=1
        o=o*ds.z.values
        plt.gcf().axes[ii].plot(ds.time,o, marker = 'o',color = '#e41a1c',markersize=3)
        plt.gcf().axes[ii].axis([ds.time[0],ds.time[-1],-42,0])
    if ~np.isnan(IT).all():
        plt.gcf().axes[0].plot(ds.lon,ds.lat, marker = 'o',color = '#377eb8',markersize=2)
        t = IT
        t[~np.isnan(IT)]=1
        t=t*ds.z.values
        plt.gcf().axes[ii].plot(ds.time,t, marker = '.',color = '#377eb8',markersize=1)
        plt.gcf().axes[ii].axis([ds.time[0],ds.time[-1],-42,0])
    if ~np.isnan(SAL).all():
        plt.gcf().axes[0].plot(ds.lon,ds.lat, marker = 'o',color = '#dede00',markersize=2)
        s = SAL
        s[~np.isnan(SAL)]=1
        s=s*ds.z.values
        plt.gcf().axes[ii].plot(ds.time,s, marker = '.',color = '#dede00',markersize=1)
        plt.gcf().axes[ii].axis([ds.time[0],ds.time[-1],-42,0])
        
    #del IT, SAL, OXY 
    ii = ii + 1
    


fig.tight_layout()
#plt.show()

