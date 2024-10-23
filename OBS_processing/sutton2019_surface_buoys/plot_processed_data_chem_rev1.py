# Step 5: 
# plot processed data for discussion on Tuesday
# This plots surface data for Chaba buoy, Cape Elizabeth buoy and Cape Arago buoy

testing = False 

# imports
from lo_tools import Lfun, zfun, zrfun
from lo_tools import plotting_functions as pfun

#import sys 
import xarray as xr
import numpy as np
#import netCDF4 as nc
import os 
import sys
import posixpath

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
#import matplotlib.dates as mdates
from datetime import datetime
from datetime import time
import pandas as pd

out_dir = '/Users/katehewett/Documents/LKH_output/sutton2019_surface_buoys/plots'
if os.path.exists(out_dir)==False:
    Lfun.make_dir(out_dir, clean = False)
    
obs_in_dir = '/Users/katehewett/Documents/LKH_data/sutton2019_surface_buoys/daily/add_Oag'   
model_in_dir = '/Users/katehewett/Documents/LKH_data/sutton2019_surface_buoys/LO_surface_extraction'

sn_name_dict = {
    'CHABA':'Chaba',
    #'CAPEELIZABETH':'Cape Elizabeth',
    'CAPEARAGO':'Cape Arago'
}

if testing:
    sn_list = ['CAPEARAGO']
else:    
    sn_list = list(sn_name_dict.keys())
    
#numyrs = 5 # 2013 - 2017
sn_list = list(sn_name_dict.keys())
numsites = len(sn_list)

# update with input tags for start and stop 
# make some datetimes for plotting 
#mdates1 = pd.date_range(start='2013-01-01',end='2024-01-01', freq='6M')
#mdates2 = pd.date_range(start='2013-07-01',end='2023-07-01', freq='6M')

# initialize plot 


# SA (measured; modeled)
# CT (measured; modeled)
# DO um (measured; modeled)
# pH (calculated)
# Oag (calculated)
# pCO2 (measured; calculated)

##CLEAN THIS UPPPP

for sn in sn_list:
    
    obs_fn = posixpath.join(obs_in_dir, (sn+'_daily_Oag.nc'))
    model_fn = posixpath.join(model_in_dir, ('LO_'+sn + '_surface.nc'))
    fn_out1 = posixpath.join(out_dir, (sn + '_1.png'))
    fn_out2 = posixpath.join(out_dir, (sn + '_2.png'))
    
    obs_ds = xr.open_dataset(obs_fn, decode_times=True) 
    lo_ds = xr.open_dataset(model_fn, decode_times=True) 


    ltime = pd.to_datetime(lo_ds.time_utc)
    ot = pd.to_datetime(obs_ds.datetime_utc)
    
    Otstart = (ot[0] - pd.DateOffset(months=1)).normalize()
    LO_tstart = (ltime[0] - pd.DateOffset(months=1)).normalize()
    if (LO_tstart<Otstart):
        tstart = Otstart
    else: 
        tstart = LO_tstart
        
    tend = (ot[-1] - pd.DateOffset(months=1)).normalize()
    mdates1 = pd.date_range(start=tstart,end=tend, freq='9M')
    
    # CT SA DO first 
    plt.close('all')
    fs=12
    plt.rc('font', size=fs)
    date_format = DateFormatter('%d%b%y')
    fig = plt.figure(figsize=(16,8))
    
    ax0 = plt.subplot2grid((3,1),(0,0),colspan=1,rowspan=1)
    plt.plot(ltime,lo_ds.SA,color = 'navy',marker='.',
             linestyle='-',linewidth=3,alpha=0.6,markeredgecolor='none',label ='LO')
    plt.plot(ot,obs_ds.SA.values,color = 'dodgerblue',marker='.',
             linestyle='-',linewidth=3,alpha=0.6,markeredgecolor='none',label ='obs')
    ax0.set_xticklabels([])
    ax0.set_ylim([25,35])
    ax0.set_yticks(np.arange(25,36,2.5), minor=False) 
    ax0.set_xlim([mdates1[0],mdates1[-1]])
    ax0.set_xticks(mdates1)
    ax0.set_title((sn+': LO model output data (~'+str(lo_ds.Depth)+'); surface data obs (~0.5m)'))
    ax0.set_ylabel('SA g/kg')
    ax0.legend(loc="best",frameon=False)
    plt.grid(True)
    
    ax1 = plt.subplot2grid((3,1),(1,0),colspan=1,rowspan=1)
    plt.plot(ltime,lo_ds.CT,color = 'navy',marker='.',
             linestyle='-',linewidth=3,alpha=0.6,markeredgecolor='none',label ='LO')
    plt.plot(ot,obs_ds.CT.values,color = 'dodgerblue',marker='.',
             linestyle='-',linewidth=3,alpha=0.6,markeredgecolor='none',label ='obs')
    ax1.set_xticklabels([])
    ax1.set_ylim([6,22])
    ax1.set_yticks(np.arange(6,23,4), minor=False)
    ax1.set_xlim([mdates1[0],mdates1[-1]])
    ax1.set_xticks(mdates1)
    ax1.set_ylabel('CT deg C')
    plt.grid(True)
    
    ax2 = plt.subplot2grid((3,1),(2,0),colspan=1,rowspan=1)
    plt.plot(ltime,lo_ds['DO (uM)'].values,color = 'navy',marker='.',
             linestyle='-',linewidth=3,alpha=0.6,markeredgecolor='none',label ='LO')
    plt.plot(ot,obs_ds['DO (uM)'].values,color = 'dodgerblue',marker='.',
             linestyle='-',linewidth=3,alpha=0.6,markeredgecolor='none',label ='obs')
    ax2.set_ylim([100,520])
    ax2.set_yticks(np.arange(100,521,200), minor=False)
    ax2.set_xlim([mdates1[0],mdates1[-1]])
    ax2.set_xticks(mdates1)
    ax2.set_ylabel('DO uM')
    
    plt.grid(True)
    
    plt.gcf().tight_layout()
    plt.gcf().savefig(fn_out1)
    
    fs=12
    fig3 = plt.figure(figsize=(16,8))
    
    ax3 = plt.subplot2grid((3,1),(0,0),colspan=1,rowspan=1)
    plt.plot(ltime,lo_ds['pH_total'].values,color = 'navy',marker='.',
             linestyle='-',linewidth=3,alpha=0.6,markeredgecolor='none',label ='LO')
    plt.plot(ot,obs_ds['pH'].values,color = 'dodgerblue',marker='.',
             linestyle='-',linewidth=3,alpha=0.6,markeredgecolor='none',label ='obs')
    plt.plot(ot,obs_ds['new_pH'].values,color = 'yellow',marker='.',
             linestyle='-',linewidth=3,alpha=0.6,markeredgecolor='none',label ='new_pH/obs calc')
    ax3.set_xticklabels([])
    ax3.set_ylim([7.2,8.6])
    ax3.set_yticks(np.arange(7.2,8.7,.4), minor=False)
    ax3.set_xlim([mdates1[0],mdates1[-1]])
    ax3.set_xticks(mdates1)
    ax3.set_ylabel('pH total')
    
    ax4 = plt.subplot2grid((3,1),(1,0),colspan=1,rowspan=1)
    plt.plot(ltime,lo_ds.ARAG,color = 'navy',marker='.',
             linestyle='-',linewidth=3,alpha=0.6,markeredgecolor='none',label ='LO')
    plt.plot(ot,obs_ds['ARAG'].values,color = 'dodgerblue',marker='.',
             linestyle='-',linewidth=3,alpha=0.6,markeredgecolor='none',label ='obs')
    ax4.set_xticklabels([])
    ax4.set_ylim([0.5,3.5])
    ax4.set_yticks(np.arange(0.5,3.6,0.5), minor=False)
    ax4.set_xlim([mdates1[0],mdates1[-1]])
    ax4.set_xticks(mdates1)
    #ax4.set_xticks([mdates2], minor=True)
    ax4.set_ylabel('O ARAG')
    
    ax5 = plt.subplot2grid((3,1),(2,0),colspan=1,rowspan=1)
    plt.plot(ltime,lo_ds.pCO2,color = 'navy',marker='.',
             linestyle='-',linewidth=3,alpha=0.6,markeredgecolor='none',label ='LO')
    plt.plot(ot,obs_ds['pCO2_sw'].values,color = 'dodgerblue',marker='.',
             linestyle='-',linewidth=3,alpha=0.6,markeredgecolor='none',label ='obs')
    ax5.set_ylim([50,1350])
    ax5.set_yticks(np.arange(50,1351,260), minor=False)
    ax5.set_xlim([mdates1[0],mdates1[-1]])
    ax5.set_xticks(mdates1)
    ax5.set_ylabel('pCO2 sw uatm')
    
    plt.gcf().tight_layout()
    
    plt.gcf().savefig(fn_out2)
