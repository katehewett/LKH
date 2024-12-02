'''
step 6:
Plot time-series / year 

'''

import pandas as pd
import numpy as np
import posixpath
import datetime 
import gsw 
import xarray as xr
import os
import sys
from lo_tools import Lfun, zfun
import PyCO2SYS as pyco2

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter

from lo_tools import plotting_functions as pfun

Ldir = Lfun.Lstart()

# output/input locations
mooring_in_dir = '/Users/katehewett/Documents/LKH_data/sutton2019_surface_buoys/daily/add_Oag'
model_in_dir = '/Users/katehewett/Documents/LKH_data/sutton2019_surface_buoys/LO_surface_extraction'

if os.path.exists(mooring_in_dir)==False:
    print('input path for obs data does not exist')
    sys.exit()
if os.path.exists(model_in_dir)==False:
    print('input path for model output data does not exist')
    sys.exit()    

fig_out_dir = '/Users/katehewett/Documents/LKH_output/sutton2019_surface_buoys/plots/obs_model_comparisons/time_series'
if os.path.exists(fig_out_dir)==False:
    Lfun.make_dir(fig_out_dir, clean = False)

'''
sn_name_dict = {
    'CHABA':'Chaba',
    'CAPEELIZABETH':'Cape Elizabeth',
    'CAPEARAGO':'Cape Arago'
}
'''

sn_name_dict = {
    'CHABA':'Chaba'
}

sn_list = list(sn_name_dict.keys())

'''ASSIGN TIMES AND LOAD FILES'''
# there is one obs .nc file for each mooring [daily values]
obs_fn = posixpath.join(mooring_in_dir, (sn_list[0] +'_daily_Oag.nc'))
obs_ds = xr.open_dataset(obs_fn, decode_times=True)

# 1 LO file:
LO_fn = posixpath.join(model_in_dir, ('LO_'+sn_list[0] + '_surface.nc'))
LO_ds = xr.open_dataset(LO_fn, decode_times=True)

# Organize time and data 
obs_time = pd.to_datetime(obs_ds.datetime_utc.values) #obs
lo_time = pd.to_datetime(LO_ds.time_utc.values)      #LO

if sn_list[0] == 'CHABA':
    year_list = [2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022] 
elif sn_list[0] == 'CAPEARAGO':
    year_list = [2017, 2018, 2019, 2020, 2021] 
elif sn_list[0] == 'CAPEELIZABETH':
    year_list = [2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]    
      
date_format = mdates.DateFormatter('%b %y')

'''FIGURE 1'''
for yr in year_list:    
    print(yr)
    plt.close('all')
    fs=12
    plt.rc('font', size=fs)
    fig1 = plt.figure(figsize=(12,8))
    ax0 = plt.subplot2grid((4,1),(0,0),colspan=1,rowspan=1)
    ax1 = plt.subplot2grid((4,1),(1,0),colspan=1,rowspan=1)
    ax2 = plt.subplot2grid((4,1),(2,0),colspan=1,rowspan=1)
    ax3 = plt.subplot2grid((4,1),(3,0),colspan=1,rowspan=1)
    
    fig_out1 = posixpath.join(fig_out_dir,(sn_list[0]+'_1_obsmod_timeseries_'+str(yr)+'.png'))
    
    itime = lo_time[lo_time.year==yr]
    otime = obs_time[obs_time.year==yr]
    
    tstart = itime[0] 
    tend = itime[-1] + pd.DateOffset(months=1)
    mdates1 = pd.date_range(start=tstart,end=tend, freq='MS')
    
    # SALT     
    ivar = LO_ds['SA'].values[lo_time.year==yr]
    ovar = obs_ds['SA'].values[obs_time.year==yr]
    
    ax0.plot(itime,ivar,color = 'DodgerBlue', marker='.', alpha=1, linestyle='-', label = 'LO')
    ax0.plot(otime,ovar,color = 'Crimson', marker='.', alpha=0.3, linestyle='-', label = 'OBS')
    ax0.set_ylabel('SA ['+str(LO_ds['SA'].units)+']')
    
    if np.all(np.isnan(ovar))==False:
        fstdev = np.floor(np.nanstd(ovar))
        ymin = np.floor(np.min([np.nanmin(ovar),np.nanmin(ivar)]))-fstdev
        ymax = np.ceil(np.max([np.nanmax(ovar),np.nanmax(ivar)]))+fstdev
    else:
        fstdev = np.floor(np.nanstd(ivar))
        ymin = np.floor(np.min([np.nanmin(ivar)]))-fstdev
        ymax = np.ceil(np.max([np.nanmax(ivar)]))+fstdev
        
    yticks = np.linspace(ymin,ymax,5)
    ax0.set_ylim([ymin,ymax])
    ax0.set_yticks(yticks)
    ax0.set_xlim([mdates1[0],mdates1[-1]])
    ax0.set_xticks(mdates1)
    ax0.xaxis.set_major_formatter(date_format)
    ax0.set_title(sn_list[0]+' ('+str(yr)+'): Sutton et al. 2019 (obs); surface extraction cas7_t0_x4b (LO)')
    
    ax0.grid()     
    ax0.grid(zorder=0)
    
    ax0.legend(loc="best")
    del ymin, ymax, yticks
           
    # TEMP     
    ivar = LO_ds['CT'].values[lo_time.year==yr]
    ovar = obs_ds['CT'].values[obs_time.year==yr]
   
    ax1.plot(itime,ivar,color = 'DodgerBlue', marker='.', alpha=1, linestyle='-', label = 'LO')
    ax1.plot(otime,ovar,color = 'Crimson', marker='.', alpha=0.3, linestyle='-', label = 'OBS')
    ax1.set_ylabel('CT ['+str(LO_ds['CT'].units)+']')
   
    if (np.all(np.isnan(ovar))==False):
        fstdev = np.floor(np.nanstd(ovar))
        ymin = np.floor(np.min([np.nanmin(ovar),np.nanmin(ivar)]))-fstdev
        ymax = np.ceil(np.max([np.nanmax(ovar),np.nanmax(ivar)]))+fstdev
    if (np.all(np.isnan(ovar))==True):
        fstdev = np.floor(np.nanstd(ivar))
        ymin = np.floor(np.min([np.nanmin(ivar)]))-fstdev
        ymax = np.ceil(np.max([np.nanmax(ivar)]))+fstdev
    
    yticks = np.linspace(ymin,ymax,5)   
    ax1.set_ylim([ymin,ymax])
    ax1.set_yticks(yticks)
    ax1.set_xlim([mdates1[0],mdates1[-1]])
    ax1.set_xticks(mdates1)
    ax1.xaxis.set_major_formatter(date_format)
   
    ax1.grid()     
    ax1.grid(zorder=0)
    
    # SIG0     
    ivar = LO_ds['SIG0'].values[lo_time.year==yr]
    ovar = obs_ds['SIG0'].values[obs_time.year==yr]
   
    ax2.plot(itime,ivar,color = 'DodgerBlue', marker='.', alpha=1, linestyle='-', label = 'LO')
    ax2.plot(otime,ovar,color = 'Crimson', marker='.', alpha=0.3, linestyle='-', label = 'OBS')
    ax2.set_ylabel('SIG0 ['+str(LO_ds['SIG0'].units)+']')
   
    if np.all(np.isnan(ovar))==False:
        fstdev = np.floor(np.nanstd(ovar))
        ymin = np.floor(np.min([np.nanmin(ovar),np.nanmin(ivar)]))-fstdev
        ymax = np.ceil(np.max([np.nanmax(ovar),np.nanmax(ivar)]))+fstdev
    else:
        fstdev = np.floor(np.nanstd(ivar))
        ymin = np.floor(np.min([np.nanmin(ivar)]))-fstdev
        ymax = np.ceil(np.max([np.nanmax(ivar)]))+fstdev
    yticks = np.linspace(ymin,ymax,5)
    ax2.set_ylim([ymin,ymax])
    ax2.set_yticks(yticks)
    ax2.set_xlim([mdates1[0],mdates1[-1]])
    ax2.set_xticks(mdates1)
    ax2.xaxis.set_major_formatter(date_format)
   
    ax2.grid()     
    ax2.grid(zorder=0)   

    if sn_list[0] != 'CAPEELIZABETH':
        # DO     
        ivar = LO_ds['DO (uM)'].values[lo_time.year==yr]
        ovar = obs_ds['DO (uM)'].values[obs_time.year==yr]
   
        ax3.plot(itime,ivar,color = 'DodgerBlue', marker='.', alpha=1, linestyle='-', label = 'LO')
        ax3.plot(otime,ovar,color = 'Crimson', marker='.', alpha=0.3, linestyle='-', label = 'OBS')
        ax3.set_ylabel('DO uM')
   
        if np.all(np.isnan(ovar))==False:
            fstdev = np.floor(np.nanstd(ovar))
            ymin = np.floor(np.min([np.nanmin(ovar),np.nanmin(ivar)]))-fstdev
            ymax = np.ceil(np.max([np.nanmax(ovar),np.nanmax(ivar)]))+fstdev
        else:
            fstdev = np.floor(np.nanstd(ivar))
            ymin = np.floor(np.min([np.nanmin(ivar)]))-fstdev
            ymax = np.ceil(np.max([np.nanmax(ivar)]))+fstdev
        
        yticks = np.linspace(ymin,ymax,5)
        ax3.set_ylim([ymin,ymax])
        ax3.set_yticks(yticks)
        ax3.set_xlim([mdates1[0],mdates1[-1]])
        ax3.set_xticks(mdates1)
        ax3.xaxis.set_major_formatter(date_format)
   
        ax3.grid()     
        ax3.grid(zorder=0)  
    else:
        # DO     
        ivar = LO_ds['DO (uM)'].values[lo_time.year==yr]
   
        ax3.plot(itime,ivar,color = 'DodgerBlue', marker='.', alpha=1, linestyle='-', label = 'LO')
        ax3.set_ylabel('DO uM')
   
        if np.all(np.isnan(ovar))==False:
            fstdev = np.floor(np.nanstd(ovar))
            ymin = np.floor(np.min([np.nanmin(ovar),np.nanmin(ivar)]))-fstdev
            ymax = np.ceil(np.max([np.nanmax(ovar),np.nanmax(ivar)]))+fstdev
        else:
            fstdev = np.floor(np.nanstd(ivar))
            ymin = np.floor(np.min([np.nanmin(ivar)]))-fstdev
            ymax = np.ceil(np.max([np.nanmax(ivar)]))+fstdev
        
        yticks = np.linspace(ymin,ymax,5)
        ax3.set_ylim([ymin,ymax])
        ax3.set_yticks(yticks)
        ax3.set_xlim([mdates1[0],mdates1[-1]])
        ax3.set_xticks(mdates1)
        ax3.xaxis.set_major_formatter(date_format)
   
        ax3.grid()     
        ax3.grid(zorder=0)  
           
    plt.gcf().tight_layout()

    plt.gcf().savefig(fig_out1)


'''FIGURE 2'''
for yr in year_list:    
    print(yr)
    plt.close('all')
    fs=12
    plt.rc('font', size=fs)
    fig1 = plt.figure(figsize=(12,8))
    ax0 = plt.subplot2grid((4,1),(0,0),colspan=1,rowspan=1)
    ax1 = plt.subplot2grid((4,1),(1,0),colspan=1,rowspan=1)
    ax2 = plt.subplot2grid((4,1),(2,0),colspan=1,rowspan=1)
    ax3 = plt.subplot2grid((4,1),(3,0),colspan=1,rowspan=1)
    
    fig_out2 = posixpath.join(fig_out_dir,(sn_list[0]+'_2_obsmod_timeseries_'+str(yr)+'.png'))
    
    itime = lo_time[lo_time.year==yr]
    otime = obs_time[obs_time.year==yr]
    
    tstart = itime[0] 
    tend = itime[-1] + pd.DateOffset(months=1)
    mdates1 = pd.date_range(start=tstart,end=tend, freq='MS')
    
    # pCO2_sw     
    ivar = LO_ds['pCO2'].values[lo_time.year==yr]
    ovar = obs_ds['pCO2_sw'].values[obs_time.year==yr]
    
    ax0.plot(itime,ivar,color = 'DodgerBlue', marker='.', alpha=1, linestyle='-', label = 'LO')
    ax0.plot(otime,ovar,color = 'Crimson', marker='.', alpha=0.3, linestyle='-', label = 'OBS')
    ax0.set_ylabel('pCO2_sw ['+str(LO_ds['pCO2'].units)+']')
    
    if np.all(np.isnan(ovar))==False:
        fstdev = np.floor(np.nanstd(ovar))
        ymin = np.floor(np.min([np.nanmin(ovar),np.nanmin(ivar)]))-fstdev
        ymax = np.ceil(np.max([np.nanmax(ovar),np.nanmax(ivar)]))+fstdev
    else:
        fstdev = np.floor(np.nanstd(ivar))
        ymin = np.floor(np.min([np.nanmin(ivar)]))-fstdev
        ymax = np.ceil(np.max([np.nanmax(ivar)]))+fstdev
        
    yticks = np.linspace(ymin,ymax,5)
    ax0.set_ylim([ymin,ymax])
    ax0.set_yticks(yticks)
    ax0.set_xlim([mdates1[0],mdates1[-1]])
    ax0.set_xticks(mdates1)
    ax0.xaxis.set_major_formatter(date_format)
    ax0.set_title(sn_list[0]+' ('+str(yr)+'): Sutton et al. 2019 (obs); surface extraction cas7_t0_x4b (LO)')
    
    ax0.grid()     
    ax0.grid(zorder=0)
    
    ax0.legend(loc="best")
   
    # pH     
    ivar = LO_ds['pH_total'].values[lo_time.year==yr]
    ovar = obs_ds['pH_total'].values[obs_time.year==yr]
   
    ax1.plot(itime,ivar,color = 'DodgerBlue', marker='.', alpha=1, linestyle='-', label = 'LO')
    ax1.plot(otime,ovar,color = 'Crimson', marker='.', alpha=0.3, linestyle='-', label = 'OBS')
    ax1.set_ylabel('pH_total')
   
    if np.all(np.isnan(ovar))==False:
        fstdev = np.floor(np.nanstd(ovar))
        ymin = np.floor(np.min([np.nanmin(ovar),np.nanmin(ivar)]))-fstdev
        ymax = np.ceil(np.max([np.nanmax(ovar),np.nanmax(ivar)]))+fstdev
    else:
        fstdev = np.floor(np.nanstd(ivar))
        ymin = np.floor(np.min([np.nanmin(ivar)]))-fstdev
        ymax = np.ceil(np.max([np.nanmax(ivar)]))+fstdev
        
    yticks = np.linspace(ymin,ymax,5)
    ax1.set_ylim([ymin,ymax])
    ax1.set_yticks(yticks)
    ax1.set_xlim([mdates1[0],mdates1[-1]])
    ax1.set_xticks(mdates1)
    ax1.xaxis.set_major_formatter(date_format)
   
    ax1.grid()     
    ax1.grid(zorder=0)
    
    # ARAG     
    ivar = LO_ds['ARAG'].values[lo_time.year==yr]
    ovar = obs_ds['ARAG'].values[obs_time.year==yr]
   
    ax2.plot(itime,ivar,color = 'DodgerBlue', marker='.', alpha=1, linestyle='-', label = 'LO')
    ax2.plot(otime,ovar,color = 'Crimson', marker='.', alpha=0.3, linestyle='-', label = 'OBS')
    ax2.set_ylabel('ARAG')
   
    if np.all(np.isnan(ovar))==False:
        fstdev = np.floor(np.nanstd(ovar))
        ymin = np.floor(np.min([np.nanmin(ovar),np.nanmin(ivar)]))-fstdev
        ymax = np.ceil(np.max([np.nanmax(ovar),np.nanmax(ivar)]))+fstdev
    else:
        fstdev = np.floor(np.nanstd(ivar))
        ymin = np.floor(np.min([np.nanmin(ivar)]))-fstdev
        ymax = np.ceil(np.max([np.nanmax(ivar)]))+fstdev
        
    yticks = np.linspace(ymin,ymax,5)
    ax2.set_ylim([ymin,ymax])
    ax2.set_yticks(yticks)
    ax2.set_xlim([mdates1[0],mdates1[-1]])
    ax2.set_xticks(mdates1)
    ax2.xaxis.set_major_formatter(date_format)
   
    ax2.grid()     
    ax2.grid(zorder=0)   

    # TA     
    ivar = LO_ds['ALK'].values[lo_time.year==yr]
   
    ax3.plot(itime,ivar,color = 'DodgerBlue', marker='.', alpha=1, linestyle='-', label = 'LO')
    ax3.set_ylabel('TA μEq/L')
   
    fstdev = np.floor(np.nanstd(ivar))
    ymin = np.floor(np.min([np.nanmin(ivar)]))-fstdev
    ymax = np.ceil(np.max([np.nanmax(ivar)]))+fstdev
        
    yticks = np.linspace(ymin,ymax,5)
    ax3.set_ylim([ymin,ymax])
    ax3.set_yticks(yticks)
    ax3.set_xlim([mdates1[0],mdates1[-1]])
    ax3.set_xticks(mdates1)
    ax3.xaxis.set_major_formatter(date_format)
   
    ax3.grid()     
    ax3.grid(zorder=0)  
        
    plt.gcf().tight_layout()

    plt.gcf().savefig(fig_out2)
