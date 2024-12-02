'''
Explore chem
not flexible code 


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

testing = False 

# output/input locations
mooring_in_dir = '/Users/katehewett/Documents/LKH_data/sutton2019_surface_buoys/daily/add_Oag'
model_in_dir = '/Users/katehewett/Documents/LKH_data/sutton2019_surface_buoys/LO_surface_extraction'

if os.path.exists(mooring_in_dir)==False:
    print('input path for obs data does not exist')
    sys.exit()
if os.path.exists(model_in_dir)==False:
    print('input path for model output data does not exist')
    sys.exit()    

fig_out_dir = '/Users/katehewett/Documents/LKH_output/sutton2019_surface_buoys/plots/obs_model_comparisons/property_property'
if os.path.exists(fig_out_dir)==False:
    Lfun.make_dir(fig_out_dir, clean = False)

plt.close('all')
fs=12
plt.rc('font', size=fs)
fig = plt.figure(figsize=(9,9))
ax0 = plt.subplot2grid((3,3),(0,0),colspan=1,rowspan=1)
ax1 = plt.subplot2grid((3,3),(0,1),colspan=1,rowspan=1)
ax2 = plt.subplot2grid((3,3),(0,2),colspan=1,rowspan=1)
ax3 = plt.subplot2grid((3,3),(1,0),colspan=1,rowspan=1)
ax4 = plt.subplot2grid((3,3),(1,1),colspan=1,rowspan=1)
ax6 = plt.subplot2grid((3,3),(2,0),colspan=1,rowspan=1)
ax7 = plt.subplot2grid((3,3),(2,1),colspan=1,rowspan=1)

ax0.set_box_aspect(1)
ax1.set_box_aspect(1)
ax2.set_box_aspect(1)
ax3.set_box_aspect(1)
ax4.set_box_aspect(1)
ax6.set_box_aspect(1)
ax7.set_box_aspect(1)

# map + load file w/ depths 
fn_mask =  Ldir['parent'] / 'LKH_data' / 'shelf_masks' / 'OA_indicators' / 'OA_indicators_shelf_mask_15_200m_noEstuaries.nc'
dmask = xr.open_dataset(fn_mask) 
h = dmask.h
xrho = dmask.lon_rho
yrho = dmask.lat_rho

ax = plt.subplot2grid((3,3),(1,2),colspan=1,rowspan=2)
pfun.add_coast(ax)
pfun.dar(ax)
ax.axis([-126, -123.5, 42.75, 48.75])

ax.contour(xrho,yrho,h, [40],
    colors=['grey'], linewidths=1, linestyles='solid',alpha=0.4)
ax.contour(xrho,yrho,h, [80],
    colors=['black'], linewidths=1, linestyles='solid',alpha=0.4)
#ax.contour(xrho,yrho,h, [200],
#    colors=['black'], linewidths=1, linestyles='solid',alpha=0.6)
#ax.contour(xrho,yrho,h, [800],
#    colors=['black'], linewidths=1, linestyles='solid')

ax.text(1.2,.10,'40 m',color='grey',weight='bold',transform=ax.transAxes,ha='right')
ax.text(1.2,.07,'80 m',color='black',weight='bold',transform=ax.transAxes,ha='right')
#ax.text(1.2,.04,'200 m',color='black',weight='bold',transform=ax.transAxes,ha='right')
#ax.text(1.2,.01,'800 m',color='black',weight='bold',transform=ax.transAxes,ha='right')
      
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_yticks([42.75,43,44,45,46,47,48,48.75])
ax.set_yticklabels(['42.75','43','44','45','46','47','48','48.75'])
ax.set_xticks([-126.5, -125.5, -124.5, -123.5])
ax.xaxis.set_ticklabels([-126.5, -125.5, -124.5, -123.5])
ax.grid(False)


sn_name_dict = {
    'CHABA':'Chaba',
    'CAPEELIZABETH':'Cape Elizabeth',
    'CAPEARAGO':'Cape Arago'
}

'''
sn_name_dict = {
    'CHABA':'Chaba'
}
'''
sn_list = list(sn_name_dict.keys())


fig_out = posixpath.join(fig_out_dir,('Sutton_obs_model.png'))
#fig_out = posixpath.join(fig_out_dir,(sn_list[0]+'_obs_model.png'))



for sn in sn_list:    
    print(sn)
    # there is one obs .nc file for each mooring [daily values]
    obs_fn = posixpath.join(mooring_in_dir, (sn +'_daily_Oag.nc'))
    obs_ds = xr.open_dataset(obs_fn, decode_times=True)

    # 1 LO file:
    LO_fn = posixpath.join(model_in_dir, ('LO_'+sn + '_surface.nc'))
    LO_ds = xr.open_dataset(LO_fn, decode_times=True)

    # Organize time and data 
    otime = pd.to_datetime(obs_ds.datetime_utc.values) #obs
    ltime = pd.to_datetime(LO_ds.time_utc.values)      #LO

    # make sure all are daily data and spaced daily 
    if (np.unique(np.diff(ltime)/pd.Timedelta(days=1)) != 1) | (np.unique(np.diff(otime)/pd.Timedelta(days=1)) != 1) : 
        print('issue with times')
        sys.exit 

    # liveocean sampled thru 2023; Sutton et al. observations end before 2023     
    end_time = otime[-1] 
    if otime[0] < ltime[0]:
        start_time = ltime[0]
        print('LO start time')
    else: 
        start_time = otime[0]
        print('OBS start time')

    # take LO values at shared times 
    start_index = np.argmin(np.abs(ltime-start_time))  
    stop_index = np.argmin(np.abs(ltime-end_time)) 
    LOtime_utc = ltime[start_index:stop_index]  
    LO_SA = LO_ds['SA'].values[start_index:stop_index] 
    LO_TA = LO_ds['ALK'].values[start_index:stop_index] 
    LO_CT = LO_ds['CT'].values[start_index:stop_index] 
    LO_SIG0 = LO_ds['SIG0'].values[start_index:stop_index] 
    LO_pCO2 = LO_ds['pCO2'].values[start_index:stop_index] 
    LO_pH = LO_ds['pH_total'].values[start_index:stop_index]   
    LO_ARAG = LO_ds['ARAG'].values[start_index:stop_index]  
    if str(sn) != 'CAPEELIZABETH':
        LO_DO = LO_ds['DO (uM)'].values[start_index:stop_index] 
        
    # take obs values at shared times 
    start_index = np.argmin(np.abs(otime-start_time))  
    stop_index = np.argmin(np.abs(otime-end_time)) 
    obstime_utc = otime[start_index:stop_index]  

    if np.all(LOtime_utc==obstime_utc):
        time_utc = LOtime_utc
        del LOtime_utc, obstime_utc
        print('times okay')
    else:
        print('issue with times')
        sys.exit()   
    
    OBS_SA = obs_ds['SA'].values[start_index:stop_index] 
    OBS_CT = obs_ds['CT'].values[start_index:stop_index] 
    OBS_SIG0 = obs_ds['SIG0'].values[start_index:stop_index] 
    OBS_pCO2 = obs_ds['pCO2_sw'].values[start_index:stop_index] 
    OBS_pH = obs_ds['pH_total'].values[start_index:stop_index]   
    OBS_ARAG = obs_ds['ARAG'].values[start_index:stop_index]
    if str(sn) != 'CAPEELIZABETH':
        OBS_DO = obs_ds['DO (uM)'].values[start_index:stop_index] 
        
    if str(sn)=='CHABA':
        rcolor = '#56B4E9' # blue
        lab = '  ChaBa'
    elif str(sn) == 'CAPEELIZABETH':
        rcolor = '#E69F00' # orange 
        lab = 'Cape Elizabeth  '
    elif str(sn) == 'CAPEARAGO':
        rcolor = '#009E73' # green
        lab = 'Cape Arago  '
    
    if str(sn)!='CHABA':
        ax.plot(obs_ds.attrs['lon'],obs_ds.attrs['lat'],color = rcolor,marker = 'o')
        plt.text(obs_ds.attrs['lon'],obs_ds.attrs['lat'],lab,weight='bold',color = rcolor, ha='right', va='center')
    else:
        #ax.plot(obs_ds.attrs['lon'],obs_ds.attrs['lat'],color = 'Navy',marker = 'o')
        ax.plot(-124.95,47.97,color = rcolor,marker = 'o')
        plt.text(-124.95,47.97,lab,weight='bold',color = rcolor, ha='left', va='center')
        
    # TS: plot(x,y) X SALT, Y TEMP!!
    ax0.plot(OBS_SA,OBS_CT,color = rcolor, marker='o', alpha=0.2, markeredgecolor ='none', linestyle='none', label = str(sn))
    ax0.plot(LO_SA,LO_CT,color = 'Indigo', marker='o', alpha=0.1, markeredgecolor ='none', linestyle='none', label = str(sn))
    ax0.set_ylabel('CT ['+str(obs_ds['CT'].units)+']')
    ax0.set_xlabel('SA ['+str(obs_ds['SA'].units)+']')
    
    xmin = 20 #np.floor(np.min(obs_ds['SA']))-2
    xmax = 36 #np.ceil(np.max(obs_ds['SA']))+2
    ymin = 6 #np.floor(np.min(obs_ds['SA']))-2
    ymax = 20 #np.ceil(np.max(obs_ds['SA']))+2
    yticks = np.arange(ymin,ymax+1,2)
    xticks = np.arange(xmin,xmax+1,2)
    ax0.set_ylim([ymin,ymax])
    ax0.set_xlim([xmin,xmax])
    ax0.set_yticks(yticks)
    ax0.set_xticks(xticks)
    ax0.set_title(sn)
    ax0.grid(True)
    
    # pCO2 - SA : plot(x,y) x SA; y pCO2, !!     
    ax1.plot(OBS_SA,OBS_pCO2,color = rcolor, marker='o', alpha=0.2, markeredgecolor ='none', linestyle='none')
    ax1.plot(LO_SA,LO_pCO2,color = 'Indigo', marker='o', alpha=0.1, markeredgecolor ='none', linestyle='none', label = str(sn))
    ax1.set_xlabel('SA ['+str(obs_ds['SA'].units)+']')
    ax1.set_ylabel('pCO2 ['+str(obs_ds['pCO2_sw'].units)+']')

    xmin = 20 #np.floor(np.min(obs_ds['SA']))-2
    xmax = 36 #np.ceil(np.max(obs_ds['SA']))+2
    ymin = 100 #np.floor(np.min(obs_ds['SIG0']))-1
    ymax = 1400 #np.ceil(np.max(obs_ds['SIG0']))+1
    yticks = np.arange(ymin,ymax+200,200) # pCO2
    xticks = np.arange(xmin,xmax+1,2)     # SALT!! 
    ax1.set_ylim([ymin,ymax])
    ax1.set_xlim([xmin,xmax])
    ax1.set_yticks(yticks)
    ax1.set_xticks(xticks)
    ax1.grid(True)
    
    # y=TA x=SA      plot(x,y)
    ax2.plot(OBS_SA,LO_TA,color = rcolor, marker='o', alpha=0.2, markeredgecolor ='none', linestyle='none')
    ax2.plot(LO_SA,LO_TA,color = 'Indigo', marker='o', alpha=0.2, markeredgecolor ='none', linestyle='none')
    ax2.set_ylabel('TA (LO) [uEq/L]')
    ax2.set_xlabel('SA ['+str(obs_ds['SIG0'].units)+']')

    xmin = 20 #np.floor(np.min(obs_ds['SA']))-2
    xmax = 36 #np.ceil(np.max(obs_ds['SA']))+2
    ymin = 1800 #TA
    ymax = 2300 
    yticks = np.arange(ymin,ymax+100,100)
    xticks = np.arange(xmin,xmax+1,2)
    ax2.set_ylim([ymin,ymax])
    ax2.set_xlim([xmin,xmax])
    ax2.set_yticks(yticks)
    ax2.set_xticks(xticks)
    ax2.grid(True)
    
    # y=O2 x=CT      plot(x,y)
    if str(sn) != 'CAPEELIZABETH': 
        ax3.plot(OBS_CT,OBS_DO,color = rcolor, marker='o', alpha=0.2, markeredgecolor ='none', linestyle='none')
        ax3.plot(LO_CT,LO_DO,color = 'Indigo', marker='o', alpha=0.2, markeredgecolor ='none', linestyle='none')
        ax3.set_ylabel('DO ['+str(obs_ds['DO (uM)'].units)+']')
        ax3.set_xlabel('CT ['+str(obs_ds['CT'].units)+']')

        ymin = 100 #OXYGEN
        ymax = 500 
        xmin = 6 #CT
        xmax = 20 
        yticks = np.arange(ymin,ymax+1,50)
        xticks = np.arange(xmin,xmax+1,2)
        ax3.set_ylim([ymin,ymax])
        ax3.set_xlim([xmin,xmax])
        ax3.set_yticks(yticks)
        ax3.set_xticks(xticks)
        #ax3.set_xticklabels(['100', ' ', '200', ' ', '300', ' ', '400', ' ', '500'])
        #ax3.set_yticklabels(['100', ' ', '200', ' ', '300', ' ', '400', ' ', '500'])
        ax3.grid(True)
        
        del LO_DO, OBS_DO

    plt.gcf().tight_layout()
    sys.exit()
    # TA y DIC x   
    ax4.plot(LO_DIC,LO_TA,color = rcolor, marker='o', alpha=0.2, markeredgecolor ='none', linestyle='none')
    ax4.set_ylabel('LO pCO2_sw ['+str(LO_ds['pCO2'].units)+']')
    ax4.set_xlabel('pCO2_sw ['+str(obs_ds['pCO2_sw'].units)+']')

    smin = 100 #np.floor(np.min(obs_ds['SIG0']))-1
    smax = 1400 #np.ceil(np.max(obs_ds['SIG0']))+1
    yticks = np.arange(smin,smax+200,200)
    ax4.set_ylim([smin,smax])
    ax4.set_xlim([smin,smax])
    ax4.set_yticks(yticks)
    ax4.set_xticks(yticks)
    labels = [item.get_text() for item in ax4.get_xticklabels()]
    ax4.set_xticklabels(['100', ' ', '500', ' ', '900', ' ', '1300', ' '])
    #ax4.set_yticklabels(['100', ' ', '500', ' ', '900', ' ', '1300', '1500'])
    ax4.grid(True)
    
    # pH     
    ax6.plot(OBS_pH,LO_pH,color = rcolor, marker='o', alpha=0.2, markeredgecolor ='none', linestyle='none')
    ax6.set_ylabel('LO pH total')
    ax6.set_xlabel('pH total')

    smin = 7.4 #np.floor(np.min(obs_ds['SIG0']))-1
    smax = 8.6 #np.ceil(np.max(obs_ds['SIG0']))+1
    yticks = np.arange(smin,smax+0.1,.2)
    ax6.set_ylim([smin,smax])
    ax6.set_xlim([smin,smax])
    ax6.set_yticks(yticks)
    ax6.set_xticks(yticks)
    ax6.set_xticklabels(['7.4','','7.8','','8.2','','8.6'])
    ax6.grid(True)
    
    # ARAG      
    ax7.plot(OBS_ARAG,LO_ARAG,color = rcolor, marker='o', alpha=0.2, markeredgecolor ='none', linestyle='none')
    ax7.set_ylabel('LO ARAG')
    ax7.set_xlabel('ARAG')

    smin = 0 #np.floor(np.min(obs_ds['SIG0']))-1
    smax = 4.75 #np.ceil(np.max(obs_ds['SIG0']))+1
    yticks = np.arange(smin,smax+0.75,.75)
    ax7.set_ylim([smin,smax])
    ax7.set_xlim([smin,smax])
    ax7.set_yticks(yticks)
    ax7.set_xticks(yticks)
    ax7.set_xticklabels(['0.00', ' ', '1.50', ' ', '3.00', ' ', '4.50', ' '])
    ax7.grid(True)
    
    plt.gcf().tight_layout()

#ax0.legend(loc="lower right")


ax.plot(-124.95,47.97,'kx') # Chaba 
ax.plot(-124.73,47.35,'rx') # CE checked 
ax.plot(-124.5,43.3,'rx') # CA checkeed 

sys.exit()
plt.gcf().savefig(fig_out)

