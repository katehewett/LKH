'''
Step 5
Calc Oag for ...
* moorning data (pCO2_sw, S, T) using LO TA 
* LO mooring extractions (TA + TIC, S, T)


TODO: update and use Ldir so can run on apogee 
a/t/m this code is hardcoded for kh's computer only
**update testing tag + command line args 
**update so can do multiple years
'''

testing = True 
yYear = 2019

import pandas as pd
import numpy as np
import posixpath
import datetime 
import gsw 
import xarray as xr
import os
import sys
from datetime import datetime 

from lo_tools import Lfun, zfun

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter

# output/input locations
#datapath = 
mooring_in_dir = '/Users/katehewett/Documents/LKH_data/sutton2019_surface_buoys/py_files'
model_in_dir = '/Users/katehewett/Documents/LO_output/extract/cas7_t0_x4b/moor/Sutton_etal_2019/'
out_dir = '/Users/katehewett/Documents/LKH_output/'
if os.path.exists(mooring_in_dir)==False:
    print('input path for obs data does not exist')
    sys.exit()
if os.path.exists(model_in_dir)==False:
    print('input path for model output data does not exist')
    sys.exit()

# header_keyword = 'datetime_utc	SST	SSS	pCO2_sw	pCO2_air	xCO2_air	pH_sw	DOXY	CHL	NTU'
sn_name_dict = {
    'CHABA':'Chaba',
    'CAPEELIZABETH':'Cape Elizabeth',
    'CAPEARAGO':'Cape Arago'
}

if testing:
    sn_list = ['CHABA']
else:    
    sn_list = list(sn_name_dict.keys())
    
for sn in sn_list:
    print(sn)
    obs_fn = posixpath.join(mooring_in_dir, (sn +'.nc'))
    LO_fn = posixpath.join(model_in_dir, (sn + '_'+str(yYear)+'.01.01_'+str(yYear)+'.12.31.nc'))
    out_fn = posixpath.join(model_in_dir, fig_nm)
    
    obs_ds = xr.open_dataset(obs_fn, decode_times=True) 
    LO_ds = xr.open_dataset(LO_fn, decode_times=True) 
    
    # LO MODEL OUTPUT DATA HERE::
    lat = LO_ds.lat_rho.values
    lon = LO_ds.lon_rho.values
    LO_time = pd.to_datetime(LO_ds['ocean_time'])
    
    DO = LO_ds['oxygen'].values[:,-1]     # only surface; i think the output units need to be fixed on the LO extraction from 
                                          # umol/m3 --> uM or umol/L??? 
    SP = LO_ds['salt'].values[:,-1]
    IT = LO_ds['temp'].values[:,-1]
    z_rho = LO_ds.z_rho.values[:,-1]
    z_w = LO_ds.z_w.values[:,-1]
    Z = z_w-z_rho
    P = gsw.p_from_z(Z,lat)
    SA = gsw.SA_from_SP(SP, P, lon, lat)
    CT = gsw.CT_from_t(SA, IT, P)
    #SIG0 = gsw.sigma0(SA,CT) # potential density anomaly w/ ref. pressure of 0 [RHO-1000 kg/m^3]
    #DENS = SIG0 + 1000
    
    # MOORING DATA HERE::
    obs_time = pd.to_datetime(obs_ds['time'])
    
    target_time1 = datetime(2019,1,1,0,17)
    start_idx = np.where(obs_time==target_time1)
    
    target_time2 = datetime(2019,11,13,3,17)
    stop_idx = np.where(obs_time==target_time2)
    
    idx = np.arange(14029,(16549+1)) # fix this!! the datetimes were annoying me with the search 
    ot = obs_time[idx] 
    oSA = obs_ds['SA'][idx]
    oCT = obs_ds['CT'][idx]
    #opCO2 = obs_ds['pCO2_sw'][idx]
    oDO = obs_ds['DO (uM)'][idx]
    
    
    # initialize plot 
    plt.close('all')
    fs=12
    plt.rc('font', size=fs)
    fig = plt.figure(figsize=(16,8))

    ax0 = plt.subplot2grid((3,1),(0,0),colspan=1,rowspan=1)
    plt.plot(LO_time,SA,color = 'navy',marker='.',linestyle='none',linewidth=3,alpha=0.6,markeredgecolor='none',label ='LO')
    plt.plot(ot,oSA,color = 'dodgerblue',marker='.',linestyle='none',linewidth=3,alpha=0.6,markeredgecolor='none',label ='obs')
    ax0.set_xticklabels([])
    ax0.set_ylim([26,34])
    ax0.set_yticks(np.arange(26,35,2), minor=False) 
    ax0.legend(loc="best",frameon=False)
    ax0.set_title('Cha ba: LO model output data (~0.5m); surface data obs (~0.5m)')
    ax0.set_ylabel('SA g/kg')
    
    ax1 = plt.subplot2grid((3,1),(1,0),colspan=1,rowspan=1)
    plt.plot(LO_time,CT,color = 'navy',marker='.',linestyle='none',linewidth=3,alpha=0.6,markeredgecolor='none',label ='LO')
    plt.plot(ot,oCT,color = 'dodgerblue',marker='.',linestyle='none',linewidth=3,alpha=0.6,markeredgecolor='none',label ='obs')
    ax1.set_xticklabels([])
    ax1.set_ylim([7,21])
    ax1.set_yticks(np.arange(7,22,2), minor=False)     
    ax1.legend(loc="best",frameon=False)
    ax1.set_ylabel('CT deg C')
    
    ax2 = plt.subplot2grid((3,1),(2,0),colspan=1,rowspan=1)
    plt.plot(LO_time,DO,color = 'navy',marker='.',linestyle='none',linewidth=3,alpha=0.6,markeredgecolor='none',label ='LO')
    plt.plot(ot,oDO,color = 'dodgerblue',marker='.',linestyle='none',linewidth=3,alpha=0.6,markeredgecolor='none',label ='obs')
    #ax2.set_xticklabels([])
    ax2.set_ylim([120,420])
    ax2.set_yticks(np.arange(120,421,40), minor=False)     
    ax2.legend(loc="best",frameon=False)
    ax2.set_ylabel('DO uM')
    
    plt.gcf().savefig(fig_nm)