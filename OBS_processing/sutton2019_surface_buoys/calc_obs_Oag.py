'''
Step 4: 
Calc Oag for ...
* moorning data (pCO2_sw, S, T) using LO TA 

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

testing = False 

#sn_name_dict = {
#    'CHABA':'Chaba',
#    'CAPEELIZABETH':'Cape Elizabeth',
#    'CAPEARAGO':'Cape Arago'
#}

#sn_name_dict = {
#    'CHABA':'Chaba'
#}

sn = 'CAPEARAGO'

# output/input locations
mooring_in_dir = '/Users/katehewett/Documents/LKH_data/sutton2019_surface_buoys/daily'
model_in_dir = '/Users/katehewett/Documents/LKH_data/sutton2019_surface_buoys/LO_surface_extraction'

if os.path.exists(mooring_in_dir)==False:
    print('input path for obs data does not exist')
    sys.exit()
if os.path.exists(model_in_dir)==False:
    print('input path for model output data does not exist')
    sys.exit()    

out_dir = '/Users/katehewett/Documents/LKH_data/sutton2019_surface_buoys/daily/add_Oag'
if os.path.exists(out_dir)==False:
    Lfun.make_dir(out_dir, clean = False)
    
# there is one obs .nc file for each mooring [daily values]
obs_fn = posixpath.join(mooring_in_dir, (sn +'_daily.nc'))
obs_ds = xr.open_dataset(obs_fn, decode_times=True)

out_fn = posixpath.join(out_dir, (sn+'_daily_Oag.nc'))

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
LO_ALK = LO_ds.ALK.values[start_index:stop_index]  

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
    
oSA = obs_ds['SA'].values[start_index:stop_index] 
oCT = obs_ds['CT'].values[start_index:stop_index] 
opCO2 = obs_ds['pCO2_sw'].values[start_index:stop_index] 
p = np.array(gsw.p_from_z(0.5,obs_ds.lat))
oP = np.ones(np.shape(oSA))*p
oSP = obs_ds['SP'].values[start_index:stop_index] 
ti = gsw.t_from_CT(oSA, oCT, oP) # in situ temperature [degC]
rho = gsw.rho(oSA, oCT, oP) # in situ density [kg m-3]

# Convert from micromol/L to micromol/kg using in situ dentity because these are the
# units expected by pyco2.
LO_ALK = 1000 * LO_ALK / rho

# I'm not sure if this is needed. In the past a few small values of these variables had
# caused big slowdowns in the MATLAB version of CO2SYS.
LO_ALK[LO_ALK < 100] = 100

CO2dict = pyco2.sys(par1=LO_ALK, par1_type=1, par2=opCO2, par2_type=4,
    salinity=oSP, temperature=ti, pressure=oP,
    total_silicate=50, total_phosphate=2, opt_k_carbonic=10, opt_buffers_mode=0)
        
oARAG = CO2dict['saturation_aragonite']
opH_total = CO2dict['pH_total']

#initialize arrays to be the size of the obs data set 
ARAG = np.ones(np.shape(otime))*np.nan
new_pH = np.ones(np.shape(otime))*np.nan

ARAG[start_index:stop_index] = oARAG
new_pH[start_index:stop_index] = opH_total

'''
plt.close('all')
fs=12
plt.rc('font', size=fs)
fig = plt.figure(figsize=(16,8))
ax0 = plt.subplot2grid((3,1),(0,0),colspan=1,rowspan=1)
plt.plot(otime,ARAG,color = 'black',marker='o',linestyle='none',linewidth=1)
plt.plot(time_utc,oARAG,color = 'red',marker='.',linestyle='-',linewidth=1)
ax0.set_ylabel('ARAG')
plt.grid(True)

ax0 = plt.subplot2grid((3,1),(1,0),colspan=1,rowspan=1)
plt.plot(otime,new_pH,color = 'black',marker='o',linestyle='none',linewidth=1)
plt.plot(time_utc,opH_total,color = 'red',marker='.',linestyle='-',linewidth=1)
ax0.set_ylabel('pH')
plt.grid(True)
'''    

if np.shape(obs_ds['SA']) == np.shape(ARAG):
    obs_ds['ARAG'] = xr.DataArray(ARAG, dims=('time'),
        attrs={'units':' ', 'long_name':'pyco2sys saturation state of aragonite w/ LO ALK'})
    
    obs_ds['pH_total'] = xr.DataArray(new_pH, dims=('time'),
        attrs={'units':' ', 'long_name':'pyco2sys pH on total scale w/ LO ALK'})
        
else: 
    print('error with array shapes')
    sys.exit()
 
  
if not testing:
    obs_ds.to_netcdf(out_fn, unlimited_dims='time')
    print('saved')
        


