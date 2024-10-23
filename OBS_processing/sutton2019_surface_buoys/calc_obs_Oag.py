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

sn = 'CAPEELIZABETH'

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

# take sample LO TA at obs times
otime = pd.to_datetime(obs_ds.datetime_utc.values)
ot = otime[otime.year>2012] # first day of model is 2013 Jan 01

ltime = pd.to_datetime(LO_ds.time_utc.values)
lALK = LO_ds.ALK.values

if ot[-1]>ltime[-1]:
    print('need to fix end date of obs')
    sys.exit()

# sample LO ALK at obs times (ot)
oALK = np.interp(ot,ltime,lALK)

oSA = obs_ds['SA'].values[otime.year>2012]
oCT = obs_ds['CT'].values[otime.year>2012]
opCO2 = obs_ds['pCO2_sw'].values[otime.year>2012]
p = np.array(gsw.p_from_z(0.5,obs_ds.lat))
oP = np.ones(np.shape(oSA))*p
oSP = obs_ds['SP'].values[otime.year>2012]
ti = gsw.t_from_CT(oSA, oCT, oP) # in situ temperature [degC]
rho = gsw.rho(oSA, oCT, oP) # in situ density [kg m-3]

# Convert from micromol/L to micromol/kg using in situ dentity because these are the
# units expected by pyco2.
oALK1 = 1000 * oALK / rho

# I'm not sure if this is needed. In the past a few small values of these variables had
# caused big slowdowns in the MATLAB version of CO2SYS.
oALK1[oALK1 < 100] = 100

CO2dict = pyco2.sys(par1=oALK1, par1_type=1, par2=opCO2, par2_type=4,
    salinity=oSP, temperature=ti, pressure=oP,
    total_silicate=50, total_phosphate=2, opt_k_carbonic=10, opt_buffers_mode=0)
        
oARAG = CO2dict['saturation_aragonite']
opH_total = CO2dict['pH_total']

#initialize arrays
ARAG = np.ones(np.shape(otime))*np.nan
new_pH = np.ones(np.shape(otime))*np.nan

# this indexing was driving me mad!! fix this
#if sn =='CHABA':
#    start_idx = 456 #np.where(ot[0]==otime)
#    stop_idx = 2270 #np.where(ot[-1]==otime)
#elif sn =='CAPEELIZABETH':
#    start_idx = 2044 #np.where(ot[0]==otime)
#    stop_idx = 3401 #np.where(ot[-1]==otime)
#elif sn =='CAPEARAGO':
#    start_idx = 0 #np.where(ot[0]==otime)
#    stop_idx = 1089 #np.where(ot[-1]==otime)

#ARAG[start_idx:start_idx+len(oARAG)] = oARAG
#new_pH[start_idx:start_idx+len(opH_total)] = opH_total

if np.shape(ARAG) == np.shape(oARAG):
    ARAG = oARAG
    new_pH = opH_total

if np.shape(obs_ds['SA']) == np.shape(ARAG):
    obs_ds['ARAG'] = xr.DataArray(ARAG, dims=('time'),
        attrs={'units':' ', 'long_name':'pyco2sys saturation state of aragonite'})
    
    obs_ds['new_pH'] = xr.DataArray(new_pH, dims=('time'),
        attrs={'units':' ', 'long_name':'pyco2sys pH on total scale'})
        
else: 
    print('error with array shapes')
    sys.exit()
    
if not testing:
    obs_ds.to_netcdf(out_fn, unlimited_dims='time')
        


