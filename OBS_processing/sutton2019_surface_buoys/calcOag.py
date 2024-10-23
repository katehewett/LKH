'''
Calc Oag for ...
* moorning data (pCO2_sw, S, T) using LO TA 
* LO mooring extractions (TA + TIC, S, T)


TODO: update and use Ldir so can run on apogee 
a/t/m this code is hardcoded for kh's computer only
**update testing tag + command line args 
**update so can do multiple years
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

testing = True 

#sn_name_dict = {
#    'CHABA':'Chaba',
#    'CAPEELIZABETH':'Cape Elizabeth',
#    'CAPEARAGO':'Cape Arago'
#}

#sn_name_dict = {
#    'CHABA':'Chaba'
#}

sn = 'CHABA'

# output/input locations
mooring_in_dir = '/Users/katehewett/Documents/LKH_data/sutton2019_surface_buoys/daily'
model_in_dir = '/Users/katehewett/Documents/LO_output/extract/cas7_t0_x4b/moor/Sutton_etal_2019/'

if os.path.exists(mooring_in_dir)==False:
    print('input path for obs data does not exist')
    sys.exit()
if os.path.exists(model_in_dir)==False:
    print('input path for model output data does not exist')
    sys.exit()    


# there is one obs .nc file for each mooring [daily values]
obs_fn = posixpath.join(mooring_in_dir, (sn +'_daily.nc'))
obs_ds = xr.open_dataset(obs_fn, decode_times=True)

# and 3 LO files
LO_fn1 = posixpath.join(model_in_dir, (sn + '_2013.01.01_2016.12.31'))
LO_fn2 = posixpath.join(model_in_dir, (sn + '_2017.01.01_2020.12.31'))
LO_fn3 = posixpath.join(model_in_dir, (sn + '_2021.01.01_2023.12.31'))

sys.exit()
# setup LO vars and calc SA CT SIG
lat = LO_ds.lat_rho.values
lon = LO_ds.lon_rho.values

DO = LO_ds['oxygen'].values[:,-1]     # only surface
SP = LO_ds['salt'].values[:,-1]
IT = LO_ds['temp'].values[:,-1]
z_rho = LO_ds.z_rho.values[:,-1]
z_w = LO_ds.z_w.values[:,-1]
Z = z_w-z_rho
P = gsw.p_from_z(Z,lat)
SA = gsw.SA_from_SP(SP, P, lon, lat)
CT = gsw.CT_from_t(SA, IT, P)
SIG0 = gsw.sigma0(SA,CT) # potential density anomaly w/ ref. pressure of 0 [RHO-1000 kg/m^3]
DENS = SIG0 + 1000

obs_time = pd.to_datetime(obs_ds['time'])
LO_time = pd.to_datetime(LO_ds['ocean_time'])

target_time1 = pd.datetime(2019,1,1,0,17)
start_idx = np.where(obs_time==target_time1)

target_time2 = pd.datetime(2019,11,13,3,17)
stop_idx = np.where(obs_time==target_time2)

idx = np.arange(14029,(16549+1)) # fix this!! the datetimes were annoying me with the search 
ot = obs_time[idx] 
oSA = obs_ds['SA'][idx]
oCT = obs_ds['CT'][idx]
opCO2 = obs_ds['pCO2_sw'][idx]
oDO = obs_ds['DO (uM)'][idx]

A = np.interp(ot[1],LO_time,SA)


#df = read_data_with_header(in_fn, header_keyword)
#print(df)
    
    
    
    
    