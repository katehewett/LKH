'''
Step 3: 
organize model output - 
grab surface 
calc carb family 
and save

rev1 = you need to not be so sloppy on your z's 

_rev1_chaba.py = we had to rerun chaba because sutton et al. 2019 
had mistake in offshore distance. 

we extracted 2013 - 2023 so only need to open 1 file

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
#    'CHABA':'Chaba'
#}

sn = 'CHABA'

# output/input locations
#mooring_in_dir = '/Users/katehewett/Documents/LKH_data/sutton2019_surface_buoys/daily'
model_in_dir = '/Users/katehewett/Documents/LO_output/extract/cas7_t0_x4b/moor/Sutton_etal_2019/'

#if os.path.exists(mooring_in_dir)==False:
#    print('input path for obs data does not exist')
#    sys.exit()
if os.path.exists(model_in_dir)==False:
    print('input path for model output data does not exist')
    sys.exit()    

out_dir = '/Users/katehewett/Documents/LKH_data/sutton2019_surface_buoys/LO_surface_extraction'
if os.path.exists(out_dir)==False:
    Lfun.make_dir(out_dir, clean = False)
    
# there is one obs .nc file for each mooring [daily values]
#obs_fn = posixpath.join(mooring_in_dir, (sn +'_daily.nc'))
#obs_ds = xr.open_dataset(obs_fn, decode_times=True)

# and 1 LO files
LO_fn1 = posixpath.join(model_in_dir, (sn + '_2013.01.01_2023.12.31.nc'))
LO_ds = xr.open_dataset(LO_fn1, decode_times=True)

# grab all the LO data surface data and put to one array 
        
lat = LO_ds.lat_rho.values
lon = LO_ds.lon_rho.values
    
LO_time = pd.to_datetime(LO_ds['ocean_time'].values)
        
DO = np.array(LO_ds['oxygen'].values[:,-1])     # only surface
SP = np.array(LO_ds['salt'].values[:,-1])
PT = np.array(LO_ds['temp'].values[:,-1])
z_rho = np.array(LO_ds.z_rho.values[:,-1])
z_w = np.array(LO_ds.z_w.values[:,-1])
Z = np.array(LO_ds.z_w.values[:,-1]-LO_ds.z_w.values[:,-2])
P = np.array(gsw.p_from_z(Z,lat))
SA = np.array(gsw.SA_from_SP(SP, P, lon, lat))
CT = np.array(gsw.CT_from_pt(SA, PT))
SIG0 = np.array(gsw.sigma0(SA,CT))
DENS = np.array(SIG0 + 1000)
ALK = np.array(LO_ds['alkalinity'].values[:,-1]) 
TIC = np.array(LO_ds['TIC'].values[:,-1])
        
plt.plot(LO_time,ALK)

# calculate Oag and pH
ti = gsw.t_from_CT(SA, CT, P) # in situ temperature [degC]
rho = gsw.rho(SA, CT, P) # in situ density [kg m-3]
# alkalinity [milli equivalents m-3 = micro equivalents L-1]
# TIC [millimol C m-3 = micromol C L-1]
# Convert from micromol/L to micromol/kg using in situ dentity because these are the
# units expected by pyco2.
# Convert from micromol/L to micromol/kg using in situ dentity because these are the
# units expected by pyco2.
ALK1 = 1000 * ALK / rho
TIC1 = 1000 * TIC / rho
# I'm not sure if this is needed. In the past a few small values of these variables had
# caused big slowdowns in the MATLAB version of CO2SYS.
ALK1[ALK1 < 100] = 100
TIC1[TIC1 < 100] = 100

CO2dict = pyco2.sys(par1=ALK1, par1_type=1, par2=TIC1, par2_type=2,
        salinity=SP, temperature=ti, pressure=P,
        total_silicate=50, total_phosphate=2, opt_k_carbonic=10, opt_buffers_mode=0)
        
ARAG = CO2dict['saturation_aragonite']
pH_total = CO2dict['pH_total']
#xCO2 = CO2dict['xCO2']
pCO2 = CO2dict['pCO2']
#fCO2 = CO2dict['fCO2']

#initialize new dataset and fill
fn_out = posixpath.join(out_dir, ('LO_'+sn + '_surface.nc'))

coords = {'time_utc':('time_utc',LO_time)}
ds = xr.Dataset(coords=coords, attrs={'Station Name':sn,'lon':lon,'lat':lat,
           'Depth':'surface cell thickness: '+str(np.round(np.nanmean(Z),1))+'m',
           'Source file':str(model_in_dir),'data processed': 'DailyAverages centered 12UTC'})

ds['SA'] = xr.DataArray(SA, dims=('time'),
    attrs={'units':'g kg-1', 'long_name':'Absolute Salinity'})
    
ds['SP'] = xr.DataArray(SP, dims=('time'),
    attrs={'units':' ', 'long_name':'Practical Salinity'})
    
ds['IT'] = xr.DataArray(ti, dims=('time'),
    attrs={'units':'degC', 'long_name':'Insitu Temperature'})
    
ds['CT'] = xr.DataArray(CT, dims=('time'),
    attrs={'units':'degC', 'long_name':'Conservative Temperature'})

ds['SIG0'] = xr.DataArray(SIG0, dims=('time'),
    attrs={'units':'kg/m3', 'long_name':'Potential Density Anomaly'})

ds['rho'] = xr.DataArray(rho, dims=('time'),
    attrs={'units':'kg/m3', 'long_name':'insitu density'})
            
ds['DO (uM)'] = xr.DataArray(DO, dims=('time'),
    attrs={'units':'uM', 'long_name':'Dissolved Oxygen'})

ds['ALK'] = xr.DataArray(ALK, dims=('time'),
    attrs={'units':'milli equivalents m-3 = micro equivalents L-1', 'long_name':'Total Alkalinity'})

ds['ARAG'] = xr.DataArray(ARAG, dims=('time'),
    attrs={'units':' ', 'long_name':'pyco2sys saturation state of aragonite'})
             
ds['pCO2'] = xr.DataArray(pCO2, dims=('time'),
    attrs={'units':'uatm', 'long_name':'pyco2sys sw partial pressure of CO2'})
    
ds['pH_total'] = xr.DataArray(pH_total, dims=('time'),
    attrs={'units':' ', 'long_name':'pyco2sys pH on total scale'})
    
if not testing:
    ds.to_netcdf(fn_out, unlimited_dims='time')
        







    
    
    
    