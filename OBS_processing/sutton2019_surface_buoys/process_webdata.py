'''
This code is to pull and process data from a NOAA data product for 5 buoys:
    Chaba buoy : https://www.pmel.noaa.gov/co2/timeseries/CHABA.txt
    Cape Elizabeth buoy : https://www.pmel.noaa.gov/co2/timeseries/CAPEELIZABETH.txt
    Cape Arago buoy : https://www.pmel.noaa.gov/co2/timeseries/CAPEARAGO.txt
    CCE2 buoy : https://www.pmel.noaa.gov/co2/timeseries/CCE2.txt
    CCE1 buoy : https://www.pmel.noaa.gov/co2/timeseries/CCE1.txt

and then convert and save in LO format for model-obs comparisons.

see README.txt for more information and data source information

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

testing = False 

# def function to read .txt with text before the header lines
# need to put in seperate file under /LKH so can call ::
def read_data_with_header(fn_i, header_keyword):

    with open(fn_i, 'r') as f:
        lines = f.readlines()

    # Find the header line
    header_index = None
    for i, line in enumerate(lines):
        if header_keyword in line:
            header_index = i
            break

    if header_index is None:
        raise ValueError(f"Header keyword '{header_keyword}' not found in file.")

    # Read the data into a dataframe
    df = pd.read_csv(fn_i, skiprows=header_index, delimiter='\t')

    return df

# output/input locations
datapath = '/Users/katehewett/Documents/LKH_data/sutton2019_surface_buoys'
out_dir = posixpath.join(datapath, 'py_files')
in_dir = posixpath.join(datapath, 'data_files')

if os.path.exists(out_dir)==False:
    Lfun.make_dir(out_dir, clean = False)
if os.path.exists(in_dir)==False:
    print('input path does not exist')
    sys.exit()

# run the 5 files; all have the same header order 
header_keyword = 'datetime_utc	SST	SSS	pCO2_sw	pCO2_air	xCO2_air	pH_sw	DOXY	CHL	NTU'
sn_name_dict = {
    'CHABA':'Chaba',
    'CAPEELIZABETH':'Cape Elizabeth',
    'CAPEARAGO':'Cape Arago',
    'CCE1':'California Current Ecosystem 1', 
    'CCE2':'California Current Ecosystem 2'
}

if testing:
    sn_list = ['CHABA']
else:    
    sn_list = list(sn_name_dict.keys())
    
for sn in sn_list:
    print(sn)
    in_fn = posixpath.join(in_dir, (sn +'.txt'))
    out_fn = posixpath.join(out_dir, (sn + '.nc'))
    
    df = read_data_with_header(in_fn, header_keyword)
    print(df)
    
    dt = pd.to_datetime(df.datetime_utc)
    IT = np.array(df.SST)
    SP = np.array(df.SSS)
    pCO2_sw = np.array(df.pCO2_sw)
    pCO2_air = np.array(df.pCO2_air)
    xCO2_air = np.array(df.xCO2_air)
    pH = np.array(df.pH_sw)
    DOXY = np.array(df.DOXY)
    CHL = np.array(df.CHL)
    NTU = np.array(df.NTU)
    
    del lat,lon
    if sn == 'CHABA':
        lat = 47.936
        lon = -125.958
    elif sn == 'CAPEELIZABETH':
        lat = 47.353
        lon = -124.731
    elif sn == 'CAPEARAGO':
        lat = 43.320
        lon = -124.500
    elif sn == 'CCE1':
        lat = 33.48
        lon = -122.51
    elif sn == 'CCE2':
        lat = 34.324
        lon = -120.814

    Z = 0.5 # m
    P = gsw.p_from_z(Z,lat)
    SA = gsw.SA_from_SP(SP, P, lon, lat)
    CT = gsw.CT_from_t(SA, IT, P)
    SIG0 = gsw.sigma0(SA,CT) # potential density anomaly w/ ref. pressure of 0 [RHO-1000 kg/m^3]
    DENS = SIG0 + 1000
    
    # DOXY saved as umol/kg ; and LO saved as uM (convert using potential density)
    DO = DOXY * DENS * (1/1000) # DOXY[umol/kg] * DENS[kg/m3] * [m3/L] 
    
    #initialize new dataset and fill
    coords = {'time':('time',dt)}
    ds = xr.Dataset(coords=coords, 
        attrs={'Station Name':sn_name_dict[sn],'lon':lon,'lat':lat,
               'Depth':'surface 0.5m',
               'Source file':str(in_fn),'data citation': 'Sutton et al. 2019'})
    
    ds['SA'] = xr.DataArray(SA, dims=('time'),
        attrs={'units':'g kg-1', 'long_name':'Absolute Salinity'})
        
    ds['SP'] = xr.DataArray(SP, dims=('time'),
        attrs={'units':' ', 'long_name':'Practical Salinity', 'depth':str('0.5m')})
        
    ds['IT'] = xr.DataArray(IT, dims=('time'),
        attrs={'units':'degC', 'long_name':'Insitu Temperature', 'depth':str('0.5m')})
        
    ds['CT'] = xr.DataArray(CT, dims=('time'),
        attrs={'units':'degC', 'long_name':'Conservative Temperature', 'depth':str('0.5m')})

    ds['SIG0'] = xr.DataArray(SIG0, dims=('time'),
        attrs={'units':'kg/m3', 'long_name':'Potential Density Anomaly'})
                
    ds['DO (uM)'] = xr.DataArray(DO, dims=('time'),
        attrs={'units':'uM', 'long_name':'Dissolved Oxygen'})
       
    ds['DOXY (uM)'] = xr.DataArray(DOXY, dims=('time'),
        attrs={'units':'umol/kg', 'long_name':'Dissolved Oxygen', 'depth':str('0.5m')})
         
    ds['pCO2_sw'] = xr.DataArray(pCO2_sw, dims=('time'),
        attrs={'units':'uatm', 'long_name':'seawater pCO2', 'depth':str('<0.5m')})
        
    ds['pCO2_air'] = xr.DataArray(pCO2_air, dims=('time'),
        attrs={'units':'uatm', 'long_name':'air pCO2', 'depth':str('0.5-1m')})
        
    ds['xCO2_air'] = xr.DataArray(xCO2_air, dims=('time'),
        attrs={'units':'uatm', 'long_name':'air xCO2', 'depth':str('0.5-1m')})   
    
    ds['pH'] = xr.DataArray(pH, dims=('time'),
        attrs={'units':' ', 'long_name':'seawater pH', 'depth':str('0.5-1m')})
    
    ds['CHL'] = xr.DataArray(CHL, dims=('time'),
        attrs={'units':'ug/L', 'long_name':'fluorescence-based nighttime chlorophyll-a', 'depth':str('0.5m')})  
        
    ds['Turbidity'] = xr.DataArray(NTU, dims=('time'),
        attrs={'units':'NTU', 'long_name':'turbidity', 'depth':str('0.5m')})

    if not testing:
        ds.to_netcdf(out_fn, unlimited_dims='time')




                  

