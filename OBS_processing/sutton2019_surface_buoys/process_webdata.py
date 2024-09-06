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

testing = True 

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
   
# run the 5 files; all have the same header order 
header_keyword = 'datetime_utc	SST	SSS	pCO2_sw	pCO2_air	xCO2_air	pH_sw	DOXY	CHL	NTU'
sn_name_dict = {
    'CHABA':'Chaba',
    'CE':'Cape Elizabeth',
    'CA':'Cape Arago',
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
    IT = df.SST
    SP = df.SSS
    pCO2_sw = df.pCO2_sw
    pCO2_air = df.pCO2_air
    xCO2_air = df.xCO2_air
    pH = df.pH_sw
    DOXY = df.DOXY
    CHL = df.CHL
    NTU = df.NTU
    
    if sn == 'CHABA':
        lat = 47.936
        lon = -125.958
    elif sn == 'CE':
        lat = 47.353
        lon = -124.731
    elif sn == 'CA':
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
    SIG0 = gsw.sigma0(SA,CT) # potential density relative to 0 dbar, minus 1000 kg m-3
    DENS = SIG0 + 1000
    
    DO = DOXY * DENS * (1/1000) # DOXY[umol/kg] * DENS[kg/m3] * [m3/L] 
    
    #initialize new dataset and fill
    coords = {'time':('time',dt)}
    ds = xr.Dataset(coords=coords, attrs={'Station Name':sn_name_dict[sn],'lon':lon,'lat':lat})
    
    ds['SA'] = xr.DataArray(SA, dims=('time','z'),
        attrs={'units':'g kg-1', 'long_name':'Absolute Salinity'})
    ds['SP'] = xr.DataArray(SA, dims=('time','z'),
        attrs={'units':' ', 'long_name':'Practical Salinity'})
    ds['IT'] = xr.DataArray(IT, dims=('time','z'),
        attrs={'units':'degC', 'long_name':'Insitu Temperature'})
    ds['CT'] = xr.DataArray(CT, dims=('time','z'),
        attrs={'units':'degC', 'long_name':'Conservative Temperature'})
    ds['DO (uM)'] = xr.DataArray(DO, dims=('time','z'),
        attrs={'units':'uM', 'long_name':'Dissolved Oxygen'})
    ds['SIG0'] = xr.DataArray(SIG0, dims=('time','z'),
        attrs={'units':'kg m-3', 'long_name':'Sigma0'})
        
    
    
    
    
    






                  

