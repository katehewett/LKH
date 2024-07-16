'''
plotting seasonal maps of 
N2max + depth 
sml 


'''

# imports
from lo_tools import Lfun, zfun

import sys 
import xarray as xr
import netCDF4 as nc
import pandas as pd

Ldir = Lfun.Lstart()

fn_o = Ldir['parent'] / 'plotting' / 'clines' 
# lopass test 1 
#fn_i = Ldir['LOo'] / 'extract' / 'cas7_t0_x4b' / 'clines' / 'shelf_box_2017.12.12_2017.12.13' 
fn_i = Ldir['LOo'] / 'extract' / 'cas7_t0_x4b' / 'clines' / 'shelf_box_2014.01.01_2014.12.31' 


#fnb_in = fn_i / 'box_000000.nc'
#fnc_in = fn_i / 'cline_000310.nc'
#fnf_in = fn_i / 'shelf_box_pycnocline_2017.12.12_2017.12.13.nc'
fnf_in = fn_i / 'shelf_box_pycnocline_2014.01.01_2014.12.31.nc'

#dsb = xr.open_dataset(fnb_in, decode_times=True)  
#dsc = xr.open_dataset(fnc_in, decode_times=True)  
dsd = xr.open_dataset(fnf_in, decode_times=True) 

ot = pd.to_datetime(dsd.ocean_time.values)
