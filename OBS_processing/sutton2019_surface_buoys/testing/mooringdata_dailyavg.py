'''
This code is to estimate daily averages of mooring data from NOAA data product for 
3 moornings: 
    Chaba 
    Cape Elizabeth 
    Cape Arago 
 
iput .nc files were created by running process_webdata.py and then saved here: 
/Users/katehewett/Documents/LKH_data/sutton2019_surface_buoys/py_files/

daily files saved here:
/Users/katehewett/Documents/LKH_data/sutton2019_surface_buoys/daily/

Ran thru this exercise so could compare with lowpass mooring extractions.

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

Ldir = Lfun.Lstart()

# processed data location
source = 'ocnms'
otype = 'moor' 
in_dir = Ldir['parent'] / 'LKH_data' / 'sutton2019_surface_buoys' / 'py_files'
out_dir = Ldir['parent'] / 'LKH_data' / 'sutton2019_surface_buoys' / 'daily'

if os.path.exists(out_dir)==False:
    Lfun.make_dir(out_dir, clean = False)
if os.path.exists(in_dir)==False:
    print('input path does not exist')
    sys.exit()

sn_name_dict = {
    'CHABA':'Chaba',
    'CAPEELIZABETH':'Cape Elizabeth',
    'CAPEARAGO':'Cape Arago',
}

if testing:
    sn_list = ['CHABA']
else:    
    sn_list = list(sn_name_dict.keys())
    
for sn in sn_list:
    print(sn)
    in_fn = posixpath.join(in_dir, (sn +'.nc'))
    out_fn = posixpath.join(out_dir, (sn + '_daily.nc'))
    
    df = read_data_with_header(in_fn, header_keyword)
    print(df)
    
    dt = pd.to_datetime(df.datetime_utc)
