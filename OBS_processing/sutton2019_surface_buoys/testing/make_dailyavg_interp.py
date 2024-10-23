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

testing = True 

import os
import sys
import pandas as pd
import numpy as np
import posixpath
import datetime 
import gsw 
import xarray as xr

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter

from lo_tools import Lfun, zfun

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
    fn_in = posixpath.join(in_dir, (sn +'.nc'))
    fn_out = posixpath.join(out_dir, (sn + '_daily.nc'))
    
    ds = xr.open_dataset(fn_in, decode_times=True) 
    
    dt = pd.to_datetime(ds.time_utc,format = '%YYYY-%m-%d HH:MM',errors='coerce')
    
    # interpolate so we know the spacing and can run our scripts with lil more easily. 
    # tried a few diff techniques, but the dt's change (sometimes mid timeseries chunk) 
    # and this is just a time saver that works well enough:
    a = datetime.datetime(dt[0].year,dt[0].month,dt[0].day,00,17,00)
    b = datetime.datetime(dt[-1].year,dt[-1].month,dt[-1].day,21,17,00)
    
    start_time = pd.to_datetime(a)
    end_time = pd.to_datetime(b)
    time_array = pd.date_range(start_time,end_time,freq='3H')
    
    SA = ds.SA.values
    SAi = np.interp(time_array,dt,SA)

    datas = pd.Series(SAi)
    rw = datas.rolling(8,min_periods=7,center=True)
    rw2 = datas.rolling(8*3,min_periods=7*3,center=True)
    varmean = rw.mean()
    varmean2 = rw2.mean()
    
    # initialize plot 
    plt.close('all')
    fs=12
    plt.rc('font', size=fs)
    fig = plt.figure(figsize=(16,8))
    
    ax0 = plt.subplot2grid((2,1),(0,0),colspan=1,rowspan=1)
    plt.plot(dt,SA,color = 'dodgerblue',marker='.',linestyle='none',linewidth=3,label ='SA')
    plt.plot(time_array,SAi,color = 'red',marker='x',linestyle='none',linewidth=3,label ='SA int')
    #plt.plot(dt,varmean,color = 'crimson',marker='none',linestyle='-',linewidth=3,label ='daily')
    #plt.plot(dt,varmean,color = 'black',marker='.',linestyle='none',linewidth=3,label ='daily')
    #plt.plot(dt,varmean2,color = 'green',marker='.',linestyle='none',linewidth=3,label ='daily')
    
    plt.gcf().tight_layout()    
    
    #df.set_index('time_utc')
        

