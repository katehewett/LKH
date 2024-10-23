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
    
    df = pd.DataFrame({'time_utc':dt,'SA':ds.SA.values})
    df['time_diff'] = df['time_utc'].diff()
    df['dates'] = dt.date
    
    df['SA'] = ds.SA.values
    
    daily_avg = df.groupby('dates')['SA'].mean()
    Tdaily_avg = df.groupby('dates')['time_utc'].mean()
    #df.set_index('time_utc')
    
    # A fix: Can't just search for when: 
    # df['3 hour'] = (df.time_diff == pd.Timedelta('3 hour')) 
    # because some diffs are 2 hr 59 min; 3 hour 1 min [ex: CHABA], etc.
    # So we use a 15 min threhsold to catch them all 
    threshold = pd.Timedelta('15 minutes') 
    df['3 hour'] = (abs(df['time_diff']-pd.Timedelta('3 hour'))) <= threshold
    
    # find start stop indicies of continious 3 hour data 
    x = df['3 hour']
    s = pd.Series(x)
    grp = s.eq(False).cumsum()
    arr = grp.loc[s.eq(True)].groupby(grp).apply(lambda x: [x.index.min()-1,x.index.max()])
    a = np.array(arr) # its an array of start and stop indicies
                      # a[0] = index of first chunk of data: (start,end) aka (a[0][0],a[0][1]) 
    
    #df['run'] = df['3 hour']*df['SA']
    #df['run'][df['run']==0]= np.nan
    
    datas = pd.Series(ds.SA.values)
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
    plt.plot(dt,ds.SA,color = 'dodgerblue',marker='.',linestyle='none',linewidth=3,label ='SA')
    plt.plot(Tdaily_avg,daily_avg,color = 'red',marker='x',linestyle='none',linewidth=3,label ='SA')
    #plt.plot(dt,varmean,color = 'crimson',marker='none',linestyle='-',linewidth=3,label ='daily')
    #plt.plot(dt,varmean,color = 'black',marker='.',linestyle='none',linewidth=3,label ='daily')
    #plt.plot(dt,varmean2,color = 'green',marker='.',linestyle='none',linewidth=3,label ='daily')
    
    plt.gcf().tight_layout()    
    
    sys.exit()
    
    for i in range(0,np.shape(a)[0]):
        T = dt[a[i][0]:a[i][1]]
        S = ds.SA[a[i][0]:a[i][1]]
        plt.plot(T,S,color = 'red',marker='x',linestyle='none',linewidth=3,label ='SA f')
    
    df.set_index('time_utc')
        

