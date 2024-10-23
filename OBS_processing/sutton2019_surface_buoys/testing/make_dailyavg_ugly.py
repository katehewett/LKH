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

this code needs to be cleaned up 
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
 
# initialize plot 
plt.close('all')
fs=12
plt.rc('font', size=fs)
fig = plt.figure(figsize=(16,8))
ax0 = plt.subplot2grid((2,1),(0,0),colspan=1,rowspan=1)
    
for sn in sn_list:
    print(sn)
    fn_in = posixpath.join(in_dir, (sn +'.nc'))
    fn_out = posixpath.join(out_dir, (sn + '_daily.nc'))
    
    ds = xr.open_dataset(fn_in, decode_times=True) 
    
    dt = pd.to_datetime(ds.time_utc,format = '%YYYY-%m-%d HH:MM',errors='coerce')
    
    df = pd.DataFrame({'time_utc':dt,'SA':ds.SA.values})
    df['time_diff'] = df['time_utc'].diff()
    # A fix: Can't just search for when: 
    # df['3 hour'] = (df.time_diff == pd.Timedelta('3 hour')) 
    # because some diffs are 2 hr 59 min; 3 hour 1 min [ex: CHABA], etc.
    # So we use a 15 min threhsold to catch them all 
    threshold = pd.Timedelta('15 minutes') 
    df['3 hour'] = (abs(df['time_diff']-pd.Timedelta('3 hour'))) <= threshold
    # second option - find all the time deltas under 3 hours 15 mins (bc some short dt's present)
    df['LT 3 hour'] = df['time_diff']<pd.Timedelta('3 hours 15 minutes')
    
    # find start & stop indicies of all continious data
    x = df['LT 3 hour']
    s = pd.Series(x)
    grp = s.eq(False).cumsum()
    arr = grp.loc[s.eq(True)].groupby(grp).apply(lambda x: [x.index.min()-1,x.index.max()])
    a = np.array(arr) # its an array of start and stop indicies
                      # a[0] = index of first chunk of data: (start,end) aka (a[0][0],a[0][1]) 
    

    # initialize - rolling mean at 8PM UTC 
    mtime = {}   
    mSA = {}
    mSP = {}
    mIT = {}
    mCT = {}
    mSIG0 = {}
    mDO = {}
    mDOXY = {}
    mpCO2 = {}
    mpH = {}

    # step thru the chunks of continious data 
    for idx in range(0,np.shape(a)[0]):
        itime = dt[a[idx][0]:a[idx][1]]
        iSA = ds.SA.values[a[idx][0]:a[idx][1]]
        iSP = ds.SP.values[a[idx][0]:a[idx][1]]
        iIT = ds.IT.values[a[idx][0]:a[idx][1]]
        iCT = ds.CT.values[a[idx][0]:a[idx][1]]
        iSIG0 = ds.SIG0.values[a[idx][0]:a[idx][1]]
        iDO = ds['DO (uM)'].values[a[idx][0]:a[idx][1]]
        iDOXY = ds.DOXY.values[a[idx][0]:a[idx][1]]
        ipCO2 = ds.pCO2_sw.values[a[idx][0]:a[idx][1]]
        ipH = ds.pH.values[a[idx][0]:a[idx][1]]
        
        # if dt is not 3 hourly then resample  
        idt = df['3 hour'][a[idx][0]:a[idx][1]]
        if np.any(idt == False):  # not all dt's == 3 hourly; resample:
            start_time = itime[0] 
            end_time = itime[-1] 
            time_array[idx] = pd.date_range(start_time,end_time,freq='3H')
            nSA = pd.Series(np.interp(time_array,itime,iSA))
            nSP = pd.Series(np.interp(time_array,itime,iSP))
            nIT = pd.Series(np.interp(time_array,itime,iIT))
            nCT = pd.Series(np.interp(time_array,itime,iCT))
            nSIG0 = pd.Series(np.interp(time_array,itime,iSIG0))
            nDO = pd.Series(np.interp(time_array,itime,iDO))
            nDOXY = pd.Series(np.interp(time_array,itime,iDOXY))
            npCO2 = pd.Series(np.interp(time_array,itime,ipCO2))
            npH = pd.Series(np.interp(time_array,itime,ipH))
            
            mtime[idx] = pd.date_range(pd.to_datetime(start_time.date())+pd.Timedelta(hours=12),pd.to_datetime(end_time.date())+pd.Timedelta(hours=12),freq='D')
        elif np.all(idt==True): # good to go 
            time_array = itime
            nSA = pd.Series(iSA)
            nSP = pd.Series(iSP)
            nIT = pd.Series(iIT)
            nCT = pd.Series(iCT)
            nSIG0 = pd.Series(iSIG0)
            nDO = pd.Series(iDO)
            nDOXY = pd.Series(iDOXY)
            npCO2 = pd.Series(ipCO2)
            npH = pd.Series(ipH)
            
            start_time = itime[0] 
            end_time = itime[-1]
            mtime[idx] = pd.date_range(pd.to_datetime(start_time.date())+pd.Timedelta(hours=12),pd.to_datetime(end_time.date())+pd.Timedelta(hours=12),freq='D')
        
        SArw = nSA.rolling(9,min_periods=8,center=True)
        SPrw = nSP.rolling(9,min_periods=8,center=True)
        ITrw = nIT.rolling(9,min_periods=8,center=True)
        CTw = nCT.rolling(9,min_periods=8,center=True)
        SIG0rw = nSIG0.rolling(9,min_periods=8,center=True)
        DOrw = nDO.rolling(9,min_periods=8,center=True)
        DOXYrw = nDOXY.rolling(9,min_periods=8,center=True)
        pCO2rw = npCO2.rolling(9,min_periods=8,center=True)
        pHrw = npH.rolling(9,min_periods=8,center=True)
        
        SAmean = SArw.mean()
        SPmean = SPrw.mean()
        ITmean = ITrw.mean()
        CTmean = CTrw.mean()
        SIG0mean = SIG0rw.mean()
        DOmean = DOrw.mean()
        DOXYmean = DOXYrw.mean()
        pCO2mean = pCO2rw.mean()
        pHmean = pHrw.mean()
        
        mSA[idx] = pd.Series(np.interp(mtime[idx],time_array[idx],SAmean))
        mSP[idx] = pd.Series(np.interp(mtime[idx],time_array[idx],SPmean))
        mIT[idx] = pd.Series(np.interp(mtime[idx],time_array[idx],ITmean))
        mCT[idx] = pd.Series(np.interp(mtime[idx],time_array[idx],CTmean))
        mSIG0[idx] = pd.Series(np.interp(mtime[idx],time_array[idx],SIG0mean))
        mDO[idx] = pd.Series(np.interp(mtime[idx],time_array[idx],DOmean))
        mDOXY[idx] = pd.Series(np.interp(mtime[idx],time_array[idx],DOXYmean))
        mpCO2[idx] = pd.Series(np.interp(mtime[idx],time_array[idx],pCO2mean))
        mpH[idx] = pd.Series(np.interp(mtime[idx],time_array[idx],pHmean))
        
        plt.plot(time_array[idx],datas[idx],color = 'green',marker='x',linestyle='-',linewidth=3,label ='daily')
        #plt.plot(time_array[idx],varmean[idx],color = 'red',marker='.',linestyle='-',linewidth=3,label ='daily')
        plt.plot(mtime[idx],mSA[idx],color = 'blue',marker='s',linestyle='-',linewidth=3,label ='daily')
        
    #plt.plot(dt,ds.SA,color = 'dodgerblue',marker='.',linestyle='none',linewidth=3,label ='SA')

    sys.exit()

    # initialize plot 
    plt.close('all')
    fs=12
    plt.rc('font', size=fs)
    fig = plt.figure(figsize=(16,8))
    
    ax0 = plt.subplot2grid((2,1),(0,0),colspan=1,rowspan=1)
    #plt.plot(dt,varmean,color = 'crimson',marker='none',linestyle='-',linewidth=3,label ='daily')
    for idx in range(0,np.shape(a)[0]):
        plt.plot(time_array[idx],varmean[idx],color = 'red',marker='x',linestyle='-',linewidth=3,label ='daily')
    #plt.plot(dt,varmean2,color = 'green',marker='.',linestyle='none',linewidth=3,label ='daily')
    plt.plot(dt,ds.SA,color = 'dodgerblue',marker='.',linestyle='none',linewidth=3,label ='SA')
    plt.gcf().tight_layout()    
    
    #for i in range(0,np.shape(a)[0]):
    #    T = dt[a[i][0]:a[i][1]]
    #    S = ds.SA[a[i][0]:a[i][1]]
    #    plt.plot(T,S,color = 'red',marker='x',linestyle='none',linewidth=3,label ='SA f')
    
    df.set_index('time_utc')
        

