'''
Step 2: 
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

TODO this code needs to be cleaned up, but I also think that there are some data that
need to be filtered out. Need to talk with A Sutton about the dataset (Sept 18 2024)
Maybe we should do like PL33 or some filter instead of moving average (right now)
'''

testing = False 

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

#sn_name_dict = {
#    'CHABA':'Chaba',
#    'CAPEELIZABETH':'Cape Elizabeth',
#    'CAPEARAGO':'Cape Arago',
#}

#if testing:
#    sn_list = ['CAPEELIZABETH']
#else:    
#    sn_list = list(sn_name_dict.keys())

 
sn_name_dict = {
    'CAPEARAGO':'Cape Arago'
} 
sn_list = list(sn_name_dict.keys())

cdx = 1 

for sn in sn_list:
    print(sn)
    # initialize plot 
    plt.close('all')
    fs=12
    plt.rc('font', size=fs)
    fig = plt.figure(figsize=(16,8))
    ax0 = plt.subplot2grid((2,1),(0,0),colspan=1,rowspan=1)
    
    fn_in = posixpath.join(in_dir, (sn +'.nc'))
    fn_out = posixpath.join(out_dir, (sn + '_daily.nc'))
    
    ds = xr.open_dataset(fn_in, decode_times=True) 
    
    sys.exit()
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
    
    #t1 = datetime.datetime(dt[0].year,dt[0].month,dt[0].day,00,17,00)
    #t2 = datetime.datetime(dt[-1].year,dt[-1].month,dt[-1].day,21,17,00)
    
    #start_time = pd.to_datetime(datetime.datetime(dt[0].year,dt[0].month,dt[0].day,00,17,00))
    #end_time = pd.to_datetime(datetime.datetime(dt[-1].year,dt[-1].month,dt[-1].day,21,17,00))
    #time_array = pd.date_range(start_time,end_time,freq='3H')
    
    #SA = ds.SA.values
    #SAi = np.interp(time_array,dt,SA)
    
    varmean = {}
    time_array = {}
    datas = {}
    rw = {}
    
    mSA = {}
    mSP = {}
    mIT = {}
    mCT = {}
    mSIG0 = {}
    mDO = {}
    mDOXY = {}
    mpCO2 = {}
    mpH = {}
    
    mtime = {}
    
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
            time_array = pd.date_range(start_time,end_time,freq='3H')
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
        
        plt.plot(time_array,nSA,color = 'green',marker='.',linestyle='-',linewidth=3,label ='SA')
        
        SArw = nSA.rolling(9,min_periods=8,center=True)
        SPrw = nSP.rolling(9,min_periods=8,center=True)
        ITrw = nIT.rolling(9,min_periods=8,center=True)
        CTrw = nCT.rolling(9,min_periods=8,center=True)
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
        
        if np.all(np.isnan(SAmean)) == False: # if true wasn't enough data and array saved is almost all NaN --> skip 
            mSA[idx] = pd.Series(np.interp(mtime[idx],time_array,SAmean))
            mSP[idx] = pd.Series(np.interp(mtime[idx],time_array,SPmean))
            mIT[idx] = pd.Series(np.interp(mtime[idx],time_array,ITmean))
            mCT[idx] = pd.Series(np.interp(mtime[idx],time_array,CTmean))
            mSIG0[idx] = pd.Series(np.interp(mtime[idx],time_array,SIG0mean))
            mDO[idx] = pd.Series(np.interp(mtime[idx],time_array,DOmean))
            mDOXY[idx] = pd.Series(np.interp(mtime[idx],time_array,DOXYmean))
            mpCO2[idx] = pd.Series(np.interp(mtime[idx],time_array,pCO2mean))
            mpH[idx] = pd.Series(np.interp(mtime[idx],time_array,pHmean))
        
            plt.plot(mtime[idx],mSA[idx],color = 'blue',marker='x',linestyle='-',linewidth=3,label ='daily')
            
            if 'SA' in locals():
                SA = np.concatenate((SA,np.array(mSA[idx])))
                SP = np.concatenate((SP,np.array(mSP[idx])))
                IT = np.concatenate((IT,np.array(mIT[idx])))
                CT = np.concatenate((CT,np.array(mCT[idx])))
                SIG0 = np.concatenate((SIG0,np.array(mSIG0[idx])))
                DO = np.concatenate((DO,np.array(mDO[idx])))
                DOXY = np.concatenate((DOXY,np.array(mDOXY[idx])))
                pCO2 = np.concatenate((pCO2,np.array(mpCO2[idx])))
                pH = np.concatenate((pH,np.array(mpH[idx])))
            
                ichunk = np.ones(np.shape(mSA[idx]))*cdx
                chunk = np.concatenate((chunk,ichunk))
                ctime = np.concatenate((ctime,mtime[idx]))
                
            else: 
                SA = np.array(mSA[idx])
                SP = np.array(mSP[idx])
                IT = np.array(mIT[idx])
                CT = np.array(mCT[idx])
                SIG0 = np.array(mSIG0[idx])
                DO = np.array(mDO[idx])
                DOXY = np.array(mDOXY[idx])
                pCO2 = np.array(mpCO2[idx])
                pH = np.array(mpH[idx])
            
                chunk = np.ones(np.shape(SA))*cdx # an index of what chunk/deployment it is 
                ctime = mtime[idx]
                
            cdx = cdx+1
          
        #plt.plot(time_array[idx],varmean[idx],color = 'red',marker='.',linestyle='-',linewidth=3,label ='daily')
        
        
    plt.plot(ctime,SA,color = 'yellow',marker='.',linestyle='none',linewidth=3,label ='12UTC')   
    
    #initialize new dataset and fill
    coords = {'time_utc':('time_utc',ctime)}
    ds = xr.Dataset(coords=coords, 
        attrs={'Station Name':sn_name_dict[sn],'lon':ds.lon,'lat':ds.lat,
               'Depth':'surface 0.5m',
               'Source file':str(fn_in),'data processed': 'DailyAverages centered 12UTC'})
    
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
       
    ds['DOXY'] = xr.DataArray(DOXY, dims=('time'),
        attrs={'units':'umol/kg', 'long_name':'Dissolved Oxygen', 'depth':str('0.5m')})
         
    ds['pCO2_sw'] = xr.DataArray(pCO2, dims=('time'),
        attrs={'units':'uatm', 'long_name':'seawater pCO2', 'depth':str('<0.5m')})
        
    #ds['pCO2_air'] = xr.DataArray(pCO2_air, dims=('time'),
    #    attrs={'units':'uatm', 'long_name':'air pCO2', 'depth':str('0.5-1m')})
        
    #ds['xCO2_air'] = xr.DataArray(xCO2_air, dims=('time'),
    #    attrs={'units':'uatm', 'long_name':'air xCO2', 'depth':str('0.5-1m')})   
    
    ds['pH'] = xr.DataArray(pH, dims=('time'),
        attrs={'units':' ', 'long_name':'seawater pH', 'depth':str('0.5m')})
    
    #ds['CHL'] = xr.DataArray(CHL, dims=('time'),
    #    attrs={'units':'ug/L', 'long_name':'fluorescence-based nighttime chlorophyll-a', 'depth':str('0.5m')})  
        
    #ds['Turbidity'] = xr.DataArray(NTU, dims=('time'),
    #    attrs={'units':'NTU', 'long_name':'turbidity', 'depth':str('0.5m')})
      
    ds['chunk'] = xr.DataArray(chunk, dims=('time'),
        attrs={'units':'index', 'long_name':'chunk-deployment'})
          
    if not testing:
        ds.to_netcdf(fn_out, unlimited_dims='time')
        
