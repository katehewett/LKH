# This plots OCNMS "midshelf" moorings from LO mooring extraction 
# cas7_t0_x4b. Want to grab Drho Dthermo and calc N2 S2 + Ri 
# this is to show parker an idea, so the locations are ~random but 
# something that I had already extracted

# imports
from lo_tools import Lfun, zfun, zrfun
from lo_tools import plotting_functions as pfun

#import sys 
import xarray as xr
import numpy as np
#import netCDF4 as nc
import os 
import gsw

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
import cmcrameri.cm as cmc
#import cmocean as cm
#import matplotlib.dates as mdates
from datetime import datetime
from datetime import time
import pandas as pd

Ldir = Lfun.Lstart()

# processed data location
source = 'ocnms'
otype = 'moor' 
data_dir = Ldir['LOo'] / 'obs' / source / otype

yr_list = [2017] 
numyrs = np.size(yr_list) # 2017 (2018 - 2023 running now)
runB = 'cas7_t0_x4b'

model_dir = Ldir['LOo'] / 'extract' / runB / otype / 'OCNMS_moorings_current' 

sn_name_dict = {
    'MB042':0
}

sn_list = list(sn_name_dict.keys())
numsites = len(sn_list)

# update with input tags for start and stop 
# make some datetimes for plotting 
mdates1 = pd.date_range(start='2017-01-01',end='2018-01-01', freq='MS')
mdates2 = pd.date_range(start='2017-01-01',end='2018-01-01', freq='16D')

for sn in sn_list:
    print(sn)
    
    # initialize plot 
    plt.close('all')
    fs=12
    plt.rc('font', size=fs)
    fig = plt.figure(figsize=(16,8))

    ax1 = plt.subplot2grid((4,1),(0,0),colspan=1,rowspan=1)
    ax2 = plt.subplot2grid((4,1),(1,0),colspan=1,rowspan=1)
    ax3 = plt.subplot2grid((4,1),(2,0),colspan=1,rowspan=1)
    ax4 = plt.subplot2grid((4,1),(3,0),colspan=1,rowspan=1)

    #add model data 
    in_fn = model_dir / (sn + '_' + str(yr_list[0]) + '.01.01_'+ str(yr_list[0]) + '.12.31.nc')
    if os.path.isfile(in_fn):
        ds2 = xr.open_dataset(in_fn)
        oxygen = ds2['oxygen']
        SP = ds2['salt']
        PT = ds2['temp']
        u = ds2['u']
        v = ds2['v']
        AKs = ds2['AKs']
        AKv = ds2['AKv']
        z_w = ds2['z_w'] 
        z_rho = ds2['z_rho'] 
        ot = ds2['ocean_time'].values[:]
        lat = ds2.lat_rho.values
        lon = ds2.lon_rho.values
        
        P = gsw.p_from_z(z_rho.values,lat)
        SA = gsw.SA_from_SP(SP.values, P, lon, lat)
        CT = gsw.CT_from_pt(SA, PT.values)
        SIG0 = gsw.sigma0(SA,CT)
        
        #po = np.nanmean(SIG0)+1000
        g = 9.81
        po = 1024
        
        # u and v are on rho points
        dz = np.diff(z_rho.values,axis=1)
        du = np.diff(u.values,axis=1)
        dv = np.diff(v.values,axis=1)
        drho = np.diff(SIG0,axis=1)
        dT = np.diff(CT,axis=1)
        dS = np.diff(SA,axis=1)
        dudz = du/dz
        dvdz = dv/dz
        dpdz = drho/dz
        dtdz = dT/dz
        dsdz = dS/dz
        S2 = np.square(dudz) + np.square(dvdz)
        N2 = -(g/po)*dpdz
        Rig = abs(N2)/S2
        Rig_N = np.log10(Rig/0.25)
        
        # for plotting N2 S2 dudz's 
        z = z_w.values[:,1:-1];  # chop the surface most value 
        
        Kdudz = dudz*AKv.values[:,1:-1];  
        Kdvdz = dvdz*AKv.values[:,1:-1]; 
        
        NZ = np.shape(z)[1]
        x = mdates.date2num(ot)
        df = pd.DataFrame(x)
        xr = pd.concat([df] * (NZ), axis=1, ignore_index=True)
        
        # initialize plot 
        plt.close('all')
        fs=12
        plt.rc('font', size=fs)
        fig = plt.figure(figsize=(16,8))

        ax1 = plt.subplot2grid((4,1),(0,0),colspan=1,rowspan=1)
        ax2 = plt.subplot2grid((4,1),(1,0),colspan=1,rowspan=1)
        ax3 = plt.subplot2grid((4,1),(2,0),colspan=1,rowspan=1)
        ax4 = plt.subplot2grid((4,1),(3,0),colspan=1,rowspan=1)
    
        
        # plots
        axnum = 0 
        iRi = plt.gcf().axes[axnum].pcolormesh(xr,z,Kdudz,cmap=cmc.roma) #, vmin = -1, vmax = 1)
        plt.gcf().axes[axnum].set_title(sn+': 2017')
        plt.gcf().axes[axnum].set_ylabel('depth m')
        
        plt.gcf().axes[axnum].set_xlim([mdates1[0],mdates1[-1]])
        plt.gcf().axes[axnum].set_xticks(mdates2)  
        date_form = mdates.DateFormatter('%D')
        plt.gcf().axes[axnum].xaxis.set_major_formatter(date_form)
        plt.gcf().axes[axnum].set_xticklabels([])
        
        fig.colorbar(iRi,shrink=0.75,ticks=[-1, -0.5, 0, 0.5, 1],label='Kv*dudz')
        
        axnum = 1 
        YY = np.log10(abs(N2)) #*60*60
        iN2 = plt.gcf().axes[axnum].pcolormesh(xr,z,YY,cmap=cmc.roma) #, vmin = -10, vmax = -1)  
        plt.gcf().axes[axnum].set_ylabel('depth m')
        
        fig.colorbar(iN2,shrink=0.75,label='log10(N2) s^-1')
        
        plt.gcf().axes[axnum].set_xlim([mdates1[0],mdates1[-1]])
        plt.gcf().axes[axnum].set_xticks(mdates2)  
        date_form = mdates.DateFormatter('%D')
        plt.gcf().axes[axnum].xaxis.set_major_formatter(date_form)
        plt.gcf().axes[axnum].set_xticklabels([])
        
        axnum = 2 
        idtdz = plt.gcf().axes[axnum].pcolormesh(xr,z,dtdz,cmap=cmc.roma) #, vmin = -4, vmax = 1)
        plt.gcf().axes[axnum].set_ylabel('depth m')
        
        fig.colorbar(idtdz,shrink=0.75,label='dtdz')
        
        plt.gcf().axes[axnum].set_xlim([mdates1[0],mdates1[-1]])
        plt.gcf().axes[axnum].set_xticks(mdates2)  
        date_form = mdates.DateFormatter('%D')
        plt.gcf().axes[axnum].set_xticklabels([])
        
        axnum = 3 
        idsdz = plt.gcf().axes[axnum].pcolormesh(xr,z,dsdz,cmap=cmc.roma) #, vmin = -4, vmax = 1)
        plt.gcf().axes[axnum].set_ylabel('depth m')
        
        fig.colorbar(idsdz,shrink=0.75,label='dSdz')
                
        plt.gcf().axes[axnum].set_xlim([mdates1[0],mdates1[-1]])
        plt.gcf().axes[axnum].set_xticks(mdates2)  
        date_form = mdates.DateFormatter('%D')
        plt.gcf().axes[axnum].xaxis.set_major_formatter(date_form)
        
        fig.autofmt_xdate()
        
        #axnum = 2 
        #iRig_N = plt.gcf().axes[axnum].pcolormesh(xr,z,Rig_N,cmap=cmc.roma, vmin = -1, vmax = 1) 
        
        #axnum = 3 
        #iKdudz = plt.gcf().axes[axnum].pcolormesh(xr,z,Kdudz,cmap=cmc.roma, vmin = -0.0001, vmax = 0.0001)
        
    
    

