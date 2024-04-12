# This plots OCNMS "midshelf" moorings from LO mooring extraction 
# cas7_t0_x4b. Want to plot mixing and then DO as a line on top 
# this is to show parker an idea, so the locations are random but 
# something that I had already extracted

# imports
from lo_tools import Lfun, zfun, zrfun
from lo_tools import plotting_functions as pfun

#import sys 
import xarray as xr
import numpy as np
#import netCDF4 as nc
import os 

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
        AKs = ds2['AKs']
        AKv = ds2['AKv']
        z_w = ds2['z_w'] 
        z_rho = ds2['z_rho'] 
        ot = ds2['ocean_time'].values[:]

        # Convert dates to numbers
        NW = np.shape(z_w)[1]
        NRho = np.shape(SP)[1]
        x = mdates.date2num(ot)
        df = pd.DataFrame(x)
        xw = pd.concat([df] * (NW), axis=1, ignore_index=True)
        xr = pd.concat([df] * (NRho), axis=1, ignore_index=True)
        
        yw = z_w.values
        yr = z_rho.values
        Cmix = np.log(AKs.values)
        Csalt = SP.values
        Ctemp = PT.values
        Coxy = oxygen.values *(22.381/1000) # 2x check this

        # Create salt plot
        axnum = 0
        isalt = plt.gcf().axes[axnum].pcolormesh(xr,yr,Csalt,cmap=cmc.lapaz, vmin = 24, vmax = 34)
        plt.gcf().axes[axnum].set_title(sn+': 2017')
        
        fig.colorbar(isalt,shrink=0.75,ticks=[24,28,32,34],label='salinity P')
        plt.gcf().axes[axnum].set_ylabel('depth m')
        
        plt.gcf().axes[axnum].set_xlim([mdates1[0],mdates1[-1]])
        plt.gcf().axes[axnum].set_xticks(mdates2)  
        date_form = mdates.DateFormatter('%D')
        plt.gcf().axes[axnum].xaxis.set_major_formatter(date_form)
        plt.gcf().axes[axnum].set_xticklabels([])

        # Create temp plot
        axnum = 1
        itemp = plt.gcf().axes[axnum].pcolormesh(xr,yr,Ctemp,cmap=cmc.davos, vmin = 6, vmax = 16)
        
        fig.colorbar(itemp,shrink=0.75,ticks=[6,8,10,12,14,16],label='temp deg.C')
        plt.gcf().axes[axnum].set_ylabel('depth m')
        
        plt.gcf().axes[axnum].set_xlim([mdates1[0],mdates1[-1]])
        plt.gcf().axes[axnum].set_xticks(mdates2)  
        date_form = mdates.DateFormatter('%D')
        plt.gcf().axes[axnum].xaxis.set_major_formatter(date_form)
        plt.gcf().axes[axnum].set_xticklabels([])

        # Create oxygen plot
        axnum = 2
        ioxygen = plt.gcf().axes[axnum].pcolormesh(xr,yr,Coxy,cmap=cmc.hawaii, vmin = 0, vmax = 10)
        
        fig.colorbar(ioxygen,shrink=0.75,ticks=[0,2,4,6,8,10],label='(DO ml/L)')
        plt.gcf().axes[axnum].set_ylabel('depth m')
        
        plt.gcf().axes[axnum].set_xlim([mdates1[0],mdates1[-1]])
        plt.gcf().axes[axnum].set_xticks(mdates2)  
        date_form = mdates.DateFormatter('%D')
        plt.gcf().axes[axnum].xaxis.set_major_formatter(date_form)
        plt.gcf().axes[axnum].set_xticklabels([])
                        
        # Ks plot
        axnum = 3
        imix = plt.gcf().axes[axnum].pcolormesh(xw,yw,Cmix,cmap=cmc.batlowK, vmin = -12, vmax = -1)
        fig.colorbar(imix,shrink=0.75,ticks=[-12, -6, -3, -1],label='log Ks m2/s')
        plt.gcf().axes[axnum].set_ylabel('depth m')
        
        
        plt.gcf().axes[axnum].set_xlim([mdates1[0],mdates1[-1]])
        plt.gcf().axes[axnum].set_xticks(mdates2)  
        date_form = mdates.DateFormatter('%D')
        plt.gcf().axes[axnum].xaxis.set_major_formatter(date_form)
        
        
        
        # Rotates the labels to fit: fig.autofmt_xdate()

        
        
        #plt.gcf().plot(ot,LOoxy,color = 'crimson',marker='none',linestyle='-',linewidth=2,alpha=0.7,label ='cas7_t0_x4b')
        

        #plt.gcf().tight_layout()
        

# have to do backwards or else if remove 8 first then 9 becomes 8 and so on        
#plt.gcf().axes[9].remove()   # no obs data at CA042 in 2016 / 2017
#plt.gcf().axes[8].remove() 

#plt.gcf().axes[7].legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')

#fig_nm = '/Users/katehewett/Documents/LKH_output/draft_ocnms_cas7_t0_x4b.png'
#plt.gcf().savefig(fig_nm)


    
    

    
    

