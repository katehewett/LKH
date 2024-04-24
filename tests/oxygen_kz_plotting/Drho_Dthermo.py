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
    'CE042':0
}

sn_list = list(sn_name_dict.keys())
numsites = len(sn_list)

# update with input tags for start and stop 
# make some datetimes for plotting 
mdates1 = pd.date_range(start='2017-01-01',end='2018-01-01', freq='MS')
mdates2 = pd.date_range(start='2017-01-01',end='2018-01-01', freq='16D')

for sn in sn_list:
    print(sn)

    #add model data 
    in_fn = model_dir / (sn + '_' + str(yr_list[0]) + '.01.01_'+ str(yr_list[0]) + '.12.31.nc')
    if os.path.isfile(in_fn):
        ds2 = xr.open_dataset(in_fn)
        oxygen = ds2['oxygen']
        SP = ds2['salt']
        PT = ds2['temp']
        u = ds2['u'].values
        ubar = ds2['ubar'].values
        vbar = ds2['vbar'].values
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
        drho = np.diff(SIG0,axis=1)
        dpdz = drho/dz
        dpdz_est = (SIG0[:,1]-SIG0[:,20])/(z_rho.values[:,1]-z_rho.values[:,20])
        N2 = -(g/po)*dpdz
        N2_est = -(g/po)*dpdz_est
        
        # for plotting N2 S2 dudz's 
        z = z_w.values[:,1:-1];  # chop the surface most value 
        zr = z_rho.values 
        
        NZ = np.shape(z)[1]
        NR = np.shape(zr)[1]
        x = mdates.date2num(ot)
        df = pd.DataFrame(x)
        xz = pd.concat([df] * (NZ), axis=1, ignore_index=True)
        xr = pd.concat([df] * (NR), axis=1, ignore_index=True)
        
        # initialize plot 
        plt.close('all')
        fs=12
        plt.rc('font', size=fs)
        fig = plt.figure(figsize=(16,8))
        

        ax1 = plt.subplot2grid((4,1),(0,0),colspan=1,rowspan=1)
        ax2 = plt.subplot2grid((4,1),(1,0),colspan=1,rowspan=1)
        ax3 = plt.subplot2grid((4,1),(2,0),colspan=1,rowspan=1)
        ax4 = plt.subplot2grid((4,1),(3,0),colspan=1,rowspan=1)
		
		# dpdz plot 
        axnum = 0 
        iRHO = plt.gcf().axes[axnum].pcolormesh(xr,zr,SIG0,cmap=cmc.roma_r, vmin = 16, vmax = 28)
        plt.gcf().axes[axnum].set_title(sn+': 2017')

        axnum = 1 
        iRHO = plt.gcf().axes[axnum].pcolormesh(xr,zr,CT,cmap=cmc.roma_r, vmin = 6, vmax = 18)
        
        axnum = 2 
        iRHO = plt.gcf().axes[axnum].pcolormesh(xr,zr,SA,cmap=cmc.roma_r, vmin = 20, vmax = 35)
        
        axnum = 3 
        iRHO = plt.gcf().axes[axnum].plot(x,ubar) #, vmin = 20, vmax = 35)
        iRHO = plt.gcf().axes[axnum].plot(x,vbar) #, vmin = 20, vmax = 35)
        
        #LN2 = np.log10(N2_est)
        #botO2 = oxygen[:,1]*(22.381/1000) # 2x check this
        #surfSA = SA[:,20]
        
        #df2 = pd.DataFrame({'DO':botO2,'N2e':LN2,'SS':surfSA,'year':pd.DatetimeIndex(ot).year,'month':pd.DatetimeIndex(ot).month})
        #mask_dict = {}
        #mask_dict = (df2.month>4) | (df2.month<9)
        
        #plt.scatter(df2.DO,df2.N2e,s=100,c=df2.SS,cmap = cmc.roma_r)

        
        



    

