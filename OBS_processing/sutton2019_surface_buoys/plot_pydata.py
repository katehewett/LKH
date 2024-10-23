# This plots surface data for Chaba buoy, Cape Elizabeth buoy and Cape Arago buoy
# that was extracted by running process_webdata.py, which placed files here:
# /LKH_data/sutton2019_surface_buoys/py_files/

testing = True 

# imports
from lo_tools import Lfun, zfun, zrfun
from lo_tools import plotting_functions as pfun

#import sys 
import xarray as xr
import numpy as np
#import netCDF4 as nc
import os 
import sys
import posixpath

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
#import matplotlib.dates as mdates
from datetime import datetime
from datetime import time
import pandas as pd

Ldir = Lfun.Lstart()

# processed data location
source = 'ocnms'
otype = 'moor' 
in_dir = Ldir['parent'] / 'LKH_data' / 'sutton2019_surface_buoys' / 'py_files'
out_dir = Ldir['parent'] / 'LKH_output' / 'sutton2019_surface_buoys' / 'plots'

if os.path.exists(out_dir)==False:
    Lfun.make_dir(out_dir, clean = False)
if os.path.exists(in_dir)==False:
    print('input path does not exist')
    sys.exit()
       
#model_dir = Ldir['LOo'] / 'extract/cas7_t0_x4b/moor/OCNMS_moorings_current' # update with tags

sn_name_dict = {
    'CHABA':'Chaba',
    'CAPEELIZABETH':'Cape Elizabeth',
    'CAPEARAGO':'Cape Arago'
}

if testing:
    sn_list = ['CAPEELIZABETH']
else:    
    sn_list = list(sn_name_dict.keys())
    
#numyrs = 5 # 2013 - 2017
sn_list = list(sn_name_dict.keys())
numsites = len(sn_list)

# update with input tags for start and stop 
# make some datetimes for plotting 
#mdates1 = pd.date_range(start='2013-05-01',end='2017-05-01', freq='AS-MAY')
#mdates2 = pd.date_range(start='2013-11-01',end='2017-11-01', freq='AS-NOV')

# initialize plot 
plt.close('all')
fs=12
plt.rc('font', size=fs)
fig = plt.figure(figsize=(16,8))

for sn in sn_list:
    fn_in = posixpath.join(in_dir, (sn + '.nc'))
    fn_out = posixpath.join(out_dir, (sn + '.png'))
    ds = xr.open_dataset(fn_in, decode_times=True) 

    ax0 = plt.subplot2grid((8,1),(0,0),colspan=1,rowspan=1)
    plt.plot(ds.time,ds.SA,color = 'dodgerblue',marker='.',linestyle='none',linewidth=3,label ='SA')
    ax0.set_xticklabels([])
    ax0.set_ylim([25,35])
    ax0.set_yticks(np.arange(25,36,5), minor=False) 
    
    ax1 = plt.subplot2grid((8,1),(1,0),colspan=1,rowspan=1)
    plt.plot(ds.time,ds.CT,color = 'dodgerblue',marker='.',linestyle='none',linewidth=3,label ='CT')
    ax1.set_xticklabels([])
    ax1.set_ylim([7,21])
    ax1.set_yticks(np.arange(7,22,7), minor=False)
    
    ax2 = plt.subplot2grid((8,1),(2,0),colspan=1,rowspan=1)
    plt.plot(ds.time,ds.pH,color = 'dodgerblue',marker='.',linestyle='none',linewidth=3,label ='pH')
    ax2.set_xticklabels([])
    ax2.set_ylim([7.6,8.6])
    ax2.set_yticks(np.arange(7.6,8.7,.2), minor=False)
    
    ax3 = plt.subplot2grid((8,1),(3,0),colspan=1,rowspan=1)
    plt.plot(ds.time,ds['DO (uM)'],color = 'dodgerblue',marker='.',linestyle='none',linewidth=3,label ='DO uM')
    ax3.set_xticklabels([])
    ax3.set_ylim([100,520])
    ax3.set_yticks(np.arange(100,521,200), minor=False)
    
    ax4 = plt.subplot2grid((8,1),(4,0),colspan=1,rowspan=1)
    plt.plot(ds.time,ds.pCO2_air,color = 'black',marker='.',linestyle='none',linewidth=3,label ='pCO2_air')
    plt.plot(ds.time,ds.xCO2_air,color = 'red',marker='.',linestyle='none',linewidth=3,label ='xCO2_air')
    ax4.set_xticklabels([])
    ax4.set_ylim([360,480])
    ax4.set_yticks(np.arange(360,481,100), minor=False)
    
    ax5 = plt.subplot2grid((8,1),(5,0),colspan=1,rowspan=1)
    plt.plot(ds.time,ds.pCO2_sw,color = 'dodgerblue',marker='.',linestyle='none',linewidth=3,label ='pCO2_sw')
    ax5.set_xticklabels([])
    ax5.set_ylim([70,1090])
    
    ax6 = plt.subplot2grid((8,1),(6,0),colspan=1,rowspan=1)
    plt.plot(ds.time,ds.CHL,color = 'dodgerblue',marker='.',linestyle='none',linewidth=3,label ='CHL')
    ax6.set_xticklabels([])
    ax6.set_ylim([0,15])
    
    ax7 = plt.subplot2grid((8,1),(7,0),colspan=1,rowspan=1)
    plt.plot(ds.time,ds.Turbidity,color = 'dodgerblue',marker='.',linestyle='none',linewidth=3,label ='Turbidity')
    ax7.set_ylim([0,5])
    
    plt.gcf().tight_layout()
    
