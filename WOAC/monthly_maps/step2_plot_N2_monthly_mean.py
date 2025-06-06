'''
Plot monthly means of percent water column corrosive 
w/ diff thresholds 
ARAG < = 1 (0.5; 1.7) 

example call:
run step2_plot_N2_monthly_mean -gtx cas7_t0_x4b -y0 2020 -job OA_indicators
'''

# imports
import os 
import sys
import argparse
from lo_tools import plotting_functions as pfun
from lo_tools import Lfun
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import xarray as xr
from time import time
import numpy as np

import matplotlib.pyplot as plt
#import matplotlib.cm as cm
import cmcrameri.cm as cm
import pickle 

# command line arugments
parser = argparse.ArgumentParser()
# which run was used:
parser.add_argument('-gtx', '--gtagex', type=str)   # e.g. cas7_t0_x4b
#parser.add_argument('-ro', '--roms_out_num', type=int) # 2 = Ldir['roms_out2'], etc.
# select years 
parser.add_argument('-y0', '--ys0', type=str) # e.g. 2014
parser.add_argument('-job', '--job_type', type=str) # job 
# Optional: set max number of subprocesses to run at any time
parser.add_argument('-Nproc', type=int, default=10)
# Optional: for testing
parser.add_argument('-test', '--testing', default=False, type=Lfun.boolean_string)
# get the args and put into Ldir
args = parser.parse_args()
# test that main required arguments were provided
argsd = args.__dict__
for a in ['gtagex']:
    if argsd[a] == None:
        print('*** Missing required argument: ' + a)
        sys.exit()
gridname, tag, ex_name = args.gtagex.split('_')
# get the dict Ldir
Ldir = Lfun.Lstart(gridname=gridname, tag=tag, ex_name=ex_name)
# add more entries to Ldir
for a in argsd.keys():
    if a not in Ldir.keys():
        Ldir[a] = argsd[a]
        
Ldir = Lfun.Lstart()

# setup in and out dirs w/ filenames
in_dir = Ldir['parent'] / 'LKH_output' / 'WOAC' / args.gtagex / 'Rig_maps' / 'N2_monthly_means'
pn = 'shelf_monthly_mean_N2_' + args.ys0 + '.pkl'
picklepath = in_dir / pn

out_dir = Ldir['parent'] / 'LKH_output' / 'WOAC' / args.gtagex / 'maps_monthly_mean' / 'N2' / 'plots' 
fn = 'shelf_monthly_mean_N2_' + args.ys0 + '.png'
fn_o = out_dir/fn

if os.path.exists(out_dir)==False:
    Lfun.make_dir(out_dir, clean = False)

if os.path.isfile(picklepath)==False:
    print('no file named: '+fn_in)
    sys.exit()
else:   
    with open(picklepath, 'rb') as fp3:
        N2mean = pickle.load(fp3)
        print('loaded file')

# load the shelf mask / assign vars 
fn_mask =  Ldir['parent'] / 'LKH_data' / 'shelf_masks' / 'OA_indicators' / 'OA_indicators_shelf_mask_15_200m_noEstuaries.nc'
dmask = xr.open_dataset(fn_mask) 
mask_shelf = dmask.mask_shelf.values
mask_rho = dmask.mask_rho.values
h = dmask.h
xrho = dmask.lon_rho
yrho = dmask.lat_rho

# PLOTTING    
plt.close('all')
fs=12
plt.rc('font', size=fs)
fig = plt.figure(figsize=(18,10))
fig.set_size_inches(18,10, forward=False)

xmin = -125.5 
xmax = -123.5

month_list = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep', 'Oct','Nov','Dec']

for idx in range(0,12):
    axw = plt.subplot2grid((2,14), (0,idx), colspan=2,rowspan=1)
    pfun.add_coast(axw)
    pfun.dar(axw)
    axw.contour(xrho,yrho,h, [40],
    colors=['white'], linewidths=1, linestyles='solid',alpha=0.4)
    axw.contour(xrho,yrho,h, [80],
    colors=['black'], linewidths=1, linestyles='solid',alpha=0.4)
    axw.contour(xrho,yrho,h, [200],
    colors=['black'], linewidths=1, linestyles='solid',alpha=0.6)
    axw.contour(xrho,yrho,h, [1000],
    colors=['black'], linewidths=1, linestyles='solid')
    axw.set_title(month_list[idx])
    axw.set_xlim([xmin, xmax])
    axw.set_ylim([42.75,48.75])
    axw.set_yticks([42.75,43,44,45,46,47,48,48.75])
    axw.set_yticklabels(['42.75','43','44','45','46','47','48','48.75'])

    if idx>0 & idx<11: 
        axw.set_yticklabels([])
    
    if (idx == 0) | (idx == 11): 
        axw.set_title(month_list[idx] + ' ' + str(args.ys0))
 
    '''
    if idx == 11:
        axw.yaxis.set_label_position("right")
        axw.yaxis.tick_right()
        axw.set_yticklabels(['42.75','43','44','45','46','47','48','48.75'])
    '''
    if (idx % 2 == 0): 
        xxticks = [-125.5, -124.5, -123.5]
        xxticklabels = ['-125.5', '-124.5','-123.5']
        axw.set_xticks(xxticks)
        axw.xaxis.set_ticklabels(xxticklabels,rotation=270, ha='center')
    else:  
        xxticks = [-125.5, -124.5, -123.5]
        xxticklabels = [' ', ' ', ' ']
        axw.set_xticks(xxticks)
        axw.xaxis.set_ticklabels(xxticklabels,rotation=270, ha='center')

    N2 = N2mean[idx+1]
    N2[mask_shelf==0] = np.nan 
    N2[mask_rho==0] = np.nan

    cw = axw.pcolormesh(N2mean['lon_rho'], N2mean['lat_rho'], np.log10(N2),vmin=-5,vmax=-2,cmap=cm.roma_r)

    if idx == 11:
        fig.colorbar(cw, ax=axw, location='right', orientation='vertical',label = 'log10 N2')
    
    axw.grid(False)

fig.tight_layout()
fig.savefig(fn_o)

