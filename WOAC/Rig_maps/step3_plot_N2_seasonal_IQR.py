'''
Plot seasonal N2 p50 (+IQR) calculated 
using step2_calc_N2_seasonal.py

run step3_plot_N2_seasonal_B -gtx cas7_t0_x4b -y0 2013 -y1 2023
'''

# imports
import os 
import sys
import argparse
import xarray as xr
import numpy as np
import pickle 

from lo_tools import plotting_functions as pfun
from lo_tools import Lfun
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from time import time

import matplotlib.pyplot as plt
#import matplotlib.cm as cm
import cmcrameri.cm as cm


parser = argparse.ArgumentParser()
# these flags get only surface or bottom fields if True
# - cannot have both True - It plots one or the other to avoid a 2x loop 
parser.add_argument('-gtx', '--gtagex', type=str)   # e.g. cas7_t0_x4b
# select years 
parser.add_argument('-y0', '--ys0', type=str) # e.g. 2014
parser.add_argument('-y1', '--ys1', type=str) # e.g. 2015
# get the args and put into Ldir
args = parser.parse_args()

Ldir = Lfun.Lstart()

# input paths and open pickled data 
fn_i = Ldir['parent'] / 'LKH_output' / 'WOAC' / args.gtagex / 'Rig_maps' / 'N2_seasonal_pickled_values'
pw = 'OA_indicators_N2_winter_'+str(args.ys0)+'_'+str(args.ys1)+'.pkl'
psp = 'OA_indicators_N2_spring_'+str(args.ys0)+'_'+str(args.ys1)+'.pkl'
psu = 'OA_indicators_N2_summer_'+str(args.ys0)+'_'+str(args.ys1)+'.pkl'
pf = 'OA_indicators_N2_fall_'+str(args.ys0)+'_'+str(args.ys1)+'.pkl'

# WINTER!!!!! 
picklepath = fn_i/pw
with open(picklepath, 'rb') as fp:
    W = pickle.load(fp)
    print('loaded file')

# Spring!!!!! 
picklepath = fn_i/psp
with open(picklepath, 'rb') as fp1:
    SP = pickle.load(fp1)
    print('loaded file')

# Summer!!!!! 
picklepath = fn_i/psu
with open(picklepath, 'rb') as fp2:
    SU = pickle.load(fp2)
    print('loaded file')

# Fall!!!!! 
picklepath = fn_i/pf
with open(picklepath, 'rb') as fp3:
    F = pickle.load(fp3)
    print('loaded file')

# output paths + final figure name
fn_o = Ldir['parent'] / 'LKH_output' / 'WOAC' / 'cas7_t0_x4b' / 'Rig_maps' / 'N2_seasonal_maps' 
figname = 'N2_seasonal_IQR_'+str(args.ys0)+'_'+str(args.ys1)+'_OA_indicators.png'
if os.path.exists(fn_o)==False:
    Lfun.make_dir(fn_o, clean=False)
    
# load the shelf mask / assign vars 
fn_mask =  Ldir['parent'] / 'LKH_data' / 'shelf_masks' / 'OA_indicators' / 'OA_indicators_shelf_mask_15_200m_noEstuaries.nc'
dmask = xr.open_dataset(fn_mask) 
mask_shelf = dmask.mask_shelf 
h = dmask.h
xrho = dmask.lon_rho
yrho = dmask.lat_rho

# PLOTTING
# map
plt.close('all')
fs=14
plt.rc('font', size=fs)
fig = plt.figure(figsize=(18,10))
fig.set_size_inches(18,10, forward=False)

xmin = -125.5 
xmax = -123
xxticks = [-125.5, -124.5, -123.5]
xxticklabels = ['-125.5', '-124.5', '-123.5']

# WINTER!!!!! 
axw = plt.subplot2grid((3,5), (0,0), colspan=1,rowspan=3)
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

axw.set_title('Jan - Mar')
axw.set_xlabel('Longitude')
axw.set_ylabel('Latitude')
axw.set_xlim([xmin, xmax])
axw.set_ylim([42.75,48.75])
axw.set_xticks(xxticks)
axw.set_yticks([42.75,43,44,45,46,47,48,48.75])
axw.set_yticklabels(['42.75','43','44','45','46','47','48','48.75'])
axw.xaxis.set_ticklabels(xxticklabels)
axw.grid(False)

# SPRING!!!!! 
axs = plt.subplot2grid((3,5), (0,1), colspan=1,rowspan=3)
pfun.add_coast(axs)
pfun.dar(axs)
axs.contour(xrho,yrho,h, [40],
colors=['white'], linewidths=1, linestyles='solid',alpha=0.4)
axs.contour(xrho,yrho,h, [80],
colors=['black'], linewidths=1, linestyles='solid',alpha=0.4)
axs.contour(xrho,yrho,h, [200],
colors=['black'], linewidths=1, linestyles='solid',alpha=0.6)
axs.contour(xrho,yrho,h, [1000],
colors=['black'], linewidths=1, linestyles='solid')

axs.set_title('Apr - Jun')
axs.set_xlabel('Longitude')
axs.set_xlim([xmin, xmax])
axs.set_ylim([42.75,48.75])
axs.set_xticks(xxticks)
axs.set_yticks([42.75,43,44,45,46,47,48,48.75])
axs.set_yticklabels(['42.75','43','44','45','46','47','48','48.75'])
axs.xaxis.set_ticklabels(xxticklabels)
axs.grid(False)

# Summer!!!!! 
axsu = plt.subplot2grid((3,5), (0,2), colspan=1,rowspan=3)
pfun.add_coast(axsu)
pfun.dar(axsu)
axsu.contour(xrho,yrho,h, [40],
colors=['white'], linewidths=1, linestyles='solid',alpha=0.4)
axsu.contour(xrho,yrho,h, [80],
colors=['black'], linewidths=1, linestyles='solid',alpha=0.4)
axsu.contour(xrho,yrho,h, [200],
colors=['black'], linewidths=1, linestyles='solid',alpha=0.6)
axsu.contour(xrho,yrho,h, [1000],
colors=['black'], linewidths=1, linestyles='solid')

axsu.set_title('Jul - Sep')
axsu.set_xlabel('Longitude')
axsu.set_xlim([xmin, xmax])
axsu.set_ylim([42.75,48.75])
axsu.set_xticks(xxticks)
axsu.set_yticks([42.75,43,44,45,46,47,48,48.75])
axsu.set_yticklabels(['42.75','43','44','45','46','47','48','48.75'])
axsu.xaxis.set_ticklabels(xxticklabels)
axsu.grid(False)

# Fall!!!!! 
axf = plt.subplot2grid((3,5), (0,3), colspan=1,rowspan=3)
pfun.add_coast(axf)
pfun.dar(axf)
axf.contour(xrho,yrho,h, [40],
colors=['white'], linewidths=1, linestyles='solid',alpha=0.4)
axf.contour(xrho,yrho,h, [80],
colors=['black'], linewidths=1, linestyles='solid',alpha=0.4)
axf.contour(xrho,yrho,h, [200],
colors=['black'], linewidths=1, linestyles='solid',alpha=0.6)
axf.contour(xrho,yrho,h, [1000],
colors=['black'], linewidths=1, linestyles='solid')

axf.set_title('Oct - Dec')
axf.set_xlabel('Longitude')
axf.set_xlim([xmin, xmax])
axf.set_ylim([42.75,48.75])
axf.set_xticks(xxticks)
axf.set_yticks([42.75,43,44,45,46,47,48,48.75])
axf.set_yticklabels(['42.75','43','44','45','46','47','48','48.75'])
axf.xaxis.set_ticklabels(xxticklabels)
axf.grid(False)

# plot data 
IQRw = W['p75'] - W['p25']
IQRsp = SP['p75'] - SP['p25']
IQRsu = SU['p75'] - SU['p25']
IQRf = F['p75'] - F['p25']

cpw = axw.pcolormesh(W['lon_rho'], W['lat_rho'], np.log10(IQRw),vmin=-5,vmax=-3,cmap=cm.roma_r)
cbaxes = inset_axes(axw, width="5%", height="40%", loc='lower right')
fig.colorbar(cpw, cax=cbaxes, location='left', orientation='vertical',label = 'log10 IQRN2')

cpsp = axs.pcolormesh(SP['lon_rho'], SP['lat_rho'], np.log10(IQRsp),vmin=-5,vmax=-3,cmap=cm.roma_r)
cbaxes = inset_axes(axs, width="5%", height="40%", loc='lower right')
fig.colorbar(cpsp, cax=cbaxes, location='left', orientation='vertical',label = 'log10 IQRN2')

cpsu = axsu.pcolormesh(SU['lon_rho'], SU['lat_rho'], np.log10(IQRsu),vmin=-5,vmax=-3,cmap=cm.roma_r)
cbaxes = inset_axes(axsu, width="5%", height="40%", loc='lower right')
fig.colorbar(cpsu, cax=cbaxes, location='left', orientation='vertical',label = 'log10 IQRN2')

cpf = axf.pcolormesh(F['lon_rho'], F['lat_rho'], np.log10(IQRf),vmin=-5,vmax=-3,cmap=cm.roma_r)
cbaxes = inset_axes(axf, width="5%", height="40%", loc='lower right')
fig.colorbar(cpf, cax=cbaxes, location='left', orientation='vertical',label = 'log10 IQRN2')

fig.tight_layout()

# fix details
axw.tick_params(axis = "x", labelsize = 14, labelrotation = 0)

axw.set_title('January - March')
axs.set_title('April - June')
axsu.set_title('July - September')
axf.set_title('October - December')

#axf.text(0.82, 0.85,'contours',color='black',weight='normal',fontstyle='italic',transform=axf.transAxes,ha='right')
axf.text(0.82, 0.83,'40 m',color='darkgrey',weight='light',fontstyle='italic',transform=axf.transAxes,ha='right')
axf.text(0.82, 0.81,'80 m',color='grey',weight='light',fontstyle='italic',transform=axf.transAxes,ha='right')
axf.text(0.82, 0.79,'200 m',color='black',weight='light',fontstyle='italic',transform=axf.transAxes,ha='right')
axf.text(0.82, 0.77,'1000 m',color='black',weight='light',fontstyle='italic',transform=axf.transAxes,ha='right')

axsu.yaxis.set_label_position("right")
axsu.yaxis.tick_right()
axsu.set_ylabel(' ')

axs.set_yticklabels([])
axsu.set_yticklabels([])
axf.set_yticklabels([])

fig.savefig(fn_o / figname)
