"""
Plots cumulative shelf volume for each region 
at Oag<1 and <0.5 

Sum the cumulative percent volume / region
grab the last value and put in a stacked bar graph

example calls:  run calc_cumulative_pvolume_byRegion -mvar ARAG1
                run calc_cumulative_pvolume_byRegion -mvar ARAG05
"""
# imports
from lo_tools import Lfun, zfun
from lo_tools import plotting_functions as pfun

import os 
import sys 
import argparse
import xarray as xr
import netCDF4 as nc
from time import time
import numpy as np
import pandas as pd
import pickle 
import matplotlib.pyplot as plt
import matplotlib.cm as cm
#import cmocean
import matplotlib.dates as mdates
from datetime import datetime
import seaborn as sns

# add a var type so can call arg threshold in command line
parser = argparse.ArgumentParser()
parser.add_argument('-mvar', '--variable', type=str) # select variable 
args = parser.parse_args()

Ldir = Lfun.Lstart()

# load the shelf mask / assign vars 
fn_mask =  Ldir['parent'] / 'LKH_data' / 'shelf_masks' / 'OA_indicators' / 'OA_indicators_shelf_mask_15_200m_noEstuaries.nc'
dmask = xr.open_dataset(fn_mask) 
mask_shelf = dmask.mask_shelf 
h = dmask.h
xrho = dmask.lon_rho
yrho = dmask.lat_rho

# input location for pickled files 
fn_i = Ldir['parent'] / 'LKH_output' / 'OA_indicators' / 'cas7_t0_x4b' / 'volumes_by_threshold' / 'byregion' / args.variable

#plotting details
mask_dict = {}
mask_dict[48] = (yrho >=47.75) & (yrho < 48.75) & (mask_shelf == 1)
mask_dict[47] = (yrho >=46.75) & (yrho < 47.75) & (mask_shelf == 1)
mask_dict[46] = (yrho >=45.75) & (yrho < 46.75) & (mask_shelf == 1)
mask_dict[45] = (yrho >=44.75) & (yrho < 45.75) & (mask_shelf == 1)
mask_dict[44] = (yrho >=43.75) & (yrho < 44.75) & (mask_shelf == 1)
mask_dict[43] = (yrho >42.75) & (yrho < 43.75) & (mask_shelf == 1)
NMASK = len(mask_dict)

yr_list = [year for year in range(2013,2024)]
numyrs = len(yr_list)

lat_list = [48, 47, 46, 45, 44, 43]
# Rcolors = ['#73210D','#9C6B2F','#C5B563','#A9A9A9','#77BED0','#4078B3','#112F92'] #roma w grey not green #idx 3
Rcolors = ['#73210D','#9C6B2F','#C5B563','#77BED0','#4078B3','#112F92'] #no grey 

# Lcolors = sns.color_palette() #for lines 2014 - 2023 

plt.close('all')
fs=11
plt.rc('font', size=fs)
height_of_image = 15 
width_of_image = 8 
fig1 = plt.figure(figsize=(height_of_image,width_of_image))
fig1.set_size_inches(height_of_image,width_of_image, forward=False)
frac=0.047 * (height_of_image / (width_of_image/13))

# map
ax = plt.subplot2grid((2,3), (0,0), colspan=1, rowspan=2)
#ax = plt.subplot2grid((2,10), (0,0), colspan=1,rowspan=10)
pfun.add_coast(ax)
pfun.dar(ax)
ax.axis([-125.5, -123.5, 42.75, 48.75])

ax.contour(xrho,yrho,h, [40],
colors=['white'], linewidths=1, linestyles='solid',alpha=0.4)
ax.contour(xrho,yrho,h, [80],
colors=['black'], linewidths=1, linestyles='solid',alpha=0.4)
ax.contour(xrho,yrho,h, [200],
colors=['black'], linewidths=1, linestyles='solid',alpha=0.6)
ax.contour(xrho,yrho,h, [1000],
colors=['black'], linewidths=1, linestyles='solid')

ax.text(0.95,.10,'40 m',color='lightgrey',weight='bold',transform=ax.transAxes,ha='right')
ax.text(0.95,.07,'80 m',color='black',weight='bold',transform=ax.transAxes,ha='right')
ax.text(0.95,.04,'*200 m',color='black',weight='bold',fontstyle = 'italic',transform=ax.transAxes,ha='right')
ax.text(0.95,.01,'1000 m',color='black',weight='bold',transform=ax.transAxes,ha='right')
      
ax.set_title('Area of Calculation')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_xticks([-125.5, -124.5, -123.5])
ax.set_yticks([42.75,43,44,45,46,47,48,48.75])
ax.set_yticklabels(['42.75','43','44','45','46','47','48','48.75'])
ax.xaxis.set_ticklabels([-125.5, -124.5 , -123.5])
ax.xaxis.set_ticklabels([-125.5, -124.5 , -123.5])
ax.grid(False)

# shade regions
ii = 0
for mm in lat_list:
    pts2 = ax.scatter(xrho*mask_dict[mm], yrho*mask_dict[mm], s=2, c = Rcolors[ii])
    ii = ii+1

fig1.tight_layout()

#initialize for entering the 365th day cumsum value + year 
R1 = np.full(numyrs, np.nan)
R2 = np.full(numyrs, np.nan)
R3 = np.full(numyrs, np.nan)
R4 = np.full(numyrs, np.nan)
R5 = np.full(numyrs, np.nan)
R6 = np.full(numyrs, np.nan)
RYEARS = np.full(numyrs, np.nan)

for ydx in range(0,numyrs): 
    pn = 'byregion_'+args.variable+'_regional_volumes_'+str(yr_list[ydx])+'.pkl'
    picklepath = fn_i / pn 
        
    if os.path.isfile(picklepath)==False:
        print('no file named: ' + pn)
        sys.exit()
    
    # Load the dictionary from the file
    with open(picklepath, 'rb') as fp:
        vol = pickle.load(fp)
        print('loaded'+str(yr_list[ydx]))
            
    V_corr = vol['Vtotal_corr'] 
    V_shelf = vol['Vtotal_shelf']
    
    P = {}
    region_cumsum = {}
    for mm in lat_list:
        P[mm] = (V_corr[mm]/V_shelf[mm])
        region_cumsum[mm] = np.cumsum(P[mm])
    
    R1[ydx] = region_cumsum[48][364] # always take the 365th so leap years don't have another day of volume 
    R2[ydx] = region_cumsum[47][364]
    R3[ydx] = region_cumsum[46][364]
    R4[ydx] = region_cumsum[45][364]
    R5[ydx] = region_cumsum[44][364]
    R6[ydx] = region_cumsum[43][364]
    RYEARS[ydx] = yr_list[ydx]

ax1 = plt.subplot2grid((2,3), (0,1), colspan=1, rowspan = 1)  # WA

# create data
x = RYEARS
y1 = np.array([10, 20, 10, 30])
y2 = np.array([20, 25, 15, 25])
y3 = np.array([12, 15, 19, 6])
y4 = np.array([10, 29, 13, 19])
 
# plot bars in stack manner
plt.bar(x, R1, color=Rcolors[0])
plt.bar(x, R2, bottom=R1, color=Rcolors[1])
plt.bar(x, R3, bottom=R1+R2, color=Rcolors[2])
#plt.bar(x, y4, bottom=y1+y2+y3, color='g')
sys.exit()

plt.xlabel("cumsum")
plt.ylabel("years")
plt.legend(["Round 1", "Round 2", "Round 3", "Round 4"])
plt.title("Scores by Teams in 4 Rounds")



ax2 = plt.subplot2grid((2,3), (1,1), colspan=1, rowspan = 1)  # OR

''''
    ot = vol['ocean_time']
    yd = np.arange(1,len(ot)+1,1) # lazy yearday

    if args.variable == 'arag1':
        ax1.set_title('$\mathregular{Washington^{+}}$: \u03A9 ag < 1')
        ax2.set_title('Oregon')
    elif args.variable == 'arag05':
        ax1.set_title('$\mathregular{Washington^{+}}$: \u03A9 ag < 0.5')
        ax2.set_title('Oregon')
            
    ax1.plot(yd,region_cumsum[48],c = Lcolors[ydx],linewidth=3, alpha=0.8,label = str(yr_list[ydx]))
    ax1.set_ylabel('cumulative (corrosive volume / vol WA shelf)')
    
    #ax2.plot(yd,OR_cumsum,c = Rcolors[ydx],linewidth=3, alpha=0.8,label = str(yr_list[ydx]))
    #ax2.set_ylabel('cumulative (corrosive volume / vol OR shelf)')
    
    ax1.set_xlim(1,366)
    ax2.set_xlim(1,366)
    
    ax1.set_xticks([1,32,60,91,121,152,182,213,244,274,305,335,365])
    ax2.set_xticks([1,32,60,91,121,152,182,213,244,274,305,335,365])
    
    ax2.set_xlabel('year day')
    
    ax1.legend(frameon=False)
    ax2.legend(frameon=False)
    
    ax1.grid(True)
    ax2.grid(True)
    
    if args.variable == 'arag1':    
        ax1.set_ylim(0,250)
        ax2.set_ylim(0,250)        
    elif args.variable == 'arag05':  
        ax1.set_ylim(0,60)
        ax2.set_ylim(0,60) 


sys.exit()

if args.variable == 'arag1':    
    fig1.savefig('/Users/katehewett/Documents/LKH_output/OA_indicators/cas7_t0_x4b/volumes_by_threshold/bystate/arag1/plots/PERCENT_Oag_state_above1_NEW.png')
elif args.variable == 'arag05':    
    fig1.savefig('/Users/katehewett/Documents/LKH_output/OA_indicators/cas7_t0_x4b/volumes_by_threshold/bystate/arag05/plots/PERCENT_Oag_state_above05_NEW.png')
'''
