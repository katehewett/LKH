"""
Plots cumulative shelf volume for each state 
at Oag<1 and <0.5 

To run this code, you need to have first run both:
run calc_cumulative_pvolume_byState -mvar ARAG05
run calc_cumulative_pvolume_byState -mvar ARAG1
 
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

Ldir = Lfun.Lstart()

# load the shelf mask / assign vars 
fn_mask =  Ldir['parent'] / 'LKH_data' / 'shelf_masks' / 'OA_indicators' / 'OA_indicators_shelf_mask_15_200m_noEstuaries.nc'
dmask = xr.open_dataset(fn_mask) 
mask_shelf = dmask.mask_shelf 
h = dmask.h
xrho = dmask.lon_rho
yrho = dmask.lat_rho

# input location for pickled files 
fn_i1 = Ldir['parent'] / 'LKH_output' / 'OA_indicators' / 'cas7_t0_x4b' / 'volumes_by_threshold' / 'bystate' / 'ARAG1' / 'cumulative_sum'
fn_i05 = Ldir['parent'] / 'LKH_output' / 'OA_indicators' / 'cas7_t0_x4b' / 'volumes_by_threshold' / 'bystate' / 'ARAG05' / 'cumulative_sum'

#plotting details
mask_dict = {}
mask_dict['WA'] = (yrho > 46.25) & (mask_shelf == 1)
mask_dict['OR'] = (yrho <= 46.25) & (mask_shelf == 1)
NMASK = len(mask_dict)

yr_list = [year for year in range(2013,2024)]
numyrs = len(yr_list)

# Colors for plotting
WAcolor = '#B79346'  # yellow
ORcolor = '#294066'  # blue 

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

ax.text(0.95,.10,'40m',color='DarkGrey',weight='bold',transform=ax.transAxes,ha='right')
ax.text(0.95,.07,'80m',color='black',weight='bold',transform=ax.transAxes,ha='right')
ax.text(0.95,.04,'200m',color='black',weight='bold',transform=ax.transAxes,ha='right')
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
ax.scatter(xrho*mask_dict['WA'], yrho*mask_dict['WA'], s=2, c = WAcolor) 
ax.scatter(xrho*mask_dict['OR'], yrho*mask_dict['OR'], s=2, c = ORcolor)

# LOAD DATA 
# open ARAG1 and ARAG05 
pn1 = 'bystate_ARAG1_365_cumsum_2013_2023.pkl'
pn05 = 'bystate_ARAG05_365_cumsum_2013_2023.pkl'
picklepath1 = fn_i1 / pn1 
picklepath05 = fn_i05 / pn05
        
if os.path.isfile(picklepath05)==False | os.path.isfile(picklepath1)==False:
    print('no picklefiles')
    sys.exit()

# Load the dictionary from the file
with open(picklepath1, 'rb') as fp:
    ARAG1 = pickle.load(fp)
    print('loaded pickled ARAG1')  

with open(picklepath05, 'rb') as fp:
    ARAG05 = pickle.load(fp)
    print('loaded pickled ARAG05')  
     
# PLOT 
barWidth = 0.25
# set height of bars
R1 = ARAG1['WA']  # for my brain WA = 1 
R2 = ARAG1['OR']  #              OR = 2

R1h = ARAG05['WA']
R2h = ARAG05['OR']

# Set position of bar on X axis
br1 = np.arange(len(R1)) 
br2 = [x + barWidth for x in br1] 
#br3 = [x + barWidth for x in br2]

# Make the subplots      
ax1 = plt.subplot2grid((2,3), (0,1), colspan=2, rowspan = 1)  # WA and OR
plt.bar(br1, R1, color = WAcolor, width = barWidth, 
        edgecolor = WAcolor, label ='R1',zorder=3) 
plt.bar(br1, R1h, color ='w', alpha = 0.4, width = barWidth, 
        edgecolor = 'k', label ='R1 <0.5',zorder=3) 

plt.bar(br2, R2, color = ORcolor, width = barWidth, 
        edgecolor = ORcolor, label ='R2',zorder=3) 
plt.bar(br2, R2h, color ='w', alpha = 0.4, width = barWidth, 
        edgecolor = 'k', label ='R2 <0.5',zorder=3) 

# Adding Xticks 
year_strings = [str(x) for x in yr_list]
plt.xticks([r + barWidth for r in range(len(R1))], 
        year_strings)
                       
ax1.grid()
ax1.grid(zorder=0)

ax1.set_ylim(0,250)
 
tks = np.arange(0, 251, 25)   
ax1.set_yticks(tks)

ax1.set_title('Cumulative corrosive volumes across WA(OR)')
ax1.set_ylabel('cumulative fractional volume')

fig1.tight_layout()

fig1.savefig('/Users/katehewett/Documents/LKH_output/OA_indicators/cas7_t0_x4b/volumes_by_threshold/bystate/arag1/plots/cumulative_fractVols_lt_ARAG1_ARAG05.png')