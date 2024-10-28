"""
Plots cumulative shelf volume for each region 
at Oag<1 and <0.5 

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
fn_i1 = Ldir['parent'] / 'LKH_output' / 'OA_indicators' / 'cas7_t0_x4b' / 'volumes_by_threshold' / 'byregion' / 'ARAG1' / 'cumulative_sum'
fn_i05 = Ldir['parent'] / 'LKH_output' / 'OA_indicators' / 'cas7_t0_x4b' / 'volumes_by_threshold' / 'byregion' / 'ARAG05' / 'cumulative_sum'

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
Rcolors = ['#73210D','#9C6B2F','#C5B563','#77BED0','#4078B3','#112F92'] #no grey 

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

# LOAD DATA 
# open ARAG1 and ARAG05 
pn1 = 'byregion_ARAG1_365_cumsum_2013_2023.pkl'
pn05 = 'byregion_ARAG05_365_cumsum_2013_2023.pkl'
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
R1 = ARAG1['R1']
R2 = ARAG1['R2']
R3 = ARAG1['R3']
R4 = ARAG1['R4']
R5 = ARAG1['R5']
R6 = ARAG1['R6']

R1h = ARAG05['R1']
R2h = ARAG05['R2']
R3h = ARAG05['R3']
R4h = ARAG05['R4']
R5h = ARAG05['R5']
R6h = ARAG05['R6']

# Set position of bar on X axis
br1 = np.arange(len(R1)) 
br2 = [x + barWidth for x in br1] 
br3 = [x + barWidth for x in br2]

# Make the subplots      
ax1 = plt.subplot2grid((2,3), (0,1), colspan=2, rowspan = 1)  # R1:R3
plt.bar(br1, R1, color = Rcolors[0], width = barWidth, 
        edgecolor = Rcolors[0], label ='R1',zorder=3) 
plt.bar(br1, R1h, color ='w', alpha = 0.4, width = barWidth, 
        edgecolor = 'k', label ='R1 <0.5',zorder=3) 

plt.bar(br2, R2, color = Rcolors[1], width = barWidth, 
        edgecolor = Rcolors[1], label ='R2',zorder=3) 
plt.bar(br2, R2h, color ='w', alpha = 0.4, width = barWidth, 
        edgecolor = 'k', label ='R2 <0.5',zorder=3) 
                
plt.bar(br3, R3, color = Rcolors[2], width = barWidth, 
        edgecolor = Rcolors[2], label ='R3',zorder=3)
plt.bar(br3, R3h, color ='w', alpha = 0.4, width = barWidth, 
        edgecolor = 'k', label ='R3 <0.5',zorder=3)

# Adding Xticks 
year_strings = [str(x) for x in yr_list]
plt.xticks([r + barWidth for r in range(len(R1))], 
        year_strings)
                       
ax2 = plt.subplot2grid((2,3), (1,1), colspan=2, rowspan = 1)  # R4:R5  
plt.bar(br1, R4, color = Rcolors[3], width = barWidth, 
        edgecolor = Rcolors[3], label ='R4',zorder=3) 
plt.bar(br1, R4h, color ='w', alpha = 0.4, width = barWidth, 
        edgecolor = 'k', label ='R4 <0.5',zorder=3) 

plt.bar(br2, R5, color = Rcolors[4], width = barWidth, 
        edgecolor = Rcolors[4], label ='R5',zorder=3) 
plt.bar(br2, R5h, color ='w', alpha = 0.4, width = barWidth, 
        edgecolor = 'k', label ='R5 <0.5',zorder=3) 
                
plt.bar(br3, R6, color = Rcolors[5], width = barWidth, 
        edgecolor = Rcolors[5], label ='R6',zorder=3)
plt.bar(br3, R6h, color ='w', alpha = 0.4, width = barWidth, 
        edgecolor = 'k', label ='R6 <0.5',zorder=3)
   
plt.xticks([r + barWidth for r in range(len(R1))], 
        year_strings)

ax1.grid()
ax1.grid(zorder=0)
ax2.grid()     
ax2.grid(zorder=0)

ax1.set_ylim(0,250)
ax2.set_ylim(0,250)
 
tks = np.arange(0, 251, 25)   
ax1.set_yticks(tks)
ax2.set_yticks(tks)

ax1.set_title('Cumulative corrosive volumes across regions and years')
ax1.set_ylabel('cumulative fractional volume')
ax2.set_ylabel('cumulative fractional volume')

fig1.tight_layout()

#ax2.set_ylabel('cumulative fractional volume (corrosive volume at R# / shelf volume at R#)',loc="bottom")  

#ax2.yaxis.set_label_coords(-0.1, 0.5)    
            
fig1.savefig('/Users/katehewett/Documents/LKH_output/OA_indicators/cas7_t0_x4b/volumes_by_threshold/byregion/arag1/plots/cumulative_fractVols_lt_ARAG1_ARAG05.png')
