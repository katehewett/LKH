"""
Based on the cumulative shelf volume for each state 
at Oag<1  
Make a stoplight plot for WA+/OR

I don't love this plot. It's kind of ugly. SOS.

To run this code, you need to have first run:
run calc_cumulative_pvolume_byState -mvar ARAG1
 
"""
# imports
from lo_tools import Lfun

import xarray as xr
import numpy as np
import pickle 
import matplotlib.pyplot as plt
#import cmocean
from matplotlib.colors import ListedColormap
import matplotlib as mpl

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

#plotting details
yr_list = [year for year in range(2013,2024)]
numyrs = len(yr_list)

# Colors for plotting
WAcolor = '#B79346'  # yellow
ORcolor = '#294066'  # blue 

# LOAD DATA 
# open ARAG1 and ARAG05 
pn1 = 'bystate_ARAG1_365_cumsum_2013_2023.pkl'
picklepath1 = fn_i1 / pn1 

# Load the dictionary from the file
with open(picklepath1, 'rb') as fp:
    ARAG1 = pickle.load(fp)
    print('loaded pickled ARAG1')  

Cstack = np.vstack((ARAG1['WA'],ARAG1['OR']))
YRstack = np.vstack((ARAG1['RYEARS'],ARAG1['RYEARS']))
Y = np.ones(np.shape(Cstack))
Y[0,:] = Y[0,:]*2

# PLOT 
''' Lt yellow to dark red::
FBE7A2
F8DA78
EF857E
EA3324
B02518
'''
levels = [100,130,160,190,220,250]
#cmap = ListedColormap(['#FBE7A3', '#F9DA77', '#EF857E','#EA3324','#B02518'])
cmap = ListedColormap(['LemonChiffon', '#F9DA77', '#EF857E','#EA3324','#B02518'])
norm = mpl.colors.BoundaryNorm(levels, ncolors=cmap.N, clip=True)

plt.close('all')
fs=11
plt.rc('font', size=fs)
height_of_image = 15 
width_of_image = 8 
fig1 = plt.figure(figsize=(height_of_image,width_of_image))
fig1.set_size_inches(height_of_image,width_of_image, forward=False)

ax = plt.subplot2grid((2,3), (0,1), colspan=2) # surface

# Create the pcolormesh plot
pcm = ax.pcolormesh(YRstack, Y, Cstack, edgecolors='Grey', linewidths = 4, cmap=cmap, norm=norm)
cbar = fig1.colorbar(pcm)
cbar.set_label('cumulative fractional volume')

ax.set_xticks(yr_list)
ax.set_yticks([1,2])
ax.set_yticklabels(['OR','WA+'])
string_numbers = [str(num) for num in yr_list]
ax.set_xticklabels(string_numbers)

ax.set_title('Shelf OA status based on cumulative corrosive volumes per state (\u03A9 ag < 1)')

fig1.tight_layout()

#fn_o = Ldir['parent'] / 'LKH_output' / 'OA_indicators' / args.gtagex / 'oag_h40m_plots' / 'surf_h40m' 
#Lfun.make_dir(fn_o, clean=False)

fig1.savefig('/Users/katehewett/Documents/LKH_output/OA_indicators/cas7_t0_x4b/volumes_by_threshold/bystate/arag1/plots/stoplight.png')