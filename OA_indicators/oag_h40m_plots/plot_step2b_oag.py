"""
Plots ~h40m data mask from step2b
across all years with thresholds colored
months are grouped and each month end label is the monthly average for that month 

run plot_step2b_oag -bot True < plots bottom 
run plot_step2b_oag -surf True < plots surface 

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
from matplotlib.colors import BoundaryNorm
import matplotlib.ticker as ticker

# command line arugments
parser = argparse.ArgumentParser()
# these flags get only surface or bottom fields if True
# - cannot have both True - It plots one or the other to avoid a 2x loop 
parser.add_argument('-surf', default=False, type=Lfun.boolean_string)
parser.add_argument('-bot', default=False, type=Lfun.boolean_string)
# get the args and put into Ldir
args = parser.parse_args()

# check for input conflicts:
if args.surf==True and args.bot==True:
    print('Error: cannot have surf and bot both True.')
    sys.exit()
    
Ldir = Lfun.Lstart()

# organize and set paths before summing volumes 
#yr_list = [year for year in range(int(args.ys0), int(args.ys1)+1)]
#numyrs = len(yr_list)


# load the shelf mask / assign vars 
fn_mask =  Ldir['parent'] / 'LKH_data' / 'shelf_masks' / 'OA_indicators' / 'OA_indicators_shelf_mask_15_200m_noEstuaries.nc'
dmask = xr.open_dataset(fn_mask) 
mask_shelf = dmask.mask_shelf 
h = dmask.h
xrho = dmask.lon_rho
yrho = dmask.lat_rho

# load the h mask to flag data between 35<h<45m
fn_hmask =  Ldir['parent'] / 'LKH_data' / 'shelf_masks' / 'OA_indicators' / 'OA_indicators_h40m_mask.nc'
hmask = xr.open_dataset(fn_hmask) 
mask_h40m = hmask.mask_h40m.values    # 0 outside; 1 h=35-45m 


# input location for pickled files 
fn_i = Ldir['parent'] / 'LKH_output' / 'OA_indicators' / 'cas7_t0_x4b' / 'oag_h40m_plots' 

yr_list = [year for year in range(2013,2024)]
numyrs = len(yr_list)

plt.close('all')
fs=11
plt.rc('font', size=fs)
height_of_image = 15 
width_of_image = 8 
fig1 = plt.figure(figsize=(height_of_image,width_of_image))
fig1.set_size_inches(height_of_image,width_of_image, forward=False)
frac=0.047 * (height_of_image / (width_of_image/13))

# map
for idx in range(0,1):
    ax = plt.subplot2grid((2,3), (idx,0), colspan=1, rowspan=1)
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
    
    ax.set_title('h40m')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_xticks([-125.5, -124.5, -123.5])
    ax.set_yticks([42.75,43,44,45,46,47,48,48.75])
    ax.set_yticklabels(['42.75','43','44','45','46','47','48','48.75'])
    ax.xaxis.set_ticklabels([-125.5, ' ' , -123.5])
    ax.grid(False)

    pts2 = ax.scatter(xrho*mask_h40m, yrho*mask_h40m, s=1, c = 'mediumseagreen')
    plt.axhline(y = 48.4, color = 'Purple', linestyle = '-')

# 2nd comment is when omit map to make space
if args.surf==True: 
    axp = plt.subplot2grid((2,3), (0,1), colspan=2) # surface
    #axp = plt.subplot2grid((2,1), (0,0), colspan=2) # surface
elif args.bot==True: 
    axp = plt.subplot2grid((2,3), (0,1), colspan=2) # bottom
    #axp = plt.subplot2grid((2,1), (1,0), colspan=2) # surface

fig1.tight_layout()

# Load the lat lon for one file 
if args.surf==True:
    fn_p = fn_i / 'surf_h40m' 
    pn = 'surf_Oag_h40m_mean_2013_2023.pkl' 
elif args.bot==True:
    fn_p = fn_i / 'bot_h40m' 
    pn = 'bot_Oag_h40m_mean_2013_2023.pkl'
picklepath = fn_p/pn

if os.path.isfile(picklepath)==False:
    print('no file named: ' + pn)
    sys.exit()

with open(picklepath, 'rb') as fp3:
    A = pickle.load(fp3)
    print('loaded file')

ll = A['lat_rho'][0,:]
lat_rho = np.expand_dims(ll,axis=0) # just need the first row
del A, fn_p, pn, picklepath



# Load the giganto df (sorted by month then year (2013 - 2023 monthly averages of ARAG))      
if args.surf==True: 
    pn = 'SURF_monthly_means_sorted_Oag_h40m_2013_2023.pkl'
    picklepath = fn_i / 'surf_h40m' / 'monthly' / pn
elif args.bot==True: 
    pn = 'BOT_monthl_means_sorted_Oag_h40m_2013_2023.pkl'
    picklepath = fn_i / 'bot_h40m' / 'monthly' / pn

if os.path.isfile(picklepath)==False:
    print('no file named: ' + pn)
    sys.exit()

# Load the dictionary from the file
with open(picklepath, 'rb') as fp3:
    df = pickle.load(fp3)
    print('loaded file')

# take just the arags
df2 = df.copy()
del df2['year']
del df2['month']
ARAG = np.array(df2) # shape 132x941 . 12 months/year * 11 years = 132 

NR = np.shape(ARAG)[0] 
NC = np.shape(ARAG)[1]

# set up for plotting 
Y = np.tile(lat_rho,(NR,1))
x = np.arange(1, NR+1)
x = np.expand_dims(x,axis=1)
X = np.tile(x,(1,NC))

levels = [0, 0.25, 0.5, 1, 1.5, 1.7, 3, 3.5]
cmap = plt.get_cmap('RdBu')
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

# Create the pcolormesh plot
pcm = axp.pcolor(X, Y, ARAG, cmap=cmap, norm=norm)
#axp.colorbar()
#plt.ylim([42.75, 48.75])
axp.set_yticks([42.75,43,44,45,46,47,48,48.75])
axp.set_yticklabels(['42.75','43','44','45','46','47','48','48.75'])

axp.set_xlim([0.5, 132.5])

plt.axvline(x = 1, color = 'k', label = 'axvline - full height')
plt.axvline(x = 12, color = 'k', label = 'axvline - full height')
plt.axvline(x = 23, color = 'k', label = 'axvline - full height')
plt.axvline(x = 34, color = 'k', label = 'axvline - full height')
plt.axvline(x = 45, color = 'k', label = 'axvline - full height')
plt.axvline(x = 56, color = 'k', label = 'axvline - full height')
plt.axvline(x = 67, color = 'k', label = 'axvline - full height')
plt.axvline(x = 78, color = 'k', label = 'axvline - full height')
plt.axvline(x = 89, color = 'k', label = 'axvline - full height')
plt.axvline(x = 100, color = 'k', label = 'axvline - full height')
plt.axvline(x = 111, color = 'k', label = 'axvline - full height')
plt.axvline(x = 122, color = 'k', label = 'axvline - full height')
plt.axvline(x = 132, color = 'k', label = 'axvline - full height')

plt.axhline(y = 48.40, color = 'Purple', linestyle = '-')

fig1.tight_layout()

axp.set_xticks([1,12,23,34,45,56,67,78,89,100,111,122,132])

axp.set_xticklabels(['Jan 2013','Feb 2013','Mar 2013', 'Apr 2013','May 2013',
                     'Jun 2013','Jul 2013', 'Aug 2013','Sep 2013','Oct 2013',
                     'Nov 2013', 'Dec 2013','Dec 2023'])

plt.xticks(rotation='vertical')
axp.xaxis.set_minor_locator(ticker.MultipleLocator(1))
axp.xaxis.grid(True, which='minor', linestyle=':')

axp.grid(True)

if args.surf==True: 
    axp.set_title('Surface layer \u03A9 ag')
elif args.bot==True: 
    axp.set_title('Bottom layer \u03A9 ag')
    
axp.set_ylabel('Latitude')        

if args.surf==True: 
    fig1.savefig('/Users/katehewett/Documents/LKH_output/OA_indicators/cas7_t0_x4b/oag_h40m_plots/plot_output/GroupedMonthlyMean_surf_map_Oag_NEW.png')
elif args.bot==True: 
    fig1.savefig('/Users/katehewett/Documents/LKH_output/OA_indicators/cas7_t0_x4b/oag_h40m_plots/plot_output/GroupedMonthlyMean_bot_map_Oag_NEW.png')





