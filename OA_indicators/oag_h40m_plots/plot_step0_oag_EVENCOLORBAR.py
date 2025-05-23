"""
Plots ~h40m data mask from step0
across all years with thresholds colored

run plot_step0_oag -bot True < plots bottom 
run plot_step0_oag -surf True < plots surface 

"""
# imports
from lo_tools import Lfun
from lo_tools import plotting_functions as pfun

import os 
import sys 
import argparse
import xarray as xr
import pickle 
import matplotlib.pyplot as plt
#import matplotlib.cm as cm
#import cmocean
from datetime import datetime
from matplotlib.colors import BoundaryNorm
import matplotlib.pyplot as plt

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
fn_b = fn_i / 'bot_h40m'
fn_s = fn_i / 'surf_h40m'

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
for idx in range(0,2):
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

#fig1.savefig('/Users/katehewett/Documents/LKH_output/OA_indicators/cas7_t0_x4b/oag_h40m_plots/plot_output/map_NEW.png')

if args.surf==True: 
    axp = plt.subplot2grid((2,3), (0,1), colspan=2) # surface
elif args.bot==True: 
    axp = plt.subplot2grid((2,3), (1,1), colspan=2) # bottom

fig1.tight_layout()
        
for ydx in range(0,numyrs): 
    
    pn = 'OA_indicators_Oag_h40m_'+str(yr_list[ydx])+'.pkl'
    if args.surf==True: 
        picklepath = fn_s/pn
    elif args.bot==True: 
        picklepath = fn_b/pn
    
    if os.path.isfile(picklepath)==False:
        print('no file named: ' + pn)
        sys.exit()
    
    # Load the dictionary from the file
    with open(picklepath, 'rb') as fp3:
        A = pickle.load(fp3)
        print('loaded'+str(yr_list[ydx]))
    
    y = A['lat_rho']
    x = A['ocean_time']
    ARAG = A['ARAG']

    '''
    #cmap = ['#5E0E20','#BB5047','#EDBB9E','#F7F6F7','#AFCFE3','#4D83B8','#132F5E'] THRESHOLD COLORBAR REDS -> blues
    #levels = [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5,1.75, 2, 2.5,2.75, 3]
    #levels = [0.25, 0.5, 1, 1.5, 2, 2.5, 2.75]
    Rcolors = ['#5E0E20','#BB5047','#EDBB9E','#F7F6F7','#AFCFE3','#4D83B8','#132F5E']
    cmap = matplotlib.colors.ListedColormap(Rcolors[1:-2],"")
    cmap.set_over(Rcolors[-1])
    cmap.set_under(Rcolors[0])
    levels = [0.5, 1, 1.5, 2, 2.5]
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=False)
    '''
    levels = [0.25, 0.5, 1, 1.5, 1.7, 2, 2.5, 3]
    cmap = plt.get_cmap('RdBu')
    cmap.set_extremes(over = 'White',under='Black')
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=False)

    # Create the pcolormesh plot
    pcm = axp.pcolormesh(x, y, ARAG, cmap=cmap, norm=norm)
    '''
    if ydx==(numyrs-1):
        cbar = fig1.colorbar(pcm, extend = 'min')
        cbar.set_label('\u03A9 ag')
    '''
    
    #axp.colorbar()
    #plt.ylim([42.75, 48.75])
    axp.set_yticks([42.75,43,44,45,46,47,48,48.75])
    axp.set_yticklabels(['42.75','43','44','45','46','47','48','48.75'])
        
    axp.set_xlim(datetime(2013,1,1), datetime(2024,1,1))
    
    plt.axvline(x = datetime(2013,1,1), color = 'k', label = 'axvline - full height')
    plt.axvline(x = datetime(2014,1,1), color = 'k', label = 'axvline - full height')
    plt.axvline(x = datetime(2015,1,1), color = 'k', label = 'axvline - full height')
    plt.axvline(x = datetime(2016,1,1), color = 'k', label = 'axvline - full height')
    plt.axvline(x = datetime(2017,1,1), color = 'k', label = 'axvline - full height')
    plt.axvline(x = datetime(2018,1,1), color = 'k', label = 'axvline - full height')
    plt.axvline(x = datetime(2019,1,1), color = 'k', label = 'axvline - full height')
    plt.axvline(x = datetime(2020,1,1), color = 'k', label = 'axvline - full height')
    plt.axvline(x = datetime(2021,1,1), color = 'k', label = 'axvline - full height')
    plt.axvline(x = datetime(2022,1,1), color = 'k', label = 'axvline - full height')
    plt.axvline(x = datetime(2023,1,1), color = 'k', label = 'axvline - full height')
    
    axp.set_xticks([datetime(2013,1,1),datetime(2013,7,1),datetime(2014,1,1), datetime(2014,7,1),
    datetime(2015,1,1),datetime(2015,7,1),datetime(2016,1,1), datetime(2016,7,1),
    datetime(2017,1,1),datetime(2017,7,1),datetime(2018,1,1), datetime(2018,7,1),
    datetime(2019,1,1),datetime(2019,7,1),datetime(2020,1,1),datetime(2020,7,1),
    datetime(2021,1,1),datetime(2021,7,1),datetime(2022,1,1),datetime(2022,7,1),
    datetime(2023,1,1),datetime(2023,7,1),datetime(2023,12,31)])
    
    axp.set_xticklabels(['Jan13','Jul','Jan14', 'Jul',
    'Jan15','Jul','Jan16', 'Jul',
    'Jan17','Jul','Jan18', 'Jul',
    'Jan19','Jul','Jan20','Jul',
    'Jan21','Jul','Jan22','Jul',
    'Jan23','Jul','Jan24'])
    
    axp.grid(True)

    if args.surf==True: 
        axp.set_title('Surface layer \u03A9 ag')
    elif args.bot==True: 
        axp.set_title('Bottom layer \u03A9 ag')
        
    axp.set_ylabel('Latitude')        

if args.surf==True: 
    'p'+str(NT)'/Users/katehewett/Documents/LKH_output/OA_indicators/cas7_t0_x4b/oag_h40m_plots/plot_output/Ysurf_map_Oag_NEW_EVENCOLORBAR.png')
elif args.bot==True: 
    fig1.savefig('/Users/katehewett/Documents/LKH_output/OA_indicators/cas7_t0_x4b/oag_h40m_plots/plot_output/Ybot_map_Oag_NEW_EVENCOLORBAR.png')





