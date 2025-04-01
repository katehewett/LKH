"""
Plots phytoplankton 
1 plot per day - going to animate to movie

run plot_step2b_oag_EVENCOLORBAR -bot True < plots bottom 
run plot_step2b_oag_EVENCOLORBAR -surf True < plots surface 

"""
# imports
from lo_tools import Lfun, zfun, zrfun

import sys
import argparse
import xarray as xr
from time import time
import numpy as np
import pandas as pd

import pickle 
import matplotlib.pyplot as plt
from cmcrameri import cm as cmc
#import cmocean
import matplotlib.dates as mdates
from datetime import datetime
from matplotlib.colors import BoundaryNorm
import matplotlib.ticker as ticker

# import why wont this save 

# command line arugments
parser = argparse.ArgumentParser()
# which run was used:
parser.add_argument('-gtx', '--gtagex', type=str)   # e.g. cas7_t0_x4b
#parser.add_argument('-ro', '--roms_out_num', type=int) # 2 = Ldir['roms_out2'], etc.
# select years 
parser.add_argument('-y0', '--ys0', type=str) # e.g. 2014
#parser.add_argument('-y1', '--ys1', type=str) # e.g. 2015
parser.add_argument('-job', '--job_type', type=str) # job 
# these flags get only surface or bottom fields if True
# - cannot have both True -
parser.add_argument('-surf', default=False, type=Lfun.boolean_string)
parser.add_argument('-bot', default=False, type=Lfun.boolean_string)
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

# organize and set paths before summing volumes 
#yr_list = [year for year in range(int(args.ys0), int(args.ys1)+1)]
#numyrs = len(yr_list)
thisYR = int(args.ys0)

# name the input+output file where files will be dumped
fn_i = Ldir['parent'] / 'LKH_output' / 'WOAC' / 'cas7_t0_x4b' / 'CE_transect'
fn_name = 'CE_transect_wARAG_'+str(thisYR)+'.01.01_'+str(thisYR)+'.12.31'+'.nc'
fn_in = fn_i / fn_name

m = 'm' + str(thisYR)
fn_o = fn_i / 'plots' / 'TSOOag' / m
fno_name = 'CE_transect_TSDO_ARAG_'+str(thisYR)+'.01.01_'+str(thisYR)+'.12.31'+'.png'
fn_out = fn_o / fno_name

ds = xr.open_dataset(fn_in, decode_times=True) 

NT,NR,NC = np.shape(ds.salt.values)

Y = ds.z_rho.values
X = np.tile(ds.lon_rho.values,(NR,1))

Tlevels = [7.5,8,8.5,9,9.5,10,10.5,11,11.5] #temp scaled
Slevels = [25.5,25.75,26,26.25,26.5,26.75,27,27.25,27.5] #sigma dens + temp 
smap=cmc.batlow.with_extremes(under='grey',over='Crimson')
norm = BoundaryNorm(Slevels, ncolors=smap.N, clip=False)
norm2 = BoundaryNorm(Tlevels, ncolors=smap.N, clip=False)

crmap=cmc.roma_r.with_extremes(under='Navy',over='Maroon')  

Oaglevels = [0.5,0.75,1,1.25,1.75,2,2.25,2.5,2.75,3] #oag
cmapoag=cmc.roma.with_extremes(under='black',over='Navy')  
normoag = BoundaryNorm(Oaglevels, ncolors=cmapoag.N, clip=False)

Olevels = [4.5,5,5.5,6,6.5,7,7.5,8] #oxygen
cmap=cmc.roma.with_extremes(under='black',over='Navy')  
norm3 = BoundaryNorm(Olevels, ncolors=cmap.N, clip=False)

O2levels = [0,0.5,1,1.5,2,2.5,3,3.5,4,4.5] #oxygen
cmap2=cmc.roma.with_extremes(under='black',over='Navy')  
norm4 = BoundaryNorm(O2levels, ncolors=cmap2.N, clip=False)

dates = ds.ocean_time.values
da = xr.DataArray(dates, dims='time', name='time')
months = da.dt.month.values.tolist()
days = da.dt.day.values.tolist()
month_names = [da.dt.strftime('%B').values[i] for i in range(len(da))]

for idx in range(0,NT): 
    plt.close('all')
    fs=11
    plt.rc('font', size=fs)
    height_of_image = 15
    width_of_image = 8
    fig1 = plt.figure(figsize=(height_of_image,width_of_image))
    fig1.set_size_inches(height_of_image,width_of_image, forward=False)
    frac=0.047 * (height_of_image / (width_of_image/13))

    ax1 = plt.subplot2grid((4,4), (0,0), colspan=2)  # T 
    ax2 = plt.subplot2grid((4,4), (1,0), colspan=2)  # S
    ax3 = plt.subplot2grid((4,4), (2,0), colspan=2)  # O2
    ax4 = plt.subplot2grid((4,4), (3,0), colspan=2)  # Oag

    ax5 = plt.subplot2grid((4,4), (0,2), colspan=2)  # T scaled
    ax6 = plt.subplot2grid((4,4), (1,2), colspan=2)  # sigma
    ax7 = plt.subplot2grid((4,4), (2,2), colspan=2)  # o2
    ax8 = plt.subplot2grid((4,4), (3,2), colspan=2)  # o2

    pcm = ax1.pcolormesh(X, Y[idx,:,:].squeeze(), ds.CT.values[idx,:,:], cmap=smap, shading='gouraud',vmin=7.5, vmax=15) #, norm=norm)
    cbar_t = fig1.colorbar(pcm, extend = 'both',ticks=[7.5,9.5,11.5,13.5,15])
    cbar_t.set_label('CT degC')

    ax1.text(-124.5, -75, "Temp.", fontsize=12, color='red', style='italic')
    ax1.text(-124.5, -95, str(days[idx])+' '+month_names[idx]+' '+str(thisYR), fontsize=12, color='black', style='italic')
    fig1.tight_layout()

    pcm2 = ax2.pcolormesh(X, Y[idx,:,:].squeeze(), ds.SA.values[idx,:,:], cmap=crmap, shading='gouraud',vmin=31.5, vmax=35) #, norm=norm)
    cbar_salt = fig1.colorbar(pcm2, extend = 'both',ticks=[31.5,32,32.5,33,33.5,34,34.5,35])
    cbar_salt.set_label('SA g/kg')

    ax2.text(-124.5, -75, "Salinity", fontsize=12, color='red', style='italic')

    # oxygen 
    DO = ds.oxygen.values[idx,:,:]/43.57
    pcm3 = ax3.pcolormesh(X, Y[idx,:,:].squeeze(), DO, cmap=cmap, shading='gouraud',vmin=0, vmax=8) #, norm=norm)
    cbar_o = fig1.colorbar(pcm3, extend = 'both')#,ticks=[25.2,26,26.5,27,27.5])
    cbar_o.set_label('DO ml/L')

    ax3.text(-124.5, -75, "oxygen", fontsize=12, color='red', style='italic')

    # oag 
    pcm4 = ax4.pcolormesh(X, Y[idx,:,:].squeeze(), ds.ARAG.values[idx,:,:], cmap=cmap, shading='gouraud',vmin=0.5, vmax=2) #, norm=norm)
    cbar_ag = fig1.colorbar(pcm4, extend = 'both')#,ticks=[25.2,26,26.5,27,27.5])
    cbar_ag.set_label('\u03A9 ag')

    ax4.text(-124.5, -75, "\u03A9 ag", fontsize=12, color='red', style='italic')

    #temp scaled
    pcm5 = ax5.pcolormesh(X, Y[idx,:,:].squeeze(), ds.CT.values[idx,:,:], cmap=smap, norm=norm2, shading='gouraud') 
    cbar_temp2 = fig1.colorbar(pcm5, extend = 'both')#,ticks=[25.5,26,26.5,27,27.5])
    cbar_temp2.set_label('temp deg C')

    ax5.text(-124.5, -75, "Temp", fontsize=12, color='red', style='italic')

    # sigma scaled
    pcm6 = ax6.pcolormesh(X, Y[idx,:,:].squeeze(), ds.rho.values[idx,:,:]-1000, cmap=smap, norm=norm, shading='gouraud') 
    cbar_rho = fig1.colorbar(pcm6, extend = 'both',ticks=[25.5,26,26.5,27,27.5])
    cbar_rho.set_label('dens-1000 kg/m^3')

    ax6.text(-124.5, -75, "sigma density", fontsize=12, color='red', style='italic')

    # oxygen 2
    pcm7 = ax7.pcolormesh(X, Y[idx,:,:].squeeze(), DO, cmap=cmap, norm=norm3, shading='gouraud') 
    cbar_o2 = fig1.colorbar(pcm7, extend = 'both')#,ticks=[25.2,26,26.5,27,27.5])
    cbar_o2.set_label('DO ml/L')

    ax7.text(-124.5, -75, "oxygen", fontsize=12, color='red', style='italic')

    # oxygen 2
    pcm8 = ax8.pcolormesh(X, Y[idx,:,:].squeeze(), DO, cmap=cmap, norm=norm4, shading='gouraud') 
    cbar_o2l = fig1.colorbar(pcm8, extend = 'both')#,ticks=[25.2,26,26.5,27,27.5])
    cbar_o2l.set_label('DO ml/L')

    ax8.text(-124.5, -75, "oxygen", fontsize=12, color='red', style='italic')

    ax1.set_ylim([-125, 0])
    ax2.set_ylim([-125, 0])
    ax3.set_ylim([-125, 0])
    ax4.set_ylim([-125, 0])
    ax5.set_ylim([-125, 0])
    ax6.set_ylim([-125, 0])
    ax7.set_ylim([-125, 0])

    fnn = 'p'+str(idx)+'.png'
    FO = fn_o / fnn
    fig1.savefig(str(FO))





'''
# 2nd comment is when omit map to make space
if args.surf==True: 
    axp = plt.subplot2grid((2,3), (0,1), colspan=2) # surface
    #axp = plt.subplot2grid((2,1), (0,0), colspan=2) # surface
elif args.bot==True: 
    axp = plt.subplot2grid((2,3), (0,1), colspan=2) # bottom
    #axp = plt.subplot2grid((2,1), (1,0), colspan=2) # surface

fig1.tight_layout()



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

levels = [0.25, 0.5, 1, 1.5, 1.7, 2, 2.5, 3]
cmap = plt.get_cmap('RdBu')
cmap.set_extremes(over = 'White',under='Black')
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=False)

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
'''

'''
if args.surf==True: 
    fig1.savefig('/Users/katehewett/Documents/LKH_output/OA_indicators/cas7_t0_x4b/oag_h40m_plots/plot_output/GroupedMonthlyMean_surf_map_Oag_EVENCOLORBAR_NEW.png')
elif args.bot==True: 
    fig1.savefig('/Users/katehewett/Documents/LKH_output/OA_indicators/cas7_t0_x4b/oag_h40m_plots/plot_output/GroupedMonthlyMean_bot_map_Oag_EVENCOLORBAR_NEW.png')

'''



