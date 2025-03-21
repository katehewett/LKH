"""
After Running, step0_calc_rho.py
Use this code to find a rough average dpdz for each day of the year for each grid cell 
it uses the surface most and bottom most cells 

run step1_calc_meanYD_dpdz 

"""
# imports
from lo_tools import Lfun

import os 
import sys 
import argparse
import pandas as pd
import numpy as np
import pandas as pd
import pickle 

import warnings

Ldir = Lfun.Lstart()

# input location for pickled files 
fn_i = Ldir['parent'] / 'LKH_output' / 'OA_indicators' / 'cas7_t0_x4b' / 'rho_h40m_plots' 
fn_s = fn_i / 'surf_h40m'
fn_b = fn_i / 'bot_h40m'

# name the output file where files will be dumped

fn_o = Ldir['parent'] / 'LKH_output' / 'OA_indicators' / 'cas7_t0_x4b' / 'rho_h40m_plots' / 'dpdz_h40m' 
Lfun.make_dir(fn_o, clean=False)

yr_list = [year for year in range(2013,2024)]
numyrs = len(yr_list)

# super specific to our grid, careful!! 
NT = 11    # all years
NR = 366   # max leap year num time
NC = 941   # num n/s points
sRHO = np.full([NT,NR,NC],np.nan)
bRHO = np.full([NT,NR,NC],np.nan)
PRES = np.full([NT,NR,NC],np.nan)

for ydx in range(0,numyrs): 
    
    pn = 'OA_indicators_rho_h40m_'+str(yr_list[ydx])+'.pkl'
    surf_picklepath = fn_s/pn
    bot_picklepath = fn_b/pn
    
    if os.path.isfile(surf_picklepath)==False:
        print('no file named: ' + pn)
        sys.exit()

    if os.path.isfile(bot_picklepath)==False:
        print('no file named: ' + pn)
        sys.exit()

    # Load the dictionary from the file
    with open(surf_picklepath, 'rb') as fp3:
        surfA = pickle.load(fp3)
        print('loaded'+str(yr_list[ydx]))

    with open(bot_picklepath, 'rb') as fp3:
        botA = pickle.load(fp3)
        print('loaded'+str(yr_list[ydx]))

    # It's packed weird becuase we averaged across the 35-45m depth range (EW)... 
    # and b/c it's going to help later to have it packed like this
    # np.shape(ARAG) = (365, 941) = each column corresponds to a position unique position N-S; each row a unique yearday 

    if np.shape(surfA['SA'])[0] == 365:
        sRHO[ydx,0:365,:] = surfA['RHO']
        bRHO[ydx,0:365,:] = botA['RHO']
        PRES[ydx,0:365,:] = botA['PRES']
    else: 
        sRHO[ydx,:,:] = surfA['RHO']
        bRHO[ydx,:,:] = botA['RHO']
        PRES[ydx,:,:] = botA['PRES']

        y = surfA['lat_rho']
        yy = botA['lat_rho']
        x = surfA['ocean_time']
        xx = botA['ocean_time']

if np.shape(xx)!=np.shape(x): 
    sys.exit()
if np.shape(yy)!=np.shape(y): 
    sys.exit()

dpdz1 = (bRHO-sRHO)/PRES

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    m_dpdz = np.nanmean(dpdz1,axis=0,keepdims=False)

# Create a sample DataFrame, convert to year day and drop the year 
df = pd.DataFrame({'date': x[:,0]})
df['date'] = pd.to_datetime(df['date'])
df['day_of_year'] = df['date'].dt.dayofyear
df.drop('date', axis=1, inplace=True)

arr = np.expand_dims(np.array(df['day_of_year']),axis=1) 
arr2 = np.tile(arr,NC)

dpdz = {}
dpdz['year_day'] = arr2
dpdz['lat_rho'] = y
dpdz['m_dpdz'] = m_dpdz
dpdz['source_file'] = 'data from '+str(fn_i)
dpdz['calc_region'] = botA['calc_region']
dpdz['date_note'] = 'mean values across 2013 - 2023, year is arbitrary'

pn = 'simple_dpdz_h40m_mean_2013_2023.pkl'
picklepath = fn_o/pn

with open(picklepath, 'wb') as fm:
    pickle.dump(dpdz, fm)
    print('Pickled year %0.0f' % yr_list[ydx])
    sys.stdout.flush()