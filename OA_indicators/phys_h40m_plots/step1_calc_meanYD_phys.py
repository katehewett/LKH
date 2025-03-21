"""
After Running, step0_calc_rho.py
Use this code to find the average across each day of the year for each grid cell 

run step1_calc_meanYD_phys -bot True < plots bottom 
run step1_calc_meanYD_phys -surf True < plots surface 

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

# input location for pickled files 
fn_i = Ldir['parent'] / 'LKH_output' / 'OA_indicators' / 'cas7_t0_x4b' / 'rho_h40m_plots' 

if args.surf==True: 
    fn_s = fn_i / 'surf_h40m'
elif args.bot==True: 
    fn_b = fn_i / 'bot_h40m'

# name the output file where files will be dumped
if args.surf==True:
    fn_o = Ldir['parent'] / 'LKH_output' / 'OA_indicators' / 'cas7_t0_x4b' / 'rho_h40m_plots' / 'surf_h40m' 
    Lfun.make_dir(fn_o, clean=False)
elif args.bot==True:
    fn_o = Ldir['parent'] / 'LKH_output' / 'OA_indicators' / 'cas7_t0_x4b' / 'rho_h40m_plots' / 'bot_h40m' 
    Lfun.make_dir(fn_o, clean=False)

yr_list = [year for year in range(2013,2024)]
numyrs = len(yr_list)

# super specific to our grid, careful!! 
NT = 11    # all years
NR = 366   # max leap year num time
NC = 941   # num n/s points
SA = np.full([NT,NR,NC],np.nan)
CT = np.full([NT,NR,NC],np.nan)
RHO = np.full([NT,NR,NC],np.nan)
if args.bot==True: 
    PRES = np.full([NT,NR,NC],np.nan)
elif args.surf==True: 
    PRES = np.full([NT,NR,NC],0) # sloppy but these are rough calcs
        
for ydx in range(0,numyrs): 
    
    pn = 'OA_indicators_rho_h40m_'+str(yr_list[ydx])+'.pkl'
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

    # It's packed weird becuase we averaged across the 35-45m depth range (EW)... 
    # and b/c it's going to help later to have it packed like this
    # np.shape(ARAG) = (365, 941) = each column corresponds to a position unique position N-S; each row a unique yearday 

    if np.shape(A['SA'])[0] == 365:
        SA[ydx,0:365,:] = A['SA']
        CT[ydx,0:365,:] = A['CT']
        RHO[ydx,0:365,:] = A['RHO']
        if args.bot==True: 
            PRES[ydx,0:365,:] = A['PRES']
    else: 
        SA[ydx,:,:] = A['SA']
        CT[ydx,:,:] = A['CT']
        RHO[ydx,:,:] = A['RHO']
        if args.bot==True: 
            PRES[ydx,:,:] = A['PRES']
        y = A['lat_rho']
        x = A['ocean_time']

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    m_smat = np.nanmean(SA,axis=0,keepdims=False)
    m_cmat = np.nanmean(CT,axis=0,keepdims=False)
    m_rmat = np.nanmean(RHO,axis=0,keepdims=False)
    m_pmat = np.nanmean(PRES,axis=0,keepdims=False)

# Create a sample DataFrame, convert to year day and drop the year 
df = pd.DataFrame({'date': x[:,0]})
df['date'] = pd.to_datetime(df['date'])
df['day_of_year'] = df['date'].dt.dayofyear
df.drop('date', axis=1, inplace=True)

arr = np.expand_dims(np.array(df['day_of_year']),axis=1) 
arr2 = np.tile(arr,NC)

PHYS = {}
PHYS['year_day'] = arr2
PHYS['lat_rho'] = y
PHYS['mSA'] = m_smat
PHYS['mCT'] = m_cmat
PHYS['mRHO'] = m_rmat
PHYS['mPRES'] = m_pmat
if args.surf == True:
    PHYS['level'] = 'surf'   
elif args.bot ==True:
    PHYS['level'] = 'bot'  
PHYS['source_file'] = 'data from '+str(fn_i)
PHYS['calc_region'] = A['calc_region']
PHYS['date_note'] = 'mean values across 2013 - 2023, year is arbitrary'

if args.surf==True: 
    pn = 'surf_rho_h40m_mean_2013_2023.pkl'
    picklepath = fn_s/pn
elif args.bot==True: 
    pn = 'bot_rho_h40m_mean_2013_2023.pkl'
    picklepath = fn_b/pn

with open(picklepath, 'wb') as fm:
    pickle.dump(PHYS, fm)
    print('Pickled year %0.0f' % yr_list[ydx])
    sys.stdout.flush()

'''
if args.surf==True: 
    fig1.savefig('/Users/katehewett/Documents/LKH_output/OA_indicators/cas7_t0_x4b/oag_h40m_plots/plot_output/Ysurf_map_Oag_NEW.png')
elif args.bot==True: 
    fig1.savefig('/Users/katehewett/Documents/LKH_output/OA_indicators/cas7_t0_x4b/oag_h40m_plots/plot_output/Ybot_map_Oag_NEW.png')

'''



