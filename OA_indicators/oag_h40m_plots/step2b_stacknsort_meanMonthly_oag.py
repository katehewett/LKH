"""
After Running, step2_calc_meanMonthly_oag (surf and bot)
Use this code to open each pickled monthly file and stack them all together to a giant dataframe
with all the years...
sort by month, year...
save giganto file 
Goal is to plot all the january's together (febs, march-s, etc...). 

run step2b_stacknsort_meanMonthly_oag -bot True < plots bottom 
run step2b_stacknsort_meanMonthly_oag -surf True < plots surface 

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

# starter location for pickled files 
fn_i = Ldir['parent'] / 'LKH_output' / 'OA_indicators' / 'cas7_t0_x4b' / 'oag_h40m_plots' 

# name the input and output location where files will be dumped
if args.surf==True:
    fn_p = fn_i / 'surf_h40m' / 'monthly'
    #Lfun.make_dir(fn_o, clean=False)
elif args.bot==True:
    fn_p = fn_i / 'bot_h40m' / 'monthly'
    #Lfun.make_dir(fn_o, clean=False)

yr_list = [year for year in range(2013,2024)]
numyrs = len(yr_list)

# super specific to our grid, careful!! 
#NT = 11
#NR = 366
#NC = 941 
#amat = np.full([NT,NR,NC],np.nan)
        
for ydx in range(0,numyrs): 
    
    if args.surf==True: 
        pn = 'SURF_monthly_mean_Oag_h40m_'+str(yr_list[ydx])+'.pkl'
    elif args.bot==True: 
        pn = 'BOT_monthly_mean_Oag_h40m_'+str(yr_list[ydx])+'.pkl'

    picklepath = fn_p/pn
    
    if os.path.isfile(picklepath)==False:
        print('no file named: ' + pn)
        sys.exit()
    
    # Load the dictionary from the file
    with open(picklepath, 'rb') as fp3:
        A = pickle.load(fp3)
        print('loaded'+str(yr_list[ydx]))

    if ydx == 0: 
        df = A
    else: 
        df = pd.concat([df,A],axis=0)

# extract month and year and then sort 
df['month'] = df.index.month
df['year'] = df.index.year
df3 = df.sort_values(['month','year'])

if args.surf==True: 
    pn = 'SURF_monthly_means_sorted_Oag_h40m_2013_2023.pkl'
elif args.bot==True: 
    pn = 'BOT_monthl_means_sorted_Oag_h40m_2013_2023.pkl'

picklepath = fn_p/pn

with open(picklepath, 'wb') as fm:
    pickle.dump(df3, fm)
    print('Pickled year %0.0f' % yr_list[ydx])
    sys.stdout.flush()

