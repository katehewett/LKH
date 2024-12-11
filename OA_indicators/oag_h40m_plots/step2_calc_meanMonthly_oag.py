"""
After Running, step0_calc_Oag.py
Use this code to find the average across each month of the year for each grid cell 
Then plot all the january's together (febs, march-s, etc...). 

run step2_calc_meanMonthly_oag -bot True < plots bottom 
run step2_calc_meanMonthly_oag -surf True < plots surface 

"""
# imports
from lo_tools import Lfun

import os 
import sys 
import argparse
import pandas as pd
import pandas as pd
import pickle 


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
fn_i = Ldir['parent'] / 'LKH_output' / 'OA_indicators' / 'cas7_t0_x4b' / 'oag_h40m_plots' 

if args.surf==True: 
    fn_s = fn_i / 'surf_h40m'
elif args.bot==True: 
    fn_b = fn_i / 'bot_h40m'

# name the output file where files will be dumped
if args.surf==True:
    fn_o = Ldir['parent'] / 'LKH_output' / 'OA_indicators' / 'cas7_t0_x4b' / 'oag_h40m_plots' / 'surf_h40m' / 'monthly'
    Lfun.make_dir(fn_o, clean=False)
elif args.bot==True:
    fn_o = Ldir['parent'] / 'LKH_output' / 'OA_indicators' / 'cas7_t0_x4b' / 'oag_h40m_plots' / 'bot_h40m' / 'monthly'
    Lfun.make_dir(fn_o, clean=False)

yr_list = [year for year in range(2013,2024)]
numyrs = len(yr_list)

# super specific to our grid, careful!! 
#NT = 11
#NR = 366
#NC = 941 
#amat = np.full([NT,NR,NC],np.nan)
        
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

    # pack to dataframe so we can take monthly avgs 
    otime = pd.to_datetime(A['ocean_time'][:,0])    # take all rows in first column = our dates
    df = pd.DataFrame(A['ARAG'])
    df['date'] = otime
    df = df.set_index('date') 

    df2 = df.resample('ME').mean()

    if args.surf==True: 
        pn = 'SURF_monthly_mean_Oag_h40m_'+str(yr_list[ydx])+'.pkl'
    elif args.bot==True: 
        pn = 'BOT_monthly_mean_Oag_h40m_'+str(yr_list[ydx])+'.pkl'

    picklepath = fn_o/pn

    with open(picklepath, 'wb') as fm:
        pickle.dump(df2, fm)
        print('Pickled year %0.0f' % yr_list[ydx])
        sys.stdout.flush()
