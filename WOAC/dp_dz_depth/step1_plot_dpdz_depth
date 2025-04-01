'''
test file from step2b


run step2_calc_percenttime_corrosive_seasonal.py -bot True -gtx cas7_t0_x4b -job OA_indicators

''' 
# imports and set up command line arugments
from lo_tools import Lfun
import sys 
import argparse
import pandas as pd
import numpy as np
import pickle 

parser = argparse.ArgumentParser()
# these flags get only surface or bottom fields if True
# - cannot have both True - It plots one or the other to avoid a 2x loop 
parser.add_argument('-surf', default=False, type=Lfun.boolean_string)
parser.add_argument('-bot', default=False, type=Lfun.boolean_string)
parser.add_argument('-gtx', '--gtagex', type=str)   # e.g. cas7_t0_x4b
parser.add_argument('-job', '--job_type', type=str) # job 
# get the args and put into Ldir
args = parser.parse_args()

if args.surf==True and args.bot==True:
    print('Error: cannot have surf and bot both True.')
    sys.exit()

Ldir = Lfun.Lstart()

# add this in as a command line argument 
threshold = 1

# name the input and output location where files will be dumped
if args.surf==True:
    fn_i = Ldir['parent'] / 'LKH_output' / 'OA_indicators' / args.gtagex / 'surface_maps'    
    #Lfun.make_dir(fn_o, clean=False)
elif args.bot==True:
    fn_i = Ldir['parent'] / 'LKH_output' / 'OA_indicators' / args.gtagex / 'bottom_maps' 
    fn_o = Ldir['parent'] / 'LKH_output' / 'OA_indicators' / args.gtagex / 'bottom_maps' / 'percent_time' 
    Lfun.make_dir(fn_o, clean=False)

# add to loop 
ys0 = 2013 
ys1 = 2023
yr_list = [year for year in range(int(ys0), int(ys1)+1)]
numyrs = len(yr_list)

# 1. concat all the years for ARAG and otime 
for ydx in range(0,numyrs): 
    if args.surf==True:
        pn = args.job_type+'_Oag_surface_'+str(yr_list[ydx])+'.pkl'
    elif args.bot==True:
        pn = args.job_type+'_Oag_bottom_'+str(yr_list[ydx])+'.pkl'

    picklepath = fn_i/pn

    # Load the dictionary from the file
    with open(picklepath, 'rb') as fp3:
        A = pickle.load(fp3)
        print('loaded'+str(yr_list[ydx]))

    a = A['ARAG']
    ot = pd.to_datetime(A['ocean_time'].values)

    if ydx == 0:
        ARAG = a
        otime = ot
    else: 
        otime = np.concatenate([otime,ot])
        ARAG = np.concatenate([ARAG,a])

lat_rho=A['lat_rho'].values
lon_rho=A['lon_rho'].values

corrosive = np.copy(ARAG)
corrosive[ARAG > threshold] = 0
corrosive[ARAG <= threshold] = 1

# grab seasonal indicies to sum over 
df = pd.DataFrame({'date': otime})
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.month        # Extract the month from the datetime column

wdx = np.where((df['month']>=1) & (df['month']<=3)); wdx = np.squeeze(wdx)
spdx = np.where((df['month']>=4) & (df['month']<=6)); spdx = np.squeeze(spdx)
sudx = np.where((df['month']>=7) & (df['month']<=9)); sudx = np.squeeze(sudx)
fdx = np.where((df['month']>=10) & (df['month']<=12)); fdx = np.squeeze(fdx)

winter = corrosive[wdx,:,:]
winter_count = np.sum(winter,axis=0)
winter_percent = (winter_count/len(wdx))*100

spring = corrosive[spdx,:,:]
spring_count = np.sum(spring,axis=0)
spring_percent = (spring_count/len(spdx))*100

summer = corrosive[sudx,:,:]
summer_count = np.sum(summer,axis=0)
summer_percent = (summer_count/len(sudx))*100

fall = corrosive[fdx,:,:]
fall_count = np.sum(fall,axis=0)
fall_percent = (fall_count/len(fdx))*100

#assign values and save
percent_time = {}
percent_time['winter'] = winter_percent
percent_time['spring'] = spring_percent
percent_time['summer'] = summer_percent
percent_time['fall'] = fall_percent
percent_time['lat_rho'] = lat_rho
percent_time['lon_rho'] = lon_rho
percent_time['years_included'] = yr_list
percent_time['winter_months'] = [1,2,3]
percent_time['spring_months'] = [4,5,6]
percent_time['summer_months'] = [7,8,9]
percent_time['fall_months'] = [10,11,12]
percent_time['threshold_used'] = threshold


del picklepath 
del pn 

if args.surf==True:
    pn = args.job_type+'_Oag_surface_'+str(yr_list[ydx])+'.pkl'
elif args.bot==True:
    pn = 'seasonal_percent_time_ARAG'+str(threshold)+'_bottom_'+args.job_type+'.pkl'

picklepath = fn_o/pn

with open(picklepath, 'wb') as fm:
    pickle.dump(percent_time, fm)
    print('Pickled file')
    sys.stdout.flush()
