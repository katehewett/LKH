'''
Step 2: 
This code is to process data from OOI for the WA(and OR) surface mooring velocity data.

The ADCP data has already been binned by OOI/Axiom and is provides u,v,w (and error, see README)


(1) opens a LO mooring extraction to grab bin edges 
(2) opens associated velocity data and bins accordingly
(3) organizes and plots data to see what is there

need to enter the info you entered for a grid extraction (just one - to grab the bin edges for the mooring):
For example (I have already extracted): -gtx cas7_t0_x4b -ro 2 -0 2017.01.01 -1 2017.12.31 -lt lowpass -job OOI_WA_SM

-gtx cas7_t0_x4b (this is the grid I compared)
-y0 2017 (this is the year moors were extracted)
-job OOI_WA_SM (OOI_OR_SM)



'/Users/katehewett/Documents/LKH_data/OOI/CE/coastal_moorings/CE06ISSM/velocity/mfd'
ooi-ce06issm-mfd35-01-vel3dd000_21b5_1859_99b7.nc  ooi-ce06issm-mfd35-04-adcptm000_dad4_820b_2d26.nc
'''

import xarray as xr
import numpy as np
import pandas as pd 
import sys

ds = xr.open_dataset('/Users/katehewett/Documents/LKH_data/OOI/CE/coastal_moorings/CE06ISSM/velocity/mfd/ooi-ce06issm-mfd35-04-adcptm000_dad4_820b_2d26.nc', decode_times=True)

z = np.unique(ds.z.values)
ot = np.unique(ds.time.values)
NT = np.shape(ot)[0]
NZ = np.shape(z)[0]

print('len Z: '+ str(np.shape(ds.z.values)[0]))
print('len Time: '+ str(np.shape(ds.time.values)[0]))
print('len NZ*NT: ' + str(NT*NZ))

# OOI download provides data in a long 1D array 
# want to convert to row/z and col/time
df = pd.DataFrame({'datetimez':ds.time.values})
df['datez'] = df['datetimez'].dt.date
df['z'] = ds.z.values

if df['datetimez'].is_monotonic_increasing == False:
    print('issue with times')
    sys.exit()
else: 
    print('times pass')


amat = np.nan * np.ones((NZ,NT))

'''
umat = np.nan * np.ones((NZ,NT))
vmat = np.nan * np.ones((NZ,NT))
emat = np.nan * np.ones((NZ,NT))
'''