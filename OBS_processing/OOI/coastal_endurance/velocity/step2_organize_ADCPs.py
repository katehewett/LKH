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

import sys
import xarray as xr
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

# ds = xr.open_dataset('/Users/katehewett/Documents/LKH_data/OOI/CE/coastal_moorings/CE06ISSM/velocity/mfd/ooi-ce06issm-mfd35-04-adcptm000_dad4_820b_2d26.nc', decode_times=True)
ds = xr.open_dataset('/Users/katehewett/Documents/LKH_data/OOI/CE/coastal_moorings/CE09OSSM/velocity/mfd/ooi-ce09ossm-mfd35-04-adcpsj000_ff2d_359a_11eb.nc', decode_times=True)
                     
z = np.unique(ds.z.values)
ot = np.unique(ds.time.values)
NT = np.shape(ot)[0]
NZ = np.shape(z)[0]

print('unique z, NZ: '+ str(NZ))
print('unique time, NT: '+ str(NT))
print('length z: ' + str(np.shape(ds.z.values)[0]))
print('length time: ' + str(np.shape(ds.time.values)[0]))

# OOI download provides data in a long 1D array 
# want to convert to row/z and col/time
df = pd.DataFrame({'datetimes':ds.time.values})
df['date'] = df['datetimes'].dt.date

df['z'] = ds.z.values
df['u'] = ds.eastward_sea_water_velocity.values
df['v'] = ds.northward_sea_water_velocity.values
df['w'] = ds.upward_sea_water_velocity.values
df['velprof'] = ds.velprof_evl.values

plt.plot(df['datetimes'],df['z'],'.')
zmin = df.groupby('datetimes')['z'].min()
plt.plot(zmin,'x')

#df = df.set_index('datetimes')
# group the bins by sampletime and flag unreasonable depths for removal 
# example: There are not data being collected at 8,000m ... so the all data for those samples are removed
Zgroup = df.groupby('datetimes')['z'].apply(list).reset_index(name='z')  



sys.exit()

# groupby times 
       
Ugroup = df.groupby('datetimes')['z'].apply(list).reset_index(name='u')  
Vgroup = df.groupby('datetimes')['z'].apply(list).reset_index(name='v')  
Wgroup = df.groupby('datetimes')['z'].apply(list).reset_index(name='w')  
Egroup = df.groupby('datetimes')['z'].apply(list).reset_index(name='velprof')  

'''if len(grouped) == NT: print('okay times')
else: print('werid times')'''

#format plotting space 
plt.close('all')
fs=11
plt.rc('font', size=fs)
height_of_image = 6.5
width_of_image = 10

sys.exit()
# upward ADCP 
fig1 = plt.figure(figsize=(width_of_image,height_of_image))
fig1.set_size_inches(width_of_image,height_of_image, forward=False)

ax0 = plt.subplot2grid((4,1), (0,0), colspan=1)
ax0.plt(df['datetimes'],Zgroup)

sys.exit() 

# extra check lengths and make a nan array to fill 
max_group_size = max(len(group) for _, group in grouped)
amat = np.nan * np.ones((max_group_size, NT))

for i, (name,group) in enumerate(grouped):
    amat[:len(group),i] = group['z'].values

'''
if df['datetimez'].is_monotonic_increasing == False:
    print('issue with times - not monotonic increase')
else: 
    print('times pass')
'''

amat = np.nan * np.ones((NZ,NT))

'''
umat = np.nan * np.ones((NZ,NT))
vmat = np.nan * np.ones((NZ,NT))
emat = np.nan * np.ones((NZ,NT))
'''