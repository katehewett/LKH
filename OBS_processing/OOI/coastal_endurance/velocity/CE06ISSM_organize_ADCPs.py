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

moor = 'CE06ISSM'

ds = xr.open_dataset('/Users/katehewett/Documents/LKH_data/OOI/CE/coastal_moorings/CE06ISSM/velocity/mfd/ooi-ce06issm-mfd35-04-adcptm000_dad4_820b_2d26.nc', decode_times=True)

zu = np.unique(ds.z.values)
ot = np.unique(ds.time.values)
NT = np.shape(ot)[0]
NZ = np.shape(zu)[0]

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

Zgroup = df.groupby('datetimes')['z'].apply(list).reset_index(name='z')  
Ugroup = df.groupby('datetimes')['u'].apply(list).reset_index(name='u')
Vgroup = df.groupby('datetimes')['v'].apply(list).reset_index(name='v')
Wgroup = df.groupby('datetimes')['w'].apply(list).reset_index(name='w')

# there are duplicate entries on Timestamp('2023-01-15 12:00:00')
# fixing by hand 
Zgroup['z'][198254] = np.array([-28.0,
 -27.0,
 -26.0,
 -25.0,
 -24.0,
 -23.0,
 -22.0,
 -21.0,
 -20.0,
 -19.0,
 -18.0,
 -17.0,
 -16.0,
 -15.0,
 -14.0,
 -13.0,
 -12.0,
 -11.0,
 -10.0,
 -9.0,
 -8.0,
 -7.0,
 -6.0,
 -5.0,
 -4.0,
 -3.0,
 -2.0,
 -1.0,
 -0.0,
 1.0])

Ugroup['u'][198254] = np.array([-0.012618391891911426,
 -0.013389326875534834,
 -0.007981788243742972,
 -0.019260319764142025,
 -0.039801064834449965,
 -0.031237296376374856,
 -0.05771053048065161,
 -0.06963700469395083,
 -0.09953444069881331,
 -0.12858845182826276,
 -0.10405904580268655,
 -0.09766310251190241,
 -0.09487584955778885,
 -0.08330083105700974,
 -0.0417558866595785,
 -0.001025852231482709,
 0.026011840927476586,
 0.06967410809326517,
 0.09719724300189583,
 0.1604558255647121,
 0.21021308270769834,
 0.22760859169815106,
 0.19822511234903747,
 0.23321367359247092,
 0.2568556797569221,
 0.2022728641964761,
 0.13197486198318198,
 -0.22571354911154504,
 -0.40466563104317343,
 -0.5037125689123003
])

Vgroup['v'][198254] = [0.08096157228007708,
 0.07806232078166829,
 0.08283291046698797,
 0.09100571456003738,
 0.07991167147572319,
 0.08487774334355572,
 0.08260444704519111,
 0.09612329362467077,
 0.10821226878119881,
 0.11179897162501339,
 0.13011039538266875,
 0.10357566513303038,
 0.07903526536102613,
 0.1069998670336161,
 0.11147396973855571,
 0.17514550415925373,
 0.19899845258585216,
 0.2153265396123014,
 0.19041716296812747,
 0.18704792979968177,
 0.2379715526245235,
 0.23334594272278306,
 0.2473677522115837,
 0.4139714754056183,
 0.5028818546901543,
 0.5349866245148135,
 0.4729689586056574,
 0.36527030230703145,
 0.6456738550169351,
 0.8841349715511604]

Wgroup['w'][198254] =[-0.002,
 0.0,
 0.006,
 0.006,
 0.004,
 0.004,
 0.002,
 0.0,
 -0.002,
 0.0,
 0.003,
 0.001,
 0.0,
 0.001,
 0.0,
 -0.008,
 -0.003,
 -0.009,
 -0.004,
 -0.001,
 0.002,
 0.009,
 0.003,
 -0.019,
 0.003,
 -0.006,
 0.0,
 -0.009,
 -0.042,
 -0.116]

#Zgroup['ZU'] = [zu] * len(Zgroup)

zdict = {}
zdict['z'] = Zgroup['z']

# id where the z's are in the full bin range, zu
zbool = np.ones([NZ,NT])*np.nan
zb = np.ones([NZ,NT])*np.nan
ub = np.ones([NZ,NT])*np.nan
vb = np.ones([NZ,NT])*np.nan
wb = np.ones([NZ,NT])*np.nan

for i in range(NT):
    ind = np.isin(zu,zdict['z'][i])
    zbool[:,i] = ind

    zb[np.where(zbool[:,i]),i] = Zgroup['z'][i]
    #ub[:,i][np.where(zbool[:,i])]=np.array(Ugroup['u'][i])
    #vb[:,i][np.where(zbool[:,i])]=np.array(Vgroup['v'][i])
    #wb[:,i][np.where(zbool[:,i])]=np.array(Wgroup['w'][i])



    





sys.exit()

for i in enumerate(zdict['z'],start=0):
    zi = zdict['z'][i]
    ind = np.where(np.isin(zu,zi)) 
    zdict['zidx'][i] = ind


sys.exit()

def compare_arrays(row): 
    matches = []
    for i,val in enumerate(row['z']):
        if val in row['ZU']:
            matches.append(i)
    return matches

Zgroup['zindex'] = Zgroup.apply(compare_arrays,axis=1)

def test(row):
    for i,val in enumerate(row['z']):
        print(Zgroup['z'])

sys.exit()

def zidx(data,func):
    return {key:func(value) for key, value in data.items()}




for index,row in Zgroup.iterrows():
    array1 = np.array(row['ZU'])
    array2 = np.array(row['z'])

    matching_idx = np.where(np.isin(array1,array2))[0]

    #if matching_ind.size > 0: 
        
'''
def find_z_index(row,search_term,output_col):
    if search_term in row[' ']
    return np.where(np.isin(zu, row['z']))

Zgroup['z_index'] = df.apply(find_z_index,axis=0)
np.where(np.isin(array1,array2))
'''
Zgroup = Zgroup.apply(lambda row: np.where(np.isin(zu,row['z'],axis=0)))
zg = Zgroup['z'][NT-1]
zga = np.array(zg)
ind = np.where(np.isin(zu,zga))

sys.exit()
#plots 
plt.close('all')
fs=11
plt.rc('font', size=fs)
height_of_image = 6.5
width_of_image = 10
fig1 = plt.figure(figsize=(width_of_image,height_of_image))
fig1.set_size_inches(width_of_image,height_of_image, forward=False)

ax = plt.subplot2grid((3,1),(0,0),colspan=1,rowspan=1)
plt.plot(df['datetimes'],df['z'],'.')

sys.exit()
zmin = df.groupby('datetimes')['z'].min().reset_index(name = 'z')
plt.plot(zmin,'x')


# group the bins by sampletime and flag unreasonable depths for removal 
# example: There are not data being collected at 8,000m ... so the all data for those samples are removed
#zmin['flag_toodeep'] = zmin['z']<-540
zmin['flag_depth'] = (zmin['z']>-510) | (zmin['z']<-540)

#df = df.set_index('datetimes')

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


'''
Pmfn = xr.open_dataset('/Users/katehewett/Documents/LKH_data/OOI/CE/coastal_moorings/CE06ISSM/pressure/mfd/ooi-ce06issm-mfd37-03-ctdbpc000_5d2b_5c05_8b78.nc', decode_times=True)
#Pnsif = xr.open_dataset('/Users/katehewett/Documents/LKH_data/OOI/CE/coastal_moorings/CE09OSSM/pressure/nsif/ooi-ce09ossm-rid27-03-ctdbpc000_9def_2170_af4e.nc', decode_times=True)
Pseafloor = xr.open_dataset('/Users/katehewett/Documents/LKH_data/OOI/CE/coastal_moorings/CE06ISSM/pressure/mfd/ooi-ce06issm-mfd35-02-presfa000_2d0a_0cb6_f73c.nc', decode_times=True)

df_p = pd.DataFrame({'datetimes':Pmf.time.values})


ds = xr.open_dataset('/Users/katehewett/Documents/LKH_data/OOI/CE/coastal_moorings/CE09OSSM/velocity/mfd/ooi-ce09ossm-mfd35-04-adcpsj000_ff2d_359a_11eb.nc', decode_times=True)                   
Pmfn = xr.open_dataset('/Users/katehewett/Documents/LKH_data/OOI/CE/coastal_moorings/CE09OSSM/pressure/mfd/ooi-ce09ossm-mfd37-03-ctdbpe000_9def_2170_af4e.nc', decode_times=True)
Pnsif = xr.open_dataset('/Users/katehewett/Documents/LKH_data/OOI/CE/coastal_moorings/CE09OSSM/pressure/nsif/ooi-ce09ossm-rid27-03-ctdbpc000_9def_2170_af4e.nc', decode_times=True)
Pseafloor = xr.open_dataset('/Users/katehewett/Documents/LKH_data/OOI/CE/coastal_moorings/CE09OSSM/pressure/mfd/ooi-ce09ossm-mfd35-02-presfc000_e0c5_bc24_0ec6.nc', decode_times=True)


#format plotting space 
plt.close('all')
fs=11
plt.rc('font', size=fs)
height_of_image = 6.5
width_of_image = 10
fig1 = plt.figure(figsize=(width_of_image,height_of_image))
fig1.set_size_inches(width_of_image,height_of_image, forward=False)

ax = plt.subplot2grid((3,1),(0,0),colspan=1,rowspan=1)
plt.plot(Pseafloor.time.values[Pseafloor.sea_water_pressure_at_sea_floor_tide_qc_agg==1],Pseafloor.sea_water_pressure_at_sea_floor_tide.values[Pseafloor.sea_water_pressure_at_sea_floor_tide_qc_agg==1],'k.',label = 'sea floor')
plt.plot(Pmfn.time.values[Pmfn.sea_water_pressure_qc_agg==1],Pmfn.sea_water_pressure.values[Pmfn.sea_water_pressure_qc_agg==1],'b.',label = 'mfn')
plt.legend(loc="best")
ax.set_ylabel('pressure decibars')
ax.set_title(moor + '  only QC = 1')
plt.gcf().tight_layout()

dsw = xr.open_dataset('/Users/katehewett/Documents/LKH_data/OOI/CE/coastal_moorings/CE09OSSM/wave_sw/surfacebuoy/ooi-ce09ossm-sbd12-05-wavssa000_5364_5a59_7501.nc', decode_times=True)
ax1 = plt.subplot2grid((3,1),(1,0),colspan=1,rowspan=1)
plt.plot(dsw.time.values,dsw.sea_surface_wave_maximum_height.values,'g.',label = 'all')
plt.plot(dsw.time.values[dsw.sea_surface_wave_maximum_height_qc_agg==1],dsw.sea_surface_wave_maximum_height.values[dsw.sea_surface_wave_maximum_height_qc_agg==1],'m.',label = 'qc=1')
plt.legend()
ys.exit()                   
'''