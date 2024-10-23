
import os
import sys
import pandas as pd
import numpy as np
import xarray as xr
import posixpath
import datetime 
from datetime import timedelta
import statistics

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
import matplotlib.pyplot as plt

# Create a sample DataFrame
data = {'values': [32.07825719, 32.08127159, 32.099358  , 32.094334  , 32.0862956 ,
       32.0883052 , 32.04710838, 32.02299316, 32.01495476, 32.11443001,
       32.13955003, 32.0923244 , 32.0913196 , 32.02198836, 32.05816118,
       32.0862956 , 31.97375794, 31.95366193, 31.93858992, 31.97275314,
       31.9064363 , 31.8973931 , 31.92753711, 32.04911798, 32.08127159,
       32.01495476, 31.99385395, 32.0852908 , 32.08026679, 32.00088755,
       32.10136761, 32.16567484, 32.22897727, 30, 32.21591487,
       32.23701568, 32.26012609, 32.21290046, 32.20285246, 32.29529411,
       32.26615489, 32.26414529, 32.30735172, 32.37065415, 32.31237572,
       32.38773576, 32.30936132, 32.18074685, 32.25409729, 32.2721837 ,
       32.22194367, 32.2711789 , 32.2832365 , 32.30936132, 32.17773245,
       32.11744441, 32.05213238, 32.0973484 , 32.05313718, 32.03404597,
       32.04811318, 30.5  , 27, 30, 32.0862956 ,
       32.07222839, 32.07323319, 32.10136761, 32.15964604, 32.13452602,
       32.06519479, 32.07524279, 32.12246842, 32.38773576, 32.30132291,
       32.25811649, 32.26816449, 32.17270844, 32.20486206, 32.20687166,
       32.2721837 , 32.31840452, 32.24907328, 32.31739972, 32.33146693,
       32.33850053, 32.37165895, 32.40079817, 32.33850053, 32.43998539,
       32.37668295, 32.31940932, 32.34553414, 32.41185097, 32.2872557 ,
       32.40682697, 32.36864455, 32.38170696, 32.42089418, 32.42390858,
       32.44802379, 32.41787977, 32.42089418, 32.41486537, 32.42189898,
       32.4600814 , 32.38371656, 32.39677896, 32.41084617, 32.39476936,
       32.39376456, 32.30232771, 32.34452934, 32.40079817, 32.41185097,
       32.34051013, 32.33850053, 32.35759174, 32.39175496, 32.4661102 ]}
            
df = pd.DataFrame(data)

# Calculate the rolling range with a window size of 3
window_size = 30 # L
min_samples = 20 

#DataFrame.rolling(window, min_periods=None, center=False, win_type=None, on=None, axis=_NoDefault.no_default,      closed=None, step=None, method='single')    
rw = df['values'].rolling(window_size,center=True,closed='both')

df['rolling_range'] = rw.apply(lambda x: max(x) - min(x))

p25 = rw.quantile(0.25)
p75 = rw.quantile(0.75)
iqr = p75 - p25
lower_bound = p25 - 1.5 * iqr
upper_bound = p75 + 1.5 * iqr

df['outliers'] = (df['values'] < lower_bound) | (df['values'] > upper_bound)
df['outlier_vals'] = df['outliers'].astype(int)*df['values']
df['outlier_vals'].replace(0, np.nan, inplace=True)

# plot 
x =  np.arange(1, 121)

plt.close('all') 
fs=12
plt.rc('font', size=fs)
fig = plt.figure(figsize=(16,8))
ax0 = plt.subplot2grid((3,1),(0,0),colspan=1,rowspan=1)
ax0.fill_between(x, lower_bound, upper_bound,color = 'dodgerblue',alpha=0.2)
plt.plot(x,df['values'],color = 'black',marker='.',linestyle='-',linewidth=1)
plt.plot(x,df['outlier_vals'],color = 'red',marker='x',linestyle='-',linewidth=1)

#print(outliers)

print(df)







