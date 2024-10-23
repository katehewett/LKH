

import pandas as pd

# Create a sample DataFrame
data = {'values': [32.07825719, 32.08127159, 32.099358  , 32.094334  , 32.0862956 ,
       32.0883052 , 32.04710838, 32.02299316, 32.01495476, 32.11443001,
       32.13955003, 32.0923244 , 32.0913196 , 32.02198836, 32.05816118,
       32.0862956 , 31.97375794, 31.95366193, 31.93858992, 31.97275314,
       31.9064363 , 31.8973931 , 27, 32.04911798, 32.08127159,
       32.01495476, 31.99385395, 32.0852908 , 32.08026679, 32.00088755,
       32.10136761, 32.16567484, 32.22897727, 32.21892927, 32.21591487,
       32.23701568, 32.26012609, 32.21290046, 32.20285246, 32.29529411,
       32.26615489, 32.26414529, 32.30735172, 32.37065415, 32.31237572,
       32.38773576, 32.30936132, 32.18074685, 32.25409729, 32.2721837 ,
       32.22194367, 32.2711789 , 32.2832365 , 32.30936132, 32.17773245,
       32.11744441, 32.05213238, 32.0973484 , 32.05313718, 32.03404597]}
       
df = pd.DataFrame(data)
       
# Calculate the rolling range with a window size of 3
window_size = 30 # L
threshold = 0.005
N = 11

'''
1) The time series is divided into windows of len L (an odd integer
number). Then, window by window, each value is compared to its 
neighboring values: a range of these values is computed 
(max.minus min.), and replaced with the measurement accuracy threshold if threshold>range.
'''

#DataFrame.rolling(window, min_periods=None, center=False, win_type=None, on=None, axis=_NoDefault.no_default,      closed=None, step=None, method='single')
    
df['rolling_range'] = df['values'].rolling(window_size).apply(lambda x: max(x) - min(x))
df['range'] = np.copy(df['rolling_range'])
df['range'] = df['range'].clip(lower=threshold)

''''
2) A value is presumed to be good, i.e. no spike, if it deviates from the
mean of the peers by less than a multiple of the range, N*max(R,ACC).
'''
df['spike_thresh'] = df['range']*N
df['mean_range'] = df['values'].rolling(window_size).mean()
df['test'] = df['mean_range']-df['values']

#df['p2'] = rw.quantile(0.02)
df['p25'] = df['values'].rolling(window_size,min_periods=20,center=True).quantile(0.25)
df['p50'] = df['values'].rolling(window_size,min_periods=20,center=True).quantile(0.5)
df['p75'] = df['values'].rolling(window_size,min_periods=20,center=True).quantile(0.75)
#df['p92'] = rw.quantile(0.98)

print(df)







