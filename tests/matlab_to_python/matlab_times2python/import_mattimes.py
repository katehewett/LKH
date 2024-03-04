# Example of importing times from matlab to python 
# original from R Signell, modified Feb 2023 by kmh

from scipy.io import loadmat
import pandas as pd
import datetime as dt
import urllib

# In Matlab, I created this sample 20-day time series:
# t = datenum(2013,8,15,17,11,31) + [0:0.1:20];
# x = sin(t)
# y = cos(t)
# plot(t,x)
# datetick
# save sine.mat

urllib.urlretrieve('http://geoport.whoi.edu/data/sine.mat','sine.mat');

# If you don't use squeeze_me = True, then Pandas doesn't like 
# the arrays in the dictionary, because they look like an arrays
# of 1-element arrays.  squeeze_me=True fixes that.

mat_dict = loadmat('sine.mat',squeeze_me=True)

# make a new dictionary with just dependent variables we want
# (we handle the time variable separately, below)
my_dict = { k: mat_dict[k] for k in ['x','y']}

def matlab2datetime(matlab_datenum):
    day = dt.datetime.fromordinal(int(matlab_datenum))
    dayfrac = dt.timedelta(days=matlab_datenum%1) - dt.timedelta(days = 366)
    return day + dayfrac

# convert Matlab variable "t" into list of python datetime objects
my_dict['date_time'] = [matlab2datetime(tval) for tval in mat_dict['t']]

# print df
<class 'pandas.core.frame.DataFrame'>
DatetimeIndex: 201 entries, 2013-08-15 17:11:30.999997 to 2013-09-04 17:11:30.999997
Data columns (total 2 columns):
x    201  non-null values
y    201  non-null values
dtypes: float64(2)

# plot with Pandas
df = pd.DataFrame(my_dict)
df = df.set_index('date_time')
df.plot()