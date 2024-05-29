# Example of importing times from matlab to python 

from scipy.io import loadmat
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt

# In Matlab, I created 2 sample time series
#
# t = datenum(2013,8,15,17,11,31) + [0:0.1:20];
# x = sin(t)
# y = cos(t)
# plot(t,x)
# datetick
#
# cd /Users/katehewett/Documents/LKH/tests/matlab_to_python/matlab_times2python
# save sine.mat
#
# clear all 
# % put one in with fequency of 12 days multi year 
# t = datenum(datetime(2013,8,15):days(12):datetime(2022,8,15))
# x = sin(t)
# y = cos(t)
# plot(t,x)
# datetick
#
# save sine2.mat


# cd /Users/katehewett/Documents/LKH/tests/matlab_to_python/matlab_times2python

# If you don't use squeeze_me = True, then Pandas might not like 
# the arrays in the dictionary, because they look like an arrays
# of 1-element arrays.  squeeze_me=True should fix that.

mat_dict = loadmat('sine2.mat',squeeze_me=True)

# make a new dictionary with just dependent variables we want
# handle the time variable separately
my_dict = { k: mat_dict[k] for k in ['x','y']}

# The next few steps import the time
# we round to the nearest second and 719529 is the datenum value of the Unix epoch start (1970-01-01)
# i.e. in matlab: datenum(datetime(1970,01,01)) = 719529
matlab_datetimes = mat_dict['t']                                            # an array of datenums from matlab 
dti = pd.to_datetime(matlab_datetimes-719529,unit='d',utc=True).round('s')  # datetime index

# to querry the frequency in datetime index
pd.infer_freq(dti) # for sine2 returns 12D

 





return
### 
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