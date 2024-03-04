% In Matlab, create a sample 20-day time series

t = datenum(2013,8,15,17,11,31) + [0:0.1:20];
x = sin(t)
y = cos(t)
plot(t,x)
datetick

cd /Users/katehewett/Documents/LKH/tests/matlab_to_python/matlab_times2python
save sine.mat

clear all 

% put one in with fequency of 12 days
t = datenum(datetime(2013,8,15):days(12):datetime(2022,8,15))
x = sin(t)
y = cos(t)
plot(t,x)
datetick

cd /Users/katehewett/Documents/LKH/tests/matlab_to_python/matlab_times2python
save sine2.mat
