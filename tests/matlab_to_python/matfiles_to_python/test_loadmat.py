# test to load .mat to python 
#
# ran 
# script to save some data in a .mat
# a = [1, 2, 3; 4, 5, 6];
# S.b = [7, 8, 9; 10, 11, 12];
# M(1).c = [2, 4, 6; 8, 10, 12];
# M(2).c = [1, 3, 5; 7, 9, 11];
# save('data.mat','a','S','M')
# data located here: /Users/katehewett/Documents/LKH_data/test/matlab_to_python/matfiles_to_python/data.mat

import OS
import scipy.io as spio

os.getcwd('/Users/katehewett/Documents/LKH_data/test/matlab_to_python/matfiles_to_python')
mat = spio.loadmat('data.mat', squeeze_me=True)

a = mat['a'] # array 
S = mat['S'] # structure
M = mat['M'] # array of structures

print a[:,:]
print S['b'][()][:,:] # structures need [()]
print M[0]['c'][()][:,:]
print M[1]['c'][()][:,:]