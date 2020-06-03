import random

import numpy as np

"""
Various useful functions for use within TPG, and for using TPG.
"""

"""
Coin flips, at varying levels of success based on prob.
"""
def flip(prob):
	return random.uniform(0.0,1.0) < prob

def sign(number):
	return -1 if number < 0 else 1

# From https://stackoverflow.com/questions/38170188/generate-a-n-dimensional-array-of-coordinates-in-numpy
def ndim_grid(start,stop):
    # Set number of dimensions
    ndims = len(start)

    # List of ranges across all dimensions
    L = [np.arange(start[i],stop[i]) for i in range(ndims)]

    # Finally use meshgrid to form all combinations corresponding to all 
    # dimensions and stack them as M x ndims array
    return np.hstack((np.meshgrid(*L))).swapaxes(0,1).reshape(ndims,-1).T