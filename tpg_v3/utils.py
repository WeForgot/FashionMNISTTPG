import random

from numba import njit
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

@njit
def pad_array(A, length):
	arr = np.zeros(length)
	arr[:len(A)] = A
	return arr