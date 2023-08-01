from WDRO.Algorithm.DROalgorithm import DRO
from scipy.optimize import minimize_scalar
import numpy as np
import time
import copy



'''
Simple Example inverting running the DRO reformulation on arbitrary random data.

This script uses Gaussian data, but there are minimal assumptions required for the 
underlying distribution. For more information, please reference the following papers:


Mohajerin Esfahani, P., Kuhn, D. Data-driven distributionally robust optimization using the 
Wasserstein metric: performance guarantees and tractable reformulations. Math. Program. 
171, 115–166 (2018). https://doi.org/10.1007/s10107-017-1172-1.

C. Duan, W. Fang, L. Jiang, L. Yao and J. Liu, "Distributionally Robust Chance-Constrained 
Approximate AC-OPF With Wasserstein Metric," in IEEE Transactions on Power Systems, vol. 33, 
no. 5, pp. 4924-4936, Sept. 2018, doi: 10.1109/TPWRS.2018.2807623.

'''



# Generate random data:
num_samples = 150  # number of datums
random_data = np.random.normal(loc=0.0, scale=1.00, size = (2, num_samples))
random_data[1,:] = random_data[1,:] # +2  # optional offset
random_dat_1d = random_data[0,:]
random_dat_1d2 = random_data[1,:]
random_data = np.abs(random_data)

# Invert empirical CDF:
eta = 0.95  # probability threshold for chance constraint

sort_samples = np.sort(random_dat_1d)
invert_cdf_loc = int(np.floor(eta*num_samples))

ss2 = np.sort(random_dat_1d2)
icdf2 = int(np.floor(eta*num_samples))

invert_cdf = sort_samples[invert_cdf_loc]
ic2 = ss2[invert_cdf_loc]

invert_cdf = np.array([[invert_cdf],[ic2]])



# DRO:
beta = 0.99  # probability that ambiguity set includes true distribution

DRO_object = DRO(random_data, eta, beta, verbosity=True)
q = DRO_object.return_q()

print('The inverted CDF for the provided probability threshold is:')
print('Offset (no DRO) = '+str(invert_cdf))  # 'F^{-1}(\rho) = '+str(









