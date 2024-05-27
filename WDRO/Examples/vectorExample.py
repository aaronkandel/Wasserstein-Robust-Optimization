from WDRO.Algorithm.DROalgorithm import DRO
from scipy.optimize import minimize_scalar
import numpy as np
import time
import copy



'''
Simple Example inverting running the DRO reformulation on arbitrary random data.

This script uses uniform data, but there are minimal assumptions required for the 
underlying distribution. For more information, please reference the following papers:


Mohajerin Esfahani, P., Kuhn, D. Data-driven distributionally robust optimization using the 
Wasserstein metric: performance guarantees and tractable reformulations. Math. Program. 
171, 115–166 (2018). https://doi.org/10.1007/s10107-017-1172-1.

C. Duan, W. Fang, L. Jiang, L. Yao and J. Liu, "Distributionally Robust Chance-Constrained 
Approximate AC-OPF With Wasserstein Metric," in IEEE Transactions on Power Systems, vol. 33, 
no. 5, pp. 4924-4936, Sept. 2018, doi: 10.1109/TPWRS.2018.2807623.

'''



def main():

	# Generate random data:
	num_samples = 500  # number of datums
	support = 10
	random_data = support*np.random.uniform(size=(1, num_samples))
	random_dat_1d = np.array(random_data)[0,:]
	

	# Convert to list of samples (needed for DRO package):
	random_data = random_data.tolist()[0]
	random_dat_1d = random_dat_1d.tolist()[0]


	# Demonstrate DRO package with Uniform synthetic data:
	# Invert empirical CDF:
	eta = 0.95 # probability threshold for chance constraint

	# DRO:
	beta = 0.9  # probability that ambiguity set includes true distribution

	DRO_object = DRO(random_data, eta, beta, verbosity=True, known_support=True, support=support)
	q = DRO_object.return_q()

	
	# Compare to inverting empirical CD:
	sort_samples = np.sort(random_data)
	invert_cdf_loc = int(np.floor(eta*num_samples))
	invert_cdf = sort_samples[invert_cdf_loc]

	print('The inverted CDF for the provided probability threshold is:')
	print('Offset (no DRO) = '+str([invert_cdf]))

	print('Using the -known support- radius option, the results have added conservatism.')
	print('There are currently instabilities with the scipy minimize_scalar function in this application. Troubleshooting is ongoing.')





if __name__ == "__main__":
    main()











