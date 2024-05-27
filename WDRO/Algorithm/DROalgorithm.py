from scipy.optimize import minimize_scalar
from collections import OrderedDict
import pickle
import os
import sys
import time
import numpy as np
import copy


class DRO(object):
    """
    Description: This class represents the object that calculates and provides
    access to the DRO offset.

    """

    def __init__(self, residuals: np.array, eta: float, beta: float, verbosity: bool):
        """
        Description: Initializes the DRO class object.
        
        Args:
            residuals (numpy.array): random samples comprising the empirical distribution
            eta (float): chance constraint risk metric
            beta (float): probability that the ambiguity set contains true distribution
            verbosity (bool): flag for printing results

        Returns:

        """

        self.residuals = residuals  # n x m, where n is the dimension of a sample, and
        # m is the total number of samples:
        self.n, self.m = self.residuals.shape
        self.eta = eta
        self.beta = beta
        self.numels = self.residuals
        self.verbosity = verbosity





    def norm_dat(self):
        """
        Description: Normalizes and centers the empirical distribution
        
        Args:

        Returns:

        """

        SIG = (self.residuals.std(axis=1))**2
        mu = np.reshape(np.mean(self.residuals, axis=1), (self.n, 1))
        self.mu = mu
        self.SIG = np.reshape(SIG, (self.n, 1))

        sigi = SIG**(-0.5)
        self.sigi = np.reshape(sigi, (self.n, 1))

        thet = np.multiply(self.residuals - self.mu, sigi[:,np.newaxis])


        self.thet = thet


    def radius_calc(self):
        """
        Description: Calculates the radius of the Wasserstein ambiguity set.
        
        Args:

        Returns:

        """

        def obj_c(alpha: float)-> float: 
            '''
            Objective function for radius calculation, found from:

            Chaoyue Zhao, Yongpei Guan, Data-driven risk-averse stochastic optimization 
            with Wasserstein metric, Operations Research Letters, Volume 46, Issue 2, 2018, 
            Pages 262-267, ISSN 0167-6377, https://doi.org/10.1016/j.orl.2018.01.011.

            Args: 
                alpha (float): decision variable

            Returns:
                J (float): objective function

            '''

            test = np.absolute(self.thet)

            J = np.sqrt(np.absolute( (1/(2*alpha))*(1+np.log(1/self.m*np.sum(np.exp(alpha*test**2)))  )))

            return J

        alphaX = minimize_scalar(obj_c, method = 'bounded', bounds = (0.001, 100), tol=1e-5)
        C = 2*alphaX.x
        Dd = 2*C

        self.epsilon = Dd*np.sqrt((2/self.m)*np.log10(1/(1-self.beta)))


    def h(self, sig: float, lam: float, epsilon: float, thet: np.array):
        '''
        Dual function for DRO reformulation, found from:

        C. Duan, W. Fang, L. Jiang, L. Yao and J. Liu, "Distributionally Robust Chance-Constrained 
        Approximate AC-OPF With Wasserstein Metric," in IEEE Transactions on Power Systems, vol. 33, 
        no. 5, pp. 4924-4936, Sept. 2018, doi: 10.1109/TPWRS.2018.2807623.


        Args: 
            sig (float): decision variable
            lam (float): decision variable
            epsilon (float): Wasserstein radius
            thet (np.array): empirical distribution

        Returns:
            h (float): objective function

        '''
        initi = lam*epsilon
        result = 0
        for k in range(self.m-2): 
            sumterm = np.maximum((1 - (lam*np.maximum(  (sig-np.linalg.norm(thet[:, k], ord=np.inf))  , 0))), 0)  

            result += sumterm

        return initi + result/self.m



    def trisearch(self, lb, ub, epsilon, thet, sigmax):
        '''
        Scalar convex search for DRO reformulation. Calculates sigma from:

        C. Duan, W. Fang, L. Jiang, L. Yao and J. Liu, "Distributionally Robust Chance-Constrained 
        Approximate AC-OPF With Wasserstein Metric," in IEEE Transactions on Power Systems, vol. 33, 
        no. 5, pp. 4924-4936, Sept. 2018, doi: 10.1109/TPWRS.2018.2807623.

        Args: 
            lb (float): starting lower bound
            ub (float): starting upper bound
            epsilon (float): Wasserstein radius
            thet (np.array): empirical distribution
            sigmax (float): assumed largest sigma value

        '''

        sigu = 0
        sighat = sigmax

        # Conduct trisection search over convex function h to find sigma:
        while (sighat - sigu) > 1e-5:
            sig = (sighat + sigu) / 2.0
            ub2 = ub
            lb2 = lb
            while (ub2-lb2) > 1e-4:
                x_u = lb2 + (ub2-lb2)/3
                x_v = lb2 + 2*(ub2-lb2)/3
                h_u = self.h(sig,x_u,self.epsilon, self.thet)
                h_v = self.h(sig,x_v,self.epsilon, self.thet)
                if h_u < h_v:
                    ub2 = x_v
                else:
                    lb2 = x_u
            gamma = h_v
            if gamma > self.eta:
                sigu = sig
            else:
                sighat = sig
        self.sigma = sig
 

    def return_q(self):
        '''
        Calculates and stores the DRO offset.

        '''

        if self.verbosity:
            print('#########################################')
            print('###   Running DRO Reformulation...   ####')
            print('#########################################')
            
            print('#########################################')
            print('###        Normalizing Data...        ###')
            print('#########################################')
            self.norm_dat()
            print('\n')


            print('#########################################')
            print('###       Calculating Radius...       ###')
            print('#########################################')
            print('\n')
            self.radius_calc()
            print('Wasserstein Radius = ' + str(self.epsilon))
            print('\n')

            print('#########################################')
            print('###       Running sigma Search:       ###')
            print('#########################################')
            self.trisearch(0.13, 100, self.epsilon, self.thet, 100)
            print('\n')
            print('sigma = '+str(self.sigma))
            # print(self.sigma)
            # print(self.mu)
            # print(self.sigi)
            print('\n')
            q = np.absolute((self.SIG**0.5)*self.sigma + self.mu)
            print('Offset = '+str(q))
            self.q = q
        else: 

            self.norm_dat()

            self.radius_calc()

            self.trisearch(0.13, 100, self.epsilon, self.thet, 100)

            q = np.absolute((self.SIG**0.5)*self.sigma + self.mu)
            self.q = q

        return q










