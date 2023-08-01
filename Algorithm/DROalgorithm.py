from scipy.optimize import minimize_scalar
from collections import OrderedDict
import pickle
import os
import sys
import time
import numpy as np
import copy

# This code is not optimized, only god can judge me.


class DRO(object):
    def __init__(self, residuals, eta, beta):
        self.residuals = residuals
        self.eta = eta
        self.beta = beta


    def norm_dat(self):
        SIG = (self.residuals.std(axis=1))**2
        mu = np.mean(self.residuals, axis=1)
        # print(mu, SIG)
        self.SIG = np.expand_dims(SIG, axis=1)
        self.mu = np.expand_dims(mu, axis=1)
        sigi = SIG**(-0.5)
        self.sigi = np.expand_dims(sigi, axis=1)
        # print(sigi.shape)

        t1 = np.multiply(self.residuals[0,:]-mu[0], sigi[0])
        t2 = np.multiply(self.residuals[1,:]-mu[1], sigi[1])
        thet = np.vstack((t1, t2))


        self.thet = thet
        dummy, self.N = self.thet.shape

    def radius_calc(self):
        def obj_c(alpha):
            test = np.absolute(self.thet-self.mu)

            J = np.sqrt(np.absolute( (1/(2*alpha))*(1+np.log(1/self.N*np.sum(np.exp(alpha*test**2)))  )))

            return J

        alphaX = minimize_scalar(obj_c, method = 'bounded', bounds = (0.001, 100))
        C = 2*alphaX.x
        Dd = 2*C

        dummy, N = self.residuals.shape
        self.epsilon = Dd*np.sqrt((2/self.N)*np.log10(1/(1-self.beta)))


    def h(self,sig,xrc,epsilon,thet):
        result = xrc*epsilon

        for k in range(self.N-2): 
            sumterm = np.maximum((1 - (xrc*np.maximum(  (sig-np.linalg.norm(thet[:, k], ord=np.inf))  , 0)))/self.N, 0)  

            result += sumterm
        return result



    def trisearch(self, lb, ub, epsilon, thet, sigmax):
        sigu = 0
        sighat = sigmax
        while (sighat - sigu) > 1e-5:
            sig = (sighat + sigu) / 2.0
            #f = lambda x : self.h(sig,x,epsilon,thet)
            ub2 = ub
            lb2 = lb
            while (ub2-lb2) > 1e-3:
                x_u = lb2 + (ub2-lb2)/3
                x_v = lb2 + 2*(ub2-lb2)/3
                h_u = self.h(sig,x_u,self.epsilon,self.thet)
                h_v = self.h(sig,x_v,self.epsilon,self.thet)
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
        self.norm_dat()
        self.radius_calc()
        self.trisearch(0, 100, self.epsilon, self.thet, 100)
        # print(self.sigma)
        q = np.absolute((self.sigi)*self.sigma + self.mu)
        self.q = q
        return q










