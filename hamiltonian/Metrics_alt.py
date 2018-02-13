__author__ = 'nt357'
import numpy as np
from statsmodels.tsa import stattools # Remember to install this package
import matplotlib.pyplot as plt
from rpy2 import robjects

def autocorr(x):
    """
    Computes the ( normalised) auto-correlation function of a
    one dimensional sequence of numbers.
    
    Utilises the numpy correlate function that is based on an efficient
    convolution implementation.
    
    Inputs:
    x - one dimensional numpy array
    
    Outputs:
    Vector of autocorrelation values for a lag from zero to max possible
    """
    
    # normalise, compute norm
    xunbiased = x - np.mean(x)
    xnorm = np.sum(xunbiased ** 2)
    
    # convolve with itself
    acor = np.correlate(xunbiased, xunbiased, mode='full')
    
    # use only second half, normalise
    acor = acor[len(acor) / 2:] / xnorm
    
    return acor

class RCodaTools(object):

    @staticmethod
    def ess_coda(data):
        """
        Computes the effective samples size of a 1d-array using R-coda via
        an external R call. The python package rpy2 and the R-library
        "library(coda)" have to be installed. Inspired by Charles Blundell's
        neat little python script :)
        """
        robjects.r('library(coda)')
        r_ess = robjects.r['effectiveSize']
        data = robjects.r.matrix(robjects.FloatVector(data), nrow=len(data))
        return r_ess(data)[0]

    @staticmethod
    def ess_coda_vec(samples):

        sample_array = np.array(samples)
        l, h = sample_array.shape

        ess_vec = np.zeros(h)
        for i in range(h):
            ess_vec[i] = RCodaTools.ess_coda(sample_array[:, i])

        return ess_vec

    @staticmethod
    def geweke(data):

        robjects.r('library(coda)')
        r_geweke = robjects.r['geweke.diag']
        data = robjects.r.matrix(robjects.FloatVector(data), nrow=len(data))

        return r_geweke(data)[0]
        
    @staticmethod
    def gelman(data):

        robjects.r('library(coda)')
        r_geweke = robjects.r['gelman.diag']
        data = robjects.r.matrix(robjects.FloatVector(data), nrow=len(data))

        return r_geweke(data)[0]
        
