import numpy as np
from scipy.spatial.distance import pdist, squareform

"""
Compute the KSD divergence using samples, adapted from the theano code
"""

def KSD(z, Sqx):

    # compute the rbf kernel
    K, dimZ = z.shape
    sq_dist = pdist(z)
    pdist_square = squareform(sq_dist)**2
    # use median
    median = np.median(pdist_square)
    h_square = 0.5 * median / np.log(K+1.0)
    Kxy = np.exp(- pdist_square / h_square / 2.0)

    # now compute KSD
    Sqxdy = np.dot(Sqx, z.T) - np.tile(np.sum(Sqx * z, 1, keepdims=True), (1, K))
    Sqxdy = -Sqxdy / h_square

    dxSqy = Sqxdy.T
    dxdy = -pdist_square / (h_square ** 2) + dimZ / h_square
    # M is a (K, K) tensor
    M = (np.dot(Sqx, Sqx.T) + Sqxdy + dxSqy + dxdy) * Kxy

    # the following for U-statistic
    M2 = M - np.diag(np.diag(M))
    return np.sum(M2) / (K * (K - 1))
