from abc import abstractmethod

from kernel_exp_family.estimators.estimator_oop import EstimatorBase
try:
    from kernel_exp_family.estimators.parameter_search_bo import BayesOptSearch
except ImportError:
    print("Could not import BayesOptSearch.")
from kernel_exp_family.kernels.kernels import gaussian_kernel, \
    gaussian_kernel_grad, theano_available
from kernel_exp_family.tools.assertions import assert_array_shape
from kernel_exp_family.tools.log import Log
import numpy as np


if theano_available:
    from kernel_exp_family.kernels.kernels import gaussian_kernel_hessian_theano, \
        gaussian_kernel_third_order_derivative_tensor_theano
                
logger = Log.get_logger()

def compute_b_and_C(X, Y, K_XY, sigma):
    assert X.shape[1] == Y.shape[1]
    assert K_XY.shape[0] == X.shape[0]
    assert K_XY.shape[1] == Y.shape[0]
    
    XY = np.sum(np.expand_dims(X, 1)*Y, 2)	# (K, K)
    diag_XY = np.diag(np.diag(XY))
    KK_XY = np.dot(K_XY, K_XY)
    K_odot_XY = K_XY * XY
    KK_odot_XY = KK_XY * XY
        
    b = np.dot(K_XY, np.dot(diag_XY, K_XY)) + KK_odot_XY \
                - np.dot(K_odot_XY, K_XY) - np.dot(K_XY, K_odot_XY)
    b = np.sum(b, 1) #* sigma / 2.	# shape (K)
    C = XY * np.dot(K_XY, KK_XY) + np.dot(K_XY, np.dot(K_odot_XY, K_XY)) \
                - np.dot(KK_odot_XY, K_XY) - np.dot(K_XY, KK_odot_XY)	# shape (K, K)
    
    return b, C

def fit(X, Y, sigma, lmbda, K=None):
         
        # compute kernel matrix if needed
        if K is None:
            K = gaussian_kernel(X, Y, sigma=sigma)
                
        # compute helper matrices
        b, C = compute_b_and_C(X, Y, K, sigma)        

        # solve regularised linear system
        a = np.linalg.solve(C + np.eye(len(C)) * lmbda, b)
        
        return a
    
def objective(X, Y, sigma, lmbda, alpha, K=None, K_XY=None, b=None, C=None):
    # restrict shape
    #if X.shape[0] > 100:
    #    ind = np.random.permutation(range(X.shape[0]))[:100]
    #    X = X[ind]
    if X.shape[0] != Y.shape[0]:
        Y = np.copy(X)
    if K_XY is None:
        K_XY = gaussian_kernel(X, Y, sigma=sigma)
    
    if K is None and lmbda > 0:
        if X is Y:
            K = K_XY
        else:
            K = gaussian_kernel(X, sigma=sigma)
    
    if b is None or C is None:
        b, C = compute_b_and_C(X, Y, K_XY, sigma)
    
    NX = len(X)
    first = 2. / (NX * sigma) * alpha.dot(b)
    if lmbda > 0:
        second = 2. / (NX * sigma ** 2) * alpha.dot(
                                                    (C + (K + np.eye(len(C))) * lmbda).dot(alpha)
                                                    )
    else:
        second = 2. / (NX * sigma ** 2) * alpha.dot((C).dot(alpha))
    J = first + second
    return J

class KernelExpStein(EstimatorBase):
    def __init__(self, sigma, lmbda, D, N):
        self.sigma = sigma
        self.lmbda = lmbda
        self.D = D
        self.N = N
        
        # initial RKHS function is flat
        self.alpha = np.zeros(0)
        self.X = np.zeros((0, D))
    
    def fit(self, X):
        assert_array_shape(X, ndim=2, dims={1: self.D})
        
        # sub-sample if data is larger than previously set N
        if len(X) > self.N:
            inds = np.random.permutation(len(X))[:self.N]
            self.X = X[inds]
        else:
            self.X = np.copy(X)
            
        self.alpha = self.fit_wrapper_()
    
    @abstractmethod
    def fit_wrapper_(self):
        self.K = gaussian_kernel(self.X, sigma=self.sigma)
        return fit(self.X, self.X, self.sigma, self.lmbda, self.K)
    
    def log_pdf(self, x):
        assert_array_shape(x, ndim=1, dims={0: self.D})
        
        k = gaussian_kernel(self.X, x.reshape(1, self.D), self.sigma)[:, 0]
        return np.dot(self.alpha, k)
    
    def grad(self, x):
        assert_array_shape(x, ndim=1, dims={0: self.D})    
        k = gaussian_kernel_grad(x, self.X, self.sigma)
        gradient = np.dot(self.alpha, k)
        return gradient
    
    if theano_available:
        def hessian(self, x):
            """
            Computes the Hessian of the learned log-density function.
            
            WARNING: This implementation slow, so don't call repeatedly.
            """
            assert_array_shape(x, ndim=1, dims={0: self.D})
            
            H = np.zeros((self.D, self.D))
            for i, a in enumerate(self.alpha):
                H += a * gaussian_kernel_hessian_theano(x, self.X[i], self.sigma)
        
            return H
        
        def third_order_derivative_tensor(self, x):
            """
            Computes the third order derivative tensor of the learned log-density function.
            
            WARNING: This implementation is slow, so don't call repeatedly.
            """
            assert_array_shape(x, ndim=1, dims={0: self.D})
            
            G3 = np.zeros((self.D, self.D, self.D))
            for i, a in enumerate(self.alpha):
                G3 += a * gaussian_kernel_third_order_derivative_tensor_theano(x, self.X[i], self.sigma)
        
            return G3
    
    def log_pdf_multiple(self, X):
        assert_array_shape(X, ndim=2, dims={1: self.D})
        
        k = gaussian_kernel(self.X, X, self.sigma)
        return np.dot(self.alpha, k)
    
    def objective(self, X):
        assert_array_shape(X, ndim=2, dims={1: self.D})
        
        return objective(self.X, X, self.sigma, self.lmbda, self.alpha, self.K)

    def get_parameter_names(self):
        return ['sigma', 'lmbda']

class KernelExpSteinAdaptive(KernelExpStein):
    def __init__(self, sigma, lmbda, D, N,
                 num_initial_evaluations=3, num_evaluations=3, minimum_size_learning=100,
                 num_initial_evaluations_relearn=1, num_evaluations_relearn=1,
                 param_bounds={'sigma': [-3, 3]}):
        KernelExpStein.__init__(self, sigma, lmbda, D, N)
        
        self.bo = None
        self.param_bounds = param_bounds
        self.num_initial_evaluations = num_initial_evaluations
        self.num_iter = num_evaluations
        self.minimum_size_learning = minimum_size_learning
        
        self.n_initial_relearn = num_initial_evaluations_relearn
        self.n_iter_relearn = num_evaluations_relearn
        
        self.learning_parameters = False
        
    def fit(self, X):
        # avoid infinite recursion from x-validation fit call
        if not self.learning_parameters and len(X) >= self.minimum_size_learning:
            self.learning_parameters = True
            if self.bo is None:
                logger.info("Bayesian optimisation from scratch.")
                self.bo = BayesOptSearch(self, X, self.param_bounds, num_initial_evaluations=self.num_initial_evaluations)
                best_params = self.bo.optimize(self.num_iter)
            else:
                logger.info("Bayesian optimisation using prior model.")
                self.bo.re_initialise(X, self.n_initial_relearn)
                best_params = self.bo.optimize(self.n_iter_relearn)
            
            self.set_parameters_from_dict(best_params)
            self.learning_parameters = False
            logger.info("Learnt %s" % str(self.get_parameters()))
        
        # standard fit call from superclass
        KernelExpStein.fit(self, X)
