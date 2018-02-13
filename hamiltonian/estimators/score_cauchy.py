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
import scipy

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
    KK = K_XY * K_XY	# element-wise product
    KK_odot_XY = KK * XY
    KKK = KK * K_XY
    
    b = 4 * KKK + (X.shape[1] - 4) * KK
    b = np.sum(b, 1) * sigma	# shape (K)
    C = np.dot(KK, KK) * XY + np.dot(KK, np.dot(np.diag(np.diag(XY)), KK)) \
        - np.dot(KK_odot_XY, KK) - np.dot(KK, KK_odot_XY)
    
    return b, C

def Cauchy_kernel(X, Y, sigma=1.0):
    assert X.shape[1] == Y.shape[1]
    sq_dists = scipy.spatial.distance.cdist(X, Y, 'sqeuclidean')
    return (1.0 + sq_dists / 2 / sigma) ** -1.0

def fit(X, Y, sigma, lmbda, K=None):
         
        # compute kernel matrix if needed
        if K is None:
            K = Cauchy_kernel(X, Y, sigma=sigma)
                
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
        K_XY = Cauchy_kernel(X, Y, sigma=sigma)
    
    if K is None and lmbda > 0:
        if X is Y:
            K = K_XY
        else:
            K = Cauchy_kernel(X, sigma=sigma)
    
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

class KernelExpScore(EstimatorBase):
    def __init__(self, sigma, lmbda, D, N, beta = -0.49):
        self.sigma = sigma
        self.lmbda = lmbda
        self.D = D
        self.N = N
        self.beta = beta
        
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
        print "using Cauchy kernel..."
        self.K = Cauchy_kernel(self.X, self.X, sigma=self.sigma)	 # shape (K, K)
        return fit(self.X, self.X, self.sigma, self.lmbda, self.K)
    
    def log_pdf(self, x):
        assert_array_shape(x, ndim=1, dims={0: self.D})
        
        k = Cauchy_kernel(self.X, x.reshape(1, self.D), self.sigma)[:, 0]
        return np.dot(self.alpha, k)
    
    def grad(self, x):
        assert_array_shape(x, ndim=1, dims={0: self.D})
        KXx = Cauchy_kernel(self.X, x[np.newaxis, :], sigma=self.sigma)    
        xX_grad = (self.X - x) / self.sigma * KXx**2
        gradient = np.dot(self.alpha, xX_grad)
        return gradient
    
    def log_pdf_multiple(self, X):
        assert_array_shape(X, ndim=2, dims={1: self.D})
        
        k = Cauchy_kernel(self.X, X, self.sigma)
        return np.dot(self.alpha, k)
    
    def objective(self, X):
        assert_array_shape(X, ndim=2, dims={1: self.D})
        
        return objective(self.X, X, self.sigma, self.lmbda, self.alpha, self.K)

    def get_parameter_names(self):
        return ['sigma', 'lmbda']

class KernelExpScoreAdaptive(KernelExpScore):
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
