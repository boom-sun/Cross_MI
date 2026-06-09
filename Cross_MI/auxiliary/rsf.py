# Riemannian geometry-based spatial filter (RSF)
# Author: Pan Lincong
# Edition date: 5 Mar 2024

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.optimize import minimize
import scipy.linalg as la
from scipy.linalg import eigh
from pyriemann.utils.covariance import covariances
from pyriemann.utils.mean import mean_covariance
from pyriemann.spatialfilters import CSP
from pyriemann.utils.distance import distance

def optimizeRiemann(P1, P2, W0=None, N=8, maxiter=5000, collect_obj_values=False):
    M = P1.shape[0]
    obj_values = []

    if P1.shape[0] != P2.shape[0] or P1.shape[1] != P2.shape[1]:
        raise ValueError("The input data must have the same number of samples")
    if P1.shape[0] < N:
        raise ValueError("The number of samples is less than the number of filters")    

    if W0 is None:
        W0 = np.random.randn(M, N)
        W0, _ = np.linalg.qr(W0) 
    
    W0_flat = W0.ravel()
    
    def objFunc(W_flat):
        W = W_flat.reshape(M, -1)
        eigvals = eigh(W.T @ P1 @ W, W.T @ P2 @ W, eigvals_only=True)
        eigvals = np.clip(eigvals, a_min=1e-10, a_max=None) 
        return -np.sum(np.log(eigvals)**2)
    
    def callback(xk,state):
        obj_values.append(state.fun)
    
    result = minimize(objFunc, W0_flat, method='trust-constr', 
                      options={'maxiter': maxiter, 'disp': False}, 
                      callback=callback if collect_obj_values else None)
        
    W_opt = result.x.reshape(M, -1)
    d0 = -objFunc(W0_flat) 
    d1 = -objFunc(result.x) 
      
    return (W0 if d0 > d1 else W_opt), (obj_values if collect_obj_values else None)


class RSF(BaseEstimator, TransformerMixin):
    def __init__(self, dim=8, method='default', flag=False):
        """
        Initialize the RSF Transformer.

        Parameters:
        - dim (int, optional): Number of spatial filters to compute (default: 4).
        - method (str, optional): Filtering method ('default', 'csp', or 'riemann-csp').      
        """
        self.dim = dim
        self.method = method.lower() if method is not None else 'none'
        self.flag = flag

    def fit(self, X, y):
        """
        Fit the RSF Transformer to the data.

        Parameters:
        - X (array-like, shape [n_trials, n_channels, n_times]): EEG data.
        - y (array-like, shape [n_trials]): Class labels.

        Returns:
        - self: Fitted RSF Transformer instance.
        """
        if self.method != 'none':
            labeltype = np.unique(y)
            traincov = covariances(X, estimator='lwf')
            covm1 = mean_covariance(traincov[y == labeltype[0]], metric='riemann')
            covm2 = mean_covariance(traincov[y == labeltype[1]], metric='riemann')
        else:
            self.W = np.eye(X.shape[1])
            return self

        if self.method == 'csp':
            scaler = CSP(nfilter=self.dim, metric='euclid')
            CSPmodel = scaler.fit(traincov,y)
            W0 = CSPmodel.filters_.T
            self.W, self.obj_values = optimizeRiemann(covm1, covm2, W0=W0, 
                                                      collect_obj_values=self.flag)    
            
        elif self.method == 'riemann-csp':
            scaler = CSP(nfilter=self.dim, metric='riemann')
            CSPmodel = scaler.fit(traincov,y)
            W0 = CSPmodel.filters_.T
            self.W, self.obj_values = optimizeRiemann(covm1, covm2, W0=W0, 
                                                      collect_obj_values=self.flag)    
        else:
            self.W, self.obj_values = optimizeRiemann(covm1, covm2, N=self.dim, 
                                                      collect_obj_values=self.flag)

        return self

    def transform(self, X):
        """
        Transform the input EEG data using the learned RSF spatial filters.

        Parameters:
        - X (array-like, shape [n_trials, n_channels, n_times]): EEG data.

        Returns:
        - transformed_data (array-like, shape [n_trials, dim, n_times]):
        Transformed EEG data after applying RSF spatial filters.
        """
        if self.method != 'none':
            # Apply spatial filters using vectorized operations
            transformed_data = np.einsum('ij,kjl->kil', self.W.T, X)
        else:
            transformed_data = X

        return transformed_data
    

# 假设你有traindata, trainlabel, testdata作为输入
# rsf_transformer = RSF(dim=8, method='default')
# trainData_transformed = rsf_transformer.fit_transform(traindata, trainlabel)
# testData_transformed = rsf_transformer.transform(testdata)