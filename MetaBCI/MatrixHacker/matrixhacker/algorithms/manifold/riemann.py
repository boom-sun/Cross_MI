"""Most riemann methods are herited from Alexandre Barachant's pyRiemann package.

Some signatures are modified and simplified according to my personal needs.
If you prefer original methods, see https://github.com/alexandrebarachant/pyRiemann for more details.

"""
from functools import partial

import numpy as np
import autograd.numpy as anp
from scipy.linalg import eigvalsh, inv, eigh, svd
from scipy.linalg import sqrtm as scipy_sqrtm
from scipy import sparse
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin, ClusterMixin, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cluster.k_means_ import _init_centroids
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.utils.extmath import softmax
from joblib import Parallel, delayed

from pymanopt.manifolds.group import SpecialOrthogonalGroup
from pymanopt import Problem
from pymanopt.solvers import SteepestDescent

from ..utils.base import (sqrtm, invsqrtm, logm, expm, powm, whitenm, unwhitenm)


def logmap(Pi, P):
    """Logarithm map.对数映射。

    Logarithm map projects :math:`\mathbf{P}_i \in \mathcal{M}` to the tangent space point 
    :math:`\mathbf{S}_i \in \mathcal{T}_{\mathbf{P}} \mathcal{M}` at :math:`\mathbf{P} \in \mathcal{M}`.
    
    Parameters
    ----------
    Pi : ndarray
        SPD matrix.
    P : ndarray
        Reference point.
        
    Returns
    -------
    Si : ndarray
        Tangent space point (in matrix form).切空间点(矩阵形式)。
    """
    P12 = sqrtm(P)  #计算矩阵的平方根，即X*X=A。
    iP12 = invsqrtm(P)  #返回协方差矩阵的逆矩阵的平方根。
    wPi = iP12@Pi@iP12  #@为叉乘，就是数学中学的乘法
    Si = P12@logm(wPi)@P12
    return Si


def expmap(Si, P):
    """Exponential map.指数映射。

    Exponential map projects :math:`\mathbf{S}_i \in \mathcal{T}_{\mathbf{P}} \mathcal{M}` bach to the manifold
    :math:`\mathcal{M}`.
    
    Parameters
    ----------
    Si : ndarray
        Tangent space point (in matrix form).
        
    P : ndarray
        Reference point.
    
    Returns
    -------
    Pi : ndarray
        SPD matrix.
    """
    P12 = sqrtm(P)
    iP12 = invsqrtm(P)
    wSi = iP12@Si@iP12
    Pi = P12@expm(wSi)@P12
    return Pi


def geodesic(P1, P2, t):
    """Geodesic.
    任意两个SPD矩阵之间的测地线曲线
    The geodesic curve between any two SPD matrices :math:`\mathbf{P}_1,\mathbf{P}_2 \in \mathcal{M}`.

    Parameters
    ----------
    P1 : ndarray
        SPD matrix.
    P2 : ndarray
        SPD matrix, the same shape of P1.
    t : float
        :math:`0 \leq t \leq 1`.
    
    Returns
    -------
    phi : ndarray
        SPD matrix on the geodesic curve between P1 and P2.
    """
    P12 = sqrtm(P1)
    iP12 = invsqrtm(P1)
    wP2 = iP12@P2@iP12
    phi = P12@powm(wP2, t)@P12  #只找到pow函数，是返回 x^y（x的y次方） 的值
    return phi


def _get_sample_weight(sample_weight, N):
    """Get the sample weights.
    得到样本权重。
    If none provided, weights init to 1. otherwise, weights are normalized.
    如果没有提供，则将init的权重设为1。否则，权值归一化。
    """
    if sample_weight is None:
        sample_weight = np.ones(N)
    if len(sample_weight) != N:
        raise ValueError("len of sample_weight must be equal to len of data.")
    sample_weight /= np.sum(sample_weight)
    return sample_weight


def distance_riemann(A, B):
    """Riemannian distance between two covariance matrices A and B.

    Parameters
    ----------
    A : ndarray
        First positive-definite matrix, shape (n_trials, n_channels, n_channels) or (n_channels, n_channels).
    B : ndarray
        Second positive-definite matrix.

    Returns
    -------
    ndarray | float
        Riemannian distance between A and B.

    Notes
    -----
    .. math::
            d = {\left( \sum_i \log(\lambda_i)^2 \\right)}^{-1/2}

    where :math:`\lambda_i` are the joint eigenvalues of A and B.A和B的联合特征值
    """
    if A.ndim == 2:
        dist = np.sqrt((np.log(eigvalsh(A, B))**2).sum())  #eigvalsh函数：求解复 Hermitian 或实对称矩阵的标准或广义特征值问题。
    elif A.ndim == 3:
        dist = np.array([np.sqrt((np.log(eigvalsh(tmp, B))**2).sum()) for tmp in A])
    return dist


def mean_riemann(covmats, tol=1e-8, maxiter=100, init=None, sample_weight=None):
    """
    --Return the mean covariance matrix according to the Riemannian metric.
    根据黎曼度规返回平均协方差矩阵。
    Parameters--------
    covmats : ndarray
        Covariance matrices set, shape (n_trials, n_channels, n_channels).
        协方差矩阵集，形状(n_trials, n_channels, n_channels)。
    tol : float, optional
        The tolerance to stop the gradient descent (default 1e-8).
        停止梯度下降的公差(默认1e-8)。
    maxiter : int, optional
        The maximum number of iteration (default 50).
        最大迭代次数(默认为50)。
    init : None|ndarray, optional
        A covariance matrix used to initialize the gradient descent (default None), if None the arithmetic mean is used.
        一个协方差矩阵，用于初始化梯度下降(默认为None)，如果为None则使用算术平均值。
    sample_weight : None|ndarray, optional
        The weight of each sample (efault None), if None weights are 1 otherwise weights are normalized.

    Returns
    -------
    C : ndarray
        The Riemannian mean covariance matrix.
        黎曼平均协方差矩阵。
    Notes
    -----
    The procedure is similar to a gradient descent minimizing the sum of riemannian distance to the mean.
    该过程类似于使黎曼距离均值的和最小化的梯度下降。
    .. math::
        \mathbf{C} = \\arg \min{(\sum_i \delta_R ( \mathbf{C} , \mathbf{C}_i)^2)}

    where :math:\delta_R is riemann distance.
    """
    # init
    sample_weight = _get_sample_weight(sample_weight, len(covmats))
    Nt, Ne, Ne = covmats.shape
    if init is None:
        C = np.mean(covmats, axis=0)
    else:
        C = init
    k = 0
    nu = 1.0
    tau = np.finfo(np.float64).max  #finfo()函数显示浮点类型的机器限制。
    crit = np.finfo(np.float64).max
    # stop when J<10^-9 or max iteration = 50
    while (crit > tol) and (k < maxiter) and (nu > tol):
        k = k + 1
        C12 = sqrtm(C)
        Cm12 = invsqrtm(C)
        J = np.zeros((Ne, Ne))

        for index in range(Nt):
            tmp = np.dot(np.dot(Cm12, covmats[index, :, :]), Cm12) #dot函数：矩阵乘法
            J += sample_weight[index] * logm(tmp)

        crit = np.linalg.norm(J, ord='fro')  #求范数，ord为范数类型，fro为求F范数
        h = nu * crit
        C = np.dot(np.dot(C12, expm(nu * J)), C12)
        if h < tau:
            nu = 0.95 * nu
            tau = h
        else:
            nu = 0.5 * nu

    return C


def tangent_space(Pis, P):
    """Logarithm map projects SPD matrices to the tangent vectors.
    对数映射将SPD矩阵投影到切向量上。
    
    Parameters
    ----------
    Pis : ndarray
        SPD matrices, shape (n_trials, n_channels, n_channels).
    P : ndarray
        Reference point.
    
    Returns
    -------
    Sis : ndarray
        Tangent vectors, shape (n_trials, n_channels*(n_channels+1)/2).
    """
    n_trials, n_channels, n_channels = Pis.shape
    n_features = int(n_channels*(n_channels+1)/2)

    idx = np.triu_indices_from(P)
    #进行upper操作
    coeffs = (np.sqrt(2) * np.triu(np.ones((n_channels, n_channels)), 1) + np.eye(n_channels))[idx]
                            #triu函数：数组的上三角。返回矩阵的副本，其中第k个对角线以下的元素为零。
                            #eye函数：返回的是一个二维2的数组(N,M)，对角线的地方为1，其余的地方为0
    Sis = np.zeros((n_trials, n_features))

    # P12 = sqrtm(P)
    iP12 = invsqrtm(P)
    for i, Pi in enumerate(Pis):  #enumerate函数的基本应用就是用来遍历一个集合对象，它在遍历的同时还可以得到当前元素的索引位置。
        wPi = iP12@Pi@iP12
        Si = logm(wPi)
        Sis[i] = np.multiply(coeffs, Si[idx])  #矩阵对应元素相乘
    
    return Sis
    

def untangent_space(Sis, P):
    """Exponential map projects tangent vectors back to the SPD matrices.
    指数映射将切向量投影回SPD矩阵。
    
    Parameters
    ----------
    Sis : ndarray
        Tangent vectors, shape (n_trials, n_channels*(n_channels+1)/2).
    P : ndarray
        Reference point.
    
    Returns
    -------
    Pis : ndarray
        SPD matrices, shape (n_trials, n_channels, n_channels).
    """
    n_trials, n_features = Sis.shape
    n_channels = int((np.sqrt(1 + 8 * n_features) - 1) / 2)
    
    idx = np.triu_indices_from(P)
    # didx = np.diag_indices(n_channels)
    Pis = np.zeros((n_trials, n_channels, n_channels))
    Pis[:, idx[0], idx[1]] = Sis

    P12 = sqrtm(P)
    # iP12 = invsqrtm(P)
    for i, Pi in enumerate(Pis):
        triuc = np.triu(Pi, 1) / np.sqrt(2)
        Pi = (np.diag(np.diag(Pi)) +  triuc + triuc.T)
        Pis[i] = P12@expm(Pi)@P12
    return Pis


def pt_tangent(A, B, Sbs):
    """Parallel transport in the tangent space.
    
    Parameters
    ----------
    A : ndarray
        SPD matrix.
    B : ndarray
        SPD matrix
    Sbs : ndarray
        The tangent vector (matrix form) in the tangent space of manifold B, 
        shape (n_trials, n_channels, n_channels) or (n_channels, n_channels).
        在流形B的切空间中的切向量(矩阵形式)，shape (n_trials, n_channels, n_channels)或(n_channels, n_channels)。
    
    Returns
    -------
    Sas : ndarray
        The tangent vector (matrix form) in the tangent space of manifold A.
    """
    Sbs = Sbs.reshape(-1, *Sbs.shape[-2:])  # 列表前面加星号作用是将列表中所有元素解开成独立的参数，传入函数，参数数量等于len(data)
    n_trials, _, _ = Sbs.shape
    E = scipy_sqrtm(A@inv(B)) #scipy_sqrtm矩阵平方根  inv(X)：返回X的逆矩阵

    Sas = np.zeros_like(Sbs)
    for i, Sb in enumerate(Sbs):
        Sas[i] = E@Sb@E.T

    if n_trials == 1:
        Sas = Sas[0]
    
    return Sas


def pt_manifold(A, B, Pbs):
    """Parallel transport in the manifold space.
    
    Parameters
    ----------
    A : ndarray
        SPD matrix.
    B : ndarray
        SPD matrix
    Pbs : ndarray
        SPD matrix in the manifold B, shape (n_trials, n_channels, n_channels) or (n_channels, n_channels).
    
    Returns
    -------
    Pas : ndarray
        SPD matrix in the manifold A.
    """
    Pbs = Pbs.reshape(-1, *Pbs.shape[-2:])
    n_trials, _, _ = Pbs.shape
    E = scipy_sqrtm(A@inv(B))
    
    Pas = np.zeros_like(Pbs)
    for i, Pb in enumerate(Pbs):
        Pas[i] = E@Pb@E.T

    if n_trials == 1:
        Pas = Pas[0]

    return Pas


def recenter(C, M):
    """Re-center.
    
    Re-center :math:`\mathbr{C} \in \mathcal{M}` to the identity centroid.

    Parameters
    ----------
    C : ndarray
        SPD matrices, shape (n_trials, n_channels, n_channles) or (n_channels, n_channels).
    M : ndarray
        The centroid of manifold.
    
    Returns
    -------
    Cr : ndarray
        Re-centered matrices.
    """
    C = C.reshape(-1, *C.shape[-2:])
    n_trials = len(C)

    Cr = np.zeros_like(C)

    iM12 = invsqrtm(M)
    for i, Ci in enumerate(C):
        Cr[i] = iM12@Ci@iM12

    if n_trials == 1:
        Cr = Cr[0]

    return Cr


def rescale(C, s, M=None):
    """Re-scale.
    
    Re-scale :math:`\mathbr{C} \in \mathcal{M}` by scaling factor s.

    Parameters
    ----------
    C : ndarray
        SPD matrices, shape (n_trials, n_channels, n_channles) or (n_channels, n_channels).
    s : float
        Scaling factor.
    M : ndarray | None
        The centroid of manifold, defaults to the identity matrix.
    
    Returns
    -------
    Cs : ndarray
        Re-scaled matrices.
    """
    C = C.reshape(-1, *C.shape[-2:])
    n_trials, n_channels, _ = C.shape

    Cs = np.zeros_like(C)

    if M is not None:
        M12 = sqrtm(M)
        iM12 = invsqrtm(M)
    else:
        M12 = np.eye(n_channels)
        iM12 = np.eye(n_channels)
    
    for i, Ci in enumerate(C):
        Cs[i] = M12@powm(iM12@Ci@iM12, s)@M12
    
    if n_trials == 1:
        Cs = Cs[0]
    
    return Cs
        

def rotate(C, R):
    """Rotate.
    
    Rotate :math:`\mathbr{C} \in \mathcal{M}` with rotation matrix :math:`\mathbr{U}`.

    Parameters
    ----------
    C : ndarray
        SPD matrices, shape (n_trials, n_channels, n_channels) or (n_channels, n_channels).
    U : ndarray
        Rotation matrix.
    
    Returns
    -------
    Cr : ndarray
        Rotated matrices.
    """
    C = C.reshape(-1, *C.shape[-2:])
    n_trials, n_channels, _ = C.shape

    Cr = np.zeros_like(C)

    for i, Ci in enumerate(C):
        Cr[i] = R@Ci@R.T
    
    if n_trials == 1:
        Cr = Cr[0]
    
    return Cr


def get_scale_factor(Cs, Ct):
    """Get scalefactor in rescale step, transform Cs to Ct.
    
    Parameters
    ----------
    Cs : ndarray
        source covariance matrices, shape (n_trials, n_channels, n_channels)
    Ct : ndarray
        target covariance matrices, shape (n_trials, n_channels, n_channels)

    Returns
    -------
    s : float
        scale factor
    """
    Ms = mean_riemann(Cs)
    Mt = mean_riemann(Ct)

    ds = np.sum(np.square(distance_riemann(Cs, Ms)))
    dt = np.sum(np.square(distance_riemann(Ct, Mt)))

    s = np.sqrt(dt/ds)
    return s


def _fit_single(X, y=None, n_clusters=2, init='random', random_state=None,
                metric='riemann', max_iter=100, tol=1e-4, n_jobs=1):
    """helper to fit a single run of centroid."""
    # init random state if provided
    mdm = MDM(metric=metric, n_jobs=n_jobs)
    squared_nomrs = [np.linalg.norm(x, ord='fro')**2 for x in X]
    mdm.covmeans_ = _init_centroids(X, n_clusters, init,
                                    random_state=random_state,
                                    x_squared_norms=squared_nomrs)
    if y is not None:
        mdm.classes_ = np.unique(y)
    else:
        mdm.classes_ = np.arange(n_clusters)

    labels = mdm.predict(X)
    k = 0
    while True:
        old_labels = labels.copy()
        mdm.fit(X, old_labels)
        dist = mdm._predict_distances(X)
        labels = mdm.classes_[dist.argmin(axis=1)]
        k += 1
        if (k > max_iter) | (np.mean(labels == old_labels) > (1 - tol)):
            break
    inertia = sum([sum(dist[labels == mdm.classes_[i], i])
                   for i in range(len(mdm.classes_))])
    return labels, inertia, mdm

def _procruster_cost_function_euc(R, Mt, Ms):
    weights = anp.ones(len(Mt))

    c = []
    for Mti, Msi in zip(Mt, Ms):
        t1 = Msi
        t2 = anp.dot(R, anp.dot(Mti, R.T))
        ci = anp.linalg.norm(t1-t2)**2
        c.append(ci)
    c = anp.array(c)

    return anp.dot(c, weights)


def _procruster_cost_function_rie(R, Mt, Ms):
    weights = anp.ones(len(Mt))

    c = []
    for Mti, Msi in zip(Mt, Ms):
        t1 = Msi
        t2 = anp.dot(R, anp.dot(Mti, R.T))
        ci = distance_riemann(t1, t2)**2
        c.append(ci)
    c = anp.array(c)

    return anp.dot(c, weights)


def _procruster_egrad_function_rie(R, Mt, Ms):
    weights = anp.ones(len(Mt))
    
    g = []
    for Mti, Msi, wi in zip(Mt, Ms, weights):
        iMti12 = invsqrtm(Mti)
        Msi12 = sqrtm(Msi)
        term_aux = anp.dot(R, anp.dot(Msi, R.T))
        term_aux = anp.dot(iMti12, anp.dot(term_aux, iMti12))
        gi = 4 * anp.dot(anp.dot(iMti12, logm(term_aux)), anp.dot(Msi12, R))
        g.append(gi * wi)

    g = anp.sum(g, axis=0)

    return g


def get_rotation_matrix(Mt, Ms, metric='euc'):
    Mt = Mt.reshape(-1, *Mt.shape[-2:])
    Ms = Ms.reshape(-1, *Ms.shape[-2:])

    n = Mt[0].shape[0]
    # manifolds = Rotations(n)
    manifolds = SpecialOrthogonalGroup(n)

    if metric == 'euc':
        cost = partial(_procruster_cost_function_euc, Mt=Mt, Ms=Ms)  
        problem = Problem(manifold=manifolds, cost=cost, verbosity=0)
    elif metric == 'rie':
        cost = partial(_procruster_cost_function_rie, Mt=Mt, Ms=Ms)    
        egrad = partial(_procruster_egrad_function_rie, Mt=Mt, Ms=Ms) 
        problem = Problem(manifold=manifolds, cost=cost, egrad=egrad, verbosity=0) 

    solver = SteepestDescent(mingradnorm=1e-3)

    Ropt = solver.solve(problem)

    return Ropt


class TangentSpace(BaseEstimator, TransformerMixin):
    """Tangent space projection.

    Attributes
    ----------
    reference_ : ndarray
        If fit, the reference point for tangent space mapping.
    """

    def __init__(self):
        """Init."""
        pass

    def fit(self, X, y=None, sample_weight=None):
        """Estimate the reference point.

        Parameters
        ----------
        X : ndarray
            SPD matrices, shape (n_trials, n_channels,n_channels).
        y : None|ndarray, optional
            Not used, here for compatibility with sklearn API.
        sample_weight : None|ndarray, optional
            Weight of each trial (default None). If None provided, weights init to 1, otherwise, weights are normalized.

        Returns
        -------
        self : TangentSpace instance
            The TangentSpace instance.
        """
        X = X.copy()
        if y is not None:
            y = y.copy()
        # compute mean covariance
        self.reference_ = mean_riemann(X, sample_weight=sample_weight)
        return self

    def _check_data_dim(self, X):
        """Check data shape and return the size of cov mat."""
        shape_X = X.shape
        if len(X.shape) == 2:
            Ne = (np.sqrt(1 + 8 * shape_X[1]) - 1) / 2
            if Ne != int(Ne):
                raise ValueError("Shape of Tangent space vector does not"
                                 " correspond to a square matrix.")
            return int(Ne)
        elif len(X.shape) == 3:
            if shape_X[1] != shape_X[2]:
                raise ValueError("Matrices must be square")
            return int(shape_X[1])
        else:
            raise ValueError("Shape must be of len 2 or 3.")

    def _check_reference_points(self, X):
        """Check reference point status, and force it to identity if not."""
        if not hasattr(self, 'reference_'):
            self.reference_ = np.eye(self._check_data_dim(X))
        else:
            shape_cr = self.reference_.shape[0]
            shape_X = self._check_data_dim(X)

            if shape_cr != shape_X:
                raise ValueError('Data must be same size of reference point.')

    def transform(self, X):
        """Tangent space projection.

        Parameters
        ----------
        X : ndarray
            SPD matrices, shape (n_trials, n_channels, n_channels).

        Returns
        -------
        ts : ndarray
            The tangent space projection of the matrices, shape (n_trials, n_channels*(n_channels+1)/2).
        """
        X = X.copy()
        self._check_reference_points(X)
        return tangent_space(X, self.reference_)

    def inverse_transform(self, X, y=None):
        """Inverse transform.

        Project back a set of tangent space vector in the manifold.

        Parameters
        ----------
        X : ndarray
            SPD matrices, shape (n_trials, n_channels*(n_channels+1)/2).
        y : None|ndarray, optional
            Not used, here for compatibility with sklearn API.

        Returns
        -------
        cov : ndarray
            The covariance matrices corresponding to each of tangent vector, shape (n_trials, n_channels, n_channels).
        """
        X = X.copy()
        if y is not None:
            y = y.copy()
        self._check_reference_points(X)
        return untangent_space(X, self.reference_)


class Kmeans(BaseEstimator, ClassifierMixin, ClusterMixin, TransformerMixin):

    """Kmean clustering using Riemannian geometry.

    Find clusters that minimize the sum of squared distance to their centroid.
    This is a direct implementation of the kmean algorithm with a riemanian
    metric.

    Parameters
    ----------
    n_cluster: int (default: 2)
        number of clusters.
    max_iter : int (default: 100)
        The maximum number of iteration to reach convergence.
    metric : string (default: 'riemann')
        The type of metric used for centroid and distance estimation.
    random_state : integer or numpy.RandomState, optional
        The generator used to initialize the centers. If an integer is
        given, it fixes the seed. Defaults to the global numpy random
        number generator.
    init : 'k-means++', 'random' or an ndarray (default 'random')
        Method for initialization of centers.
        'k-means++' : selects initial cluster centers for k-mean
        clustering in a smart way to speed up convergence. See section
        Notes in k_init for more details.
        'random': choose k observations (rows) at random from data for
        the initial centroids.
        If an ndarray is passed, it should be of shape (n_clusters, n_features)
        and gives the initial centers.
    n_init : int, (default: 10)
        Number of time the k-means algorithm will be run with different
        centroid seeds. The final results will be the best output of
        n_init consecutive runs in terms of inertia.
    n_jobs : int, (default: 1)
        The number of jobs to use for the computation. This works by computing
        each of the n_init runs in parallel.
        If -1 all CPUs are used. If 1 is given, no parallel computing code is
        used at all, which is useful for debugging. For n_jobs below -1,
        (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one
        are used.
    tol: float, (default: 1e-4)
        the stopping criterion to stop convergence, representing the minimum
        amount of change in labels between two iterations.

    Attributes
    ----------
    mdm_ : MDM instance.
        MDM instance containing the centroids.
    labels_ :
        Labels of each point
    inertia_ : float
        Sum of distances of samples to their closest cluster center.

    Notes
    -----
    .. versionadded:: 0.2.2

    See Also
    --------
    Kmeans
    MDM
    """

    def __init__(self, n_clusters=2, max_iter=100, metric='riemann',
                 random_state=None, init='random', n_init=10, n_jobs=1,
                 tol=1e-4):
        """Init."""
        self.metric = metric
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.seed = random_state
        self.init = init
        self.n_init = n_init
        self.tol = tol
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        """Fit (estimates) the clusters.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.
        y : ndarray | None (default None)
            Not used, here for compatibility with sklearn API.

        Returns
        -------
        self : Kmeans instance
            The Kmean instance.
        """
        if (self.init != 'random') | (self.n_init == 1):
            # no need to iterate if init is not random
            labels, inertia, mdm = _fit_single(X, y,
                                               n_clusters=self.n_clusters,
                                               init=self.init,
                                               random_state=self.seed,
                                               metric=self.metric,
                                               max_iter=self.max_iter,
                                               tol=self.tol,
                                               n_jobs=self.n_jobs)
        else:
            np.random.seed(self.seed)
            seeds = np.random.randint(
                np.iinfo(np.int32).max, size=self.n_init)
            if self.n_jobs == 1:
                res = []
                for i in range(self.n_init):
                    res.append(_fit_single(X, y,
                                      n_clusters=self.n_clusters,
                                      init=self.init,
                                      random_state=seeds[i],
                                      metric=self.metric,
                                      max_iter=self.max_iter,
                                      tol=self.tol))
                labels, inertia, mdm = zip(*res)
            else:

                res = Parallel(n_jobs=self.n_jobs, verbose=0)(
                    delayed(_fit_single)(X, y,
                                         n_clusters=self.n_clusters,
                                         init=self.init,
                                         random_state=seed,
                                         metric=self.metric,
                                         max_iter=self.max_iter,
                                         tol=self.tol,
                                         n_jobs=1)
                    for seed in seeds)
                labels, inertia, mdm = zip(*res)

            best = np.argmin(inertia)
            mdm = mdm[best]
            labels = labels[best]
            inertia = inertia[best]

        self.mdm_ = mdm
        self.inertia_ = inertia
        self.labels_ = labels

        return self

    def predict(self, X):
        """get the predictions.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.

        Returns
        -------
        pred : ndarray of int, shape (n_trials, 1)
            the prediction for each trials according to the closest centroid.
        """
        return self.mdm_.predict(X)

    def transform(self, X):
        """get the distance to each centroid.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.

        Returns
        -------
        dist : ndarray, shape (n_trials, n_cluster)
            the distance to each centroid according to the metric.
        """
        return self.mdm_.transform(X)

    def centroids(self):
        """helper for fast access to the centroid.

        Returns
        -------
        centroids : list of SPD matrices, len (n_cluster)
            Return a list containing the centroid of each cluster.
        """
        return self.mdm_.covmeans_


class KmeansPerClassTransform(BaseEstimator, TransformerMixin):

    """Run kmeans for each class."""

    def __init__(self, n_clusters=2, **params):
        """Init."""
        params['n_clusters'] = n_clusters
        self.km = Kmeans(**params)
        self.metric = self.km.metric

    def fit(self, X, y):
        """fit."""
        self.covmeans_ = []
        self.classes_ = np.unique(y)
        for c in self.classes_:
            self.km.fit(X[y == c])
            self.covmeans_.extend(self.km.centroids())
        return self

    def transform(self, X):
        """transform."""
        mdm = MDM(metric=self.metric, n_jobs=self.km.n_jobs)
        mdm.covmeans_ = self.covmeans_
        return mdm._predict_distances(X)




class MDM(BaseEstimator, TransformerMixin, ClassifierMixin):

    """Classification by Minimum Distance to Mean.

    Classification by nearest centroid. For each of the given classes, a
    centroid is estimated according to the chosen metric. Then, for each new
    point, the class is affected according to the nearest centroid.

    Parameters
    ----------
    metric : string | dict (default: 'riemann')
        The type of metric used for centroid and distance estimation.
        see `mean_covariance` for the list of supported metric.
        the metric could be a dict with two keys, `mean` and `distance` in
        order to pass different metric for the centroid estimation and the
        distance estimation. Typical usecase is to pass 'logeuclid' metric for
        the mean in order to boost the computional speed and 'riemann' for the
        distance in order to keep the good sensitivity for the classification.
    n_jobs : int, (default: 1)
        The number of jobs to use for the computation. This works by computing
        each of the class centroid in parallel.
        If -1 all CPUs are used. If 1 is given, no parallel computing code is
        used at all, which is useful for debugging. For n_jobs below -1,
        (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one
        are used.

    Attributes
    ----------
    covmeans_ : list
        the class centroids.
    classes_ : list
        list of classes.

    See Also
    --------
    Kmeans
    FgMDM
    KNearestNeighbor

    References
    ----------
    [1] A. Barachant, S. Bonnet, M. Congedo and C. Jutten, "Multiclass
    Brain-Computer Interface Classification by Riemannian Geometry," in IEEE
    Transactions on Biomedical Engineering, vol. 59, no. 4, p. 920-928, 2012.

    [2] A. Barachant, S. Bonnet, M. Congedo and C. Jutten, "Riemannian geometry
    applied to BCI classification", 9th International Conference Latent
    Variable Analysis and Signal Separation (LVA/ICA 2010), LNCS vol. 6365,
    2010, p. 629-636.
    """

    def __init__(self):
        """Init."""
        # store params for cloning purpose

    def fit(self, X, y, sample_weight=None):
        """Fit (estimates) the centroids.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.
        y : ndarray shape (n_trials, 1)
            labels corresponding to each trial.
        sample_weight : None | ndarray shape (n_trials, 1)
            the weights of each sample. if None, each sample is treated with
            equal weights.

        Returns
        -------
        self : MDM instance
            The MDM instance.
        """
        X = X.copy()
        y = y.copy()

        self.classes_ = np.unique(y)
        self.covmeans_ = []

        if sample_weight is None:
            sample_weight = np.ones(X.shape[0])


        for l in self.classes_:
            self.covmeans_.append(mean_riemann(X[y == l]))
        return self

    def _predict_distances(self, covtest):
        """Helper to predict the distance. equivalent to transform."""
        covtest = covtest.copy()
        covtest = np.reshape(covtest, (-1, *covtest.shape[-2:]))

        Nc = len(self.covmeans_)

        dist = [distance_riemann(covtest, self.covmeans_[m])
                    for m in range(Nc)]

        dist = np.stack(dist)
        return dist.T

    def predict(self, covtest):
        """get the predictions.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.

        Returns
        -------
        pred : ndarray of int, shape (n_trials, 1)
            the prediction for each trials according to the closest centroid.
        """
        dist = self._predict_distances(covtest)
        return self.classes_[dist.argmin(axis=1)]

    def transform(self, X):
        """get the distance to each centroid.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.

        Returns
        -------
        dist : ndarray, shape (n_trials, n_classes)
            the distance to each centroid according to the metric.
        """
        return self._predict_distances(X)

    def fit_predict(self, X, y):
        """Fit and predict in one function."""
        self.fit(X, y)
        return self.predict(X)

    def predict_proba(self, X):
        """Predict proba using softmax.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.

        Returns
        -------
        prob : ndarray, shape (n_trials, n_classes)
            the softmax probabilities for each class.
        """
        return softmax(-1*self._predict_distances(X))


class FGDA(BaseEstimator, TransformerMixin):

    """Fisher Geodesic Discriminant analysis.

    Project data in Tangent space, apply a FLDA to reduce dimention, and
    project filtered data back in the manifold.
    For a complete description of the algorithm, see [1]

    Parameters
    ----------
    metric : string (default: 'riemann')
        The type of metric used for reference point mean estimation.
        see `mean_covariance` for the list of supported metric.
    tsupdate : bool (default False)
        Activate tangent space update for covariante shift correction between
        training and test, as described in [2]. This is not compatible with
        online implementation. Performance are better when the number of trials
        for prediction is higher.

    See Also
    --------
    FgMDM
    TangentSpace

    References
    ----------
    [1] A. Barachant, S. Bonnet, M. Congedo and C. Jutten, "Riemannian geometry
    applied to BCI classification", 9th International Conference Latent
    Variable Analysis and Signal Separation (LVA/ICA 2010), LNCS vol. 6365,
    2010, p. 629-636.

    [2] A. Barachant, S. Bonnet, M. Congedo and C. Jutten, "Classification of
    covariance matrices using a Riemannian-based kernel for BCI applications",
    in NeuroComputing, vol. 112, p. 172-178, 2013.
    """

    def __init__(self):
        """Init."""
        pass

    def _fit_lda(self, X, y, sample_weight=None):
        """Helper to fit LDA."""
        self.classes_ = np.unique(y)
        self._lda = LinearDiscriminantAnalysis(n_components=len(self.classes_) - 1,
                        solver='lsqr',
                        shrinkage='auto')

        ts = self._ts.fit_transform(X, sample_weight=sample_weight)
        self._lda.fit(ts, y)

        W = self._lda.coef_.copy()
        self._W = np.dot(
            np.dot(W.T, np.linalg.pinv(np.dot(W, W.T))), W)
        return ts

    def _retro_project(self, ts):
        """Helper to project back in the manifold."""
        ts = np.dot(ts, self._W)
        return self._ts.inverse_transform(ts)

    def fit(self, X, y, sample_weight=None):
        """Fit (estimates) the reference point and the FLDA.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.
        y : ndarray 
            Not used, here for compatibility with sklearn API.
        sample_weight : ndarray | None (default None)
            weight of each sample.

        Returns
        -------
        self : FGDA instance
            The FGDA instance.
        """
        X = X.copy()
        if y is not None:
            y = y.copy()
        self._ts = TangentSpace()
        self._fit_lda(X, y, sample_weight=sample_weight)
        return self

    def transform(self, X):
        """Filtering operation.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.

        Returns
        -------
        covs : ndarray, shape (n_trials, n_channels, n_channels)
            covariances matrices after filtering.
        """
        X = X.copy()
        ts = self._ts.transform(X)
        return self._retro_project(ts)

    def fit_transform(self, X, y, sample_weight=None):
        """Fit and transform in a single function.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.
        y : ndarray | None (default None)
            Not used, here for compatibility with sklearn API.
        sample_weight : ndarray | None (default None)
            weight of each sample.

        Returns
        -------
        covs : ndarray, shape (n_trials, n_channels, n_channels)
            covariances matrices after filtering.
        """
        X = X.copy()
        if y is not None:
            y = y.copy()
        self._ts = TangentSpace()
        ts = self._fit_lda(X, y, sample_weight=sample_weight)
        return self._retro_project(ts)


class FgMDM(BaseEstimator, TransformerMixin, ClassifierMixin):

    """Classification by Minimum Distance to Mean with geodesic filtering.

    Apply geodesic filtering described in [1], and classify using MDM algorithm
    The geodesic filtering is achieved in tangent space with a Linear
    Discriminant Analysis, then data are projected back to the manifold and
    classifier with a regular mdm.
    This is basically a pipeline of FGDA and MDM

    Parameters
    ----------
    metric : string | dict (default: 'riemann')
        The type of metric used for centroid and distance estimation.
        see `mean_covariance` for the list of supported metric.
        the metric could be a dict with two keys, `mean` and `distance` in
        order to pass different metric for the centroid estimation and the
        distance estimation. Typical usecase is to pass 'logeuclid' metric for
        the mean in order to boost the computional speed and 'riemann' for the
        distance in order to keep the good sensitivity for the classification.
    tsupdate : bool (default False)
        Activate tangent space update for covariante shift correction between
        training and test, as described in [2]. This is not compatible with
        online implementation. Performance are better when the number of trials
        for prediction is higher.
    n_jobs : int, (default: 1)
        The number of jobs to use for the computation. This works by computing
        each of the class centroid in parallel.
        If -1 all CPUs are used. If 1 is given, no parallel computing code is
        used at all, which is useful for debugging. For n_jobs below -1,
        (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one
        are used.

    See Also
    --------
    MDM
    FGDA
    TangentSpace

    References
    ----------
    [1] A. Barachant, S. Bonnet, M. Congedo and C. Jutten, "Riemannian geometry
    applied to BCI classification", 9th International Conference Latent
    Variable Analysis and Signal Separation (LVA/ICA 2010), LNCS vol. 6365,
    2010, p. 629-636.

    [2] A. Barachant, S. Bonnet, M. Congedo and C. Jutten, "Classification of
    covariance matrices using a Riemannian-based kernel for BCI applications",
    in NeuroComputing, vol. 112, p. 172-178, 2013.
    """

    def __init__(self, n_jobs=1):
        """Init."""
        self.n_jobs = n_jobs

    def fit(self, X, y):
        """Fit FgMDM.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.
        y : ndarray shape (n_trials, 1)
            labels corresponding to each trial.

        Returns
        -------
        self : FgMDM instance
            The FgMDM instance.
        """
        self._mdm = MDM(n_jobs=self.n_jobs)
        self._fgda = FGDA()
        cov = self._fgda.fit_transform(X, y)
        self._mdm.fit(cov, y)
        return self

    def predict(self, X):
        """get the predictions after FDA filtering.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.

        Returns
        -------
        pred : ndarray of int, shape (n_trials, 1)
            the prediction for each trials according to the closest centroid.
        """
        cov = self._fgda.transform(X)
        return self._mdm.predict(cov)

    def transform(self, X):
        """get the distance to each centroid after FGDA filtering.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.

        Returns
        -------
        dist : ndarray, shape (n_trials, n_cluster)
            the distance to each centroid according to the metric.
        """
        cov = self._fgda.transform(X)
        return self._mdm.transform(cov)


class TSclassifier(BaseEstimator, ClassifierMixin):

    """Classification in the tangent space.

    Project data in the tangent space and apply a classifier on the projected
    data. This is a simple helper to pipeline the tangent space projection and
    a classifier. Default classifier is LogisticRegression

    Parameters
    ----------
    metric : string | dict (default: 'riemann')
        The type of metric used for centroid and distance estimation.
        see `mean_covariance` for the list of supported metric.
        the metric could be a dict with two keys, `mean` and `distance` in
        order to pass different metric for the centroid estimation and the
        distance estimation. Typical usecase is to pass 'logeuclid' metric for
        the mean in order to boost the computional speed and 'riemann' for the
        distance in order to keep the good sensitivity for the classification.
    tsupdate : bool (default False)
        Activate tangent space update for covariante shift correction between
        training and test, as described in [2]. This is not compatible with
        online implementation. Performance are better when the number of trials
        for prediction is higher.
    clf: sklearn classifier (default LogisticRegression)
        The classifier to apply in the tangent space

    See Also
    --------
    TangentSpace

    Notes
    -----
    .. versionadded:: 0.2.4
    """

    def __init__(self, clf=LogisticRegression(solver='lbfgs', multi_class='auto')):
        """Init."""
        self.clf = clf

    def fit(self, X, y):
        """Fit TSclassifier.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.
        y : ndarray shape (n_trials, 1)
            labels corresponding to each trial.

        Returns
        -------
        self : TSclassifier. instance
            The TSclassifier. instance.
        """
        ts = TangentSpace()
        self._pipe = make_pipeline(ts, self.clf)
        self._pipe.fit(X, y)
        return self

    def predict(self, X):
        """get the predictions.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.

        Returns
        -------
        pred : ndarray of int, shape (n_trials, 1)
            the prediction for each trials according to the closest centroid.
        """
        return self._pipe.predict(X)

    def transform(self, X):

        return self._pipe.transform(X)


    def predict_proba(self, X):
        """get the probability.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.

        Returns
        -------
        pred : ndarray of ifloat, shape (n_trials, n_classes)
            the prediction for each trials according to the closest centroid.
        """
        return self._pipe.predict_proba(X)


class Potato(BaseEstimator, TransformerMixin, ClassifierMixin):

    """Artefact detection with the Riemannian Potato.

    The Riemannian Potato [1] is a clustering method used to detect artifact in
    EEG signals. The algorithm iteratively estimate the centroid of clean
    signal by rejecting every trial that have a distance greater than several
    standard deviation from it.

    Parameters
    ----------
    metric : string (default 'riemann')
        The type of metric used for centroid and distance estimation.
    threshold : int (default 3)
        The number of standard deviation to reject artifacts.
    n_iter_max : int (default 100)
        The maximum number of iteration to reach convergence.
    pos_label: int (default 1)
        The positive label corresponding to clean data
    neg_label: int (default 0)
        The negative label corresponding to artifact data

    Notes
    -----
    .. versionadded:: 0.2.3

    See Also
    --------
    Kmeans
    MDM

    References
    ----------
    [1] A. Barachant, A. Andreev and M. Congedo, "The Riemannian Potato: an
    automatic and adaptive artifact detection method for online experiments
    using Riemannian geometry", in Proceedings of TOBI Workshop IV, p. 19-20,
    2013.
    """

    def __init__(self, threshold=3, n_iter_max=100,
                 pos_label=1, neg_label=0):
        """Init."""
        self.threshold = threshold
        self.n_iter_max = n_iter_max
        if pos_label == neg_label:
            raise(ValueError("Positive and Negative labels must be different"))
        self.pos_label = pos_label
        self.neg_label = neg_label

    def fit(self, X, y=None):
        """Fit the potato from covariance matrices.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.
        y : ndarray | None (default None)
            Not used, here for compatibility with sklearn API.

        Returns
        -------
        self : Potato instance
            The Potato instance.
        """
        X = X.copy()
        if y is not None:
            y = y.copy()

        self._mdm = MDM()

        if y is not None:
            if len(y) != len(X):
                raise ValueError('y must be the same lenght of X')

            classes = np.int32(np.unique(y))

            if len(classes) > 2:
                raise ValueError('number of classes must be maximum 2')

            if self.pos_label not in classes:
                raise ValueError('y must contain a positive class')

            y_old = np.int32(np.array(y) == self.pos_label)
        else:
            y_old = np.ones(len(X))
        # start loop
        for n_iter in range(self.n_iter_max):
            ix = (y_old == 1)
            self._mdm.fit(X[ix], y_old[ix])
            y = np.zeros(len(X))
            d = np.squeeze(np.log(self._mdm.transform(X[ix])))
            self._mean = np.mean(d)
            self._std = np.std(d)
            y[ix] = self._get_z_score(d) < self.threshold

            if np.array_equal(y, y_old):
                break
            else:
                y_old = y
        return self

    def transform(self, X):
        """return the normalized log-distance to the centroid (z-score).

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.

        Returns
        -------
        z : ndarray, shape (n_epochs, 1)
            the normalized log-distance to the centroid.
        """
        d = np.squeeze(np.log(self._mdm.transform(X)))
        z = self._get_z_score(d)
        return z

    def predict(self, X):
        """predict artefact from data.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.

        Returns
        -------
        pred : ndarray of bool, shape (n_epochs, 1)
            the artefact detection. True if the trial is clean, and False if
            the trial contain an artefact.
        """
        z = self.transform(X)
        pred = z < self.threshold
        out = np.zeros_like(z) + self.neg_label
        out[pred] = self.pos_label
        return out

    def _get_z_score(self, d):
        """get z score from distance."""
        z = (d - self._mean) / self._std
        return z


class RecursiveRiemannMean(BaseEstimator, TransformerMixin):
    """Recursive Riemannian Mean Update.
    
    Parameters
    ----------
    init_M: ndarray 
        The initialization of M, shape (n_channels, n_channels).
    count: int
        The number of accumulated step.
    
    """

    def __init__(self, init_M=None, count=0):
        self.M = init_M
        self.count = count

    def fit(self, X, y=None):
        if self.M is None:
            self.M = X
        else:
            self.M = geodesic(self.M, X, 1/(self.count + 1))
            self.count = self.count + 1
        return self

    def transform(self, X=None):
        return self.M

class OnlineMDM(BaseEstimator, TransformerMixin, ClassifierMixin):
    
    def __init__(self,n_train,n_channel):
        self._mdm=MDM(use_trace=False)
        self.n_channel=n_channel
        self.trial_num=0
        self.class_num=dict()
        self.n_train=n_train
        self.fit_X=np.zeros((self.n_train,self.n_channel,self.n_channel))
        self.fit_y=np.zeros(self.n_train).astype(int)
    
    def fit(self,new_X,new_y):
        if new_y in self.class_num:
            self.class_num[new_y]=self.class_num[new_y]+1
        else:
            self.class_num[new_y]=1
        if self.trial_num<self.n_train:
            self.fit_X[self.trial_num]=new_X
            self.fit_y[self.trial_num]=new_y
            if self.trial_num==self.n_train-1:
                self._mdm.fit(self.fit_X,self.fit_y)
            self.trial_num=self.trial_num+1
        return -1
    
    def static_strategy(self,new_X,new_y):
        self.trial_num=self.trial_num+1
        return self._mdm.predict(new_X)[0]
    
    def retrained_strategy(self,new_X,new_y):
        self.trial_num=self.trial_num+1
        predict_new_X=self._mdm.predict(new_X)
        new_X=new_X[np.newaxis,:]
        self.fit_X=np.concatenate((self.fit_X,new_X))
        self.fit_y=np.append(self.fit_y,predict_new_X[0])
        self._mdm.fit(self.fit_X,self.fit_y)
        return predict_new_X[0]
    
    def incremental_strategy(self,new_X,new_y):
        self.trial_num=self.trial_num+1
        predict_new_X=self._mdm.predict(new_X)
        self.class_num[predict_new_X[0]]=self.class_num[predict_new_X[0]]+1
        if self._mdm.use_trace: new_X /= np.trace(new_X)
        temp_matrix=powm(np.dot(np.dot(invsqrtm(self._mdm.covmeans_[predict_new_X[0]]),new_X),invsqrtm(self._mdm.covmeans_[predict_new_X[0]])),
                         1/self.class_num[predict_new_X[0]])
        self._mdm.covmeans_[predict_new_X[0]]=np.dot(np.dot(sqrtm(self._mdm.covmeans_[predict_new_X[0]]),temp_matrix),
                                          sqrtm(self._mdm.covmeans_[predict_new_X[0]]))
        return predict_new_X[0]
    
    def running(self,new_X,new_y,strategy):
        new_X=new_X.copy()
        new_y=new_y.copy()
        if self.trial_num<self.n_train:
            return self.fit(new_X,new_y)
        else:
            if strategy=='static':
                return self.static_strategy(new_X,new_y)
            if strategy=='retrained':
                return self.retrained_strategy(new_X,new_y)
            if strategy=='incremental':
                return self.incremental_strategy(new_X,new_y)    


class RiemannianPotatoPatch():
    def __init__(self, tol = 0, zscore=2.5, unsup_ada = False):
        self.tol = tol
        self.zscore = zscore
        self.unsup_ada = unsup_ada
    def fit(self,raw_mat,raw_y):
        self.raw_mat = raw_mat.copy()
        self.raw_y = raw_y.copy()
        self.classes = np.unique(raw_y)
        self.class_index = []
        self.class_X = []
        for idx in range(len(self.classes)):
            self.class_index.append([])
            self.class_X.append([])
        for idx in range(len(self.classes)):
            self.class_index[idx] = np.where(raw_y == self.classes[idx])[0]
            self.class_X[idx] = raw_mat[self.class_index[idx]]
        self.raw_got = []
        self.potato_list = [[],[],[],[],[]] # class, seed, num, mean_dis, std_dis
        for idx in range(len(self.classes)):
            if len(self.classes) == 2:
                inv_idx = int(1-idx)
            self.temp_class_index = self.class_index[idx]
            self.temp_notclass_index = self.class_index[inv_idx]
            self.temp_class_X = self.class_X[idx].copy()
            self.temp_notclass_X = self.class_X[inv_idx].copy()
            
            while(len(self.temp_class_index) > 0):
                self.temp_mean = mean_riemann(self.temp_class_X)
                self.intra_distance = distance_riemann(self.temp_class_X,self.temp_mean)
                self.seed_index = np.argmax(self.intra_distance)
                self.stable_potato_seed = self.temp_class_X[self.seed_index]
                self.stable_potato_num = 1
                self.stable_potato_mean_dis = 1e-15
                self.stable_potato_std_dis = 1e-15
                self.raw_got.append(self.temp_class_index)
                self.temp_class_index = np.delete(self.temp_class_index,self.seed_index,axis=0)
                self.temp_class_X = np.delete(self.temp_class_X,self.seed_index,axis=0)


                if (len(self.temp_class_index) > 0):
                    self.stable_potato_seed_intra_distance = distance_riemann(self.temp_class_X,self.stable_potato_seed)
                    self.other_seed_index = np.argmin(self.stable_potato_seed_intra_distance)
                    self.other_seed = self.temp_class_X[self.other_seed_index]
                    self.unstable_potato_num = self.stable_potato_num+1
                    self.unstable_potato_seed = mean_riemann(np.array([self.stable_potato_seed,self.other_seed]))
                    self.unstable_potato_dis = distance_riemann(np.array([self.stable_potato_seed,self.other_seed]),
                                                                self.unstable_potato_seed)
                    self.unstable_potato_mean_dis = np.mean(self.unstable_potato_dis)
                    self.unstable_potato_std_dis = np.std(self.unstable_potato_dis)
                    self.unstable_potato_inter_dis = distance_riemann(self.temp_notclass_X, self.unstable_potato_seed)
                    self.inter_zscore = (self.unstable_potato_inter_dis - self.unstable_potato_mean_dis)/self.unstable_potato_std_dis
                    
                    while(np.sum(self.inter_zscore<self.zscore)
                          <= self.tol):
                        self.stable_potato_seed = self.unstable_potato_seed
                        self.stable_potato_num = self.unstable_potato_num
                        self.stable_potato_mean_dis = self.unstable_potato_mean_dis
                        self.stable_potato_std_dis = self.unstable_potato_std_dis
                        self.temp_class_index = np.delete(self.temp_class_index,self.other_seed_index,axis=0)
                        self.temp_class_X = np.delete(self.temp_class_X,self.other_seed_index,axis=0)
    
                        if (len(self.temp_class_index) > 0):
                            self.stable_potato_seed_intra_distance = distance_riemann(self.temp_class_X,self.stable_potato_seed)
                            self.other_seed_index = np.argmin(self.stable_potato_seed_intra_distance)
                            self.other_seed = self.temp_class_X[self.other_seed_index]
                            self.unstable_potato_num = self.stable_potato_num+1
                            self.unstable_potato_seed = geodesic(self.unstable_potato_seed, self.other_seed, 1/(self.unstable_potato_num))
                            self.seeds_distance = distance_riemann(self.unstable_potato_seed,self.other_seed)
                            self.unstable_potato_mean_dis = ((1-1/(self.unstable_potato_num))*self.unstable_potato_mean_dis)+(
                                self.seeds_distance/(self.unstable_potato_num))
                            self.unstable_potato_std_dis = np.sqrt((1-1/(self.unstable_potato_num))*self.unstable_potato_std_dis**2+(
                                self.seeds_distance-self.unstable_potato_mean_dis)**2/(self.unstable_potato_num))
                            self.unstable_potato_inter_dis = distance_riemann(self.temp_notclass_X, self.unstable_potato_seed)
                            self.inter_zscore = (self.unstable_potato_inter_dis - self.unstable_potato_mean_dis)/self.unstable_potato_std_dis
                        else:
                            break                   
                self.potato_list[0].append(self.classes[idx])
                self.potato_list[1].append(self.stable_potato_seed)
                self.potato_list[2].append(self.stable_potato_num)
                self.potato_list[3].append(self.stable_potato_mean_dis)
                self.potato_list[4].append(self.stable_potato_std_dis)
        self.init_got = []
        for potato_i_index in range(len(self.raw_got)):
            if potato_i_index < len(self.raw_got)-1:
                if self.potato_list[0][potato_i_index] == self.potato_list[0][potato_i_index+1]:
                    self.init_got.append(list(set(self.raw_got[potato_i_index])-set(self.raw_got[potato_i_index+1])))
                else:
                    self.init_got.append(list(set(self.raw_got[potato_i_index])))
            else:
                self.init_got.append(list(set(self.raw_got[potato_i_index])))
                    
          
    def predict_fit(self, new_X, new_y = None):
        self.new_X = new_X
        self.new_distance = distance_riemann(np.array(self.potato_list[1]),self.new_X)
        self.potato_index = np.argmin(self.new_distance)
        self.pre_y = self.potato_list[0][self.potato_index]
        if self.unsup_ada == False:
            return self.pre_y
        else:
            return self.pre_y
                
class RiemannianEggCarton():

    def __init__(self, ada = True, sup = False):
     
        self.ada = ada
        self.sup = sup

    def fit(self, raw_mat, raw_y):
        
        self.raw_mat = raw_mat
        self.raw_y = raw_y
        self.classes = np.unique(raw_y)      # 包含的类别
        self.class_index = []
        self.class_X = []
        for idx in range(len(self.classes)):
            self.class_index.append([])
            self.class_X.append([])
        for idx in range(len(self.classes)):
            self.class_index[idx] = np.where(self.raw_y == self.classes[idx])[0]
            self.class_X[idx] = self.raw_mat[self.class_index[idx]]
        self.raw_got = []
        self.egg_list = [[], [], [], [], []]
        for idx in range(len(self.classes)):               # 每一类循环
            if len(self.classes) == 2:
                inv_idx = int(1 - idx)
            self.temp_class_index = self.class_index[idx]           # 取其中一类的样本索引
            self.temp_notclass_index = self.class_index[inv_idx]    # 取另外一类的样本索引
            self.temp_class_X = self.class_X[idx].copy()            # 取其中一类样本
            self.temp_notclass_X = self.class_X[inv_idx].copy()     # 取另外一类样本

            while(len(self.temp_class_index) > 0):
                self.temp_mean = mean_riemann(self.temp_class_X)    # 一类的黎曼均值
                self.intra_distance = distance_riemann(self.temp_class_X, self.temp_mean)     # 一类的黎曼均值和一类内所有样本的黎曼距离
                self.seed_index = np.argmax(self.intra_distance)    # 得到一类内距离最大的样本的索引值
                self.stable_egg_seed = self.temp_class_X[self.seed_index]  # 得到一类内距离最大的样本，设为子簇中心
                self.stable_egg_num = 1                             # 子簇内样本数设为1
                self.raw_got.append(self.temp_class_index)
                self.stable_outer_bound = np.min(distance_riemann(self.temp_notclass_X, self.stable_egg_seed))# 取子簇中心与二类所有样本的距离最小的样本索引
                self.stable_inner_bound = 0
                self.egg = [self.stable_egg_seed]
                self.temp_class_index = np.delete(self.temp_class_index,self.seed_index,axis=0)    # 在一类中去除子簇中心样本索引
                self.temp_class_X = np.delete(self.temp_class_X,self.seed_index,axis=0)            # 在一类中去除子簇中心样本索引
                if (len(self.temp_class_index) > 0):                                               # 如果一类还有没进入子簇的样本
                    self.stable_egg_seed_intra_distance = distance_riemann(self.temp_class_X, self.stable_egg_seed)# 得一类与子簇中心距离
                    self.other_seed_index = np.argmin(self.stable_egg_seed_intra_distance)         # 取距离最小样本索引
                    self.egg.append(self.temp_class_X[self.other_seed_index])                      # 将距离最小样本加入该子簇
                    self.unstable_egg_num = self.stable_egg_num+1                                  # 暂时子簇内样本数加1
                    self.unstable_egg_seed = mean_riemann(np.array(self.egg))                      # 暂时求子簇中心即黎曼均值
                    self.unstable_inner_bound = np.max(distance_riemann(np.array(self.egg),self.unstable_egg_seed)) # 求新的中心与子簇内所有样本的距离最大的样本
                    self.unstable_outer_bound = np.min(distance_riemann(self.temp_notclass_X, self.unstable_egg_seed))# 取子簇中心与二类所有样本的距离最小的样本索引
                    
                    while(self.unstable_outer_bound>self.unstable_inner_bound):                    # 所有二类样本的距离均大于子簇内样本距离
                        self.stable_egg_seed = self.unstable_egg_seed                              # 确定最终的子簇中心
                        self.stable_egg_num = self.unstable_egg_num                                # 确定最终的子簇内样本数
                        self.stable_inner_bound = self.unstable_inner_bound                        # 确定最终的簇内边界
                        self.stable_outer_bound = self.unstable_outer_bound                        # 确定最终的簇外边界
                        self.temp_class_index = np.delete(self.temp_class_index,self.other_seed_index,axis=0) # 去除新加入子簇的样本
                        self.temp_class_X = np.delete(self.temp_class_X,self.other_seed_index,axis=0)

                        if (len(self.temp_class_index) > 0):
                            self.stable_egg_seed_intra_distance = distance_riemann(self.temp_class_X,self.stable_egg_seed)
                            self.other_seed_index = np.argmin(self.stable_egg_seed_intra_distance)
                            self.egg.append(self.temp_class_X[self.other_seed_index])
                            self.unstable_egg_num = self.stable_egg_num+1
                            self.unstable_egg_seed = geodesic(self.unstable_egg_seed, 
                                                               self.temp_class_X[self.other_seed_index], 1/(self.unstable_egg_num))
                            self.unstable_inner_bound = np.max(distance_riemann(np.array(self.egg),self.unstable_egg_seed))
                            self.unstable_outer_bound = np.min(distance_riemann(self.temp_notclass_X, self.unstable_egg_seed))
                        else:
                            break
                self.egg_list[0].append(self.classes[idx])
                self.egg_list[1].append(self.stable_egg_seed)
                self.egg_list[2].append(self.stable_egg_num)
                self.egg_list[3].append(self.stable_inner_bound)
                self.egg_list[4].append(self.stable_outer_bound)
        self.init_got = []
        for egg_i_index in range(len(self.raw_got)):
            if egg_i_index < len(self.raw_got) - 1:
                if self.egg_list[0][egg_i_index] == self.egg_list[0][egg_i_index + 1]:
                    self.init_got.append(
                        list(set(self.raw_got[egg_i_index]) - set(self.raw_got[egg_i_index + 1])))
                else:
                    self.init_got.append(list(set(self.raw_got[egg_i_index])))
            else:
                self.init_got.append(list(set(self.raw_got[egg_i_index])))
              
    def predict_fit(self,new_mat,new_y = None):
        self.new_mat = new_mat
        self.new_distance = distance_riemann(np.array(self.egg_list[1]), self.new_mat)
        self.pre = self.egg_list[0][np.argmin(self.new_distance)]
        return self.pre


class RiemannianEggCarton4():

    def __init__(self, ada=True, sup=False):

        self.ada = ada
        self.sup = sup

    def fit(self, raw_mat, raw_y):

        self.raw_mat = raw_mat
        self.raw_y = raw_y
        self.classes = np.unique(raw_y)  # 包含的类别

        self.class_index = []
        self.class_X = []
        for idx in range(len(self.classes)):
            self.class_index.append([])
            self.class_X.append([])
        for idx in range(len(self.classes)):
            self.class_index[idx] = np.where(self.raw_y == self.classes[idx])[0]
            self.class_X[idx] = self.raw_mat[self.class_index[idx]]

        self.raw_got = []
        self.egg_list = [[], [], [], [], []]
        for idx in range(len(self.classes)):  # 每一类循环
            inv_idx = list(range(len(self.classes)))
            del inv_idx[idx]
            self.temp_class_index = self.class_index[idx]  # 取其中一类的样本索引
            self.temp_notclass_index = np.empty(0)
            for i in range(len(inv_idx)):
                self.temp_notclass_index = np.append(self.temp_notclass_index,self.class_index[inv_idx[i]]) # 取另外一类的样本索引
            self.temp_class_X = self.class_X[idx].copy()  # 取其中一类样本
            self.temp_notclass_X = np.empty((0, self.class_X[0].shape[1], self.class_X[0].shape[2]))
            for i in range(len(inv_idx)):
                self.temp_notclass_X = np.concatenate((self.temp_notclass_X, self.class_X[inv_idx[i]]), axis=0) # 取另外一类样本

            while (len(self.temp_class_index) > 0):
                self.temp_mean = mean_riemann(self.temp_class_X)  # 一类的黎曼均值
                self.intra_distance = distance_riemann(self.temp_class_X, self.temp_mean)  # 一类的黎曼均值和一类内所有样本的黎曼距离
                self.seed_index = np.argmax(self.intra_distance)  # 得到一类内距离最大的样本的索引值
                self.stable_egg_seed = self.temp_class_X[self.seed_index]  # 得到一类内距离最大的样本，设为子簇中心
                self.stable_egg_num = 1  # 子簇内样本数设为1
                self.raw_got.append(self.temp_class_index)
                self.stable_outer_bound = np.min(
                    distance_riemann(self.temp_notclass_X, self.stable_egg_seed))  # 取子簇中心与二类所有样本的距离最小的样本索引
                self.stable_inner_bound = 0
                self.egg = [self.stable_egg_seed]
                self.temp_class_index = np.delete(self.temp_class_index, self.seed_index, axis=0)  # 在一类中去除子簇中心样本索引
                self.temp_class_X = np.delete(self.temp_class_X, self.seed_index, axis=0)  # 在一类中去除子簇中心样本索引
                if (len(self.temp_class_index) > 0):  # 如果一类还有没进入子簇的样本
                    self.stable_egg_seed_intra_distance = distance_riemann(self.temp_class_X,
                                                                           self.stable_egg_seed)  # 得一类与子簇中心距离
                    self.other_seed_index = np.argmin(self.stable_egg_seed_intra_distance)  # 取距离最小样本索引
                    self.egg.append(self.temp_class_X[self.other_seed_index])  # 将距离最小样本加入该子簇
                    self.unstable_egg_num = self.stable_egg_num + 1  # 暂时子簇内样本数加1
                    self.unstable_egg_seed = mean_riemann(np.array(self.egg))  # 暂时求子簇中心即黎曼均值
                    self.unstable_inner_bound = np.max(
                        distance_riemann(np.array(self.egg), self.unstable_egg_seed))  # 求新的中心与子簇内所有样本的距离最大的样本
                    self.unstable_outer_bound = np.min(
                        distance_riemann(self.temp_notclass_X, self.unstable_egg_seed))  # 取子簇中心与二类所有样本的距离最小的样本索引

                    while (self.unstable_outer_bound > self.unstable_inner_bound):  # 所有二类样本的距离均大于子簇内样本距离
                        self.stable_egg_seed = self.unstable_egg_seed  # 确定最终的子簇中心
                        self.stable_egg_num = self.unstable_egg_num  # 确定最终的子簇内样本数
                        self.stable_inner_bound = self.unstable_inner_bound  # 确定最终的簇内边界
                        self.stable_outer_bound = self.unstable_outer_bound  # 确定最终的簇外边界
                        self.temp_class_index = np.delete(self.temp_class_index, self.other_seed_index,
                                                          axis=0)  # 去除新加入子簇的样本
                        self.temp_class_X = np.delete(self.temp_class_X, self.other_seed_index, axis=0)

                        if (len(self.temp_class_index) > 0):
                            self.stable_egg_seed_intra_distance = distance_riemann(self.temp_class_X,
                                                                                   self.stable_egg_seed)
                            self.other_seed_index = np.argmin(self.stable_egg_seed_intra_distance)
                            self.egg.append(self.temp_class_X[self.other_seed_index])
                            self.unstable_egg_num = self.stable_egg_num + 1
                            self.unstable_egg_seed = geodesic(self.unstable_egg_seed,
                                                              self.temp_class_X[self.other_seed_index],
                                                              1 / (self.unstable_egg_num))
                            self.unstable_inner_bound = np.max(
                                distance_riemann(np.array(self.egg), self.unstable_egg_seed))
                            self.unstable_outer_bound = np.min(
                                distance_riemann(self.temp_notclass_X, self.unstable_egg_seed))
                        else:
                            break
                self.egg_list[0].append(self.classes[idx])
                self.egg_list[1].append(self.stable_egg_seed)
                self.egg_list[2].append(self.stable_egg_num)
                self.egg_list[3].append(self.stable_inner_bound)
                self.egg_list[4].append(self.stable_outer_bound)
        self.init_got = []
        for egg_i_index in range(len(self.raw_got)):
            if egg_i_index < len(self.raw_got) - 1:
                if self.egg_list[0][egg_i_index] == self.egg_list[0][egg_i_index + 1]:
                    self.init_got.append(
                        list(set(self.raw_got[egg_i_index]) - set(self.raw_got[egg_i_index + 1])))
                else:
                    self.init_got.append(list(set(self.raw_got[egg_i_index])))
            else:
                self.init_got.append(list(set(self.raw_got[egg_i_index])))

    def predict_fit(self, new_mat, new_y=None):
        self.new_mat = new_mat
        self.new_distance = distance_riemann(np.array(self.egg_list[1]), self.new_mat)
        self.pre = self.egg_list[0][np.argmin(self.new_distance)]
        return self.pre


class RiemannianPotatoPatch4():
    def __init__(self, tol=0, zscore=2.5, unsup_ada=False):
        self.tol = tol
        self.zscore = zscore
        self.unsup_ada = unsup_ada

    def fit(self, raw_mat, raw_y):
        self.raw_mat = raw_mat.copy()
        self.raw_y = raw_y.copy()
        self.classes = np.unique(raw_y)
        self.class_index = []
        self.class_X = []
        for idx in range(len(self.classes)):
            self.class_index.append([])
            self.class_X.append([])
        for idx in range(len(self.classes)):
            self.class_index[idx] = np.where(raw_y == self.classes[idx])[0]
            self.class_X[idx] = raw_mat[self.class_index[idx]]
        self.raw_got = []
        self.potato_list = [[], [], [], [], []]  # class, seed, num, mean_dis, std_dis
        for idx in range(len(self.classes)):
            inv_idx = list(range(len(self.classes)))
            del inv_idx[idx]
            self.temp_class_index = self.class_index[idx]  # 取其中一类的样本索引
            self.temp_notclass_index = np.empty(0)
            for i in range(len(inv_idx)):
                self.temp_notclass_index = np.append(self.temp_notclass_index,
                                                     self.class_index[inv_idx[i]])  # 取另外一类的样本索引
            self.temp_class_X = self.class_X[idx].copy()  # 取其中一类样本
            self.temp_notclass_X = np.empty((0, self.class_X[0].shape[1], self.class_X[0].shape[2]))
            for i in range(len(inv_idx)):
                self.temp_notclass_X = np.concatenate((self.temp_notclass_X, self.class_X[inv_idx[i]]),
                                                      axis=0)  # 取另外一类样本

            while (len(self.temp_class_index) > 0):
                self.temp_mean = mean_riemann(self.temp_class_X)
                self.intra_distance = distance_riemann(self.temp_class_X, self.temp_mean)
                self.seed_index = np.argmax(self.intra_distance)
                self.stable_potato_seed = self.temp_class_X[self.seed_index]
                self.stable_potato_num = 1
                self.stable_potato_mean_dis = 1e-15
                self.stable_potato_std_dis = 1e-15
                self.raw_got.append(self.temp_class_index)
                self.temp_class_index = np.delete(self.temp_class_index, self.seed_index, axis=0)
                self.temp_class_X = np.delete(self.temp_class_X, self.seed_index, axis=0)

                if (len(self.temp_class_index) > 0):
                    self.stable_potato_seed_intra_distance = distance_riemann(self.temp_class_X,
                                                                              self.stable_potato_seed)
                    self.other_seed_index = np.argmin(self.stable_potato_seed_intra_distance)
                    self.other_seed = self.temp_class_X[self.other_seed_index]
                    self.unstable_potato_num = self.stable_potato_num + 1
                    self.unstable_potato_seed = mean_riemann(np.array([self.stable_potato_seed, self.other_seed]))
                    self.unstable_potato_dis = distance_riemann(np.array([self.stable_potato_seed, self.other_seed]),
                                                                self.unstable_potato_seed)
                    self.unstable_potato_mean_dis = np.mean(self.unstable_potato_dis)
                    self.unstable_potato_std_dis = np.std(self.unstable_potato_dis)
                    self.unstable_potato_inter_dis = distance_riemann(self.temp_notclass_X, self.unstable_potato_seed)
                    self.inter_zscore = (
                                                    self.unstable_potato_inter_dis - self.unstable_potato_mean_dis) / self.unstable_potato_std_dis

                    while (np.sum(self.inter_zscore < self.zscore)
                           <= self.tol):
                        self.stable_potato_seed = self.unstable_potato_seed
                        self.stable_potato_num = self.unstable_potato_num
                        self.stable_potato_mean_dis = self.unstable_potato_mean_dis
                        self.stable_potato_std_dis = self.unstable_potato_std_dis
                        self.temp_class_index = np.delete(self.temp_class_index, self.other_seed_index, axis=0)
                        self.temp_class_X = np.delete(self.temp_class_X, self.other_seed_index, axis=0)

                        if (len(self.temp_class_index) > 0):
                            self.stable_potato_seed_intra_distance = distance_riemann(self.temp_class_X,
                                                                                      self.stable_potato_seed)
                            self.other_seed_index = np.argmin(self.stable_potato_seed_intra_distance)
                            self.other_seed = self.temp_class_X[self.other_seed_index]
                            self.unstable_potato_num = self.stable_potato_num + 1
                            self.unstable_potato_seed = geodesic(self.unstable_potato_seed, self.other_seed,
                                                                 1 / (self.unstable_potato_num))
                            self.seeds_distance = distance_riemann(self.unstable_potato_seed, self.other_seed)
                            self.unstable_potato_mean_dis = ((1 - 1 / (
                                self.unstable_potato_num)) * self.unstable_potato_mean_dis) + (
                                                                    self.seeds_distance / (self.unstable_potato_num))
                            self.unstable_potato_std_dis = np.sqrt(
                                (1 - 1 / (self.unstable_potato_num)) * self.unstable_potato_std_dis ** 2 + (
                                        self.seeds_distance - self.unstable_potato_mean_dis) ** 2 / (
                                    self.unstable_potato_num))
                            self.unstable_potato_inter_dis = distance_riemann(self.temp_notclass_X,
                                                                              self.unstable_potato_seed)
                            self.inter_zscore = (
                                                            self.unstable_potato_inter_dis - self.unstable_potato_mean_dis) / self.unstable_potato_std_dis
                        else:
                            break
                self.potato_list[0].append(self.classes[idx])
                self.potato_list[1].append(self.stable_potato_seed)
                self.potato_list[2].append(self.stable_potato_num)
                self.potato_list[3].append(self.stable_potato_mean_dis)
                self.potato_list[4].append(self.stable_potato_std_dis)
        self.init_got = []
        for potato_i_index in range(len(self.raw_got)):
            if potato_i_index < len(self.raw_got) - 1:
                if self.potato_list[0][potato_i_index] == self.potato_list[0][potato_i_index + 1]:
                    self.init_got.append(
                        list(set(self.raw_got[potato_i_index]) - set(self.raw_got[potato_i_index + 1])))
                else:
                    self.init_got.append(list(set(self.raw_got[potato_i_index])))
            else:
                self.init_got.append(list(set(self.raw_got[potato_i_index])))

    def predict_fit(self, new_X, new_y=None):
        self.new_X = new_X
        self.new_distance = distance_riemann(np.array(self.potato_list[1]), self.new_X)
        self.potato_index = np.argmin(self.new_distance)
        self.pre_y = self.potato_list[0][self.potato_index]
        if self.unsup_ada == False:
            return self.pre_y
        else:
            return self.pre_y

def _class_means(X, y):
    """Compute class means.
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data.
    y : array-like of shape (n_samples,) or (n_samples, n_targets)
        Target values.
    Returns
    -------
    means : array-like of shape (n_classes, n_features)
        Class means.
    """
    classes, y = np.unique(y, return_inverse=True)
    cnt = np.bincount(y)   # 求出两个类的个数
    means = np.zeros(shape=(len(classes), X.shape[1]))
    np.add.at(means, y, X)   # np.add.at()是将传入的数组中制定下标位置的元素加上指定的值
    means /= cnt[:, None]
    return means

def _class_cov(X, y):
    """Compute weighted within-class covariance matrix.
    The per-class covariance are weighted by the class priors.
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data.
    y : array-like of shape (n_samples,) or (n_samples, n_targets)
        Target values.
    Returns
    -------
    cov : array-like of shape (n_features, n_features)
        Weighted within-class covariance matrix
    """
    _, y_t = np.unique(y, return_inverse=True)
    priors = np.bincount(y_t) / float(len(y))
    classes = np.unique(y)
    cov = np.zeros((len(classes),X.shape[1], X.shape[1]))
    for idx, group in enumerate(classes):
        Xg = X[y == group, :]
        cov [idx] = priors[idx] * np.cov(Xg.T,bias = False)
    return cov

def shrunk_covariance(emp_cov, shrinkage=0.1):
    """Calculates a covariance matrix shrunk on the diagonal
    Read more in the :ref:`User Guide <shrunk_covariance>`.
    Parameters
    ----------
    emp_cov : array-like of shape (n_features, n_features)
        Covariance matrix to be shrunk
    shrinkage : float, default=0.1
        Coefficient in the convex combination used for the computation
        of the shrunk estimate. Range is [0, 1].
    Returns
    -------
    shrunk_cov : ndarray of shape (n_features, n_features)
        Shrunk covariance.
    Notes
    -----
    The regularized (shrunk) covariance is given by:
    (1 - shrinkage) * cov + shrinkage * mu * np.identity(n_features)
    where mu = trace(cov) / n_features
    """
    n_features = emp_cov.shape[0]

    mu = np.trace(emp_cov) / n_features
    shrunk_cov = (1. - shrinkage) * emp_cov
    shrunk_cov.flat[::n_features + 1] += shrinkage * mu
    return shrunk_cov

class Ada_TSNC():
    def __init__(self, ada = True, sup = False):
        self.ada = ada
        self.sup = sup
    def fit(self,raw_mat,raw_y):
        self.ky_num = np.bincount(raw_y)   # 返回一个数组，其长度等于a中元素最大值加1，每个元素值则是它当前索引值在a中出现的次数。
        self.y_num = np.sum(self.ky_num)   # 求和
        self.all_R_mean = mean_riemann(raw_mat)   # 求黎曼均值
        raw_X = tangent_space(raw_mat, self.all_R_mean)   # 求切空间映射向量
        self.classes = np.unique(raw_y)
        self.s_means = _class_means(raw_X, raw_y)   # 求出类均值（质心）
        self.R_means = untangent_space(self.s_means, self.all_R_mean)  # 指数映射将切向量投影回SPD矩阵。
    def predict_fit(self, new_mat, new_y = None):
        new_X = np.squeeze(tangent_space(np.expand_dims(new_mat, axis=0), self.all_R_mean))  ## 式3-7 # axis = 0时，[]加在最外面；axis = 1时，给每一行都加[]；axis = 2时，给每一个元素都加[]
        self.y_s_distance = np.sqrt(np.sum((new_X - self.s_means)**2, axis=1))   # **表示次方
        self.pre_y = self.classes[np.argmin(self.y_s_distance)]   # np.argmin函数：按照axis的要求返回最小的数/最大的数的下标
        if self.ada ==True:
            if self.sup == False:
                y_use = self.pre_y
            if self.sup == True:
                y_use = new_y
            new_idx = np.squeeze(np.argwhere(self.classes==y_use))
            self.all_R_mean = geodesic(self.all_R_mean, new_mat, 1/(self.y_num + 1))
            self.R_means[new_idx] = geodesic(self.R_means[new_idx], new_mat, 1/(self.ky_num[new_idx] + 1))
            self.s_means = tangent_space(self.R_means, self.all_R_mean)
            self.ky_num[new_idx] =  self.ky_num[new_idx]+1
            self.y_num = self.y_num = np.sum(self.ky_num)
            return self.pre_y
        else:
            return self.pre_y

class Ada_TSLDA():
    def __init__(self, ada = False, sup = False):
        self.ada = ada
        self.sup = sup 

    def fit(self, raw_mat, raw_y):
        self.ky_num = np.bincount(raw_y)
        self.y_num = np.sum(self.ky_num)
        self.classes_ = np.unique(raw_y)
        self.priors_ = self.ky_num / float(self.y_num)
        self.all_R_mean = mean_riemann(raw_mat)
        raw_X = tangent_space(raw_mat,self.all_R_mean)
        self._max_components = min(len(self.classes_) - 1, raw_X.shape[1])
        self.s_means = _class_means(raw_X, raw_y)
        self.R_means = untangent_space(self.s_means, self.all_R_mean)
        self.covariance_ = _class_cov(raw_X, raw_y)
        self.Sw = self.covariance_[0] + self.covariance_[1]
        self.Sb = np.dot(np.atleast_2d((self.s_means[0]-self.s_means[1])).T,
                              np.atleast_2d((self.s_means[0]-self.s_means[1])))
        evals, evecs = eigh(self.Sb, shrunk_covariance(self.Sw, shrinkage=1/(self.y_num+1)))
        evecs = evecs[:, np.argsort(evals)[::-1]]  
        self.coef_ = np.dot(self.s_means, evecs).dot(evecs.T)
        self.intercept_ = (-0.5 * np.diag(np.dot(self.s_means, self.coef_.T)) +
                           np.log(self.priors_))
        return self

    def predict_fit(self, new_mat, new_y):
        new_X = tangent_space(np.expand_dims(new_mat,axis = 0),self.all_R_mean)
        scores = np.dot(new_X, self.coef_.T) + self.intercept_
        indices = scores.argmax(axis=1)
        self.pre_y = self.classes_[indices]
        if self.ada ==True:
            if self.sup == False:
                y_use = self.pre_y
            if self.sup == True:
                y_use = new_y
            new_idx = np.squeeze(np.argwhere(self.classes_==y_use))
            self.covariance_[new_idx] = 1/(self.ky_num[new_idx]+1)*self.covariance_[new_idx]
            +self.ky_num[new_idx]*logm(recenter(new_mat,self.R_means[new_idx]))/(self.y_num*(self.ky_num[new_idx]+1))
            self.covariance_ = self.covariance_*self.y_num/(self.y_num+1)
            self.all_R_mean = geodesic(self.all_R_mean, new_mat, 1/(self.y_num + 1))
            self.R_means[new_idx] = geodesic(self.R_means[new_idx], new_mat, 1/(self.ky_num[new_idx] + 1))
            self.s_means = tangent_space(self.R_means, self.all_R_mean)
            self.ky_num[new_idx] =  self.ky_num[new_idx]+1
            self.y_num = self.y_num+1
            self.priors_ = self.ky_num / float(self.y_num)
            self.Sw = self.covariance_[0] + self.covariance_[1]
            self.Sb = np.dot(np.atleast_2d((self.s_means[0]-self.s_means[1])).T,
                                  np.atleast_2d((self.s_means[0]-self.s_means[1])))
            evals, evecs = eigh(self.Sb, shrunk_covariance(self.Sw, shrinkage=1/(self.y_num+1)))
            evecs = evecs[:, np.argsort(evals)[::-1]]  
            self.coef_ = np.dot(self.s_means, evecs).dot(evecs.T)
            self.intercept_ = (-0.5 * np.diag(np.dot(self.s_means, self.coef_.T)) +
                               np.log(self.priors_))            
            return self.pre_y
        if self.ada == False:
            return self.pre_y

