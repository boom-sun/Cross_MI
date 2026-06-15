import numpy as np
from Cross_MI.auxiliary.functions import EA, RA, RPA
from pyriemann.utils.covariance import covariances


class Alignment:
    """
    Wraps alignment methods (EA, RA, RPA) with a fit/transform interface.

    EA  – Euclidean Alignment: per-batch whitening; each split is
          normalised by its own mean covariance.
    RA  – Riemannian Alignment: whitening matrix computed from the
          training set and applied to both train and test.
    RPA – Riemannian Procrustes Analysis: requires labelled target
          samples; handled only in the joint alignment() call.
    """

    def __init__(self, srate, Alignment):
        self.srate     = np.squeeze(srate)
        self.Alignment = Alignment
        self.align_method = 'NONE'

    def _parse(self):
        if isinstance(self.Alignment, str):
            return self.Alignment.upper()
        if isinstance(self.Alignment, (list, tuple)):
            joined = ' '.join(str(a) for a in self.Alignment).upper()
            return joined
        return 'NONE'

    # ------------------------------------------------------------------
    # Fit-transform interface (used by global_cross_double pipeline)
    # ------------------------------------------------------------------
    def fit_transform(self, X_train, y_train, clabel_train=[], clabel_test=[]):
        tag = self._parse()
        if 'EA' in tag:
            self.align_method = 'EA'
            return EA(X_train.copy())
        elif 'RA' in tag:
            self.align_method = 'RA'
            _, self._RA_P1 = _compute_RA_whitener(X_train)
            return _apply_RA(X_train, self._RA_P1)
        else:
            self.align_method = 'NONE'
            return X_train

    def transform(self, X):
        if self.align_method == 'EA':
            return EA(X.copy())
        elif self.align_method == 'RA' and hasattr(self, '_RA_P1'):
            return _apply_RA(X, self._RA_P1)
        return X

    # ------------------------------------------------------------------
    # Joint interface used by Model_Framework (train + test together)
    # ------------------------------------------------------------------
    def alignment(self, X_train, X_test, y_train, clabel_train=[], clabel_test=[]):
        tag = self._parse()
        if 'EA' in tag:
            self.align_method = 'EA'
            return EA(X_train.copy()), EA(X_test.copy())
        elif 'RPA' in tag:
            self.align_method = 'RPA'
            data_train, label_train, data_test = RPA(
                X_train, X_test, y_train, clabel_train, clabel_test
            )
            return data_train, data_test
        elif 'RA' in tag:
            self.align_method = 'RA'
            _, P1 = _compute_RA_whitener(X_train)
            self._RA_P1 = P1
            return _apply_RA(X_train, P1), _apply_RA(X_test, P1)
        else:
            self.align_method = 'NONE'
            return X_train, X_test


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _compute_RA_whitener(X_train):
    """Compute mean SPD matrix and its inverse-square-root from training data."""
    from pyriemann.tangentspace import mean_covariance
    covdata = covariances(X_train)
    P = mean_covariance(covdata)
    v, Q = np.linalg.eig(P)
    V = np.diag(v ** (-0.5))
    P1 = Q @ V @ np.linalg.inv(Q)
    P1 = np.real(P1)
    return P, P1


def _apply_RA(X, P1):
    """Apply whitening matrix P1 to each trial in X."""
    return np.stack([P1 @ X[s] for s in range(X.shape[0])])
