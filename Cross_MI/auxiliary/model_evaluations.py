import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix,
    f1_score, cohen_kappa_score, matthews_corrcoef,
    recall_score, roc_auc_score,
)

METRIC_KEYS = [
    'acc', 'balanced_acc', 'macro_f1', 'kappa', 'mcc',
    'sensitivity', 'specificity', 'conf_matrix', 'auc',
]


def evaluate_model(y_test, y_pred, y_score=None):
    """Compute comprehensive classification metrics.

    Parameters
    ----------
    y_test  : array-like  Ground-truth labels.
    y_pred  : array-like  Predicted hard labels.
    y_score : array-like or None
        Continuous probability scores.
        Shape ``(n_samples,)`` for binary or ``(n_samples, n_classes)``
        for multi-class.  Required for AUC; pass None when the classifier
        does not expose probabilities.

    Returns
    -------
    dict
        Keys: acc, balanced_acc, macro_f1, kappa, mcc,
              sensitivity, specificity, conf_matrix, auc.
    """
    y_test = np.asarray(y_test)
    y_pred = np.asarray(y_pred)
    n_classes = len(np.unique(y_test))

    acc          = float(accuracy_score(y_test, y_pred))
    balanced_acc = float(balanced_accuracy_score(y_test, y_pred))
    macro_f1     = float(f1_score(y_test, y_pred, average='macro', zero_division=0))
    kappa        = float(cohen_kappa_score(y_test, y_pred))
    mcc          = float(matthews_corrcoef(y_test, y_pred))
    conf_mat     = confusion_matrix(y_test, y_pred)
    sensitivity  = float(recall_score(y_test, y_pred, average='macro', zero_division=0))
    specificity  = float(_macro_specificity(conf_mat))
    auc          = _compute_auc(y_test, y_score, n_classes)

    return {
        'acc':          acc,
        'balanced_acc': balanced_acc,
        'macro_f1':     macro_f1,
        'kappa':        kappa,
        'mcc':          mcc,
        'sensitivity':  sensitivity,
        'specificity':  specificity,
        'conf_matrix':  conf_mat,
        'auc':          auc,
    }


def _macro_specificity(conf_mat):
    """Macro-averaged specificity via one-vs-rest from a confusion matrix."""
    n = conf_mat.shape[0]
    specs = []
    for i in range(n):
        TP = conf_mat[i, i]
        FP = conf_mat[:, i].sum() - TP
        FN = conf_mat[i, :].sum() - TP
        TN = conf_mat.sum() - TP - FP - FN
        denom = TN + FP
        specs.append(float(TN / denom) if denom > 0 else 0.0)
    return float(np.mean(specs))


def _compute_auc(y_test, y_score, n_classes):
    """Return AUC from probability scores; np.nan when scores are unavailable."""
    if y_score is None:
        return np.nan
    y_score = np.asarray(y_score, dtype=float)
    try:
        if n_classes == 2:
            s = y_score[:, 1] if y_score.ndim == 2 else y_score
            return float(roc_auc_score(y_test, s))
        else:
            if y_score.ndim != 2 or y_score.shape[1] != n_classes:
                return np.nan
            return float(roc_auc_score(
                y_test, y_score, multi_class='ovr', average='macro',
            ))
    except Exception:
        return np.nan
