import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

def evaluate_model(y_test, y_pred):
    acc = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    if np.unique(y_test).size > 2:
        auc = np.zeros(np.shape(acc))
    else:
        auc = roc_auc_score(y_test, y_pred, multi_class='ovo')
    return acc, conf_matrix, auc
