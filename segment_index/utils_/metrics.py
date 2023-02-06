import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def return_result(model, X, y_true):
    y_pred = model.predict(X)
    y_pred_prob = model.predict_proba(X)

    result = {'acc': accuracy_score(y_true, y_pred),
              'precision': precision_score(y_true, y_pred),
              'recall': recall_score(y_true, y_pred),
              'f1': f1_score(y_true, y_pred, average='macro'),
              'auc': roc_auc_score(y_true, y_pred_prob[::, 1])
              }
    return result


def return_result_by_threshold(model, X, y_true, min_v, max_v):
    y_true = y_true.reset_index(drop=True)
    y_pred_prob = model.predict_proba(X)

    target_idx = np.where((y_pred_prob[::, 1]>min_v) & (y_pred_prob[::, 1]<=max_v))[0]

    result = {'pred': len(target_idx),
              'real': y_true.loc[target_idx].sum(),
              'acc': y_true.loc[target_idx].sum()/len(target_idx)}

    return result