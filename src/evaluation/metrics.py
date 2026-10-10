from __future__ import annotations

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

CLASS_ORDER = ('BCC', 'NEV', 'MEL', 'MISC', 'SK')


def softmax(logits):
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2 or logits.shape[1] != 5 or not np.isfinite(logits).all():
        raise ValueError('Expected finite logits [N,5]')
    values = np.exp(logits - logits.max(axis=1, keepdims=True))
    return values / values.sum(axis=1, keepdims=True)


def classification_metrics(logits, y_true):
    p = softmax(logits)
    y = np.asarray(y_true)
    if y.shape != (len(p),) or not len(y) or not np.issubdtype(y.dtype, np.integer):
        raise ValueError('Expected one integer diagnosis per nonempty prediction row')
    if np.any((y < 0) | (y > 4)):
        raise ValueError('Diagnosis index outside fixed five-class order')
    pred = p.argmax(1)
    precision, recall, f1, support = precision_recall_fscore_support(
        y, pred, labels=np.arange(5), zero_division=0)
    per_class = {}
    missing = []
    aucs = []
    for index, name in enumerate(CLASS_ORDER):
        binary = y == index
        auc = float(roc_auc_score(binary, p[:, index])) if binary.any() and (~binary).any() else None
        if auc is None:
            missing.append(name)
        else:
            aucs.append(auc)
        per_class[name] = {'precision': float(precision[index]), 'recall': float(recall[index]),
                           'f1': float(f1[index]), 'support': int(support[index]), 'ovr_auroc': auc}
    return {'diagnosis_macro_f1': float(f1.mean()), 'accuracy': float((pred == y).mean()),
            'balanced_accuracy': float(recall.mean()),
            'balanced_accuracy_definition': 'mean recall over fixed five labels; missing support contributes zero',
            'macro_ovr_auroc': float(np.mean(aucs)) if not missing else None,
            'auroc_unavailable_classes': missing, 'class_order': list(CLASS_ORDER),
            'n_samples': len(y), 'per_class': per_class}


def validate_predictions(artifact, expected_ids=None):
    ids = np.asarray(artifact['sample_id']).astype(str)
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate prediction sample_id')
    if tuple(artifact['class_order']) != CLASS_ORDER:
        raise ValueError('Prediction class order mismatch')
    if expected_ids is not None and set(ids) != set(map(str, expected_ids)):
        raise ValueError('Prediction sample coverage mismatch')
    p = softmax(artifact['logits'])
    if len(ids) != len(p) or len(artifact['y_true']) != len(ids):
        raise ValueError('Prediction rows are misaligned')
    if not np.allclose(p, artifact['probabilities'], atol=1e-6):
        raise ValueError('Probabilities disagree with logits')
    return classification_metrics(artifact['logits'], artifact['y_true'])
