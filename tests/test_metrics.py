import numpy as np
import pytest

from src.evaluation.metrics import CLASS_ORDER, classification_metrics, softmax, validate_predictions


def test_fixed_five_class_denominator_and_missing_auc():
    metrics = classification_metrics([[8, 0, 0, 0, 0]], np.array([0]))
    assert metrics['diagnosis_macro_f1'] == pytest.approx(.2)
    assert metrics['balanced_accuracy'] == pytest.approx(.2)
    assert metrics['macro_ovr_auroc'] is None
    assert len(metrics['auroc_unavailable_classes']) == 5


def test_duplicate_and_missing_cases_rejected():
    logits = np.eye(5)
    artifact = dict(sample_id=np.array(list('abcde')), y_true=np.arange(5),
                    logits=logits, probabilities=softmax(logits), class_order=CLASS_ORDER)
    assert validate_predictions(artifact, list('edcba'))['diagnosis_macro_f1'] == 1
    with pytest.raises(ValueError, match='coverage'):
        validate_predictions(artifact, list('abcd'))
    artifact['sample_id'] = np.array(list('aacde'))
    with pytest.raises(ValueError, match='Duplicate'):
        validate_predictions(artifact)
