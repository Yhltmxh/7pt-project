import copy

import pytest
import torch

from src.train_utils import load_checkpoint, save_checkpoint


def test_checkpoint_rejects_label_and_identity_drift(tmp_path):
    path = tmp_path / 'checkpoint.pt'
    save_checkpoint(path, {'schema_version': 1, 'identity_hash': 'abc',
        'class_order': ['BCC', 'NEV', 'MEL', 'MISC', 'SK'], 'model': {'tensor': torch.zeros(1)}})
    with pytest.raises(ValueError, match='class order'):
        load_checkpoint(path, class_order=['NEV', 'BCC', 'MEL', 'MISC', 'SK'])
    with pytest.raises(ValueError, match='identity'):
        load_checkpoint(path, expected_identity='def')


def test_complete_synthetic_training_and_resume(tmp_path):
    from scripts.smoke_test import run_smoke
    assert run_smoke(tmp_path, 'cpu')['passed']


def test_amp_overflow_skips_optimizer_and_ema_then_recovers():
    from src.train_utils import ModelEMA, train_epoch

    class Overflow(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value):
            return value

        @staticmethod
        def backward(ctx, gradient):
            return torch.full_like(gradient, float('inf'))

    class Model(torch.nn.Linear):
        def forward(self, *, clinical):
            return {'diagnosis_logits': super().forward(clinical)}

    class Criterion:
        calls = 0

        def __call__(self, outputs, targets):
            logits = outputs['diagnosis_logits'].float()
            if self.calls == 0:
                logits = Overflow.apply(logits)
            self.calls += 1
            loss = logits.square().mean()
            return {'total': loss, 'diagnosis': loss, 'attributes': loss.new_zeros(())}

    model = Model(2, 5)
    initial = model.weight.detach().clone()
    ema = ModelEMA(model, .9)
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    dataset = [({'clinical': torch.tensor([1., 2.])}, {'diagnosis': torch.tensor(0)}, {}) for _ in range(4)]
    loader = torch.utils.data.DataLoader(dataset, batch_size=2)
    scaler = torch.amp.GradScaler('cpu', init_scale=8., enabled=True)
    report = train_epoch(model, loader, optimizer, Criterion(), torch.device('cpu'), scaler,
                         {'gradient_accumulation': 1, 'max_grad_norm': 1.}, ema)
    assert report['skipped_optimizer_steps'] == 1
    assert report['optimizer_updates'] == ema.updates == 1
    assert all(torch.isfinite(p).all() for p in model.parameters())
    assert not torch.equal(model.weight, initial)
