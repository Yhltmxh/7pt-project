"""Training, EMA and complete epoch-boundary state restoration."""
from __future__ import annotations

import copy
import math
import os
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: to_device(item, device) for key, item in value.items()}
    return value


class ModelEMA:
    def __init__(self, model, decay):
        self.module = copy.deepcopy(model).eval().requires_grad_(False)
        self.decay = decay
        self.updates = 0

    @torch.no_grad()
    def update(self, model):
        source_state = model.state_dict()
        for name, value in self.module.state_dict().items():
            source = source_state[name].detach()
            if value.is_floating_point():
                value.mul_(self.decay).add_(source, alpha=1 - self.decay)
            else:
                value.copy_(source)
        self.updates += 1


def capture_rng(generator):
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(), 'loader': generator.get_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state, generator):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    generator.set_state(state['loader'].cpu())
    if state['cuda'] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda']])


def save_checkpoint(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix='.pt')
    os.close(fd)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def load_checkpoint(path, expected_identity=None, class_order=None):
    # Only load project-generated, trusted local checkpoints; legacy layouts fail closed.
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if checkpoint.get('schema_version') != 1:
        raise ValueError('Unsupported or legacy checkpoint')
    if class_order is not None and checkpoint.get('class_order') != list(class_order):
        raise ValueError('Checkpoint class order mismatch')
    if expected_identity is not None and checkpoint.get('identity_hash') != expected_identity:
        raise ValueError('Checkpoint identity mismatch: source/config/data/seed changed')
    return checkpoint


def build_optimizer(model, cfg):
    encoders = list(model.encoder_parameters())
    heads = list(model.head_parameters())
    return torch.optim.AdamW([
        {'params': encoders, 'lr': cfg['backbone_learning_rate']},
        {'params': heads, 'lr': cfg['head_learning_rate']},
    ], weight_decay=cfg['weight_decay'])


def build_scheduler(optimizer, cfg):
    epochs, warmup, minimum = cfg['epochs'], cfg['warmup_epochs'], cfg['min_learning_rate']
    def schedule(base):
        def multiplier(epoch):
            if epoch < warmup:
                return (epoch + 1) / warmup
            progress = min(1., max(0., (epoch - warmup) / max(1, epochs - warmup - 1)))
            return minimum / base + (1 - minimum / base) * .5 * (1 + math.cos(math.pi * progress))
        return multiplier
    return torch.optim.lr_scheduler.LambdaLR(optimizer, [schedule(g['lr']) for g in optimizer.param_groups])


def progress_batches(loader, description, enabled=True):
    # The server tees epoch logs to disk. Dynamic bars belong only on its console.
    stream = getattr(sys.stderr, 'console', sys.stderr)
    return tqdm(loader, desc=description, leave=False, file=stream,
                disable=not enabled or not getattr(stream, 'isatty', lambda: False)())


def mixup_inputs(inputs, targets, alpha):
    """Use one permutation and beta draw for every image modality; IDs stay outside forward."""
    labels = targets['diagnosis']
    if alpha <= 0 or len(labels) < 2:
        return inputs, targets, targets, 1.0
    lam = float(np.random.beta(alpha, alpha))
    permutation = torch.randperm(len(labels), device=labels.device)
    mixed = dict(inputs)
    for name in ('clinical', 'dermoscopic'):
        if inputs.get(name) is not None:
            image = inputs[name]
            mixed[name] = lam * image + (1 - lam) * image[permutation]
    other = dict(targets)
    other['diagnosis'] = labels[permutation]
    return mixed, targets, other, lam


def prediction_cross_entropy(predictions):
    """Hard-label evaluation CE; no augmentation, Mixup or target smoothing."""
    logits = np.asarray(predictions['logits'], dtype=np.float64)
    labels = predictions['y_true']
    shifted = logits - logits.max(axis=1, keepdims=True)
    return float((np.log(np.exp(shifted).sum(axis=1)) - shifted[np.arange(len(labels)), labels]).mean())


def train_epoch(model, loader, optimizer, criterion, device, scaler, cfg, ema=None):
    model.train()
    accumulation = cfg['gradient_accumulation']
    optimizer.zero_grad(set_to_none=True)
    totals = {'total': 0., 'diagnosis': 0., 'attributes': 0.}
    samples = correct = 0
    updates = skipped = 0
    batches = progress_batches(loader, 'Training', cfg.get('progress_bar', True))
    for step, (inputs, targets, _) in enumerate(batches):
        inputs, targets = to_device(inputs, device), to_device(targets, device)
        alpha = cfg.get('mixup_alpha', 0.0)
        if alpha and getattr(criterion, 'attribute_lambda', 0) > 0:
            raise ValueError('Mixup currently supports diagnosis-only training')
        inputs, targets_a, targets_b, lam = mixup_inputs(inputs, targets, alpha)
        batch_size = len(targets['diagnosis'])
        window_start = (step // accumulation) * accumulation * loader.batch_size
        window_samples = min(accumulation * loader.batch_size, len(loader.dataset) - window_start)
        with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
            output = model(**inputs)
            losses = criterion(output, targets_a)
            if lam != 1.0:
                other_losses = criterion(output, targets_b)
                losses = {key: lam * value + (1 - lam) * other_losses[key]
                          for key, value in losses.items()}
            loss = losses['total'] * (batch_size / window_samples)
        if not all(torch.isfinite(value).all() for value in losses.values()):
            raise FloatingPointError('Non-finite loss; stopping without a complete artifact')
        scaler.scale(loss).backward()
        for key in totals:
            totals[key] += float(losses[key].detach()) * batch_size
        samples += batch_size
        predicted = output['diagnosis_logits'].detach().argmax(1)
        correct += lam * int((predicted == targets_a['diagnosis']).sum())
        correct += (1 - lam) * int((predicted == targets_b['diagnosis']).sum())
        if (step + 1) % accumulation == 0 or step + 1 == len(loader):
            scaler.unscale_(optimizer)
            gradients_finite = all(torch.isfinite(p.grad).all().item()
                                   for p in model.parameters() if p.grad is not None)
            if gradients_finite:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['max_grad_norm'], error_if_nonfinite=True)
            elif not scaler.is_enabled():
                raise FloatingPointError('Non-finite gradients without AMP; stopping')
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= old_scale:
                updates += 1
                if ema is not None:
                    ema.update(model)
            else:
                skipped += 1
            optimizer.zero_grad(set_to_none=True)
    if updates == 0:
        raise FloatingPointError('Every optimizer step overflowed; reduce AMP scale or inspect the model')
    return {**{key: value / samples for key, value in totals.items()},
            'optimizer_updates': updates, 'skipped_optimizer_steps': skipped, 'accuracy': correct / samples}


@torch.no_grad()
def collect_predictions(model, loader, device, progress_bar=True):
    from src.evaluation.metrics import softmax
    model.eval()
    logits, labels, sample_ids, lesion_ids, source_rows = [], [], [], [], []
    for inputs, targets, ids in progress_batches(loader, 'Evaluating', progress_bar):
        output = model(**to_device(inputs, device))['diagnosis_logits'].float()
        if not torch.isfinite(output).all():
            raise FloatingPointError('Non-finite prediction logits')
        logits.append(output.cpu().numpy())
        labels.append(targets['diagnosis'].numpy())
        sample_ids.extend(ids['sample_id'])
        lesion_ids.extend(ids['lesion_id'])
        source_rows.extend(ids['source_row_index'].tolist())
    if not logits:
        raise ValueError('Empty evaluation partition')
    combined = np.concatenate(logits)
    return {'logits': combined, 'probabilities': softmax(combined), 'y_true': np.concatenate(labels),
            'sample_id': np.asarray(sample_ids, dtype=str), 'lesion_id': np.asarray(lesion_ids, dtype=str),
            'source_row_index': np.asarray(source_rows, dtype=np.int64)}


def state_hash(model):
    import hashlib
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()
