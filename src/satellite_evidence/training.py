import json
import os
import random

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF

from .common import output, sha256, write_json
from .model import UNet, segmentation_loss
from .splits import load_splits, validate_derivatives


def augment_pair(image, mask, valid, horizontal=False, vertical=False):
    transform = lambda x: TF.vflip(TF.hflip(x) if horizontal else x) if vertical else (TF.hflip(x) if horizontal else x)
    return tuple(transform(x) for x in (image, mask, valid))


class Samples(Dataset):
    def __init__(self, rows, size, augment=False, robustness=None):
        self.rows = rows.reset_index(drop=True)
        self.size, self.augment, self.robustness = size, augment, robustness

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows.iloc[index]
        image = cv2.imread(row.image_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(row.mask_path, cv2.IMREAD_GRAYSCALE)
        valid = cv2.imread(row.valid_path, cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None or valid is None:
            raise ValueError(f'Missing sample assets: {row.sample_id}')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.robustness == 'resize':
            h, w = image.shape[:2]
            small = (max(1, w // 2), max(1, h // 2))
            image = cv2.resize(cv2.resize(image, small, interpolation=cv2.INTER_AREA), (w, h), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(cv2.resize(mask, small, interpolation=cv2.INTER_NEAREST), (w, h), interpolation=cv2.INTER_NEAREST)
            valid = cv2.resize(cv2.resize(valid, small, interpolation=cv2.INTER_NEAREST), (w, h), interpolation=cv2.INTER_NEAREST)
        elif self.robustness == 'jpeg':
            ok, compressed = cv2.imencode('.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 70])
            if not ok:
                raise ValueError('JPEG encoding failed')
            image = cv2.cvtColor(cv2.imdecode(compressed, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        elif self.robustness == 'blur':
            image = cv2.GaussianBlur(image, (5, 5), 1)
        elif self.robustness is not None:
            raise ValueError('Unknown robustness transform')
        shape = (self.size, self.size)
        image = cv2.resize(image, shape, interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, shape, interpolation=cv2.INTER_NEAREST)
        valid = cv2.resize(valid, shape, interpolation=cv2.INTER_NEAREST)
        image = torch.from_numpy(image.transpose(2, 0, 1).copy()).float() / 255
        mask = torch.from_numpy((mask > 0).astype(np.float32)[None])
        valid = torch.from_numpy((valid > 0).astype(np.float32)[None])
        if self.augment:
            image, mask, valid = augment_pair(image, mask, valid, bool(torch.rand(()) < 0.5), bool(torch.rand(()) < 0.5))
        return image, mask, valid, index


def samples_frame(cfg):
    root = output(cfg)
    splits = load_splits(cfg)
    samples = pd.read_csv(root / 'samples.csv')
    truth = [json.loads(line) for line in (root / 'ground_truth/generation.jsonl').read_text().splitlines()]
    summary = json.loads((root / 'ground_truth/generation_summary.json').read_text())
    if 'samples_sha256' in summary and sha256(root / 'samples.csv') != summary['samples_sha256']:
        raise ValueError('Supervised sample manifest was changed')
    records = {item['sample_id']: item for item in truth}
    for row in samples.itertuples():
        history = records[row.sample_id]
        for path, key in [(row.image_path, 'output_sha256'), (row.mask_path, 'mask_sha256'), (row.valid_path, 'valid_sha256')]:
            if key in history and sha256(path) != history[key]:
                raise ValueError(f'Generated asset integrity mismatch: {path}')
    validate_derivatives(samples, truth, splits)
    return samples


def device_for(name):
    if name.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; set training.device: cpu')
    return torch.device(name)


def train(cfg):
    root = output(cfg)
    settings = cfg['training']
    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    torch.set_num_threads(settings.get('threads', 2))
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    device = device_for(settings['device'])
    # CUDA bilinear interpolation backward is not bitwise deterministic.
    torch.use_deterministic_algorithms(device.type == 'cpu')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(cfg['seed'])
    rows = samples_frame(cfg)
    training = Samples(rows[rows.split == 'train'], settings['size'], augment=True)
    validation = Samples(rows[rows.split == 'validation'], settings['size'])
    if not len(training) or not len(validation):
        raise ValueError('Training and validation partitions must be nonempty')
    loader = DataLoader(training, batch_size=settings['batch_size'], shuffle=True,
                        generator=torch.Generator().manual_seed(cfg['seed']), num_workers=0)
    val_loader = DataLoader(validation, batch_size=settings['batch_size'], num_workers=0)
    model = UNet(settings['base_channels'], settings['model']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings['lr'])
    amp = settings['amp'] and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    directory = root / 'training'
    directory.mkdir(exist_ok=True)
    best, steps = float('inf'), 0
    history = []
    for epoch in range(settings['epochs']):
        model.train()
        losses = []
        for image, mask, valid, _ in loader:
            image, mask, valid = image.to(device), mask.to(device), valid.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                loss = segmentation_loss(model(image), mask, valid)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite training loss; inspect data before proceeding')
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
            steps += 1
            if settings.get('max_steps') and steps >= settings['max_steps']:
                break
        model.eval()
        val_losses = []
        with torch.no_grad():
            for image, mask, valid, _ in val_loader:
                val_losses.append(float(segmentation_loss(model(image.to(device)), mask.to(device), valid.to(device))))
        record = dict(epoch=epoch + 1, steps=steps, train_loss=float(np.mean(losses)), validation_loss=float(np.mean(val_losses)))
        history.append(record)
        print(json.dumps(record), flush=True)
        checkpoint = dict(state_dict=model.state_dict(), training=settings, seed=cfg['seed'], steps=steps,
                          parameter_count=sum(p.numel() for p in model.parameters()),
                          splits_sha256=sha256(root / 'splits.csv'), normalization_sha256=sha256(root / 'normalization.json'))
        torch.save(checkpoint, directory / 'last.pt')
        if record['validation_loss'] < best:
            best = record['validation_loss']
            torch.save(checkpoint, directory / 'best.pt')
        if settings.get('max_steps') and steps >= settings['max_steps']:
            break
    write_json(directory / 'history.json', history)
    write_json(directory / 'budget.json', dict(parameters=checkpoint['parameter_count'], steps=steps,
               epochs=len(history), device=str(device), amp=amp, strict_determinism=device.type == 'cpu',
               model=settings['model'], config=cfg))
    return history


def load_model(cfg):
    root = output(cfg)
    checkpoint = torch.load(root / 'training/best.pt', map_location='cpu', weights_only=True)
    for filename, key in [('splits.csv', 'splits_sha256'), ('normalization.json', 'normalization_sha256')]:
        if sha256(root / filename) != checkpoint[key]:
            raise ValueError('Checkpoint data policy mismatch')
    settings = checkpoint['training']
    model = UNet(settings['base_channels'], settings['model'])
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model, settings
