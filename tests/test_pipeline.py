import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import torch

from satellite_evidence.common import config, sha256
from satellite_evidence.data import inventory, load_raw, normalize, inspect_tile
from satellite_evidence.fixtures import create_fixtures
from satellite_evidence.cli import run
from satellite_evidence.splits import load_splits, validate_splits, validate_derivatives
from satellite_evidence.training import augment_pair, Samples
from satellite_evidence.training import samples_frame
from satellite_evidence.model import UNet, segmentation_loss
from satellite_evidence.evaluation import localization, select_thresholds
from satellite_evidence.archive import extract
from satellite_evidence.optional import DiffusionInpaintingAdapter, load_trufor_prediction


@pytest.fixture(scope='module')
def experiment(tmp_path_factory):
    root = tmp_path_factory.mktemp('engineering')
    cfg = config('configs/smoke.yaml')
    cfg['data_root'], cfg['output_root'] = str(root / 'data'), str(root / 'run')
    create_fixtures(cfg['data_root'])
    result = run(cfg)
    return cfg, Path(cfg['output_root']), result


def test_georeferenced_inventory_and_training_only_normalization(experiment):
    cfg, root, _ = experiment
    frame = pd.read_csv(root / 'inventory.csv')
    assert len(frame) == 24
    assert (frame.dtype == 'uint16').all()
    assert (frame.building_count > 0).all()
    assert frame.annotation_alignment.eq('intersects_valid_raster').all()
    stats = json.loads((root / 'normalization.json').read_text())
    splits = load_splits(cfg)
    assert set(stats['training_tile_ids']) == set(splits[splits.split == 'train'].tile_id)
    assert not set(stats['training_tile_ids']) & set(splits[splits.split == 'test'].tile_id)
    raw, valid, masks = load_raw(frame.iloc[0], cfg)
    assert raw.max() > 255 and sum(m.sum() for m in masks) > 0
    rgb = normalize(raw, valid, stats)
    assert rgb.dtype == np.uint8
    assert np.array_equal(normalize(np.array([[[0., 1024., 2047.]]]), np.ones((1, 1), bool),
                                    {'lower': [0] * 3, 'upper': [2047] * 3}), np.array([[[0, 128, 255]]], np.uint8))
    assert len(list((root / 'previews').glob('*.png'))) == 12


def test_alignment_error_is_detected(experiment, tmp_path):
    cfg, root, _ = experiment
    row = pd.read_csv(root / 'inventory.csv').iloc[0]
    document = json.loads(Path(row.annotation_path).read_text())
    for feature in document['features']:
        for ring in feature['geometry']['coordinates']:
            for vertex in ring:
                vertex[0] += 10
    annotation = tmp_path / 'misaligned.geojson'
    annotation.write_text(json.dumps(document))
    with pytest.raises(ValueError, match='do not intersect'):
        inspect_tile(Path(row.image_path), annotation, cfg)


def test_spatial_overlap_and_duplicate_leakage_rejected(experiment):
    cfg, _, _ = experiment
    frame = load_splits(cfg)
    assert set(frame.split) >= {'train', 'validation', 'test'}
    train = frame.index[frame.split == 'train'][0]
    test = frame.index[frame.split == 'test'][0]
    duplicated = frame.copy()
    duplicated.loc[test, 'sha256'] = frame.loc[train, 'sha256']
    with pytest.raises(ValueError, match='Identical imagery'):
        validate_splits(duplicated, 5)
    overlapped = frame.copy()
    overlapped.loc[test, ['xmin', 'ymin', 'xmax', 'ymax']] = frame.loc[train, ['xmin', 'ymin', 'xmax', 'ymax']].values
    with pytest.raises(ValueError, match='Spatial leakage'):
        validate_splits(overlapped, 5)


def test_controlled_masks_no_unintended_changes_and_originals_unchanged(experiment):
    cfg, root, _ = experiment
    samples = pd.read_csv(root / 'samples.csv')
    assert samples.operation.value_counts().to_dict() == {'authentic': 24, 'insertion': 6, 'removal': 6}
    for row in samples.itertuples():
        image = cv2.imread(row.image_path)
        original = cv2.imread(row.original_path)
        mask = cv2.imread(row.mask_path, 0) > 0
        assert np.array_equal(image[~mask], original[~mask])
        if row.operation == 'authentic':
            assert not mask.any()
        else:
            assert mask.any() and np.any(image != original)
    frame = pd.read_csv(root / 'inventory.csv')
    assert all(sha256(row.image_path) == row.sha256 for row in frame.itertuples())
    truth = [json.loads(line) for line in (root / 'ground_truth/generation.jsonl').read_text().splitlines()]
    donor = next(row for row in truth if row['donor_id'])
    original_donor = donor['donor_id']
    source_split = samples.set_index('source_id').split.to_dict()[donor['source_id']]
    donor['donor_id'] = frame.merge(load_splits(cfg)[['tile_id', 'split']], on='tile_id').query('split != @source_split').iloc[0].tile_id
    with pytest.raises(ValueError, match='Donor leakage'):
        validate_derivatives(samples, truth, load_splits(cfg))
    donor['donor_id'] = original_donor
    assert list(pd.read_csv(root / 'detector_inputs.csv').columns) == ['sample_id', 'image_path', 'valid_path']


def test_cpu_training_checkpoint_and_synchronized_augmentation(experiment):
    _, root, _ = experiment
    checkpoint = torch.load(root / 'training/best.pt', weights_only=True)
    assert checkpoint['steps'] == 2
    assert all(torch.isfinite(value).all() for value in checkpoint['state_dict'].values())
    torch.manual_seed(checkpoint['seed'])
    initial = UNet(4).state_dict()
    assert any(not torch.equal(initial[key], value) for key, value in checkpoint['state_dict'].items())
    mask = torch.zeros(1, 8, 8)
    mask[0, 2, 3] = 1
    image, transformed, valid = augment_pair(mask.repeat(3, 1, 1), mask, mask, True, True)
    assert torch.equal(image[0], transformed[0]) and torch.equal(transformed, valid)
    assert transformed[0, 5, 4] == 1


def test_residual_model_backprop_and_mask_interpolation(experiment):
    _, root, _ = experiment
    model = UNet(4, 'residual')
    image = torch.rand(2, 3, 32, 32)
    target = torch.zeros(2, 1, 32, 32)
    target[0, :, 4:9, 5:10] = 1
    loss = segmentation_loss(model(image), target, torch.ones_like(target))
    loss.backward()
    assert torch.isfinite(loss) and model.residual_stem[0].weight.grad.abs().sum() > 0
    rows = pd.read_csv(root / 'samples.csv')
    for transform in ('resize', 'jpeg', 'blur'):
        _, mask, valid, _ = Samples(rows, 31, robustness=transform)[25]
        assert set(mask.unique().tolist()) <= {0, 1}
        assert set(valid.unique().tolist()) <= {0, 1}


def test_validation_thresholds_and_authentic_false_alarm_handling(experiment):
    _, root, _ = experiment
    metrics = json.loads((root / 'evaluation/metrics.json').read_text())
    assert metrics['thresholds']['fitted_split'] == 'validation'
    assert set(metrics['robustness']) == {'resize', 'jpeg', 'blur'}
    empty = np.zeros((4, 4), bool)
    result = localization(empty, np.ones((4, 4), bool), np.ones((4, 4), bool))
    assert result['iou'] is None and result['pixel_f1'] is None and result['false_positive_fraction'] == 1
    items = [{'mask': empty, 'score': 0.8, 'heatmap': np.zeros((4, 4)), 'valid': ~empty},
             {'mask': ~empty, 'score': 0.7, 'heatmap': np.ones((4, 4)), 'valid': ~empty}]
    policy = select_thresholds(items, 0)
    assert policy['fpr_threshold'] > 0.8  # No threshold can obtain recall without exceeding zero FPR.
    assert len(list((root / 'evaluation/panels').glob('*.png'))) > 0
    report = json.loads(next((root / 'reports').glob('*.json')).read_text())
    assert report['visual_evidence']['raw_score_is_probability'] is False
    assert 'ground_truth' not in json.dumps(report)
    assert report['provenance']['status'] == 'missing_credentials'


def test_incomplete_archive_is_never_opened(tmp_path, monkeypatch):
    import satellite_evidence.archive as module
    (tmp_path / module.ARCHIVE_NAME).write_bytes(b'partial')
    monkeypatch.setattr(module.tarfile, 'open', lambda *args, **kwargs: pytest.fail('Incomplete archive was read'))
    with pytest.raises(ValueError, match='incomplete/active'):
        extract(tmp_path)


def test_optional_models_require_real_local_assets(tmp_path):
    with pytest.raises(FileNotFoundError, match='no weights'):
        DiffusionInpaintingAdapter(tmp_path / 'missing')
    path = tmp_path / 'official.npz'
    np.savez(path, map=np.full((4, 5), 0.5))
    assert load_trufor_prediction(path, (4, 5)).shape == (4, 5)
    with pytest.raises(ValueError, match='dimensions'):
        load_trufor_prediction(path, (5, 4))


def test_sample_and_split_integrity_changes_rejected(experiment):
    cfg, root, _ = experiment
    manifest = root / 'splits.csv'
    original = manifest.read_bytes()
    try:
        manifest.write_bytes(original + b'\n')
        with pytest.raises(ValueError, match='manifest was changed'):
            load_splits(cfg)
    finally:
        manifest.write_bytes(original)
    samples = pd.read_csv(root / 'samples.csv')
    image = Path(samples.iloc[0].image_path)
    original = image.read_bytes()
    try:
        image.write_bytes(original + b'modification')
        with pytest.raises(ValueError, match='integrity mismatch'):
            samples_frame(cfg)
    finally:
        image.write_bytes(original)


def test_resume_reuses_frozen_manifests_and_thresholds(experiment, monkeypatch):
    import runpy
    import satellite_evidence.common as common
    cfg, root, _ = experiment
    paths = [root / name for name in ['splits.csv', 'normalization.json', 'samples.csv', 'evaluation/thresholds.json']]
    hashes = [sha256(path) for path in paths]
    monkeypatch.setattr(common, 'config', lambda _: cfg)
    runpy.run_path('scripts/integrate_real.py', run_name='__main__')
    assert hashes == [sha256(path) for path in paths]
