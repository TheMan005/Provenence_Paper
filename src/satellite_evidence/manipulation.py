"""Controlled edits; footprints select objects, edit support defines tamper masks."""
import hashlib
import json
from collections import OrderedDict

import cv2
import numpy as np
import pandas as pd

from .common import output, sha256, write_json
from .data import fit_normalization, load_raw, normalize
from .splits import load_splits, validate_derivatives


def save_rgb(path, rgb):
    if not cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
        raise IOError(f'Could not write {path}')


def choose_object(masks, valid, settings, rng):
    eligible = []
    for mask in masks:
        area = int(mask.sum())
        support = cv2.dilate(mask, np.ones((2 * settings['dilation_px'] + 1,) * 2, np.uint8))
        if settings['min_area_px'] <= area <= mask.size * settings['max_area_fraction'] and np.all(valid[support > 0]):
            eligible.append(mask)
    if not eligible:
        raise ValueError('No eligible footprint')
    return eligible[int(rng.integers(len(eligible)))]


def removal(rgb, object_mask, valid, settings):
    mask = cv2.dilate(object_mask, np.ones((2 * settings['dilation_px'] + 1,) * 2, np.uint8))
    if not np.all(valid[mask > 0]):
        raise ValueError('Edit would affect nodata')
    # TELEA is classical fast-marching inpainting, not diffusion or a learned model.
    result = cv2.inpaint(rgb, mask * 255, settings['inpaint_radius'], cv2.INPAINT_TELEA)
    result[mask == 0] = rgb[mask == 0]
    return result, mask, {'method': 'opencv_telea', 'radius': settings['inpaint_radius']}


def insertion(rgb, valid, occupied, donor_rgb, donor_mask, donor_valid, settings, rng, scale=1.0):
    margin = settings['dilation_px'] + settings['blend_px']
    support = cv2.dilate(donor_mask, np.ones((2 * margin + 1,) * 2, np.uint8))
    if not np.all(donor_valid[support > 0]):
        raise ValueError('Donor contains nodata')
    ys, xs = np.where(support > 0)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    patch = donor_rgb[y0:y1, x0:x1]
    alpha = support.astype(np.float32)
    if settings['blend_px'] > 0:
        alpha = cv2.GaussianBlur(alpha, (2 * settings['blend_px'] + 1,) * 2, 0)
        alpha *= support
    alpha = alpha[y0:y1, x0:x1]
    width = max(1, round(patch.shape[1] * scale))
    height = max(1, round(patch.shape[0] * scale))
    patch = cv2.resize(patch, (width, height), interpolation=cv2.INTER_LINEAR)
    alpha = cv2.resize(alpha, (width, height), interpolation=cv2.INTER_LINEAR)
    footprint = cv2.resize(donor_mask[y0:y1, x0:x1], (width, height), interpolation=cv2.INTER_NEAREST)
    if width >= rgb.shape[1] or height >= rgb.shape[0]:
        raise ValueError('Donor object too large for recipient')
    for _ in range(200):
        x = int(rng.integers(0, rgb.shape[1] - width + 1))
        y = int(rng.integers(0, rgb.shape[0] - height + 1))
        affected = alpha > 0
        if np.any(occupied[y:y + height, x:x + width][affected]) or not np.all(valid[y:y + height, x:x + width][affected]):
            continue
        result = rgb.copy()
        region = result[y:y + height, x:x + width]
        blended = np.rint(region * (1 - alpha[..., None]) + patch * alpha[..., None]).astype(np.uint8)
        region[affected] = blended[affected]
        mask = np.zeros(rgb.shape[:2], np.uint8)
        mask[y:y + height, x:x + width] = affected
        return result, mask, {'method': 'donor_alpha_blend', 'target_xy': [x, y],
                              'donor_box': [int(x0), int(y0), int(x1), int(y1)], 'scale': scale,
                              'blend_px': settings['blend_px'], 'inserted_footprint_pixels': int(footprint.sum())}
    raise ValueError('No unoccupied valid placement found')


def generate(cfg):
    root = output(cfg)
    if (root / 'samples.csv').exists():
        raise FileExistsError('Generation is immutable; use a new output_root for another run')
    frame = load_splits(cfg)
    frame = frame[frame.split != 'excluded'].sort_values('tile_id')
    normalization_path = root / 'normalization.json'
    stats = json.loads(normalization_path.read_text()) if normalization_path.exists() else fit_normalization(frame, cfg)
    if stats['split_sha256'] != sha256(root / 'splits.csv'):
        raise ValueError('Normalization was fitted on a different split manifest')
    directory = root / 'samples'
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError('Partial sample generation exists; preserve it and choose a fresh output_root')
    directory.mkdir(exist_ok=True)
    truth_dir = root / 'ground_truth'
    truth_dir.mkdir(exist_ok=True)
    settings = cfg['manipulation']
    cache, samples, histories, skipped = OrderedDict(), [], [], []

    def get(row):
        key = row['tile_id']
        if key not in cache:
            raw, valid, masks = load_raw(row, cfg)
            cache[key] = (normalize(raw, valid, stats), valid, masks)
            if len(cache) > settings.get('cache_tiles', 4):
                cache.popitem(last=False)
        cache.move_to_end(key)
        return cache[key]

    def record(row, sample_id, image, mask, valid, operation, donor_id, seed, parameters, original_path):
        image_path, mask_path, valid_path = [directory / f'{sample_id}_{suffix}.png' for suffix in ('rgb', 'mask', 'valid')]
        save_rgb(image_path, image)
        cv2.imwrite(str(mask_path), mask * 255)
        cv2.imwrite(str(valid_path), valid.astype(np.uint8) * 255)
        samples.append(dict(sample_id=sample_id, source_id=row.tile_id, split=row.split,
                            image_path=str(image_path), mask_path=str(mask_path), valid_path=str(valid_path),
                            original_path=str(original_path), operation=operation,
                            edited_fraction=float(mask.mean())))
        histories.append(dict(sample_id=sample_id, source_id=row.tile_id, donor_id=donor_id, operation=operation,
                              seed=seed, parameters=parameters, source_path=row.image_path,
                              source_sha256=row.sha256, image_path=str(image_path), mask_path=str(mask_path),
                              processing_history=['frozen_train_rgb_normalization', 'uint8_round', operation, 'lossless_png'],
                              output_sha256=sha256(image_path), mask_sha256=sha256(mask_path), valid_sha256=sha256(valid_path)))

    # Authentic controls undergo identical normalization, quantization, and PNG export.
    for row in frame.itertuples():
        rgb, valid, _ = get(row._asdict())
        sample_id = f'{row.tile_id}_authentic'
        path = directory / f'{sample_id}_rgb.png'
        record(row, sample_id, rgb, np.zeros(valid.shape, np.uint8), valid, 'authentic', None, cfg['seed'], {}, path)
    rng = np.random.default_rng(cfg['seed'])
    # Round-robin partitions so small smoke jobs still create validation/test edits.
    buckets = [list(group.iloc[rng.permutation(len(group))].index) for _, group in frame.groupby('split', sort=True)]
    ordering = [bucket[i] for i in range(max(map(len, buckets))) for bucket in buckets if i < len(bucket)]
    candidates = frame.loc[ordering]
    edits = 0
    # Cycle operations across spatially distributed sources, with bounded retries.
    for attempt in range(max(len(frame), settings['edits'] * 8)):
        if edits >= settings['edits']:
            break
        row = next(candidates.iloc[[attempt % len(candidates)]].itertuples())
        operation = settings['operations'][edits % len(settings['operations'])]
        seed = cfg['seed'] + int(hashlib.sha256(f'{row.tile_id}:{attempt}'.encode()).hexdigest()[:8], 16)
        local_rng = np.random.default_rng(seed)
        donor_id = None
        try:
            rgb, valid, masks = get(row._asdict())
            if operation == 'removal':
                object_mask = choose_object(masks, valid, settings, local_rng)
                edited, mask, parameters = removal(rgb, object_mask, valid, settings)
            elif operation == 'insertion':
                donors = frame[(frame.split == row.split) & (frame.tile_id != row.tile_id)]
                donors = donors[donors.resolution_m.between(row.resolution_m / settings['max_scale_ratio'],
                                                            row.resolution_m * settings['max_scale_ratio'])]
                if donors.empty:
                    raise ValueError('No same-split, scale-compatible donor')
                donor = next(donors.iloc[[int(local_rng.integers(len(donors)))]].itertuples())
                donor_rgb, donor_valid, donor_masks = get(donor._asdict())
                object_mask = choose_object(donor_masks, donor_valid, settings, local_rng)
                occupied = np.maximum.reduce(masks) if masks else np.zeros(valid.shape, np.uint8)
                edited, mask, parameters = insertion(rgb, valid, occupied, donor_rgb, object_mask, donor_valid,
                                                     settings, local_rng, donor.resolution_m / row.resolution_m)
                donor_id = donor.tile_id
            else:
                raise ValueError(f'Unsupported baseline operation: {operation}; diffusion is a separate optional adapter')
            if not np.array_equal(edited[mask == 0], rgb[mask == 0]):
                raise AssertionError('Unintended changes outside manipulation support')
            if not np.any(edited != rgb):
                raise ValueError('Edit made no pixel changes')
            record(row, f'{row.tile_id}_{operation}_{edits}', edited, mask, valid, operation, donor_id, seed,
                   parameters, directory / f'{row.tile_id}_authentic_rgb.png')
            edits += 1
        except ValueError as error:
            skipped.append({'tile_id': row.tile_id, 'operation': operation, 'reason': str(error)})
    samples = pd.DataFrame(samples)
    validate_derivatives(samples, histories, frame)
    with (root / 'samples.csv').open('x') as stream:
        samples.to_csv(stream, index=False)
    with (truth_dir / 'generation.jsonl').open('x') as stream:
        for history in histories:
            stream.write(json.dumps(history, sort_keys=True) + '\n')
    write_json(truth_dir / 'generation_summary.json', {'requested_edits': settings['edits'], 'generated_edits': edits,
               'authentic_controls': len(frame), 'skipped_attempts': skipped, 'samples_sha256': sha256(root / 'samples.csv')})
    # Detector input index omits ground-truth operation, donor, mask, and source history.
    samples[['sample_id', 'image_path', 'valid_path']].to_csv(root / 'detector_inputs.csv', index=False)
    return samples
