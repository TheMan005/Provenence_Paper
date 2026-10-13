"""Spatial blocks with exclusion corridors and overlap-component grouping."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import box
from shapely.strtree import STRtree

from .common import output, sha256, write_json


ACTIVE = ('train', 'validation', 'test')


def validate_splits(frame, buffer_m):
    if frame.tile_id.duplicated().any():
        raise ValueError('Duplicate original tile IDs')
    if not set(frame.split).issubset(set(ACTIVE) | {'excluded'}):
        raise ValueError('Unknown split label')
    active = frame[frame.split.isin(ACTIVE)]
    if active.groupby('sha256').split.nunique().gt(1).any():
        raise ValueError('Identical imagery crosses partitions')
    if 'coverage_group' in active and active.groupby('coverage_group').split.nunique().gt(1).any():
        raise ValueError('Overlapping coverage group crosses partitions')
    for _, group in active.groupby('metric_crs'):
        rows = group.reset_index(drop=True)
        shapes = [box(r.xmin, r.ymin, r.xmax, r.ymax) for r in rows.itertuples()]
        tree = STRtree(shapes)
        for i, geom in enumerate(shapes):
            for j in tree.query(geom.buffer(buffer_m + 1e-6)):
                if j <= i or rows.iloc[i].split == rows.iloc[j].split:
                    continue
                if geom.intersection(shapes[j]).area > 1e-6 or geom.distance(shapes[j]) < buffer_m - 1e-6:
                    raise ValueError(f'Spatial leakage: {rows.iloc[i].tile_id}, {rows.iloc[j].tile_id}')
    return {'active_tiles': len(active), 'excluded_tiles': len(frame) - len(active),
            'split_counts': active.split.value_counts().to_dict(), 'buffer_m': buffer_m}


def create_splits(cfg):
    root = output(cfg)
    if (root / 'splits.csv').exists():
        raise FileExistsError('Splits are immutable: choose a new output_root to create another experiment')
    frame = pd.read_csv(root / 'pilot.csv').reset_index(drop=True)
    settings = cfg['split']
    ratios = np.asarray(settings['ratios'], float)
    if len(ratios) != 3 or np.any(ratios <= 0) or not np.isclose(ratios.sum(), 1):
        raise ValueError('Three positive split ratios must sum to 1')
    block, buffer = settings['block_m'], settings['buffer_m']
    if block <= 0 or buffer < 0:
        raise ValueError('Invalid block or buffer dimensions')
    frame['split'] = 'excluded'
    frame['coverage_group'] = ''
    frame['block_id'] = ''
    assignments = settings.get('city_assignments', {})
    for (city, metric_crs), subset in frame.groupby(['city', 'metric_crs']):
        indexes = list(subset.index)
        geoms = [box(r.xmin, r.ymin, r.xmax, r.ymax) for r in subset.itertuples()]
        tree = STRtree(geoms)
        parent = list(range(len(geoms)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for i, geom in enumerate(geoms):
            for j in tree.query(geom):
                if geom.intersection(geoms[j]).area > 1e-6:
                    parent[find(int(j))] = find(i)
        components = {}
        for i in range(len(geoms)):
            components.setdefault(find(i), []).append(i)
        # Partition along the wider geographical axis, snapping cuts to spatial blocks.
        all_bounds = np.array([g.bounds for g in geoms])
        axis = 0 if np.ptp(all_bounds[:, [0, 2]]) >= np.ptp(all_bounds[:, [1, 3]]) else 1
        centers = (all_bounds[:, axis] + all_bounds[:, axis + 2]) / 2
        block_indexes = np.floor(centers / block).astype(int)
        unique_blocks = np.unique(block_indexes)
        if city not in assignments and len(unique_blocks) < 3:
            raise ValueError(f'{city}: need at least three separated spatial blocks; adjust block_m or pilot coverage')
        if city in assignments:
            if assignments[city] not in ACTIVE:
                raise ValueError(f'Invalid city assignment: {city}')
            cuts = None
        else:
            cumulative = np.cumsum(ratios)[:2]
            cut_indices = [int(np.clip(round(len(unique_blocks) * fraction), 1, len(unique_blocks) - 1)) for fraction in cumulative]
            if cut_indices[1] <= cut_indices[0]:
                cut_indices[1] = min(len(unique_blocks) - 1, cut_indices[0] + 1)
                cut_indices[0] = min(cut_indices[0], cut_indices[1] - 1)
            cuts = [float(unique_blocks[i] * block) for i in cut_indices]
        for component, members in components.items():
            b = np.array([geoms[i].bounds for i in members])
            low, high = b[:, axis].min(), b[:, axis + 2].max()
            split = 'excluded'
            if cuts is None:
                split = assignments[city]
            elif high <= cuts[0] - buffer:
                split = 'train'
            elif low >= cuts[0] + buffer and high <= cuts[1] - buffer:
                split = 'validation'
            elif low >= cuts[1] + buffer:
                split = 'test'
            for i in members:
                original = indexes[i]
                frame.loc[original, 'split'] = split
                frame.loc[original, 'coverage_group'] = f'{city}:{metric_crs}:{component}'
                frame.loc[original, 'block_id'] = f'{city}:{axis}:{block_indexes[i]}'
    checks = validate_splits(frame, buffer)
    missing = set(ACTIVE) - set(frame.split)
    if missing:
        raise ValueError(f'Empty partitions {missing}; reduce buffer/block size or increase geographic pilot coverage')
    with (root / 'splits.csv').open('x') as stream:
        frame.sort_values('tile_id').to_csv(stream, index=False)
    write_json(root / 'split_receipt.json', dict(manifest_sha256=sha256(root / 'splits.csv'),
               pilot_sha256=sha256(root / 'pilot.csv'), settings=settings, checks=checks), immutable=True)
    return frame


def load_splits(cfg):
    root = output(cfg)
    receipt = json.loads((root / 'split_receipt.json').read_text())
    if sha256(root / 'splits.csv') != receipt['manifest_sha256']:
        raise ValueError('Immutable split manifest was changed')
    if sha256(root / 'pilot.csv') != receipt['pilot_sha256']:
        raise ValueError('Pilot inventory was changed after splitting')
    frame = pd.read_csv(root / 'splits.csv')
    validate_splits(frame, receipt['settings']['buffer_m'])
    return frame


def validate_derivatives(samples, truth, splits):
    assignments = splits.set_index('tile_id').split.to_dict()
    if samples.sample_id.duplicated().any():
        raise ValueError('Duplicate derivative sample IDs')
    for row in samples.itertuples():
        if assignments.get(row.source_id) != row.split or row.split not in ACTIVE:
            raise ValueError('Derivative crosses source split')
    for row in truth:
        if row.get('donor_id') and assignments.get(row['donor_id']) != assignments.get(row['source_id']):
            raise ValueError('Donor leakage across partitions')
    return True
