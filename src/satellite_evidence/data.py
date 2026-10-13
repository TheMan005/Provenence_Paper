"""Discovery, CRS-aware annotation alignment, and frozen RGB normalization."""
import json
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import rasterio
from pyproj import CRS, Transformer
from rasterio.features import rasterize
from shapely.geometry import box, shape, mapping
from shapely.ops import transform as geometry_transform

from .common import output, sha256, write_json


def tile_id(path):
    match = re.search(r'(AOI_\d+_[A-Za-z]+_img\d+)', Path(path).stem, re.I)
    if not match:
        raise ValueError(f'Cannot derive stable SpaceNet tile ID: {path}')
    return match.group(1)


def footprints(annotation, raster_crs, fallback_crs='EPSG:4326'):
    document = json.loads(Path(annotation).read_text())
    annotation_crs = document.get('crs', {}).get('properties', {}).get('name', fallback_crs)
    converter = Transformer.from_crs(annotation_crs, raster_crs, always_xy=True)
    polygons = []
    for feature in document.get('features', []):
        if feature.get('geometry') is None:
            continue
        geom = geometry_transform(converter.transform, shape(feature['geometry']))
        if not geom.is_valid:
            geom = geom.buffer(0)
        if not geom.is_empty and geom.geom_type in ('Polygon', 'MultiPolygon'):
            polygons.append(geom)
    return polygons


def inspect_tile(image, annotation, cfg):
    with rasterio.open(image) as src:
        if src.crs is None:
            raise ValueError(f'Missing raster CRS: {image}')
        bands = cfg['inventory']['rgb_bands']
        if len(bands) != 3 or max(bands) > src.count or min(bands) < 1:
            raise ValueError(f'Invalid RGB band mapping for {image}')
        colors = [c.name for c in src.colorinterp]
        named = {'red', 'green', 'blue'}
        if named.issubset(colors) and [colors[i - 1] for i in bands] != ['red', 'green', 'blue']:
            raise ValueError('RGB mapping conflicts with explicit raster color interpretation')
        polygons = footprints(annotation, src.crs, cfg['inventory']['annotation_crs'])
        inside = [p.intersection(box(*src.bounds)) for p in polygons if p.intersects(box(*src.bounds))]
        if polygons and not inside:
            raise ValueError(f'Footprints do not intersect raster: {image}')
        mask = rasterize([(mapping(p), 1) for p in inside if not p.is_empty],
                         out_shape=(src.height, src.width), transform=src.transform,
                         fill=0, dtype='uint8') if inside else np.zeros((src.height, src.width), np.uint8)
        valid = np.all(src.read_masks(bands) > 0, axis=0)
        if inside and not np.any(mask & valid):
            raise ValueError(f'Footprints fall outside valid raster pixels: {image}')
        raw = src.read(bands)
        valid &= np.isfinite(raw).all(axis=0)
        if not valid.any():
            raise ValueError(f'No finite valid RGB pixels: {image}')
        lon, lat = Transformer.from_crs(src.crs, 'EPSG:4326', always_xy=True).transform(*src.xy(src.height // 2, src.width // 2))
        zone = int((lon + 180) // 6) + 1
        metric_crs = CRS.from_epsg((32600 if lat >= 0 else 32700) + zone)
        metric = geometry_transform(Transformer.from_crs(src.crs, metric_crs, always_xy=True).transform, box(*src.bounds))
        minx, miny, maxx, maxy = metric.bounds
        return dict(tile_id=tile_id(image), city=cfg['inventory']['city'], image_path=str(image.resolve()),
                    annotation_path=str(annotation.resolve()), sha256=sha256(image),
                    width=src.width, height=src.height, channels=src.count, dtype=src.dtypes[0],
                    crs=src.crs.to_string(), metric_crs=metric_crs.to_string(),
                    transform=json.dumps(list(src.transform)[:6]), bounds=json.dumps(list(src.bounds)),
                    nodata=json.dumps(src.nodata), colorinterp=json.dumps(colors),
                    raw_min=json.dumps([float(a[valid].min()) for a in raw]) if valid.any() else '[]',
                    raw_max=json.dumps([float(a[valid].max()) for a in raw]) if valid.any() else '[]',
                    valid_fraction=float(valid.mean()), building_count=len(inside), building_fraction=float(mask.mean()),
                    xmin=minx, ymin=miny, xmax=maxx, ymax=maxy,
                    resolution_m=float(np.sqrt(metric.area / (src.width * src.height))),
                    annotation_alignment='intersects_valid_raster' if inside else 'empty_annotation')


def inventory(cfg):
    root = Path(cfg['data_root'])
    images = sorted(p for p in root.rglob('*') if p.suffix.lower() in ('.tif', '.tiff')
                    and any(part.lower() in ('ps-rgb', 'rgb-pansharpen') for part in p.parts))
    if not images:
        raise ValueError(f'No extracted PS-RGB or RGB-PanSharpen imagery under {root}. Never read a partial archive.')
    inspection_limit = cfg['inventory'].get('inspection_limit')
    if inspection_limit and len(images) > inspection_limit:
        images = [images[i] for i in np.linspace(0, len(images) - 1, inspection_limit).astype(int)]
    annotations = {}
    for p in root.rglob('*.geojson'):
        if 'road' in str(p).lower():
            continue
        try:
            key = tile_id(p)
        except ValueError:
            continue
        if key in annotations:
            raise ValueError(f'Ambiguous annotation for {key}: {annotations[key]}, {p}')
        annotations[key] = p
    records = []
    for p in images:
        key = tile_id(p)
        if key not in annotations:
            raise ValueError(f'No building annotation matched to {key}')
        records.append(inspect_tile(p, annotations[key], cfg))
    frame = pd.DataFrame(records)
    if cfg['inventory']['city'] == 'auto':
        frame['city'] = frame.tile_id.str.extract(r'AOI_\d+_([A-Za-z]+)_')[0]
    if frame.tile_id.duplicated().any():
        raise ValueError('Duplicate imagery tile IDs')
    frame.to_csv(output(cfg) / 'inventory.csv', index=False)
    # Select throughout the density ranking rather than the first N files.
    count = min(cfg['inventory']['pilot_tiles'], len(frame))
    ordered = frame.sort_values(['building_fraction', 'tile_id'])
    pilot = ordered.iloc[np.linspace(0, len(ordered) - 1, count).astype(int)]
    pilot.to_csv(output(cfg) / 'pilot.csv', index=False)
    return frame


def load_raw(row, cfg):
    with rasterio.open(row['image_path']) as src:
        bands = cfg['inventory']['rgb_bands']
        image = src.read(bands).transpose(1, 2, 0).astype(np.float32)
        valid = np.all(src.read_masks(bands) > 0, axis=0)
        valid &= np.isfinite(image).all(axis=2)
        polygons = footprints(row['annotation_path'], src.crs, cfg['inventory']['annotation_crs'])
        object_masks = [rasterize([(mapping(p), 1)], out_shape=valid.shape,
                                 transform=src.transform, dtype='uint8') for p in polygons]
    return image, valid, [m for m in object_masks if m.any()]


def fit_normalization(frame, cfg):
    settings = cfg['normalization']
    training = frame[frame.split == 'train'].sort_values('tile_id')
    if training.empty:
        raise ValueError('No training tiles available to fit normalization')
    mode = settings['mode']
    if mode == 'fixed':
        lower, upper = settings['lower'], settings['upper']
    elif mode == 'train_percentile':
        values = []
        for _, row in training.iloc[:settings['max_tiles']].iterrows():
            image, valid, _ = load_raw(row, cfg)
            pixels = image[valid]
            if len(pixels):
                values.append(pixels[::max(1, len(pixels) // 10000)])
        if not values:
            raise ValueError('Training tiles contain no valid pixels')
        lower, upper = np.percentile(np.concatenate(values), settings['percentiles'], axis=0)
        upper = np.maximum(upper, np.asarray(lower) + 1)
    else:
        raise ValueError(f'Unknown normalization policy: {mode}')
    stats = dict(mode=mode, lower=np.broadcast_to(lower, (3,)).tolist(), upper=np.broadcast_to(upper, (3,)).tolist(),
                 rgb_bands=cfg['inventory']['rgb_bands'], fitted_split='train',
                 training_tile_ids=training.iloc[:settings['max_tiles']].tile_id.tolist(),
                 split_sha256=sha256(output(cfg) / 'splits.csv'))
    write_json(output(cfg) / 'normalization.json', stats, immutable=True)
    return stats


def normalize(image, valid, stats):
    lower, upper = np.asarray(stats['lower']), np.asarray(stats['upper'])
    if np.any(upper <= lower):
        raise ValueError('Normalization upper bounds must exceed lower bounds')
    image = np.clip((np.where(valid[..., None], image, lower) - lower) / (upper - lower), 0, 1)
    image[~valid] = 0
    return np.rint(image * 255).astype(np.uint8)


def previews(cfg, limit=12):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root = output(cfg)
    frame = pd.read_csv(root / 'pilot.csv')
    stats_path = root / 'normalization.json'
    if not stats_path.exists():
        raise ValueError('Create splits and fit training normalization before previews')
    stats = json.loads(stats_path.read_text())
    directory = root / 'previews'
    directory.mkdir(exist_ok=True)
    for _, row in frame.iloc[:limit].iterrows():
        image, valid, masks = load_raw(row, cfg)
        rgb = normalize(image, valid, stats)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(rgb)
        if masks:
            ax.contour(np.maximum.reduce(masks), levels=[0.5], colors=['lime'], linewidths=0.6)
        ax.set_title(f"{row.tile_id}: {row.building_count} footprints")
        ax.axis('off')
        fig.savefig(directory / f'{row.tile_id}.png', bbox_inches='tight', dpi=100)
        plt.close(fig)
