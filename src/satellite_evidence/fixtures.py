"""Tiny synthetic georeferenced engineering fixtures, never research data."""
import json
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin
from shapely.geometry import box, mapping
from shapely.ops import transform as geometry_transform


def create_fixtures(root, count=24):
    root = Path(root)
    image_dir, annotation_dir = root / 'RGB-PanSharpen', root / 'geojson' / 'buildings'
    image_dir.mkdir(parents=True, exist_ok=True)
    annotation_dir.mkdir(parents=True, exist_ok=True)
    converter = Transformer.from_crs('EPSG:32611', 'EPSG:4326', always_xy=True)
    for i in range(count):
        name = f'AOI_2_Synthetic_img{i + 1}'
        image_path = image_dir / f'RGB-PanSharpen_{name}.tif'
        if image_path.exists():
            raise FileExistsError('Synthetic fixture destination already populated; choose another path')
        rng = np.random.default_rng(100 + i)
        raw = rng.integers(200, 1100, (3, 64, 64), dtype=np.uint16)
        transform = from_origin(650020 + (i // 2) * 100, 4000060 + (i % 2) * 100, 0.5, 0.5)
        features = []
        for object_id in range(1 + i % 3):
            x, y = 8 + object_id * 16, 10 + object_id * 10
            raw[:, y:y + 7, x:x + 9] = np.array([1800, 1500, 1200], np.uint16)[:, None, None]
            left, top = transform * (x, y)
            right, bottom = transform * (x + 9, y + 7)
            polygon = geometry_transform(converter.transform, box(left, bottom, right, top))
            features.append({'type': 'Feature', 'properties': {'building_id': object_id}, 'geometry': mapping(polygon)})
        with rasterio.open(image_path, 'w', driver='GTiff', width=64, height=64, count=3, dtype='uint16',
                           crs='EPSG:32611', transform=transform, nodata=0) as dest:
            dest.write(raw)
            dest.update_tags(ENGINEERING_FIXTURE='true', NOT_RESEARCH_DATA='true')
        (annotation_dir / f'buildings_{name}.geojson').write_text(json.dumps({'type': 'FeatureCollection', 'features': features}))
    (root / 'SYNTHETIC_FIXTURES_ONLY.txt').write_text('Synthetic engineering fixtures. Not satellite observations or research results.\n')
    return root
