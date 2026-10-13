"""Report only detector artifacts and actually performed provenance checks; no generation truth."""
import json

import cv2
import numpy as np
import pandas as pd

from .common import output, sha256, write_json
from .provenance import verify_asset


def regions(heatmap, valid, threshold, original_shape):
    mask = ((heatmap >= threshold) & valid).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    sx, sy = original_shape[1] / mask.shape[1], original_shape[0] / mask.shape[0]
    result = []
    for label in range(1, count):
        x, y, width, height, area = stats[label]
        result.append(dict(region_id=f'R{label}', bbox_original_pixels=[int(x * sx), int(y * sy), int((x + width) * sx), int((y + height) * sy)],
                           prediction_grid_pixels=int(area), mean_localization_score=float(heatmap[labels == label].mean())))
    return result


def report(cfg, sample_id, sidecar=None):
    root = output(cfg)
    inputs = pd.read_csv(root / 'detector_inputs.csv')
    detections = pd.read_csv(root / 'evaluation/detector_results.csv')
    selected = inputs[inputs.sample_id == sample_id]
    prediction = detections[detections.sample_id == sample_id]
    if len(selected) != 1 or len(prediction) != 1:
        raise ValueError('Report requires a unique evaluated sample ID')
    image_path = selected.iloc[0].image_path
    result = prediction.iloc[0]
    policy = json.loads((root / 'evaluation/thresholds.json').read_text())
    with np.load(result.prediction_path) as arrays:
        heatmap, valid = arrays['heatmap'], arrays['valid'].astype(bool)
    rgb = cv2.imread(image_path)
    flagged = regions(heatmap, valid, policy['pixel_threshold'], rgb.shape[:2])
    provenance = verify_asset(image_path, sidecar, cfg['provenance']['trust_policy'])
    flagged_image = float(result.score) >= policy['image_threshold']
    value = dict(sample_id=sample_id, asset_sha256=sha256(image_path),
                 visual_evidence={'raw_image_score': float(result.score), 'raw_score_is_probability': False,
                                  'validation_calibrated_estimate': float(result.calibrated_estimate),
                                  'calibration_limit': 'Small pilot validation fit; reliability is not established.',
                                  'image_flagged': flagged_image, 'image_threshold': policy['image_threshold'],
                                  'localization_threshold': policy['pixel_threshold'], 'regions': flagged},
                 provenance=provenance,
                 checks_performed=['asset_sha256', 'U-Net localization', 'validation-selected image and pixel thresholds']
                                  + ['provenance:' + key for key, value in provenance['checks'].items() if value is not None],
                 checks_not_performed=['independent satellite capture authentication', 'temporal/reference-image comparison',
                                       'edit mechanism attribution', 'parent-asset integrity verification'],
                 assessment='Suspected manipulation based on visual score; mechanism and scene truth remain unresolved.' if flagged_image
                            else 'Visual score below the selected threshold; authenticity is not established.',
                 limitations=['Flagged regions cannot establish that a building was inserted or removed.',
                              'Missing credentials do not establish manipulation.',
                              'A trusted processing signature does not authenticate original satellite capture.',
                              'Custom sidecar format is not C2PA-compliant.',
                              'No demonstrated cross-city or unseen-generator performance.'],
                 unresolved_questions=['What independent source or temporal imagery corroborates the scene?',
                                       'Does the detector generalize to this geography and editing method?'])
    directory = root / 'reports'
    directory.mkdir(exist_ok=True)
    write_json(directory / f'{sample_id}.json', value)
    text = f"Evidence assessment: {sample_id}\n\n{value['assessment']}\n\n"
    text += f"Visual score: {result.score:.4f} (uncalibrated); flagged regions: {len(flagged)}.\n"
    text += f"Provenance status: {provenance['status']}.\n\nChecks performed:\n"
    text += ''.join(f'- {check}\n' for check in value['checks_performed'])
    text += '\nLimitations:\n' + ''.join(f'- {line}\n' for line in value['limitations'])
    text += '\nUnresolved questions:\n' + ''.join(f'- {line}\n' for line in value['unresolved_questions'])
    (directory / f'{sample_id}.txt').write_text(text)
    return value
