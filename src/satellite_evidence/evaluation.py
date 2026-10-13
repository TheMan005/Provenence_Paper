"""Validation-only threshold/calibration fitting, frozen test and robustness evaluation."""
import json

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (auc, average_precision_score, brier_score_loss, confusion_matrix,
                             precision_recall_curve, precision_recall_fscore_support)

from .common import output, sha256, write_json
from .training import Samples, load_model, samples_frame


def image_score(heatmap, valid, fraction=0.01):
    if not 0 < fraction <= 1:
        raise ValueError('Top-pixel fraction must be in (0, 1]')
    values = heatmap[valid.astype(bool)]
    if not len(values):
        raise ValueError('No valid pixels to score')
    count = max(1, int(np.ceil(len(values) * fraction)))
    return float(np.partition(values, -count)[-count:].mean())


def localization(mask, prediction, valid):
    target, prediction = mask.astype(bool) & valid, prediction.astype(bool) & valid
    if not target.any():
        # Empty authentic masks are not assigned perfect IoU; report false alarm pixels instead.
        return {'iou': None, 'pixel_f1': None, 'false_positive_fraction': float(prediction.sum() / max(1, valid.sum()))}
    intersection = np.logical_and(target, prediction).sum()
    union = np.logical_or(target, prediction).sum()
    return {'iou': float(intersection / max(1, union)),
            'pixel_f1': float(2 * intersection / max(1, target.sum() + prediction.sum())),
            'false_positive_fraction': None}


def select_thresholds(items, target_fpr):
    labels = np.array([item['mask'].any() for item in items], int)
    scores = np.array([item['score'] for item in items])
    if not np.any(labels == 0) or not np.any(labels == 1):
        raise ValueError('Validation needs both authentic and manipulated examples for operating-point selection')
    candidates = np.unique(np.r_[scores, np.nextafter(scores.max(), np.inf)])
    best_threshold, best_f1 = float(candidates[-1]), -1
    best_recall, operating = -1, float(candidates[-1])
    for threshold in candidates:
        predictions = scores >= threshold
        _, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary', zero_division=0)
        fpr = predictions[labels == 0].mean()
        if f1 > best_f1:
            best_f1, best_threshold = f1, float(threshold)
        if fpr <= target_fpr and (recall > best_recall or (recall == best_recall and threshold > operating)):
            best_recall, operating = recall, float(threshold)
    edited = [item for item in items if item['mask'].any()]
    pixel_candidates = np.linspace(0.05, 0.95, 19)
    values = [np.mean([localization(item['mask'], item['heatmap'] >= t, item['valid'])['iou'] for item in edited]) for t in pixel_candidates]
    return dict(image_threshold=best_threshold, fpr_threshold=operating,
                pixel_threshold=float(pixel_candidates[int(np.argmax(values))]), target_fpr=target_fpr,
                fitted_split='validation', validation_authentic_count=int((labels == 0).sum()),
                validation_manipulated_count=int(labels.sum()),
                score_definition='mean of top configured fraction of valid localization sigmoid scores; uncalibrated')


def predict(model, rows, size, fraction, robustness=None):
    dataset = Samples(rows, size, robustness=robustness)
    items = []
    with torch.no_grad():
        for index in range(len(dataset)):
            image, mask, valid, _ = dataset[index]
            heatmap = model(image[None]).sigmoid()[0, 0].numpy()
            valid_array = valid[0].numpy().astype(bool)
            mask_array = mask[0].numpy().astype(bool) & valid_array
            items.append(dict(row=dataset.rows.iloc[index].to_dict(), image=image.numpy().transpose(1, 2, 0),
                              mask=mask_array, valid=valid_array, heatmap=heatmap,
                              score=image_score(heatmap, valid_array, fraction)))
    return items


def classification(labels, scores, threshold):
    labels, scores = np.asarray(labels, int), np.asarray(scores)
    predictions = scores >= threshold
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary', zero_division=0)
    pr_auc = average_precision = None
    if len(np.unique(labels)) == 2:
        curve_precision, curve_recall, _ = precision_recall_curve(labels, scores)
        pr_auc = float(auc(curve_recall, curve_precision))
        average_precision = float(average_precision_score(labels, scores))
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    negatives = (labels == 0).sum()
    return dict(count=len(labels), pr_auc=pr_auc, average_precision=average_precision,
                precision=float(precision), recall=float(recall), f1=float(f1),
                confusion_matrix=matrix.tolist(), confusion_order=['authentic', 'manipulated'],
                false_positive_rate=float(matrix[0, 1] / negatives) if negatives else None)


def summarize(items, policy):
    labels = [item['mask'].any() for item in items]
    scores = [item['score'] for item in items]
    metrics = classification(labels, scores, policy['image_threshold'])
    operating = classification(labels, scores, policy['fpr_threshold'])
    per_image = []
    for item in items:
        mask = item['mask']
        area = mask.sum() / max(1, item['valid'].sum())
        values = localization(mask, item['heatmap'] >= policy['pixel_threshold'], item['valid'])
        per_image.append(dict(sample_id=item['row']['sample_id'], operation=item['row']['operation'], score=item['score'],
                              edited_fraction=float(area), size_bin='small' if area < 0.01 else 'medium' if area < 0.05 else 'large',
                              manipulated=bool(mask.any()), **values))
    table = pd.DataFrame(per_image)
    edited = table[table.manipulated]
    authentic = table[~table.manipulated]
    return dict(image_level=metrics, validation_selected_fpr_operating_point=operating,
                manipulated_localization={'mean_iou': float(edited.iou.mean()) if len(edited) else None,
                                          'mean_pixel_f1': float(edited.pixel_f1.mean()) if len(edited) else None},
                authentic_localization={'mean_false_positive_fraction': float(authentic.false_positive_fraction.mean()) if len(authentic) else None,
                                        'images_with_flagged_pixels': int((authentic.false_positive_fraction > 0).sum()), 'count': len(authentic)},
                by_operation={str(name): classification([i['mask'].any() for i in items if i['row']['operation'] in ('authentic', name)],
                                                       [i['score'] for i in items if i['row']['operation'] in ('authentic', name)],
                                                       policy['image_threshold']) for name in edited.operation.unique()},
                localization_by_operation={str(name): {'mean_iou': float(group.iou.mean()), 'mean_pixel_f1': float(group.pixel_f1.mean()), 'count': len(group)}
                                           for name, group in edited.groupby('operation')},
                localization_by_size={str(name): {'mean_iou': float(group.iou.mean()), 'mean_pixel_f1': float(group.pixel_f1.mean()), 'count': len(group)}
                                      for name, group in edited.groupby('size_bin')},
                image_detection_by_size={str(name): classification(
                    [True] * len(group) + [False] * len(authentic),
                    group.score.tolist() + authentic.score.tolist(), policy['image_threshold'])
                    for name, group in edited.groupby('size_bin')}, per_image=per_image)


def calibrated_scores(scores, calibration):
    x = np.clip(np.asarray(scores), 1e-6, 1 - 1e-6)
    logits = np.log(x / (1 - x))
    z = np.clip(calibration['coefficient'] * logits + calibration['intercept'], -60, 60)
    return 1 / (1 + np.exp(-z))


def panels(items, directory, pixel_threshold, size, limit=8):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory.mkdir(exist_ok=True, parents=True)
    for item in items[:limit]:
        original = cv2.cvtColor(cv2.imread(item['row']['original_path']), cv2.COLOR_BGR2RGB)
        original = cv2.resize(original, (size, size), interpolation=cv2.INTER_AREA)
        fig, axes = plt.subplots(1, 5, figsize=(15, 3))
        arrays = [original, item['image'], item['mask'], item['heatmap'], (item['heatmap'] >= pixel_threshold) & item['valid']]
        for ax, array, title in zip(axes, arrays, ['Original export', 'Evaluated image', 'Ground truth', 'Prediction heatmap', 'Thresholded prediction']):
            ax.imshow(array, cmap='magma' if title == 'Prediction heatmap' else 'gray', vmin=0, vmax=1 if array.dtype != np.uint8 else 255)
            ax.set_title(title)
            ax.axis('off')
        fig.suptitle(item['row']['sample_id'])
        fig.savefig(directory / f"{item['row']['sample_id']}.png", dpi=100, bbox_inches='tight')
        plt.close(fig)


def evaluate(cfg):
    root = output(cfg)
    directory = root / 'evaluation'
    directory.mkdir(exist_ok=True)
    rows = samples_frame(cfg)
    model, settings = load_model(cfg)
    torch.set_num_threads(cfg['training'].get('threads', 2))
    fraction = cfg['evaluation']['image_top_fraction']
    validation = predict(model, rows[rows.split == 'validation'], settings['size'], fraction)
    test = predict(model, rows[rows.split == 'test'], settings['size'], fraction)
    policy_path = directory / 'thresholds.json'
    binding = dict(checkpoint_sha256=sha256(root / 'training/best.pt'), splits_sha256=sha256(root / 'splits.csv'),
                   image_top_fraction=fraction)
    if policy_path.exists():
        policy = json.loads(policy_path.read_text())
        if any(policy.get(key) != value for key, value in binding.items()):
            raise ValueError('Frozen thresholds belong to a different model or split; use a new output_root')
    else:
        policy = {**select_thresholds(validation, cfg['evaluation']['target_fpr']), **binding}
        write_json(policy_path, policy, immutable=True)
    calibration_path = directory / 'calibration.json'
    if calibration_path.exists():
        calibration = json.loads(calibration_path.read_text())
        if calibration['checkpoint_sha256'] != binding['checkpoint_sha256']:
            raise ValueError('Calibration model binding mismatch')
    else:
        val_scores = np.clip([i['score'] for i in validation], 1e-6, 1 - 1e-6)
        labels = np.array([i['mask'].any() for i in validation], int)
        fitted = LogisticRegression(random_state=cfg['seed']).fit(np.log(val_scores / (1 - val_scores)).reshape(-1, 1), labels)
        calibration = dict(coefficient=float(fitted.coef_[0, 0]), intercept=float(fitted.intercept_[0]),
                           fitted_split='validation', validation_count=len(labels),
                           checkpoint_sha256=binding['checkpoint_sha256'], method='logistic score calibration')
        write_json(calibration_path, calibration, immutable=True)
    result = dict(thresholds=policy, validation=summarize(validation, policy), test=summarize(test, policy), robustness={})
    probabilities = calibrated_scores([i['score'] for i in test], calibration)
    labels = np.array([i['mask'].any() for i in test], int)
    result['calibration'] = dict(test_brier=float(brier_score_loss(labels, probabilities)),
                                 caveat='Validation-fitted estimate; tiny pilot calibration is not evidence of reliability.')
    result['abstention'] = []
    for margin in sorted(set([0, cfg['evaluation']['abstention_margin'], 0.2, 0.3])):
        keep = np.abs(probabilities - 0.5) >= margin
        result['abstention'].append(dict(margin=margin, coverage=float(keep.mean()),
                                         error_rate=float(((probabilities[keep] >= 0.5) != labels[keep]).mean()) if keep.any() else None))
    for transform in cfg['evaluation']['robustness']:
        perturbed = predict(model, rows[rows.split == 'test'], settings['size'], fraction, robustness=transform)
        result['robustness'][transform] = summarize(perturbed, policy)
    predictions = directory / 'predictions'
    predictions.mkdir(exist_ok=True)
    detector_results = []
    for item in validation + test:
        path = predictions / f"{item['row']['sample_id']}.npz"
        np.savez_compressed(path, heatmap=item['heatmap'], valid=item['valid'])
        detector_results.append(dict(sample_id=item['row']['sample_id'], score=item['score'], prediction_path=str(path),
                                     calibrated_estimate=float(calibrated_scores([item['score']], calibration)[0])))
    pd.DataFrame(detector_results).to_csv(directory / 'detector_results.csv', index=False)
    panels([i for i in test if i['mask'].any()] + [i for i in test if not i['mask'].any()], directory / 'panels', policy['pixel_threshold'], settings['size'])
    result['limitations'] = ['Pilot/smoke evaluation only; no cross-city or unseen-generator claim.',
                             'Empty authentic masks excluded from IoU/F1; false-positive pixels reported separately.',
                             'Resizing may erase small masks; labels for that condition use the transformed mask.',
                             'Threshold and calibration selection share validation; no independent calibration cohort yet.']
    write_json(directory / 'metrics.json', result)
    return result
