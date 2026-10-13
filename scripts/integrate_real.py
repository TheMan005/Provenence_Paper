"""Resume completed stages without rewriting split/normalization/generation manifests."""
import json

from satellite_evidence.common import config, output
from satellite_evidence.data import inventory, fit_normalization, previews
from satellite_evidence.splits import create_splits, load_splits
from satellite_evidence.manipulation import generate
from satellite_evidence.training import train, samples_frame
from satellite_evidence.evaluation import evaluate
from satellite_evidence.reporting import report

cfg = config('configs/vegas_smoke.yaml')
root = output(cfg)
if not (root / 'inventory.csv').exists():
    inventory(cfg)
if not (root / 'splits.csv').exists():
    create_splits(cfg)
splits = load_splits(cfg)
if not (root / 'normalization.json').exists():
    fit_normalization(splits, cfg)
previews(cfg)
if not (root / 'samples.csv').exists():
    generate(cfg)
samples = samples_frame(cfg)
if not (root / 'training/best.pt').exists():
    train(cfg)
# evaluate always checks frozen checkpoint/split bindings rather than refitting on test.
metrics = evaluate(cfg)
for row in samples[samples.split == 'test'].iloc[:3].itertuples():
    report(cfg, row.sample_id)
print(json.dumps({'status': 'complete', 'real_tiles': len(splits), 'samples': len(samples),
                  'output_root': cfg['output_root'], 'note': 'Short engineering smoke; no research performance claim'}), flush=True)
