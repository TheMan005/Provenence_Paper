import hashlib
import json
from pathlib import Path

import yaml


def config(path):
    cfg = yaml.safe_load(Path(path).read_text())
    cfg['data_root'] = str(Path(cfg['data_root']).resolve())
    cfg['output_root'] = str(Path(cfg['output_root']).resolve())
    return cfg


def output(cfg):
    root = Path(cfg['output_root'])
    root.mkdir(parents=True, exist_ok=True)
    return root


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value, immutable=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x' if immutable else 'w') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
