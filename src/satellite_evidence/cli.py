import argparse
import json
from pathlib import Path

from .common import config, output


def run(cfg):
    from .data import inventory, fit_normalization, previews
    from .splits import create_splits
    from .manipulation import generate
    from .training import train
    from .evaluation import evaluate
    from .reporting import report
    inventory(cfg)
    splits = create_splits(cfg)
    fit_normalization(splits, cfg)
    previews(cfg)
    samples = generate(cfg)
    train(cfg)
    metrics = evaluate(cfg)
    for row in samples[samples.split == 'test'].iloc[:3].itertuples():
        report(cfg, row.sample_id)
    return {'output_root': cfg['output_root'], 'samples': len(samples),
            'note': 'Engineering smoke/pilot outputs; no research performance claim',
            'test': metrics['test']['image_level']}


def main(argv=None):
    parser = argparse.ArgumentParser(description='Satellite evidence research baseline')
    parser.add_argument('--config', default='configs/pilot.yaml')
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ['inventory', 'split', 'normalize', 'previews', 'generate', 'train', 'evaluate', 'validate', 'run', 'status', 'extract', 'fixtures']:
        sub.add_parser(name)
    reports = sub.add_parser('report')
    reports.add_argument('--sample-id', required=True)
    reports.add_argument('--sidecar')
    keygen = sub.add_parser('keygen')
    keygen.add_argument('--key', required=True)
    sign = sub.add_parser('sign')
    sign.add_argument('--asset', required=True)
    sign.add_argument('--key', required=True)
    sign.add_argument('--sidecar', required=True)
    sign.add_argument('--operation', action='append', required=True)
    sign.add_argument('--parent', action='append', default=[])
    verify = sub.add_parser('verify')
    verify.add_argument('--asset', required=True)
    verify.add_argument('--sidecar')
    args = parser.parse_args(argv)
    cfg = config(args.config)
    command = args.command
    if command in ('status', 'extract'):
        from .archive import status, extract
        result = (status if command == 'status' else extract)(cfg['data_root'])
    elif command == 'fixtures':
        from .fixtures import create_fixtures
        result = {'synthetic_engineering_fixtures': str(create_fixtures(cfg['data_root']))}
    elif command == 'run':
        result = run(cfg)
    elif command == 'inventory':
        from .data import inventory
        result = {'inventoried_tiles': len(inventory(cfg))}
    elif command in ('split', 'normalize', 'validate'):
        from .splits import create_splits, load_splits, validate_splits
        if command == 'split':
            frame = create_splits(cfg)
            result = frame.split.value_counts().to_dict()
        else:
            frame = load_splits(cfg)
            if command == 'normalize':
                from .data import fit_normalization
                result = fit_normalization(frame, cfg)
            else:
                from .training import samples_frame
                result = validate_splits(frame, cfg['split']['buffer_m'])
                if (output(cfg) / 'samples.csv').exists():
                    result['validated_derivatives'] = len(samples_frame(cfg))
    elif command == 'previews':
        from .data import previews
        previews(cfg)
        result = {'preview_directory': str(output(cfg) / 'previews')}
    elif command == 'generate':
        from .manipulation import generate
        frame = generate(cfg)
        result = frame.operation.value_counts().to_dict()
    elif command == 'train':
        from .training import train
        result = train(cfg)
    elif command == 'evaluate':
        from .evaluation import evaluate
        result = evaluate(cfg)['test']['image_level']
    elif command == 'report':
        from .reporting import report
        result = report(cfg, args.sample_id, args.sidecar)
    else:
        from .provenance import generate_key, sign_asset, verify_asset
        if command == 'keygen':
            result = generate_key(args.key)
        elif command == 'sign':
            result = sign_asset(args.asset, args.key, args.sidecar, args.operation, args.parent)
        else:
            result = verify_asset(args.asset, args.sidecar, cfg['provenance']['trust_policy'])
    print(json.dumps(result, indent=2, default=str), flush=True)


if __name__ == '__main__':
    main()
