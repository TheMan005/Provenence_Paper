"""Custom Ed25519 sidecar prototype, explicitly NOT a C2PA credential."""
import base64
import hashlib
import json
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from .common import canonical, sha256, write_json


def generate_key(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    key = Ed25519PrivateKey.generate()
    serialized = key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
    # Create with restrictive permissions from the outset; never overwrite existing keys.
    import os
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, 'wb') as stream:
        stream.write(serialized)
    public = key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return {'public_key_base64': base64.b64encode(public).decode(), 'fingerprint': hashlib.sha256(public).hexdigest()}


def sign_asset(asset, key_path, sidecar, operations, parents=()):
    key = serialization.load_pem_private_key(Path(key_path).read_bytes(), password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise ValueError('Expected an Ed25519 private key')
    public = key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    payload = dict(format='satellite-evidence-sidecar-v1', asset_sha256=sha256(asset),
                   public_key=base64.b64encode(public).decode(), parents=[{'asset_sha256': sha256(p)} for p in parents],
                   operations=list(operations), scope='Processing history from ingestion onward; original satellite capture is not authenticated.')
    document = dict(payload=payload, signature=base64.b64encode(key.sign(canonical(payload))).decode())
    write_json(sidecar, document, immutable=True)
    return document


def verify_asset(asset, sidecar=None, trust_policy=None):
    sidecar = Path(sidecar) if sidecar else Path(str(asset) + '.provenance.json')
    if not sidecar.exists():
        return {'status': 'missing_credentials', 'checks': {'credential_present': False}, 'trusted': False}
    checks = {'credential_present': True, 'signature_valid': False, 'asset_hash_matches': None, 'signer_trusted': None}
    try:
        document = json.loads(sidecar.read_text())
        payload = document['payload']
        if payload['format'] != 'satellite-evidence-sidecar-v1':
            raise ValueError('Unsupported sidecar format')
        if not isinstance(payload['operations'], list) or not isinstance(payload['parents'], list) or not isinstance(payload['scope'], str):
            raise ValueError('Malformed signed processing history')
        if not isinstance(payload['asset_sha256'], str) or len(payload['asset_sha256']) != 64:
            raise ValueError('Malformed asset hash')
        public = base64.b64decode(payload['public_key'], validate=True)
        fingerprint = hashlib.sha256(public).hexdigest()
        Ed25519PublicKey.from_public_bytes(public).verify(base64.b64decode(document['signature'], validate=True), canonical(payload))
        checks['signature_valid'] = True
    except (ValueError, TypeError, KeyError, InvalidSignature, json.JSONDecodeError) as error:
        return {'status': 'invalid_signature', 'checks': checks, 'trusted': False, 'detail': type(error).__name__}
    checks['asset_hash_matches'] = sha256(asset) == payload['asset_sha256']
    if not checks['asset_hash_matches']:
        return {'status': 'asset_hash_mismatch', 'checks': checks, 'trusted': False, 'signer_fingerprint': fingerprint}
    trusted = []
    if trust_policy and Path(trust_policy).exists():
        policy = json.loads(Path(trust_policy).read_text())
        trusted = policy.get('trusted_ed25519_sha256', [])
    checks['signer_trusted'] = fingerprint in trusted
    return {'status': 'valid_trusted' if checks['signer_trusted'] else 'valid_untrusted', 'checks': checks,
            'trusted': checks['signer_trusted'], 'signer_fingerprint': fingerprint,
            'scope': payload['scope'], 'recorded_operations': payload['operations'],
            'recorded_parents': payload['parents'], 'parent_assets_verified': False}
