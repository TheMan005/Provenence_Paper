import json

from satellite_evidence.common import write_json
from satellite_evidence.provenance import generate_key, sign_asset, verify_asset


def test_signature_binding_trust_and_missing_states(tmp_path):
    asset, parent = tmp_path / 'image.png', tmp_path / 'parent.png'
    asset.write_bytes(b'engineering image bytes')
    parent.write_bytes(b'parent bytes')
    key, sidecar, policy = tmp_path / 'key.pem', tmp_path / 'credential.json', tmp_path / 'policy.json'
    assert verify_asset(asset, sidecar)['status'] == 'missing_credentials'
    public = generate_key(key)
    assert key.stat().st_mode & 0o777 == 0o600
    sign_asset(asset, key, sidecar, ['ingest', 'lossless export'], [parent])
    assert verify_asset(asset, sidecar)['status'] == 'valid_untrusted'
    write_json(policy, {'trusted_ed25519_sha256': [public['fingerprint']]})
    result = verify_asset(asset, sidecar, policy)
    assert result['status'] == 'valid_trusted' and result['checks']['asset_hash_matches']
    assert result['parent_assets_verified'] is False
    asset.write_bytes(b'edited bytes')
    assert verify_asset(asset, sidecar, policy)['status'] == 'asset_hash_mismatch'
    asset.write_bytes(b'engineering image bytes')
    document = json.loads(sidecar.read_text())
    document['payload']['operations'].append('forged history')
    sidecar.write_text(json.dumps(document))
    result = verify_asset(asset, sidecar, policy)
    assert result['status'] == 'invalid_signature'
    assert result['checks']['asset_hash_matches'] is None
    assert result['checks']['signer_trusted'] is None
    sidecar.write_text('malformed')
    assert verify_asset(asset, sidecar, policy)['status'] == 'invalid_signature'
