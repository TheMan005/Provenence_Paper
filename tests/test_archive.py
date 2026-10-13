import hashlib
import io
import tarfile

import pytest

from satellite_evidence import archive


def prepare(root, monkeypatch, member_name):
    path = root / archive.ARCHIVE_NAME
    with tarfile.open(path, 'w:gz') as contents:
        member = tarfile.TarInfo(member_name)
        payload = b'completed engineering fixture'
        member.size = len(payload)
        contents.addfile(member, io.BytesIO(payload))
    monkeypatch.setattr(archive, 'EXPECTED_BYTES', path.stat().st_size)
    digest = hashlib.md5(hashlib.md5(path.read_bytes()).digest()).hexdigest() + '-1'
    monkeypatch.setattr(archive, 'EXPECTED_ETAG', digest)
    return path


def test_completed_archive_checksum_and_safe_extraction(tmp_path, monkeypatch):
    prepare(tmp_path, monkeypatch, 'AOI_2_Vegas/engineering.txt')
    result = archive.extract(tmp_path)
    assert result['extracted_files'] == 1
    assert (tmp_path / 'AOI_2_Vegas/engineering.txt').read_bytes() == b'completed engineering fixture'
    assert archive.status(tmp_path)['state'] == 'extracted_verified'
    with pytest.raises(FileExistsError, match='already extracted'):
        archive.extract(tmp_path)


def test_archive_path_traversal_is_rejected(tmp_path, monkeypatch):
    prepare(tmp_path, monkeypatch, '../outside.txt')
    with pytest.raises(tarfile.FilterError):
        archive.extract(tmp_path)
    assert not (tmp_path.parent / 'outside.txt').exists()


def test_checksum_mismatch_prevents_extraction(tmp_path, monkeypatch):
    path = prepare(tmp_path, monkeypatch, 'engineering.txt')
    payload = bytearray(path.read_bytes())
    payload[-1] ^= 1
    path.write_bytes(payload)
    with pytest.raises(ValueError, match='checksum mismatch'):
        archive.extract(tmp_path)
    assert not (tmp_path / 'engineering.txt').exists()
