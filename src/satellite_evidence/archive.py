"""Read only completed downloads; checksum and safe streaming tar extraction."""
import hashlib
import tarfile
import time
from pathlib import Path

from .common import write_json


ARCHIVE_NAME = 'SN2_buildings_train_AOI_2_Vegas.tar.gz'
EXPECTED_BYTES = 25640007327
EXPECTED_ETAG = 'b1baca54010360c071963672bc67ba7d-3057'


def status(destination):
    root = Path(destination)
    archive = root / ARCHIVE_NAME
    partials = sorted(root.glob(ARCHIVE_NAME + '.*'))
    partials = [p for p in partials if p.is_file() and p.name != ARCHIVE_NAME + '.extraction.json']
    if partials:
        transferred = sum(p.stat().st_blocks * 512 for p in partials)
        state = 'downloading'
    else:
        transferred = archive.stat().st_size if archive.exists() else 0
        state = 'downloaded_unverified' if transferred == EXPECTED_BYTES else 'incomplete'
        receipt = root / (ARCHIVE_NAME + '.extraction.json')
        if receipt.exists() and transferred == EXPECTED_BYTES:
            state = 'extracted_verified'
    return dict(state=state, downloaded_bytes=transferred, expected_bytes=EXPECTED_BYTES,
                percent=round(100 * transferred / EXPECTED_BYTES, 2), destination=str(root.resolve()))


def extract(destination):
    root = Path(destination)
    archive = root / ARCHIVE_NAME
    receipt = root / (ARCHIVE_NAME + '.extraction.json')
    if receipt.exists():
        raise FileExistsError('Archive already extracted; inspect the existing extraction receipt')
    current = status(root)
    if current['state'] != 'downloaded_unverified':
        raise ValueError(f'Refusing to read an incomplete/active archive: {current}')
    chunks = []
    print('Verifying S3 multipart checksum...', flush=True)
    with archive.open('rb') as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            chunks.append(hashlib.md5(chunk).digest())
    etag = hashlib.md5(b''.join(chunks)).hexdigest() + '-' + str(len(chunks))
    if etag != EXPECTED_ETAG:
        raise ValueError(f'S3 checksum mismatch: {etag}')
    print('Checksum verified; extracting archive with safe path/link filtering...', flush=True)
    count, bytes_written, last = 0, 0, time.monotonic()
    with tarfile.open(archive, 'r|gz') as contents:
        for member in contents:
            filtered = tarfile.data_filter(member, root)
            if filtered is None:
                continue
            target = root / filtered.name
            if member.isfile() and target.exists():
                raise FileExistsError(f'Refusing to overwrite existing data: {target}')
            contents.extract(member, path=root, filter='data')
            if member.isfile():
                count += 1
                bytes_written += member.size
            if time.monotonic() - last >= 30:
                print(f'Extracted {count} files ({bytes_written / 1e9:.2f} GB)', flush=True)
                last = time.monotonic()
    result = dict(status='complete', archive_bytes=EXPECTED_BYTES, s3_etag=etag,
                  extracted_files=count, extracted_bytes=bytes_written, destination=str(root.resolve()))
    write_json(receipt, result, immutable=True)
    return result
