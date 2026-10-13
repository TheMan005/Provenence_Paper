"""Optional local continuation: wait for current download, then extract and smoke test.

Never starts/restarts a transfer or touches partial files. Run once in a separate terminal.
"""
import subprocess
import time
import fcntl
from pathlib import Path

from satellite_evidence.archive import status

root = Path(__file__).resolve().parents[1]
destination = root / 'Building_Dataset'
lock_path = root / 'outputs' / '.real_integration.lock'
lock_path.parent.mkdir(exist_ok=True)
lock_stream = lock_path.open('a')
try:
    fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    raise SystemExit('An integration continuation is already running; inspect its log rather than starting another.')
last_bytes = None
while True:
    progress = status(destination)
    if progress['state'] in ('downloaded_unverified', 'extracted_verified'):
        break
    if progress['downloaded_bytes'] != last_bytes:
        print(f"Download {progress['percent']}% ({progress['downloaded_bytes'] / 1e9:.2f} GB)", flush=True)
        last_bytes = progress['downloaded_bytes']
    time.sleep(30)
subprocess.run(['bash', str(root / 'scripts/integrate_real.sh')], cwd=root, check=True)
