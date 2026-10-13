"""Engineering baseline; synthetic smoke outputs are not research results."""
import os
import tempfile
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'satellite-evidence-matplotlib'))
