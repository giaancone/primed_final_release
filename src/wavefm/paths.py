"""Root directories, all taken from the environment set by ``setup.sh``."""
import os
from pathlib import Path

ROOT = Path(os.environ.get("WAVEFM_ROOT", Path(__file__).resolve().parents[2]))
OUTPUT = Path(os.environ.get("WAVEFM_OUTPUT", ROOT / "outputs"))
