import sys
from pathlib import Path

def get_base_dir():
    if getattr(sys, "frozen", False):
        if sys.platform == "darwin":
            # .../BabyMove/BabyMove.app/Contents/MacOS/BabyMove
            # → .../BabyMove
            return Path(sys.executable).resolve().parents[3]

        # Windows : .../BabyMove/BabyMove.exe
        return Path(sys.executable).resolve().parent

    # Développement
    return Path(__file__).resolve().parent.parent


BASE_DIR = get_base_dir()

VIDEOS_DIR = BASE_DIR / "videos"
RESULTS_DIR = BASE_DIR / "results"
MODEL_DIR = BASE_DIR / "models"
CONFIG_FILE = BASE_DIR / "config.json"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)