# config.py
import yaml
from pathlib import Path

def load_config():
    # Automatically resolve path relative to the project root
    root_dir = Path(__file__).resolve().parents[1]  # Adjust if your structure changes
    config_path = root_dir / "config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
