from pathlib import Path

def project_root() -> Path:
    """Finds parent folder where pyproject.toml lies"""
    for parent in [Path(__file__).resolve(), *Path(__file__).resolve().parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    raise FileNotFoundError("pyproject.toml not found")