from pathlib import Path


LOOKUP_TABLE_DIR = Path(__file__).resolve().parents[2] / "lookup_tables"


def lookup_table_path(filename: str) -> Path:
    path = LOOKUP_TABLE_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(f"Missing chip lookup table: {path}")
    return path
