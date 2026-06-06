from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from plie_pic.io import read_input_frame


def test_read_input_frame_accepts_csv_and_zip(tmp_path: Path) -> None:
    df = pd.DataFrame({"time": ["2024-01-01T00:00:00Z"], "price": [100.0]})
    csv_path = tmp_path / "hmm_state.csv"
    zip_path = tmp_path / "hmm_state.csv.zip"

    df.to_csv(csv_path, index=False)
    df.to_csv(zip_path, index=False, compression={"method": "zip", "archive_name": "hmm_state.csv"})

    assert read_input_frame(csv_path).to_dict("records") == df.to_dict("records")
    assert read_input_frame(zip_path).to_dict("records") == df.to_dict("records")

