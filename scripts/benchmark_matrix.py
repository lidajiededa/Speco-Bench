#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


repository_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repository_root / "src"))

from speco_bench.matrix import main  # noqa: E402


if __name__ == "__main__":
    main()
