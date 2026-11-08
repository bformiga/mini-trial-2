"""Generate Atalla GEMM inputs from the student benchmark definitions."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from atalla.benchmarks.workloads import main

if __name__ == "__main__":
    main()
