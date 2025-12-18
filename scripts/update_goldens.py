import os
import subprocess
import sys

if __name__ == "__main__":
    os.environ["UPDATE_GOLDENS"] = "1"
    cmd = [sys.executable, "-m", "pytest", "-q", "tests/voronoi2d/test_voronoi_golden.py"]
    raise SystemExit(subprocess.call(cmd))
