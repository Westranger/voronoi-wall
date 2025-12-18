import os
import subprocess
import sys


def main() -> int:
    env = dict(os.environ)
    env["UPDATE_GOLDENS"] = "1"

    cmd = [sys.executable, "-m", "pytest", "-q", "tests/voronoi3d/test_voronoi3d_golden.py"]
    print("Running:", " ".join(cmd))
    p = subprocess.run(cmd, env=env)
    return int(p.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
