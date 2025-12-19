import os
import pytest


def main() -> int:
    os.environ["UPDATE_GOLDENS"] = "1"
    return pytest.main(["-q", "tests/voronoi3d/test_hierarchy_golden.py"])


if __name__ == "__main__":
    raise SystemExit(main())
