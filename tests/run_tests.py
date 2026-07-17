"""
Test runner. Uses pytest when installed, otherwise imports the test modules
and calls every test_* callable directly.

    python tests/run_tests.py
"""

import importlib
import importlib.util
import subprocess
import sys
import traceback
from pathlib import Path

TEST_MODULES = ["tests.test_features", "tests.test_leakage", "tests.test_evaluate"]


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root))

    if importlib.util.find_spec("pytest") is not None:
        return subprocess.call([sys.executable, "-m", "pytest", "tests/", "-q"], cwd=repo_root)

    passed, failed = 0, 0
    for mod_name in TEST_MODULES:
        mod = importlib.import_module(mod_name)
        for attr in sorted(dir(mod)):
            if not attr.startswith("test_"):
                continue
            fn = getattr(mod, attr)
            if not callable(fn):
                continue
            try:
                fn()
                passed += 1
            except Exception:
                failed += 1
                print(f"FAIL {mod_name}.{attr}")
                traceback.print_exc()
    print(f"{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
