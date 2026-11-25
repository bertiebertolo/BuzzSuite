"""Run every test_* function in tests/buzzsuite/test_*.py without pytest (the app's environment has
none). Exit code 0 when all pass. Usage: python tests/buzzsuite/run_all.py [name-substring]"""
import importlib.util
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import conftest  # noqa: E402,F401  -- puts the repo root on sys.path


def main(argv):
    pick = argv[1] if len(argv) > 1 else ""
    n_ok, failed = 0, []
    t0 = time.time()
    for fname in sorted(os.listdir(HERE)):
        if not (fname.startswith("test_") and fname.endswith(".py")):
            continue
        spec = importlib.util.spec_from_file_location(fname[:-3], os.path.join(HERE, fname))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for name in sorted(n for n in dir(mod) if n.startswith("test_")):
            fn = getattr(mod, name)
            if not callable(fn) or pick not in "%s::%s" % (fname, name):
                continue
            try:
                fn()
                n_ok += 1
                print("  ok    %s::%s" % (fname, name))
            except Exception:  # noqa: BLE001
                failed.append("%s::%s" % (fname, name))
                print("  FAIL  %s::%s\n%s" % (fname, name, traceback.format_exc()))
    print("%d passed, %d failed in %.1f s (Python %s)" % (n_ok, len(failed), time.time() - t0,
                                                         sys.version.split()[0]))
    return 1 if failed or not n_ok else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
