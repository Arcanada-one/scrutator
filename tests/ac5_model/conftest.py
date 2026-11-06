import json
import os


def pytest_sessionfinish(session, exitstatus):
    out = os.environ.get("AC5_FAULTY_JSON")
    if not out:
        return
    from . import test_generation_closure_model as t

    with open(out, "w") as f:
        json.dump(t.FAULTY_RESULTS, f, indent=1, sort_keys=True)
