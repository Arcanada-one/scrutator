import json


def pytest_addoption(parser):
    parser.addoption(
        "--ac5-faulty-json", default=None, help="write the faulty-variant verdicts (AC5 model) to this path"
    )


def pytest_sessionfinish(session, exitstatus):
    out = session.config.getoption("--ac5-faulty-json")
    if not out:
        return
    from . import test_generation_closure_model as t

    with open(out, "w") as f:
        json.dump(t.FAULTY_RESULTS, f, indent=1, sort_keys=True)
