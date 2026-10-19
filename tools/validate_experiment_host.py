"""Run host qualification, explicitly recording optional external-fixture exclusions."""

import argparse
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def cases(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from cases(item)
        else:
            yield item


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--without-desktop-sb3-reference", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to((ROOT / "results").resolve()) or output.exists():
        raise ValueError("Qualification output must be a new file below results/")
    all_cases = list(cases(unittest.defaultTestLoader.discover(str(ROOT / "tests"))))
    excluded, selected = [], []
    for case in all_cases:
        if (args.without_desktop_sb3_reference
                and case.id().startswith("test_sb3_microgrid_env.SB3ReferenceEnvTest.")):
            excluded.append(case.id())
        else:
            selected.append(case)
    print(json.dumps({
        "excluded_not_passed": excluded,
        "reason": "Old SB3 Desktop reference fixture is not part of the fresh current-model matrix"
        if excluded else None,
    }), flush=True)
    result = unittest.TextTestRunner(verbosity=1).run(unittest.TestSuite(selected))
    report = {
        "tests_run": result.testsRun, "passed": result.wasSuccessful(),
        "failures": len(result.failures), "errors": len(result.errors),
        "skipped": [(case.id(), reason) for case, reason in result.skipped],
        "excluded_not_passed": excluded,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
