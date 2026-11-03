"""Offline synthetic/source/aggregate suites only. No data or models required."""
from pathlib import Path
import argparse
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SUITES = [
    ["tests"],
    ["prototype/tests"],
    ["experiments/bert_original_retrain_20261031/tests"],
    ["experiments/bert_mac_m3pro_20261101/tests"],
    ["experiments/lineage_backtranslation_mac_20261027/pipeline/tests/test_integrity_resume.py"],
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prototype-python", default=sys.executable,
                        help="Optional interpreter for an existing separate mock-UI environment")
    args = parser.parse_args()
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")
    for suite in SUITES:
        interpreter = args.prototype_python if suite == ["prototype/tests"] else sys.executable
        suite_env = dict(env)
        if interpreter != sys.executable:
            # Never inject another environment's NumPy/Arrow into the UI process.
            suite_env.pop("PYTHONPATH", None)
        subprocess.run([interpreter, "-m", "pytest", "-q", "--import-mode=importlib", *suite], cwd=ROOT, env=suite_env, check=True)
    subprocess.run([sys.executable, str(ROOT / "scripts/verify_published_results.py")], cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
