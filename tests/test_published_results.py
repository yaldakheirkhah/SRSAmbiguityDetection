import importlib.util
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
spec = importlib.util.spec_from_file_location("public_aggregate_checks", SCRIPTS / "verify_published_results.py")
aggregate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(aggregate)


def test_saved_aggregate_arithmetic_and_checkpoint_selection():
    result = aggregate.verify()
    assert result["best_epoch"] == 41
    assert result["model_inference_run"] is False
    assert result["source_datasets_read"] is False


@pytest.mark.parametrize("matrix", [[], [[True] * 6] * 6, [[-1] * 6] * 6, [[0] * 6] * 6, [[0] * 5] * 6])
def test_invalid_counts_fail(matrix):
    with pytest.raises(ValueError):
        aggregate.metrics_from_matrix(matrix)


def test_six_class_average_includes_empty_classes():
    matrix = [[0] * 6 for _ in range(6)]
    matrix[0][0] = 1
    result = aggregate.metrics_from_matrix(matrix)
    assert result["macro_f1"] == 1 / 6
    assert result["weighted_f1"] == 1


def test_mismatched_metrics_refused():
    with pytest.raises(ValueError):
        aggregate.compare_metrics({"accuracy": 0.9}, {"accuracy": 1.0})
