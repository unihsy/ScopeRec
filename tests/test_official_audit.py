import pytest

from scoperec.official_audit import check_metrics


def test_dual_metric_audit_does_not_accept_prefix_as_exact():
    correct = {"prefix": {"cold": {"recall@20": 0.1}}, "exact": {"cold": {"recall@20": 0.01}}}
    check_metrics(correct, correct)
    changed = {"prefix": {"cold": {"recall@20": 0.1}}, "exact": {"cold": {"recall@20": 0.1}}}
    with pytest.raises(ValueError, match="exact/cold"):
        check_metrics(changed, correct)
