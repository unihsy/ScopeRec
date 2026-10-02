import numpy as np
import pytest

from scoperec.audit_v2 import validate_ranking, verify_provenance


def test_ranking_audit_rejects_duplicates_and_unavailable_targets():
    predictions = np.arange(50)[None, :]
    available = np.ones(51, dtype=bool)
    validate_ranking(predictions, np.array([0]), available)
    with pytest.raises(ValueError, match="Repeated"):
        validate_ranking(np.zeros((1, 50), dtype=int), np.array([0]), available)
    available[49] = False
    with pytest.raises(ValueError, match="snapshot"):
        validate_ranking(predictions, np.array([0]), available)


def test_provenance_verification_rejects_different_checkpoint():
    verify_provenance({"base": "one", "sid": "two"}, {"base": "one"})
    with pytest.raises(ValueError, match="base"):
        verify_provenance({"base": "changed"}, {"base": "one"})
