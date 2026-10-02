import faiss
import numpy as np
import pytest

from scoperec.features import encode_static_text, item_text
from scoperec.semantic_ids import train_quantizer, unique_codes


def test_new_codes_do_not_change_old_collision_ids():
    original = np.array([[0, 1, 1], [0, 1, 1], [2, 3, 0]])
    extended = np.concatenate([original, [[0, 1, 1], [2, 3, 0]]])
    np.testing.assert_array_equal(unique_codes(original, 4, 8), unique_codes(extended, 4, 8)[:3])
    assert len(np.unique(unique_codes(extended, 4, 8), axis=0)) == 5
    with pytest.raises(ValueError, match="capacity"):
        unique_codes(extended, 4, 2)


def test_quantizer_is_fitted_only_to_training_items():
    rng = np.random.default_rng(42)
    vectors = rng.normal(size=(180, 8)).astype(np.float32)
    mask = np.arange(180) < 160
    config = {"dimensions": 4, "levels": 2, "bits": 2}
    transform, index, codes, _ = train_quantizer(vectors, mask, config, 17)
    changed = vectors.copy()
    changed[~mask] *= 100
    other_transform, other_index, other_codes, _ = train_quantizer(changed, mask, config, 17)
    np.testing.assert_array_equal(codes[mask], other_codes[mask])
    np.testing.assert_array_equal(
        faiss.vector_to_array(index.rq.codebooks), faiss.vector_to_array(other_index.rq.codebooks)
    )
    np.testing.assert_array_equal(transform.apply(vectors), other_transform.apply(vectors))


def test_item_text_excludes_rating_aggregates():
    assert item_text({"title": "Software", "rating_number": 12345}) == "Software"


def test_text_encoding_precision_is_fixed_and_restored():
    import torch

    class Encoder:
        def encode(self, texts, **kwargs):
            assert torch.get_float32_matmul_precision() == "highest"
            return np.ones((len(texts), 4))

    previous_precision = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("high")
        vectors = encode_static_text(Encoder(), ["software"], 1)
        assert vectors.dtype == np.float32
        assert torch.get_float32_matmul_precision() == "high"
    finally:
        torch.set_float32_matmul_precision(previous_precision)