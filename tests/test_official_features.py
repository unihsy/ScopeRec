import numpy as np

from scoperec.official_features import fit_released_quantizer
from scoperec.official_tokens import released_codes


def test_released_pca_and_faiss_fit_all_pca_but_only_training_quantizer():
    vectors = np.random.default_rng(2024).normal(size=(180, 12)).astype(np.float32)
    mask = np.arange(180) < 160
    pca, projected, index, semantic = fit_released_quantizer(vectors, mask, 4, 2, 2)
    np.testing.assert_allclose(pca.mean_, vectors.mean(axis=0), atol=1e-6)
    np.testing.assert_allclose(projected.var(axis=0, ddof=1), np.ones(4), atol=1e-5)
    assert index.ntotal == 180
    assert semantic.shape == (180, 2)
    assert semantic.min() >= 0 and semantic.max() < 4
    assert pca.whiten
    codes = released_codes(semantic[:10], 16)
    assert len(np.unique(codes, axis=0)) == 10
