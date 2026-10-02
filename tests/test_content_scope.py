import numpy as np
import pytest
import torch

from scoperec.content_scope import (
    ContentScopeModel,
    load_content_artifact,
    save_content_artifact,
)
from scoperec.decoding import SIDTree
from scoperec.model import GenerativeRecommender


def make_models(alpha=0.2):
    torch.manual_seed(17)
    base = GenerativeRecommender(16, d_model=16, d_ff=32, num_layers=1, num_heads=2).eval()
    codes = np.array([[3, 5, 7], [3, 5, 8], [3, 6, 9], [4, 6, 7], [4, 6, 8]])
    groups = np.array([0, 2, 2, 0, 0])
    vectors = np.random.default_rng(17).normal(size=(5, 8)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    adapted = ContentScopeModel(base, codes, groups, vectors, depth=1, alpha=alpha).eval()
    return base, adapted, codes, SIDTree(codes, np.arange(5), 16)


def test_content_mixture_preserves_outside_and_branch_mass():
    base, adapted, codes, tree = make_models()
    inputs = torch.tensor(codes[[0]]).expand(5, -1)
    with torch.no_grad():
        before = base.path_scores(inputs, torch.tensor(codes), tree).sum(dim=1).exp()
        after = adapted.path_scores(inputs, torch.tensor(codes), tree).sum(dim=1).exp()
    torch.testing.assert_close(after.sum(), torch.tensor(1.0), atol=1e-6, rtol=0)
    torch.testing.assert_close(before[3:], after[3:], atol=1e-7, rtol=0)
    torch.testing.assert_close(before[:3].sum(), after[:3].sum(), atol=1e-6, rtol=0)
    torch.testing.assert_close(after[0], before[0] * 0.8, atol=1e-6, rtol=0)
    assert after[1:3].sum() > before[1:3].sum()


def test_content_natural_beam_matches_full_path_scoring():
    _, adapted, codes, tree = make_models()
    inputs = torch.tensor(codes[[0]])
    with torch.no_grad():
        expected = adapted.path_scores(inputs.expand(5, -1), torch.tensor(codes), tree).sum(1)
        generated = adapted.generate(inputs, tree, beam_size=8)
    assert generated["items"][0].tolist() == expected.argsort(descending=True).tolist()
    torch.testing.assert_close(
        generated["scores"][0], expected.sort(descending=True).values, atol=1e-6, rtol=1e-6
    )
    assert adapted._context is None


def test_zero_mixture_is_exactly_the_base_and_no_future_target_routing():
    base, adapted, codes, tree = make_models(alpha=0)
    inputs = torch.tensor(codes[[0]])
    with torch.no_grad():
        expected = base.generate(inputs, tree, beam_size=5)
        actual = adapted.generate(inputs, tree, beam_size=5, targets=torch.tensor(codes[[4]]))
    torch.testing.assert_close(actual["scores"], expected["scores"], atol=1e-6, rtol=1e-6)
    assert torch.equal(actual["items"], expected["items"])
    assert adapted.alpha == 0


def test_confidence_gate_changes_budget_without_changing_branch_mass():
    base, adapted, codes, tree = make_models()
    adapted.confidence_threshold = 0.5
    inputs = torch.tensor(codes[[0]]).expand(5, -1)
    with torch.no_grad():
        before = base.path_scores(inputs, torch.tensor(codes), tree).sum(1).exp()
        after = adapted.path_scores(inputs, torch.tensor(codes), tree).sum(1).exp()
    torch.testing.assert_close(before[:3].sum(), after[:3].sum(), atol=1e-6, rtol=0)
    torch.testing.assert_close(before[3:], after[3:], atol=1e-7, rtol=0)
    assert after[0] >= before[0] * 0.8 - 1e-7
    assert after[0] <= before[0] + 1e-7


def test_content_artifact_roundtrip_checks_every_dependency(tmp_path):
    base, adapted, codes, tree = make_models()
    groups = np.array([0, 2, 2, 0, 0])
    vectors = adapted.embeddings.cpu().numpy()
    provenance = {
        name: f"{name}-version-1"
        for name in ("base_sha256", "sid_sha256", "embeddings_sha256", "catalog_groups_sha256")
    }
    path = tmp_path / "content.json"
    save_content_artifact(path, adapted.settings, provenance)
    loaded = load_content_artifact(base, path, codes, groups, vectors, provenance)
    inputs = torch.tensor(codes[[0]])
    expected = adapted.generate(inputs, tree, beam_size=5)
    actual = loaded.generate(inputs, tree, beam_size=5)
    torch.testing.assert_close(expected["scores"], actual["scores"], atol=0, rtol=0)
    assert torch.equal(expected["items"], actual["items"])
    for name in provenance:
        with pytest.raises(ValueError, match="version mismatch"):
            load_content_artifact(
                base, path, codes, groups, vectors, {**provenance, name: "wrong-version"}
            )
