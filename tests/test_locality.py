import numpy as np
import pytest
import torch

from scoperec.decoding import SIDTree, prefix_keys, sequence_log_probabilities
from scoperec.patches import PrefixPatch, extend_patch, load_patch, save_patch


def toy_probabilities(tree, codes, patch=None):
    hidden_size = 5
    hidden_axes = torch.arange(hidden_size, dtype=torch.float64)
    output = torch.cos(torch.arange(hidden_size * tree.vocab_size, dtype=torch.float64))
    output = output.reshape(hidden_size, tree.vocab_size)
    scores = []
    for position in range(tree.length):
        prefix = codes[:, :position]
        hidden = torch.sin(prefix_keys(prefix, tree.vocab_size)[:, None] + hidden_axes)
        logits = hidden @ output
        if patch is not None:
            logits = logits + patch(hidden, prefix)
        scores.append(logits)
    return sequence_log_probabilities(tree, torch.stack(scores, dim=1), codes).sum(1).exp()


def test_exact_outside_invariance_mass_conservation_and_ceiling():
    torch.manual_seed(17)
    codes = torch.tensor([[3, 5, 7], [3, 5, 8], [3, 6, 7], [4, 5, 7], [4, 6, 8]])
    tree = SIDTree(codes.numpy(), np.arange(len(codes)), 10)
    patch = PrefixPatch(5, 10, [[3, 5]], 2).double()
    with torch.no_grad():
        patch.output_factor.normal_()
    before = toy_probabilities(tree, codes)
    after = toy_probabilities(tree, codes, patch)
    torch.testing.assert_close(before.sum(), torch.tensor(1.0).double(), atol=1e-12, rtol=0)
    torch.testing.assert_close(after.sum(), torch.tensor(1.0).double(), atol=1e-12, rtol=0)
    torch.testing.assert_close(before[2:], after[2:], atol=1e-12, rtol=0)
    torch.testing.assert_close(before[:2].sum(), after[:2].sum(), atol=1e-12, rtol=0)
    assert torch.all(after[:2] <= before[:2].sum() + 1e-12)
    assert not torch.equal(before[:2], after[:2])
    patch.enabled = False
    torch.testing.assert_close(before, toy_probabilities(tree, codes, patch), atol=0, rtol=0)


def test_gate_never_looks_at_target_or_changes_prefix_steps():
    patch = PrefixPatch(4, 12, [[3, 5]], 2)
    with torch.no_grad():
        patch.output_factor.fill_(1)
    hidden = torch.ones(3, 4)
    assert torch.count_nonzero(patch(hidden, torch.tensor([[3], [3], [4]]))) == 0
    deltas = patch(hidden, torch.tensor([[3, 5], [3, 6], [4, 5]]))
    assert torch.count_nonzero(deltas[0]) > 0
    assert torch.count_nonzero(deltas[1:]) == 0


def test_patch_artifact_roundtrip_and_version_guard(tmp_path):
    patch = PrefixPatch(4, 12, [[3]], 2)
    with torch.no_grad():
        patch.output_factor.normal_()
    path = tmp_path / "patch.pt"
    save_patch(patch, path, "base-v1", "sid-v1")
    restored = load_patch(path, "base-v1", "sid-v1")
    hidden = torch.randn(2, 4)
    prefixes = torch.tensor([[3, 5], [4, 5]])
    torch.testing.assert_close(patch(hidden, prefixes), restored(hidden, prefixes))
    with pytest.raises(ValueError, match="version"):
        load_patch(path, "different-base", "sid-v1")


def test_mask_before_softmax_normalizes_only_legal_tokens():
    tree = SIDTree(np.array([[3, 5], [3, 6], [4, 5]]), np.array([7, 8, 9]), 8)
    logits = torch.zeros(1, 8, dtype=torch.float64)
    logits[:, 0] = 100
    probabilities = tree.log_probabilities(logits, torch.empty(1, 0, dtype=torch.long)).exp()
    assert probabilities[0, 3] == 0.5
    assert probabilities[0, 4] == 0.5
    assert probabilities[0, 0] == 0
    assert tree.resolve(torch.tensor([[3, 6], [4, 7]])).tolist() == [8, -1]


def test_ungated_single_branch_can_change_other_subtrees():
    patch = PrefixPatch(4, 12, [[3]], 2)
    with torch.no_grad():
        patch.output_factor.fill_(1)
    hidden = torch.ones(1, 4)
    prefix = torch.tensor([[4]])
    assert torch.count_nonzero(patch(hidden, prefix)) == 0
    patch.gate_enabled = False
    assert torch.count_nonzero(patch(hidden, prefix)) > 0


def test_extending_patch_preserves_old_branch_and_initializes_new_to_zero():
    torch.manual_seed(43)
    patch = PrefixPatch(4, 12, [[4, 5]], 2)
    with torch.no_grad():
        patch.output_factor.normal_()
    extended = extend_patch(patch, [[3, 6], [4, 5]])
    hidden = torch.randn(2, 4)
    prefixes = torch.tensor([[4, 5], [3, 6]])
    torch.testing.assert_close(patch(hidden, prefixes), extended(hidden, prefixes), atol=0, rtol=0)
    assert extended.branches.tolist() == [[3, 6], [4, 5]]
