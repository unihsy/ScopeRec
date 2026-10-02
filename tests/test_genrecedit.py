import numpy as np
import torch

from scoperec.decoding import SIDTree
from scoperec.genrecedit import GenRecEditModel, edit_module, optimize_deltas, solve_weight_delta
from scoperec.model import GenerativeRecommender


def test_closed_form_matches_official_inverse_equation():
    torch.manual_seed(42)
    keys = torch.randn(25, 8, dtype=torch.float64)
    deltas = torch.randn(25, 4, dtype=torch.float64)
    old = torch.randn(30, 8, dtype=torch.float64)
    covariance = old.T @ old / len(old)
    expected = deltas.T @ keys @ torch.linalg.inv(keys.T @ keys + 1000 * covariance)
    actual, residual = solve_weight_delta(keys, deltas, covariance, 1000)
    torch.testing.assert_close(actual.double(), expected, atol=1e-8, rtol=1e-6)
    assert residual < 1e-12


def test_one_one_activation_and_no_persistent_mutation():
    torch.manual_seed(17)
    base = GenerativeRecommender(16, d_model=16, d_ff=32, num_layers=3, num_heads=2).eval()
    codes = np.array([[3, 5, 7], [3, 5, 8], [4, 6, 7], [4, 6, 8]])
    tree = SIDTree(codes, np.arange(4), 16)
    inputs = torch.tensor([[3, 5, 7]])
    before = {name: value.clone() for name, value in base.state_dict().items()}
    deltas = [torch.randn(16, 32) * 0.1 for _ in range(3)]
    editor = GenRecEditModel(base, [0, 1, 2], deltas).eval()
    with torch.no_grad():
        scores = editor.path_scores(inputs.expand(4, -1), torch.tensor(codes), tree).sum(1)
        generated = editor.generate(inputs, tree, beam_size=10)
    assert generated["items"][0].tolist() == scores.argsort(descending=True).tolist()
    torch.testing.assert_close(scores.exp().sum(), torch.tensor(1.0), atol=1e-6, rtol=0)
    for name, value in base.state_dict().items():
        assert torch.equal(value, before[name])
    assert not any(edit_module(base, layer)._forward_hooks for layer in range(3))
    editor.enabled = False
    torch.testing.assert_close(
        editor.path_scores(inputs.expand(4, -1), torch.tensor(codes), tree),
        base.path_scores(inputs.expand(4, -1), torch.tensor(codes), tree),
        atol=1e-6,
        rtol=1e-6,
    )


def test_target_optimizer_keeps_base_frozen_and_filters_failures():
    torch.manual_seed(29)
    model = GenerativeRecommender(16, d_model=16, d_ff=32, num_layers=2, num_heads=2).eval()
    inputs = torch.tensor([[3, 5, 7], [4, 6, 8]])
    targets = torch.tensor([[4, 6, 7], [3, 5, 8]])
    settings = {"v_lr": 0.5, "v_num_grad_steps": 12, "v_weight_decay": 0.2, "z_vector_max": 8000}
    before = {name: value.clone() for name, value in model.state_dict().items()}
    delta, success = optimize_deltas(model, inputs, targets, 1, 1, settings)
    assert delta.shape == (2, 16)
    assert torch.isfinite(delta).all()
    assert torch.count_nonzero(delta[~success]) == 0
    assert all(parameter.grad is None for parameter in model.parameters())
    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name])


def test_disabled_editor_uses_exact_base_teacher_under_bf16():
    torch.manual_seed(43)
    base = GenerativeRecommender(16, d_model=16, d_ff=32, num_layers=2, num_heads=2).eval()
    editor = GenRecEditModel(base, [0, 1], [torch.randn(16, 32) for _ in range(2)]).eval()
    editor.enabled = False
    inputs = torch.tensor([[3, 5, 7], [4, 6, 8]])
    targets = torch.tensor([[4, 6, 7], [3, 5, 8]])
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        expected, _ = base.teacher(inputs, targets)
        actual, _ = editor.teacher(inputs, targets)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
