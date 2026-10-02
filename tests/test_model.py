import numpy as np
import torch

from scoperec.decoding import SIDTree
from scoperec.model import GenerativeRecommender, history_tokens
from scoperec.patches import PrefixPatch


def test_wide_beam_matches_exhaustive_path_scoring():
    torch.manual_seed(17)
    model = GenerativeRecommender(14, d_model=16, d_ff=32, num_layers=1, num_heads=2).eval()
    codes = np.array([[3, 5, 7], [3, 5, 8], [3, 6, 9], [4, 6, 8], [4, 6, 9]])
    tree = SIDTree(codes, np.arange(len(codes)), 14)
    inputs = torch.tensor([[3, 5, 7, 0]])
    scores = model.path_scores(inputs.expand(len(codes), -1), torch.tensor(codes), tree).sum(1)
    torch.testing.assert_close(scores.exp().sum(), torch.tensor(1.0), atol=1e-6, rtol=0)
    generated = model.generate(inputs, tree, beam_size=10, targets=torch.tensor(codes[:1]))
    assert generated["items"][0].tolist() == scores.argsort(descending=True).tolist()
    assert generated["survival"].all()


def test_t5_patch_does_not_modify_backbone_or_outside_paths():
    torch.manual_seed(29)
    model = GenerativeRecommender(14, d_model=16, d_ff=32, num_layers=1, num_heads=2).eval()
    codes = np.array([[3, 5, 7], [3, 5, 8], [4, 6, 8], [4, 6, 9]])
    tree = SIDTree(codes, np.arange(len(codes)), 14)
    inputs = torch.tensor([[4, 6, 8]]).expand(len(codes), -1)
    before = model.path_scores(inputs, torch.tensor(codes), tree)
    weights = {name: value.clone() for name, value in model.state_dict().items()}
    patch = PrefixPatch(16, 14, [[3]], 2)
    with torch.no_grad():
        patch.output_factor.normal_()
    after = model.path_scores(inputs, torch.tensor(codes), tree, patch)
    torch.testing.assert_close(before[2:], after[2:], atol=0, rtol=0)
    torch.testing.assert_close(before[:, 0], after[:, 0], atol=0, rtol=0)
    for name, value in model.state_dict().items():
        assert torch.equal(weights[name], value)


def test_history_padding_never_turns_into_a_real_item():
    codes = np.array([[3, 4], [5, 6]])
    assert history_tokens(np.array([[1, -1, -1]]), codes).tolist() == [[5, 6, 0, 0, 0, 0]]


def test_official_tiger_attention_width_and_distinct_edit_layers():
    model = GenerativeRecommender(32, d_model=128, d_ff=1024, num_layers=6, num_heads=8, d_kv=64)
    assert model.t5.config.d_kv == 64
    layers = [model.t5.decoder.block[index].layer[2].DenseReluDense.wo for index in range(4)]
    assert len({id(layer) for layer in layers}) == 4
    assert all(layer.weight.shape == (128, 1024) for layer in layers)


def test_oracle_prefix_only_returns_its_subtree():
    model = GenerativeRecommender(14, d_model=16, d_ff=32, num_layers=1, num_heads=2).eval()
    codes = np.array([[3, 5, 7], [3, 5, 8], [4, 6, 8]])
    tree = SIDTree(codes, np.arange(len(codes)), 14)
    output = model.generate(
        torch.tensor([[4, 6, 8]]), tree, targets=torch.tensor(codes[:1]), forced_depth=1
    )
    assert set(output["items"][0].tolist()) == {0, 1, -1}
