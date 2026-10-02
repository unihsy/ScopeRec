import numpy as np
import torch

from scoperec.official_content import FullVocabulary, ReleasedContentScope
from scoperec.official_model import ReleasedTIGER
from scoperec.official_tokens import encode_requests, released_codes, token_layout


def make_fixture():
    torch.manual_seed(2024)
    layout = token_layout(4)
    base = ReleasedTIGER(layout, d_model=16, d_ff=32, num_layers=1, num_heads=2, d_kv=8).eval()
    codes = released_codes(np.array([[0, 1, 2], [0, 1, 2], [1, 2, 3]]), 4)
    groups = np.array([0, 2, 0])
    vectors = np.array([[1, 0], [1, 0], [0, 1]], dtype=np.float32)
    requests = {"history": np.array([[0, -1]]), "target": np.array([1]), "user": np.array([8])}
    inputs, labels = encode_requests(requests, codes, layout)
    model = ReleasedContentScope(
        base, codes, groups, vectors, alpha=0.2, confidence_threshold=None
    ).eval()
    return base, model, torch.tensor(inputs), torch.tensor(labels), codes


def test_full_vocab_never_masks_invalid_tokens():
    distribution = FullVocabulary(5).log_probabilities(torch.zeros(1, 5), torch.tensor([[4]]))
    torch.testing.assert_close(distribution.exp(), torch.full((1, 5), 0.2))


def test_content_mixture_includes_eos_and_keeps_old_path_budget():
    base, model, inputs, labels, codes = make_fixture()
    context = model.content_context(inputs)
    assert context["weights"].shape == (1, 1)
    assert model.new_codes[0, -1] == model.layout["eos"]
    old_labels = torch.tensor(np.append(codes[0], model.layout["eos"]))[None]
    with torch.no_grad():
        before = base.path_scores(inputs, old_labels)
        after = model.path_scores(inputs, old_labels)
    torch.testing.assert_close(before[:, :1], after[:, :1], atol=0, rtol=0)
    torch.testing.assert_close(
        (after.sum(1) - before.sum(1)).exp(), torch.tensor([0.8]), atol=2e-6, rtol=1e-6
    )
    generated = model.generate(inputs, beam_size=5)
    with torch.no_grad():
        scores = model.path_scores(inputs.expand(5, -1), generated["sequences"][0]).sum(1)
    torch.testing.assert_close(scores, generated["scores"][0], atol=3e-6, rtol=1e-6)


def test_content_disable_restores_unmasked_baseline():
    base, model, inputs, labels, _ = make_fixture()
    model.enabled = False
    expected = base.generate(inputs, beam_size=5)
    actual = model.generate(inputs, beam_size=5)
    assert torch.equal(expected["sequences"], actual["sequences"])
    torch.testing.assert_close(expected["scores"], actual["scores"], atol=0, rtol=0)
