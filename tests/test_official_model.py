import torch

from scoperec.official_model import ReleasedGenRecEdit, ReleasedTIGER
from scoperec.official_tokens import token_layout


def tiny_model():
    return ReleasedTIGER(
        token_layout(4), d_model=16, d_ff=32, num_layers=4, num_heads=2, d_kv=8
    ).eval()


def test_released_model_trains_with_eos_and_generates_five_unmasked_steps():
    torch.manual_seed(2024)
    model = tiny_model()
    inputs = torch.tensor([[17, 1, 6, 11, 14, 18, 0]])
    labels = torch.tensor([[2, 7, 12, 14, 18]])
    output = model(inputs, labels)
    assert output.logits.shape == (1, 5, 19)
    assert torch.isfinite(output.loss)
    output.loss.backward()
    generated = model.generate(inputs, beam_size=5)
    assert generated["sequences"].shape == (1, 5, 5)
    assert generated["codes"].shape == (1, 5, 4)
    with torch.no_grad():
        scores = model.path_scores(inputs.expand(5, -1), generated["sequences"][0]).sum(1)
    torch.testing.assert_close(scores, generated["scores"][0], atol=2e-6, rtol=1e-6)
    assert model.t5.config.decoder_start_token_id == 0


def test_official_editor_only_activates_first_four_steps_and_restores_exactly():
    torch.manual_seed(2024)
    base = tiny_model()
    editor = ReleasedGenRecEdit(base, [0, 1, 2, 3], [torch.randn(16, 32) * 0.02 for _ in range(4)])
    inputs = torch.tensor([[17, 1, 6, 11, 14, 18]])
    labels = torch.tensor([[2, 7, 12, 14, 18]])
    before = {name: tensor.clone() for name, tensor in base.state_dict().items()}
    with torch.no_grad():
        generated = editor.generate(inputs, beam_size=5)
        scores = editor.path_scores(inputs.expand(5, -1), generated["sequences"][0]).sum(1)
    torch.testing.assert_close(scores, generated["scores"][0], atol=2e-6, rtol=1e-6)
    for name, value in base.state_dict().items():
        assert torch.equal(value, before[name])
    editor.enabled = False
    torch.testing.assert_close(
        editor.path_scores(inputs, labels), base.path_scores(inputs, labels), atol=0, rtol=0
    )
