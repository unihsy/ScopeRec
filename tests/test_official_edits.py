import numpy as np
import torch

from scoperec.official_edits import build_pseudo, covariance_moments
from scoperec.official_model import ReleasedTIGER
from scoperec.official_tokens import token_layout


def test_official_pseudo_preserves_duplicate_prefixes_and_cold_identity():
    training = {
        "history": np.array([[0, -1], [0, -1], [0, 1]]),
        "target": np.array([1, 1, 2]),
        "user": np.array([7, 7, 7]),
        "group": np.array([0, 0, 0]),
        "source_row": np.array([1, 2, 2]),
    }
    evaluation = {"target": np.array([3, 1, 3]), "group": np.array([2, 0, 2])}
    vectors = np.array([[0, 1], [1, 0], [-1, 0], [1, 0]], dtype=np.float32)
    pseudo, mapping = build_pseudo(
        training,
        evaluation,
        vectors,
        np.array([1, 1, 1, 0], bool),
        neighbors=1,
        per_item=10,
        seed=2024,
    )
    assert mapping == {3: [1]}
    assert pseudo["target"].tolist() == [3, 3]
    assert pseudo["history"].tolist() == [[0, -1], [0, -1]]
    assert pseudo["source_training_row"].tolist() == [0, 1]


def test_streamed_moment_is_uncentered_and_matches_direct_capture():
    torch.manual_seed(17)
    model = ReleasedTIGER(
        token_layout(4), d_model=16, d_ff=32, num_layers=2, num_heads=2, d_kv=8
    ).eval()
    inputs = np.array([[17, 1, 6, 11, 14, 18], [17, 2, 7, 12, 14, 18]])
    labels = np.array([[2, 7, 12, 14, 18], [1, 6, 11, 14, 18]])
    actual = covariance_moments(model, inputs, labels, [0, 1], batch_size=1)
    captured = {}
    handles = []
    for position in range(2):

        def capture(module, arguments, output, position=position):
            captured[position] = arguments[0][:, position].detach().double()

        handles.append(
            model.t5.decoder.block[position]
            .layer[2]
            .DenseReluDense.wo.register_forward_hook(capture)
        )
    try:
        with torch.no_grad():
            model(torch.tensor(inputs), torch.tensor(labels))
    finally:
        for handle in handles:
            handle.remove()
    for position in range(2):
        expected = (captured[position].T @ captured[position] / 2).float()
        torch.testing.assert_close(actual[position], expected, rtol=2e-5, atol=1e-6)
