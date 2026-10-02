import torch
from transformers import get_scheduler

from scoperec.official_model import ReleasedTIGER
from scoperec.official_tokens import token_layout


def test_accumulated_cross_entropy_matches_full_batch_without_dropout():
    torch.manual_seed(2024)
    model = ReleasedTIGER(
        token_layout(4), d_model=16, d_ff=32, num_layers=1, num_heads=2, d_kv=8
    ).eval()
    inputs = torch.tensor([[17, 1, 6, 11, 14, 18], [17, 2, 7, 12, 14, 18]])
    labels = torch.tensor([[2, 7, 12, 14, 18], [1, 6, 11, 14, 18]])
    model(inputs, labels).loss.backward()
    expected = [parameter.grad.clone() for parameter in model.parameters()]
    model.zero_grad()
    for row in range(2):
        (model(inputs[row : row + 1], labels[row : row + 1]).loss / 2).backward()
    for parameter, gradient in zip(model.parameters(), expected, strict=True):
        torch.testing.assert_close(parameter.grad, gradient, rtol=1e-4, atol=1e-6)


def test_released_warmup_exceeds_total_training_steps():
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW([parameter], lr=0.05)
    scheduler = get_scheduler("cosine", optimizer, num_warmup_steps=10000, num_training_steps=4540)
    assert scheduler.get_last_lr() == [0.0]
    for _ in range(10):
        optimizer.step()
        scheduler.step()
    assert abs(scheduler.get_last_lr()[0] - 0.05 * 10 / 10000) < 1e-12
