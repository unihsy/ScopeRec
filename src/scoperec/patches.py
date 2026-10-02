from pathlib import Path

import torch
from torch import nn

from scoperec.decoding import prefix_keys


class PrefixPatch(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int, branches, rank: int, mode="local"):
        super().__init__()
        branches = torch.as_tensor(branches, dtype=torch.long)
        if branches.ndim != 2 or not len(branches) or branches.shape[1] < 1:
            raise ValueError("Patch requires at least one nonempty prefix")
        if mode not in ("local", "global") or rank < 1:
            raise ValueError("Invalid patch mode or rank")
        if len(torch.unique(branches, dim=0)) != len(branches):
            raise ValueError("Patch prefixes must be unique")
        if torch.any(branches < 0) or torch.any(branches >= vocab_size):
            raise ValueError("Patch prefix token outside vocabulary")
        order = prefix_keys(branches, vocab_size).argsort()
        branches = branches[order]
        self.register_buffer("branches", branches)
        self.register_buffer("keys", prefix_keys(branches, vocab_size))
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.depth = branches.shape[1]
        self.rank = rank
        self.mode = mode
        self.enabled = True
        self.gate_enabled = True
        count = len(branches) if mode == "local" else 1
        self.input_factor = nn.Parameter(torch.randn(count, hidden_size, rank) * 0.02)
        self.output_factor = nn.Parameter(torch.zeros(count, vocab_size, rank))

    def forward(self, hidden: torch.Tensor, prefix: torch.Tensor) -> torch.Tensor:
        if prefix.shape[1] < self.depth or not self.enabled:
            return hidden.new_zeros((len(hidden), self.vocab_size))
        if self.mode == "global" or not self.gate_enabled:
            if self.mode == "local" and len(self.branches) != 1:
                raise ValueError("Ungated comparison requires a single local branch")
            indices = torch.zeros(len(hidden), dtype=torch.long, device=hidden.device)
            active = torch.ones(len(hidden), dtype=torch.bool, device=hidden.device)
        else:
            keys = prefix_keys(prefix[:, : self.depth], self.vocab_size)
            indices = torch.searchsorted(self.keys, keys).clamp(max=len(self.keys) - 1)
            active = self.keys[indices] == keys
        inputs = self.input_factor[indices].to(hidden.dtype)
        outputs = self.output_factor[indices].to(hidden.dtype)
        latent = torch.bmm(hidden[:, None, :], inputs)
        delta = torch.bmm(latent, outputs.transpose(1, 2)).squeeze(1)
        return delta * active[:, None]

    def specification(self) -> dict:
        return {
            "hidden_size": self.hidden_size,
            "vocab_size": self.vocab_size,
            "branches": self.branches.cpu().tolist(),
            "rank": self.rank,
            "mode": self.mode,
        }


def save_patch(patch: PrefixPatch, path: Path, base_sha256: str, sid_sha256: str, metadata=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "specification": patch.specification(),
            "state": {name: value.detach().cpu() for name, value in patch.state_dict().items()},
            "base_sha256": base_sha256,
            "sid_sha256": sid_sha256,
            "metadata": metadata or {},
        },
        path,
    )


def load_patch(path: Path, base_sha256: str, sid_sha256: str, device="cpu") -> PrefixPatch:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["format_version"] != 1:
        raise ValueError("Unsupported patch format")
    if payload["base_sha256"] != base_sha256 or payload["sid_sha256"] != sid_sha256:
        raise ValueError("Patch does not match the base checkpoint and SID version")
    patch = PrefixPatch(**payload["specification"])
    patch.load_state_dict(payload["state"])
    return patch.to(device).eval()


def extend_patch(patch: PrefixPatch, new_branches) -> PrefixPatch:
    device = patch.branches.device
    new_branches = torch.as_tensor(new_branches, dtype=torch.long, device=device)
    if new_branches.ndim != 2 or new_branches.shape[1] != patch.depth:
        raise ValueError("New branches must use the existing prefix depth")
    branches = torch.unique(torch.cat([patch.branches, new_branches]), dim=0)
    extended = PrefixPatch(
        patch.hidden_size, patch.vocab_size, branches.cpu(), patch.rank, patch.mode
    ).to(device=device, dtype=patch.input_factor.dtype)
    with torch.no_grad():
        if patch.mode == "global":
            extended.input_factor.copy_(patch.input_factor)
            extended.output_factor.copy_(patch.output_factor)
        else:
            positions = torch.searchsorted(extended.keys, patch.keys)
            extended.input_factor[positions] = patch.input_factor
            extended.output_factor[positions] = patch.output_factor
    return extended.eval()
