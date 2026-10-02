import numpy as np
import torch


def prefix_keys(codes: torch.Tensor, vocab_size: int) -> torch.Tensor:
    values = torch.zeros(codes.shape[0], dtype=torch.long, device=codes.device)
    for position in range(codes.shape[1]):
        values = values * vocab_size + codes[:, position]
    return values


def numpy_keys(codes: np.ndarray, vocab_size: int) -> np.ndarray:
    values = np.zeros(codes.shape[0], dtype=np.int64)
    for position in range(codes.shape[1]):
        values = values * vocab_size + codes[:, position]
    return values


class SIDTree:
    def __init__(self, codes: np.ndarray, item_ids: np.ndarray, vocab_size: int, device="cpu"):
        codes = np.asarray(codes, dtype=np.int64)
        if codes.ndim != 2 or not len(codes) or len(codes) != len(item_ids):
            raise ValueError("Expected nonempty aligned codes and item identifiers")
        if len(np.unique(codes, axis=0)) != len(codes):
            raise ValueError("Complete SID codes must be unique")
        if np.any(codes < 0) or np.any(codes >= vocab_size):
            raise ValueError("SID token outside the fixed vocabulary")
        if vocab_size ** codes.shape[1] > np.iinfo(np.int64).max:
            raise ValueError("SID key exceeds int64 capacity")
        self.vocab_size = vocab_size
        self.length = codes.shape[1]
        self.device = torch.device(device)
        self.keys = []
        self.masks = []
        self.prefix_counts = []
        for depth in range(self.length):
            keys, inverse = np.unique(numpy_keys(codes[:, :depth], vocab_size), return_inverse=True)
            allowed = np.zeros((len(keys) + 1, vocab_size), dtype=bool)
            allowed[inverse, codes[:, depth]] = True
            allowed[-1, 0] = True
            self.keys.append(torch.as_tensor(keys, device=self.device))
            self.masks.append(torch.as_tensor(allowed, device=self.device))
            self.prefix_counts.append(len(keys))
        keys = numpy_keys(codes, vocab_size)
        order = np.argsort(keys)
        self.leaf_keys = torch.as_tensor(keys[order], device=self.device)
        self.leaf_items = torch.as_tensor(np.asarray(item_ids)[order], device=self.device)
        self.prefix_counts.append(len(codes))

    def legal_mask(self, prefixes: torch.Tensor) -> torch.Tensor:
        depth = prefixes.shape[1]
        if depth >= self.length:
            raise ValueError("A complete SID has no successor")
        requested = prefix_keys(prefixes, self.vocab_size)
        keys = self.keys[depth]
        positions = torch.searchsorted(keys, requested).clamp(max=len(keys) - 1)
        positions = torch.where(keys[positions] == requested, positions, len(keys))
        return self.masks[depth][positions]

    def log_probabilities(self, logits: torch.Tensor, prefixes: torch.Tensor) -> torch.Tensor:
        if logits.dtype in (torch.float16, torch.bfloat16):
            logits = logits.float()
        return logits.masked_fill(~self.legal_mask(prefixes), -torch.inf).log_softmax(dim=-1)

    def resolve(self, codes: torch.Tensor) -> torch.Tensor:
        keys = prefix_keys(codes, self.vocab_size)
        positions = torch.searchsorted(self.leaf_keys, keys).clamp(max=len(self.leaf_keys) - 1)
        return torch.where(self.leaf_keys[positions] == keys, self.leaf_items[positions], -1)


def sequence_log_probabilities(tree: SIDTree, logits: torch.Tensor, codes: torch.Tensor):
    selected = []
    for position in range(tree.length):
        log_probabilities = tree.log_probabilities(logits[:, position], codes[:, :position])
        selected.append(log_probabilities.gather(1, codes[:, position:position + 1]).squeeze(1))
    return torch.stack(selected, dim=1)