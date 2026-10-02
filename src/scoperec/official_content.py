import numpy as np
import torch
from torch import nn
from torch.nn import functional as functional

from scoperec.decoding import numpy_keys, prefix_keys
from scoperec.official_model import ReleasedTIGER


class FullVocabulary:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size

    def log_probabilities(self, logits, prefix):
        return logits.float().log_softmax(dim=-1)


class ReleasedContentScope(ReleasedTIGER):
    def __init__(
        self,
        base,
        codes,
        groups,
        vectors,
        depth=1,
        alpha=0.1,
        temperature=0.05,
        confidence_threshold=0.5,
        confidence_width=0.05,
    ):
        nn.Module.__init__(self)
        if not 1 <= depth < codes.shape[1] or not 0 <= alpha < 1:
            raise ValueError("Invalid official scope depth or mixture budget")
        if temperature <= 0 or confidence_width <= 0:
            raise ValueError("Temperatures must be positive")
        self.t5 = base.t5
        self.layout = base.layout
        self.specification = base.specification
        self.depth = depth
        self.alpha = alpha
        self.temperature = temperature
        self.confidence_threshold = confidence_threshold
        self.confidence_width = confidence_width
        self.enabled = True
        self.uniform_new_distribution = False
        self._context = None
        self.tree = FullVocabulary(self.layout["vocab_size"])
        new_items = np.flatnonzero(groups == 2)
        if not len(new_items):
            raise ValueError("No cold candidates for content editing")
        sequences = np.column_stack([codes[new_items], np.full(len(new_items), self.layout["eos"])])
        self.register_buffer("new_codes", torch.as_tensor(sequences, device=base.device))
        normalized = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
        self.register_buffer("embeddings", torch.as_tensor(normalized, device=base.device))
        self.register_buffer("new_items", torch.as_tensor(new_items, device=base.device))
        keys = numpy_keys(codes, self.layout["vocab_size"])
        order = np.argsort(keys)
        self.register_buffer("history_keys", torch.as_tensor(keys[order], device=base.device))
        self.register_buffer("history_items", torch.as_tensor(order, device=base.device))
        self.settings = {
            "depth": depth,
            "alpha": alpha,
            "temperature": temperature,
            "confidence_threshold": confidence_threshold,
            "confidence_width": confidence_width,
        }

    def mix_conditionals(self, logits, prefix, context):
        position = prefix.shape[1]
        if position < self.depth or not self.enabled or self.alpha == 0:
            return logits[:, -1]
        weights = context["weights"]
        width = len(prefix) // len(weights)
        expanded = weights.repeat_interleave(width, dim=0)
        vocab_size = self.layout["vocab_size"]
        branch = prefix_keys(prefix[:, : self.depth], vocab_size)
        new_branch = prefix_keys(self.new_codes[:, : self.depth], vocab_size)
        branch_match = branch[:, None] == new_branch[None, :]
        alpha = torch.full((len(prefix),), self.alpha, device=prefix.device)
        if self.confidence_threshold is not None:
            similarities = context["similarities"].repeat_interleave(width, dim=0)
            confidence = similarities.masked_fill(~branch_match, -torch.inf).max(1).values
            alpha *= ((confidence - self.confidence_threshold) / self.confidence_width).sigmoid()
        branch_mass = (expanded * branch_match).sum(1)
        node = prefix_keys(prefix, vocab_size)
        new_node = prefix_keys(self.new_codes[:, :position], vocab_size)
        weighted = expanded * (node[:, None] == new_node[None, :])
        node_mass = weighted.sum(1)
        children = torch.zeros(len(prefix), vocab_size, device=prefix.device)
        children.scatter_add_(
            1, self.new_codes[:, position][None].expand(len(prefix), -1), weighted
        )
        base_mass = torch.zeros(len(prefix), device=prefix.device)
        for previous in range(self.depth, position):
            conditional = logits[:, previous].log_softmax(-1)
            base_mass += conditional.gather(1, prefix[:, previous : previous + 1]).squeeze(1)
        log_base = torch.log1p(-alpha) + base_mass + branch_mass.clamp_min(1e-30).log()
        log_content = alpha.clamp_min(1e-30).log()
        denominator = torch.logaddexp(log_base, log_content + node_mass.clamp_min(1e-30).log())
        log_children = children.clamp_min(1e-30).log().masked_fill(children == 0, -torch.inf)
        mixed = (
            torch.logaddexp(
                log_base[:, None] + logits[:, -1].log_softmax(-1),
                log_content[:, None] + log_children,
            )
            - denominator[:, None]
        )
        active = (branch_mass > 0) & (node_mass > 0) & (alpha > 0)
        return torch.where(active[:, None], mixed, logits[:, -1])

    def content_context(self, inputs):
        counts = inputs.ne(self.layout["padding"]).sum(dim=1)
        item_counts = torch.div(counts - 2, self.layout["sid_length"], rounding_mode="floor")
        if torch.any(item_counts < 1) or torch.any((counts - 2) % self.layout["sid_length"] != 0):
            raise ValueError("Invalid released input history structure")
        rows = torch.arange(len(inputs), device=inputs.device)
        if torch.any(inputs[rows, counts - 1] != self.layout["eos"]):
            raise ValueError("History EOS missing")
        starts = 1 + (item_counts - 1) * self.layout["sid_length"]
        columns = starts[:, None] + torch.arange(self.layout["sid_length"], device=inputs.device)
        last_codes = inputs.gather(1, columns)
        keys = prefix_keys(last_codes, self.layout["vocab_size"])
        positions = torch.searchsorted(self.history_keys, keys).clamp(
            max=len(self.history_keys) - 1
        )
        if torch.any(self.history_keys[positions] != keys):
            raise ValueError("Unregistered history SID")
        query = self.embeddings[self.history_items[positions]]
        similarities = (
            functional.normalize(query.float(), dim=1) @ self.embeddings[self.new_items].T
        )
        scores = similarities / self.temperature
        weights = (scores - scores.max(dim=1, keepdim=True).values).exp()
        if self.uniform_new_distribution:
            weights = torch.ones_like(weights)
        return {"weights": weights, "similarities": similarities}

    def decode_logits(self, encoder, attention_mask, prefix):
        full_logits, hidden = self.full_decoder(encoder, attention_mask, prefix)
        if self._context is None:
            raise RuntimeError("Content query context missing")
        return self.mix_conditionals(full_logits, prefix, self._context), hidden[:, -1]

    @torch.no_grad()
    def generate(self, inputs, beam_size=50):
        self._context = self.content_context(inputs)
        try:
            return super().generate(inputs, beam_size)
        finally:
            self._context = None

    def path_scores(self, inputs, labels):
        logits = self.forward(inputs, labels).logits
        context = self.content_context(inputs)
        mixed = torch.stack(
            [
                self.mix_conditionals(logits[:, : position + 1], labels[:, :position], context)
                for position in range(labels.shape[1])
            ],
            dim=1,
        )
        return mixed.log_softmax(-1).gather(2, labels[:, :, None]).squeeze(-1)
