import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as functional

from scoperec.decoding import SIDTree, numpy_keys, prefix_keys
from scoperec.model import GenerativeRecommender


class ContentScopeModel(GenerativeRecommender):
    def __init__(
        self,
        base,
        codes,
        groups,
        vectors,
        depth=2,
        alpha=0.05,
        temperature=0.05,
        query_mode="last",
        confidence_threshold=None,
        confidence_width=0.05,
    ):
        nn.Module.__init__(self)
        if not 0 <= alpha < 1 or temperature <= 0 or not 1 <= depth < codes.shape[1]:
            raise ValueError("Invalid subtree mixture parameters")
        if query_mode not in ("last", "mean"):
            raise ValueError("Unknown content query mode")
        if confidence_width <= 0:
            raise ValueError("Confidence width must be positive")
        self.confidence_threshold = confidence_threshold
        self.confidence_width = confidence_width
        self.t5 = base.t5
        self.specification = base.specification.copy()
        self.depth = depth
        self.alpha = alpha
        self.temperature = temperature
        self.query_mode = query_mode
        self.enabled = True
        self.uniform_new_distribution = False
        self._context = None
        device = next(base.parameters()).device
        vocab_size = self.specification["vocab_size"]
        self.tree = SIDTree(codes[groups >= 0], np.flatnonzero(groups >= 0), vocab_size, device)
        new_items = np.flatnonzero(groups == 2)
        if not len(new_items):
            raise ValueError("Content scope requires new catalog items")
        self.register_buffer("new_codes", torch.as_tensor(codes[new_items], device=device))
        self.register_buffer("embeddings", torch.as_tensor(vectors, device=device))
        self.register_buffer("new_items", torch.as_tensor(new_items, device=device))
        encoded = numpy_keys(codes, vocab_size)
        order = np.argsort(encoded)
        self.register_buffer("history_keys", torch.as_tensor(encoded[order], device=device))
        self.register_buffer("history_items", torch.as_tensor(order, device=device))
        self.settings = {
            "depth": depth,
            "alpha": alpha,
            "temperature": temperature,
            "query_mode": query_mode,
            "confidence_threshold": confidence_threshold,
            "confidence_width": confidence_width,
        }

    def content_context(self, inputs):
        length = self.new_codes.shape[1]
        if inputs.shape[1] % length:
            raise ValueError("History must contain complete SID codes")
        history = inputs.reshape(len(inputs), -1, length)
        valid = history.ne(0).any(dim=-1)
        if not valid.any(dim=1).all():
            raise ValueError("Content query requires nonempty history")
        keys = prefix_keys(history.reshape(-1, length), self.tree.vocab_size).reshape(valid.shape)
        positions = torch.searchsorted(self.history_keys, keys).clamp(
            max=len(self.history_keys) - 1
        )
        if torch.any(valid & (self.history_keys[positions] != keys)):
            raise ValueError("History contains an unregistered SID")
        values = self.embeddings[self.history_items[positions]]
        if self.query_mode == "last":
            last = valid.sum(dim=1) - 1
            query = values[torch.arange(len(inputs), device=inputs.device), last]
        else:
            query = (values * valid[:, :, None]).sum(dim=1) / valid.sum(dim=1)[:, None]
        query = functional.normalize(query.float(), dim=1)
        similarities = query @ self.embeddings[self.new_items].T
        scores = similarities.float() / self.temperature
        weights = (scores - scores.max(dim=1, keepdim=True).values).exp()
        if self.uniform_new_distribution:
            weights = torch.ones_like(weights)
        return {"weights": weights, "similarities": similarities.float()}

    def mix_conditionals(self, full_logits, prefix, context):
        weights = context["weights"]
        position = prefix.shape[1]
        if position < self.depth or not self.enabled or self.alpha == 0:
            return full_logits[:, -1]
        vocab_size = self.tree.vocab_size
        batch_size = len(weights)
        if len(prefix) % batch_size:
            raise ValueError("Beam rows are not aligned to query contexts")
        width = len(prefix) // batch_size
        expanded = weights.repeat_interleave(width, dim=0)
        branch_key = prefix_keys(prefix[:, : self.depth], vocab_size)
        new_branch_key = prefix_keys(self.new_codes[:, : self.depth], vocab_size)
        in_branch = branch_key[:, None] == new_branch_key[None, :]
        alpha = torch.full((len(prefix),), self.alpha, device=prefix.device)
        if self.confidence_threshold is not None:
            similarities = context["similarities"].repeat_interleave(width, dim=0)
            confidence = similarities.masked_fill(~in_branch, -torch.inf).max(dim=1).values
            alpha = (
                alpha * ((confidence - self.confidence_threshold) / self.confidence_width).sigmoid()
            )
        branch_mass = (expanded * in_branch).sum(dim=1)
        node_key = prefix_keys(prefix, vocab_size)
        new_node_key = prefix_keys(self.new_codes[:, :position], vocab_size)
        matching = node_key[:, None] == new_node_key[None, :]
        weighted = expanded * matching
        node_mass = weighted.sum(dim=1)
        content_child_mass = torch.zeros(
            len(prefix), vocab_size, device=prefix.device, dtype=weights.dtype
        )
        content_child_mass.scatter_add_(
            1, self.new_codes[:, position][None, :].expand(len(prefix), -1), weighted
        )
        base_log_mass = torch.zeros(len(prefix), device=prefix.device)
        for previous in range(self.depth, position):
            base_conditional = self.tree.log_probabilities(
                full_logits[:, previous], prefix[:, :previous]
            )
            base_log_mass += base_conditional.gather(1, prefix[:, previous : previous + 1]).squeeze(
                1
            )
        base = self.tree.log_probabilities(full_logits[:, -1], prefix)
        base_component = torch.log1p(-alpha) + base_log_mass + branch_mass.clamp_min(1e-30).log()
        content_component = alpha.clamp_min(1e-30).log() + node_mass.clamp_min(1e-30).log()
        fraction = (content_component - base_component).sigmoid()
        active = (branch_mass > 0) & (node_mass > 0) & (alpha > 0)
        fraction = torch.where(active, fraction, 0)
        content_conditional = content_child_mass / node_mass.clamp_min(1e-30)[:, None]
        mixed = (1 - fraction[:, None]) * base.exp() + fraction[:, None] * content_conditional
        mixed_logits = mixed.clamp_min(torch.finfo(mixed.dtype).tiny).log()
        return torch.where(active[:, None], mixed_logits, full_logits[:, -1].float())

    def decode_logits(self, encoder, attention_mask, prefix):
        if self._context is None:
            raise RuntimeError("Content context is missing")
        decoder_inputs = torch.cat(
            [torch.ones(len(prefix), 1, dtype=torch.long, device=prefix.device), prefix], dim=1
        )
        hidden = (
            self.t5.decoder(
                input_ids=decoder_inputs,
                encoder_hidden_states=encoder,
                encoder_attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            ).last_hidden_state
            * self.t5.config.d_model**-0.5
        )
        logits = self.t5.lm_head(hidden)
        return self.mix_conditionals(logits, prefix, self._context), hidden[:, -1]

    @torch.no_grad()
    def generate(self, inputs, tree, beam_size=50, patch=None, targets=None, forced_depth=0):
        if patch is not None:
            raise ValueError("Use an explicit ablation to combine learned and content patches")
        self._context = self.content_context(inputs)
        try:
            return super().generate(inputs, tree, beam_size, None, targets, forced_depth)
        finally:
            self._context = None

    def teacher(self, inputs, targets, patch=None):
        if patch is not None:
            raise ValueError("Content mixture does not accept an additional patch")
        logits, hidden = super().teacher(inputs, targets)
        context = self.content_context(inputs)
        mixed = torch.stack(
            [
                self.mix_conditionals(logits[:, : position + 1], targets[:, :position], context)
                for position in range(targets.shape[1])
            ],
            dim=1,
        )
        return mixed, hidden


def save_content_artifact(path: Path, settings, provenance):
    required = ("base_sha256", "sid_sha256", "embeddings_sha256", "catalog_groups_sha256")
    if any(not provenance.get(name) for name in required):
        raise ValueError("Content patch requires base, SID, embeddings and catalog versions")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "kind": "scoped_content_mixture",
        "settings": settings,
        "provenance": provenance,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def load_content_artifact(base, path: Path, codes, groups, vectors, provenance):
    payload = json.loads(path.read_text())
    if payload["format_version"] != 1 or payload["kind"] != "scoped_content_mixture":
        raise ValueError("Unsupported content artifact")
    for name in ("base_sha256", "sid_sha256", "embeddings_sha256", "catalog_groups_sha256"):
        if not provenance.get(name) or payload["provenance"].get(name) != provenance[name]:
            raise ValueError(f"Content patch version mismatch: {name}")
    settings = payload["settings"].copy()
    uniform = settings.pop("uniform_new_distribution", False)
    model = ContentScopeModel(base, codes, groups, vectors, **settings).eval()
    model.uniform_new_distribution = uniform
    return model
