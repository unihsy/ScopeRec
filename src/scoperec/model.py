import numpy as np
import torch
from torch import nn
from transformers import T5Config, T5ForConditionalGeneration

from scoperec.decoding import SIDTree, sequence_log_probabilities
from scoperec.patches import PrefixPatch


def history_tokens(histories: np.ndarray, codes: np.ndarray) -> np.ndarray:
    present = histories >= 0
    if np.any(histories[present] >= len(codes)):
        raise ValueError("History item outside SID table")
    tokens = codes[np.maximum(histories, 0)] * present[:, :, None]
    return tokens.reshape(len(histories), -1).astype(np.int64)


class GenerativeRecommender(nn.Module):
    def __init__(
        self, vocab_size: int, d_model=128, d_ff=512, num_layers=2, num_heads=4, d_kv=None
    ):
        super().__init__()
        self.specification = {
            "vocab_size": vocab_size,
            "d_model": d_model,
            "d_ff": d_ff,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "d_kv": d_kv,
        }
        self.t5 = T5ForConditionalGeneration(
            T5Config(
                vocab_size=vocab_size,
                d_model=d_model,
                d_ff=d_ff,
                num_layers=num_layers,
                num_decoder_layers=num_layers,
                num_heads=num_heads,
                d_kv=d_kv if d_kv is not None else d_model // num_heads,
                dropout_rate=0.1,
                feed_forward_proj="gated-silu",
                decoder_start_token_id=1,
                pad_token_id=0,
                eos_token_id=2,
            )
        )

    def teacher(
        self, inputs: torch.Tensor, targets: torch.Tensor, patch: PrefixPatch | None = None
    ):
        outputs = self.t5(
            input_ids=inputs,
            attention_mask=inputs.ne(0),
            labels=targets,
            output_hidden_states=True,
            use_cache=False,
        )
        hidden = outputs.decoder_hidden_states[-1] * self.t5.config.d_model**-0.5
        logits = outputs.logits
        if patch is not None:
            logits = logits + torch.stack(
                [
                    patch(hidden[:, position], targets[:, :position])
                    for position in range(targets.shape[1])
                ],
                dim=1,
            )
        return logits, hidden

    def path_scores(self, inputs, targets, tree: SIDTree, patch=None):
        logits, _ = self.teacher(inputs, targets, patch)
        return sequence_log_probabilities(tree, logits, targets)

    def decode_logits(self, encoder, attention_mask, prefix):
        decoder_ids = torch.cat(
            [torch.ones(len(prefix), 1, dtype=torch.long, device=prefix.device), prefix], dim=1
        )
        hidden = (
            self.t5.decoder(
                input_ids=decoder_ids,
                encoder_hidden_states=encoder,
                encoder_attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            ).last_hidden_state[:, -1]
            * self.t5.config.d_model**-0.5
        )
        return self.t5.lm_head(hidden), hidden

    @torch.no_grad()
    def generate(
        self, inputs, tree: SIDTree, beam_size=50, patch=None, targets=None, forced_depth=0
    ):
        self.eval()
        batch_size = len(inputs)
        attention_mask = inputs.ne(0)
        encoder = self.t5.encoder(
            input_ids=inputs, attention_mask=attention_mask, return_dict=True
        ).last_hidden_state
        if forced_depth:
            if targets is None or not 0 < forced_depth < tree.length:
                raise ValueError("Oracle decoding requires targets and a proper prefix depth")
            paths = targets[:, None, :forced_depth].clone()
        else:
            paths = torch.empty(batch_size, 1, 0, dtype=torch.long, device=inputs.device)
        scores = torch.zeros(batch_size, 1, device=inputs.device)
        survival = []
        for position in range(forced_depth, tree.length):
            width = paths.shape[1]
            prefix = paths.reshape(batch_size * width, position)
            repeated_encoder = (
                encoder[:, None]
                .expand(-1, width, -1, -1)
                .reshape(batch_size * width, encoder.shape[1], encoder.shape[2])
            )
            repeated_mask = (
                attention_mask[:, None]
                .expand(-1, width, -1)
                .reshape(batch_size * width, attention_mask.shape[1])
            )
            logits, decoded = self.decode_logits(repeated_encoder, repeated_mask, prefix)
            if patch is not None:
                logits = logits + patch(decoded, prefix)
            conditional = tree.log_probabilities(logits, prefix)
            candidates = conditional.reshape(batch_size, width, -1) + scores[:, :, None]
            keep = min(beam_size, tree.prefix_counts[position + 1], width * tree.vocab_size)
            scores, selected = candidates.reshape(batch_size, -1).topk(keep, dim=1)
            parents = torch.div(selected, tree.vocab_size, rounding_mode="floor")
            next_tokens = selected % tree.vocab_size
            previous = paths.gather(1, parents[:, :, None].expand(-1, -1, position))
            paths = torch.cat([previous, next_tokens[:, :, None]], dim=2)
            if targets is not None:
                survived = (paths == targets[:, None, : position + 1]).all(dim=2)
                survival.append((survived & scores.isfinite()).any(dim=1))
        items = tree.resolve(paths.reshape(-1, tree.length)).reshape(batch_size, -1)
        items = items.masked_fill(~scores.isfinite(), -1)
        return {
            "items": items,
            "scores": scores,
            "codes": paths,
            "survival": torch.stack(survival, dim=1) if survival else None,
        }
