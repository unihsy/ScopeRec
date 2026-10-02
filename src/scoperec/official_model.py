import torch
from torch import nn
from torch.nn import functional as functional
from transformers import T5Config, T5ForConditionalGeneration


class ReleasedTIGER(nn.Module):
    def __init__(self, layout, d_model=128, d_ff=1024, num_layers=6, num_heads=8, d_kv=64):
        super().__init__()
        self.layout = layout
        self.specification = {
            "layout": layout,
            "d_model": d_model,
            "d_ff": d_ff,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "d_kv": d_kv,
        }
        self.t5 = T5ForConditionalGeneration(
            T5Config(
                vocab_size=layout["vocab_size"],
                d_model=d_model,
                d_ff=d_ff,
                num_layers=num_layers,
                num_decoder_layers=num_layers,
                num_heads=num_heads,
                d_kv=d_kv,
                dropout_rate=0.1,
                feed_forward_proj="gated-silu",
                decoder_start_token_id=layout["decoder_start"],
                pad_token_id=layout["padding"],
                eos_token_id=layout["eos"],
            )
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, inputs, labels):
        return self.t5(
            input_ids=inputs, attention_mask=inputs.ne(0), labels=labels, use_cache=False
        )

    def full_decoder(self, encoder, attention_mask, prefix):
        decoder_inputs = torch.cat(
            [
                torch.full(
                    (len(prefix), 1),
                    self.layout["decoder_start"],
                    dtype=torch.long,
                    device=prefix.device,
                ),
                prefix,
            ],
            dim=1,
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
        return self.t5.lm_head(hidden), hidden

    def decode_logits(self, encoder, attention_mask, prefix):
        logits, hidden = self.full_decoder(encoder, attention_mask, prefix)
        return logits[:, -1], hidden[:, -1]

    def path_scores(self, inputs, labels):
        logits = self.forward(inputs, labels).logits
        return logits.log_softmax(-1).gather(2, labels[:, :, None]).squeeze(-1)

    @torch.no_grad()
    def generate(self, inputs, beam_size=50):
        self.eval()
        batch_size = len(inputs)
        encoder = self.t5.encoder(input_ids=inputs, attention_mask=inputs.ne(0)).last_hidden_state
        paths = torch.empty(batch_size, 1, 0, dtype=torch.long, device=inputs.device)
        scores = torch.zeros(batch_size, 1, device=inputs.device)
        vocab_size = self.layout["vocab_size"]
        for position in range(self.layout["sid_length"] + 1):
            width = paths.shape[1]
            repeated_encoder = (
                encoder[:, None]
                .expand(-1, width, -1, -1)
                .reshape(batch_size * width, encoder.shape[1], encoder.shape[2])
            )
            repeated_mask = (
                inputs.ne(0)[:, None]
                .expand(-1, width, -1)
                .reshape(batch_size * width, inputs.shape[1])
            )
            prefix = paths.reshape(batch_size * width, position)
            logits, _ = self.decode_logits(repeated_encoder, repeated_mask, prefix)
            probabilities = logits.log_softmax(-1).reshape(batch_size, width, vocab_size)
            candidates = probabilities + scores[:, :, None]
            keep = min(beam_size, width * vocab_size)
            scores, chosen = candidates.reshape(batch_size, -1).topk(keep, dim=1)
            parents = torch.div(chosen, vocab_size, rounding_mode="floor")
            previous = paths.gather(1, parents[:, :, None].expand(-1, -1, position))
            paths = torch.cat([previous, (chosen % vocab_size)[:, :, None]], dim=2)
        return {
            "codes": paths[:, :, : self.layout["sid_length"]],
            "sequences": paths,
            "scores": scores,
        }


class ReleasedGenRecEdit(ReleasedTIGER):
    def __init__(self, base, position_layers, deltas):
        nn.Module.__init__(self)
        self.t5 = base.t5
        self.layout = base.layout
        self.specification = base.specification
        self.position_layers = list(position_layers)
        self.register_buffer("weight_deltas", torch.stack(deltas))
        self.enabled = True

    def decode_logits(self, encoder, attention_mask, prefix):
        position = prefix.shape[1]
        if not self.enabled or position >= len(self.position_layers):
            return super().decode_logits(encoder, attention_mask, prefix)
        module = self.t5.decoder.block[self.position_layers[position]].layer[2].DenseReluDense.wo

        def inject(module, arguments, output):
            result = output.clone()
            result[:, -1] += functional.linear(
                arguments[0][:, -1], self.weight_deltas[position].to(arguments[0].dtype)
            )
            return result

        handle = module.register_forward_hook(inject)
        try:
            return super().decode_logits(encoder, attention_mask, prefix)
        finally:
            handle.remove()

    def path_scores(self, inputs, labels):
        if not self.enabled:
            return super().path_scores(inputs, labels)
        encoder = self.t5.encoder(input_ids=inputs, attention_mask=inputs.ne(0)).last_hidden_state
        logits = torch.stack(
            [
                self.decode_logits(encoder, inputs.ne(0), labels[:, :position])[0]
                for position in range(labels.shape[1])
            ],
            dim=1,
        )
        return logits.log_softmax(-1).gather(2, labels[:, :, None]).squeeze(-1)
