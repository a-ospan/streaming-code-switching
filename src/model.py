"""Dual-head prefix-only predictor on a multilingual transformer backbone.

Backbones:
- XLM-R (bidirectional encoder, as in the proposal): loaded with `is_decoder=True`, which
  makes Hugging Face apply a causal (look-ahead) mask so position i attends only to 0..i.
  XLM-R was pretrained bidirectionally, so this is a train/pretrain mismatch to discuss.
- XGLM (natively left-to-right causal LM): no change needed.
A test checks causality directly: changing a later token must not change earlier outputs.

Heads read the hidden state at every subword; the loss uses only the prediction points
(all other positions carry IGNORE):
    L = lambda_sw * CE(switch) + lambda_dur * CE(duration, ignore_index=-1)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from transformers import AutoConfig, AutoModel

from src.labels import IGNORE

ENCODER_TYPES = {"xlm-roberta", "bert", "roberta"}  # bidirectional: needs is_decoder=True


@dataclass
class HeadsOutput:
    sw_logits: torch.Tensor   # (batch, seq, 2)
    dur_logits: torch.Tensor  # (batch, seq, 3)
    loss: torch.Tensor | None = None
    loss_sw: torch.Tensor | None = None
    loss_dur: torch.Tensor | None = None


def causal_config(name_or_config):
    config = AutoConfig.from_pretrained(name_or_config) if isinstance(name_or_config, str) else name_or_config
    if config.model_type in ENCODER_TYPES:
        config.is_decoder = True
        config.add_cross_attention = False
    return config


class SwitchPredictor(nn.Module):
    def __init__(self, backbone: nn.Module, dropout: float = 0.1, lambda_sw: float = 1.0, lambda_dur: float = 1.0,
                 sw_class_weight: list[float] | None = None):
        super().__init__()
        self.backbone = backbone
        hidden = backbone.config.hidden_size if hasattr(backbone.config, "hidden_size") else backbone.config.d_model
        self.dropout = nn.Dropout(dropout)
        self.sw_head = nn.Linear(hidden, 2)
        self.dur_head = nn.Linear(hidden, 3)
        self.lambda_sw, self.lambda_dur = lambda_sw, lambda_dur
        weight = torch.tensor(sw_class_weight, dtype=torch.float) if sw_class_weight else None
        self.register_buffer("sw_weight", weight, persistent=False)

    @classmethod
    def from_pretrained(cls, name: str, **kwargs) -> "SwitchPredictor":
        return cls(AutoModel.from_pretrained(name, config=causal_config(name)), **kwargs)

    @classmethod
    def from_config(cls, config, **kwargs) -> "SwitchPredictor":
        """Randomly initialized backbone (tests, smoke runs)."""
        return cls(AutoModel.from_config(causal_config(config)), **kwargs)

    def forward(self, input_ids, attention_mask, sw_labels=None, dur_labels=None) -> HeadsOutput:
        h = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        h = self.dropout(h)
        out = HeadsOutput(self.sw_head(h), self.dur_head(h))
        if sw_labels is not None:
            out.loss_sw = nn.functional.cross_entropy(
                out.sw_logits.flatten(0, 1), sw_labels.flatten(), weight=self.sw_weight, ignore_index=IGNORE)
            if (dur_labels != IGNORE).any():
                out.loss_dur = nn.functional.cross_entropy(
                    out.dur_logits.flatten(0, 1), dur_labels.flatten(), ignore_index=IGNORE)
            else:  # a batch with no switch points: CE would be NaN, contribute nothing instead
                out.loss_dur = out.dur_logits.sum() * 0.0
            out.loss = self.lambda_sw * out.loss_sw + self.lambda_dur * out.loss_dur
        return out
