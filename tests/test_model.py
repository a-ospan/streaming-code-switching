"""Model tests on tiny randomly initialized backbones (no downloads)."""

import pytest
import torch
from transformers import XGLMConfig, XLMRobertaConfig

from src.labels import IGNORE as X
from src.model import SwitchPredictor

TINY = {
    "xlm-r": XLMRobertaConfig(vocab_size=100, hidden_size=32, num_hidden_layers=2, num_attention_heads=2,
                              intermediate_size=64, max_position_embeddings=64, pad_token_id=1),
    "xglm": XGLMConfig(vocab_size=100, d_model=32, num_layers=2, attention_heads=2, ffn_dim=64,
                       max_position_embeddings=64, pad_token_id=1),
}


@pytest.fixture(params=list(TINY))
def model(request):
    torch.manual_seed(0)
    m = SwitchPredictor.from_config(TINY[request.param], dropout=0.0)
    return m.eval()


def test_outputs_at_t_ignore_later_tokens(model):
    """The core no-leakage property: editing tokens after position t leaves outputs at <= t unchanged."""
    ids = torch.tensor([[0, 11, 12, 13, 14, 15, 16, 2]])
    mask = torch.ones_like(ids)
    base = model(ids, mask)
    for t in range(1, ids.shape[1] - 1):
        changed = ids.clone()
        changed[0, t + 1:] = torch.tensor([50 + i for i in range(ids.shape[1] - t - 1)])
        out = model(changed, mask)
        assert torch.allclose(out.sw_logits[0, : t + 1], base.sw_logits[0, : t + 1], atol=1e-5), f"leak at {t}"
        assert torch.allclose(out.dur_logits[0, : t + 1], base.dur_logits[0, : t + 1], atol=1e-5)
        assert not torch.allclose(out.sw_logits[0, t + 1:], base.sw_logits[0, t + 1:], atol=1e-5)


def test_right_padding_does_not_change_outputs(model):
    ids = torch.tensor([[0, 11, 12, 13, 2]])
    padded = torch.tensor([[0, 11, 12, 13, 2, 1, 1]])
    a = model(ids, torch.ones_like(ids)).sw_logits
    b = model(padded, torch.tensor([[1, 1, 1, 1, 1, 0, 0]])).sw_logits
    assert torch.allclose(a[0], b[0, :5], atol=1e-5)


def test_loss_uses_only_labeled_positions_and_reaches_backbone(model):
    model.train()
    ids = torch.tensor([[0, 11, 12, 13, 2]])
    sw = torch.tensor([[X, 0, 1, X, X]])
    dur = torch.tensor([[X, X, 2, X, X]])
    out = model(ids, torch.ones_like(ids), sw, dur)
    manual_sw = torch.nn.functional.cross_entropy(out.sw_logits[0, [1, 2]], torch.tensor([0, 1]))
    assert torch.allclose(out.loss_sw, manual_sw)
    out.loss.backward()
    grads = [p.grad for p in model.backbone.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)
    assert model.dur_head.weight.grad.abs().sum() > 0


def test_batch_without_switches_has_finite_loss(model):
    ids = torch.tensor([[0, 11, 12, 2]])
    out = model(ids, torch.ones_like(ids), torch.tensor([[X, 0, 0, X]]), torch.tensor([[X, X, X, X]]))
    assert torch.isfinite(out.loss) and out.loss_dur.item() == 0.0
