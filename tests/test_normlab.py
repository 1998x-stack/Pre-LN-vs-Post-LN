import json
import numpy as np
import pytest
import torch
from normlab.data import RECALL_VOCAB, classification_data, recall_batch, lm_batch
from normlab.models import CausalTransformer, ResidualMLP, parameter_gradient_norms
from normlab.train import Config, CONDITIONS, get_loss, make_model, set_seed


def test_condition_factorial():
    assert set(CONDITIONS) == {'pre', 'post', 'pre_warm', 'post_warm'}
    assert CONDITIONS['pre'][0] == CONDITIONS['pre_warm'][0]
    assert CONDITIONS['post'][0] == CONDITIONS['post_warm'][0]


@pytest.mark.parametrize('kind', ['pre', 'post'])
def test_mlp_forward_backward(kind):
    model = ResidualMLP(16, 3, kind)
    x, y = classification_data(19, 16, 9)
    loss, acc = get_loss(model, x, y, 'simple', trace=True)
    loss.backward()
    global_norm, per_layer = parameter_gradient_norms(model)
    assert torch.isfinite(loss)
    assert 0 <= acc <= 1
    assert len(per_layer) == 3 and global_norm > 0
    assert all(x.trace['residual_delta_ratio'] >= 0 for x in model.blocks)


@pytest.mark.parametrize('kind', ['pre', 'post'])
def test_recall_is_valid_and_trains(kind):
    x, y = recall_batch(torch.Generator().manual_seed(3), 8, 4)
    assert x.max() < RECALL_VOCAB
    assert all(int(y[i]) in x[i, 2:-2:2] for i in range(8))
    model = CausalTransformer(RECALL_VOCAB, 16, 2, 4, x.size(1), kind)
    loss, accuracy = get_loss(model, x, y, 'medium', trace=True)
    loss.backward()
    assert torch.isfinite(loss) and 0 <= accuracy <= 1
    assert all('attn_delta_ratio' in block.trace for block in model.blocks)


def test_matched_initialization_for_warmup_control():
    config = Config(layers=2, width=16, heads=4)
    for a, b in [('pre', 'pre_warm'), ('post', 'post_warm')]:
        set_seed(7)
        weights_a = [p.detach().clone() for p in make_model(config, CONDITIONS[a][0]).parameters()]
        set_seed(7)
        weights_b = list(make_model(config, CONDITIONS[b][0]).parameters())
        assert all(torch.equal(u, v) for u, v in zip(weights_a, weights_b))


def test_lm_windows(tmp_path):
    x = np.arange(100, dtype=np.uint16)
    np.save(tmp_path / 'tokens.npy', x)
    a, b = lm_batch(x, torch.Generator().manual_seed(7), 5, 8)
    assert a.shape == b.shape == (5, 8)
    assert torch.equal(a[:, 1:], b[:, :-1])


def test_config_rejects_missing_corpus():
    with pytest.raises(ValueError):
        Config(stage='complex').validate()
