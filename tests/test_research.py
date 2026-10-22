from pathlib import Path
import torch
from normlab.models import ResidualMLP
from normlab.train import Config, get_loss, make_model, set_seed
from scripts.layer_ablation import ablate_one


def test_initial_gradient_independent_of_warmup_label():
    c = Config(layers=3, width=16, heads=4)
    x = torch.randn(8, 16, generator=torch.Generator().manual_seed(23))
    y = torch.randint(2, (8,), generator=torch.Generator().manual_seed(19))
    for kind in ('pre','post'):
        grads=[]
        for warm in (False, True):
            set_seed(7)
            model = make_model(c, kind)
            loss,_ = get_loss(model,x,y,'simple')
            loss.backward()
            grads.append(torch.cat([p.grad.flatten() for p in model.parameters()]))
        assert torch.equal(grads[0], grads[1])


def test_layer_ablation_restores_forward():
    from normlab.train import evaluate
    from normlab.data import classification_data
    c=Config(layers=2,width=16,heads=4,stage='simple')
    model=ResidualMLP(16,2,'pre')
    val=classification_data(222,16,16)
    before,_=evaluate(model,c,val,torch.device('cpu'))
    difference=ablate_one(model,0,c,val,torch.device('cpu'))
    after,_=evaluate(model,c,val,torch.device('cpu'))
    assert abs(before-after) < 1e-8
    assert abs(before-difference) > 1e-9
