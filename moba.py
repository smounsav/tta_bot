# https://github.com/DequanWang/tent/blob/master/tent.py

from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit

import math

import torchvision.transforms as transforms

augment = nn.Sequential(
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(224, padding=(23,23)),
)
augment = torch.jit.script(augment)

from utils.polyloss import Poly1CrossEntropyLoss

# inspired by https://github.com/bethgelab/robustness/tree/aa0a6798fe3973bae5f47561721b59b39f126ab7
def find_bns(parent, prior):
    replace_mods = []
    if parent is None:
        return []
    for name, child in parent.named_children():
        child.requires_grad_(False)
        if isinstance(child, nn.BatchNorm2d):
            module = TBR(child, prior).cuda()
            replace_mods.append((parent, name, module))
        else:
            replace_mods.extend(find_bns(child, prior))
    return replace_mods


class TBR(nn.Module):
    def __init__(self, layer, prior):
        assert prior >= 0 and prior <= 1
        super().__init__()
        self.layer = layer
        self.layer.eval()
        self.prior = prior
        self.rmax = 3.0
        self.dmax = 5.0
        self.tracked_num = 0
        self.running_mean = None
        self.running_std = None

    def forward(self, input):
        batch_mean = input.mean([0, 2, 3])
        batch_std = torch.sqrt(input.var([0, 2, 3], unbiased=False) + self.layer.eps)

        if self.running_mean is None:
            self.running_mean = batch_mean.detach().clone()
            self.running_std = batch_std.detach().clone()

        r = (batch_std.detach() / self.running_std) #.clamp_(1./self.rmax, self.rmax)
        d = ((batch_mean.detach() - self.running_mean) / self.running_std) #.clamp_(-self.dmax, self.dmax)
        
        input = (input - batch_mean[None,:,None,None]) / batch_std[None,:,None,None] * r[None,:,None,None] + d[None,:,None,None]

        self.running_mean = self.prior * self.running_mean + (1. - self.prior) * batch_mean.detach()
        self.running_std = self.prior * self.running_std + (1. - self.prior) * batch_std.detach()

        self.tracked_num+=1

        return input * self.layer.weight[None,:,None,None] + self.layer.bias[None,:,None,None]

class MOBA(nn.Module):
    """Tent adapts a model by entropy minimization during testing.
    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, optimizer, moob_factor=0.9, buffer_size=16, temperature=10.0, class_rebalancing=False, loss_type='entropy', steps=1, episodic=False, class_num=1000):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.episodic = episodic

        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        # self.model_state, self.optimizer_state = \
        #     copy_model_and_optimizer(self.model, self.optimizer)

        self.time_decay = moob_factor
        self.class_num = class_num
        self.weight_per_class = torch.zeros(self.class_num, device="cuda") + 1 / self.class_num
        # self.n_samples_seen_per_class = torch.zeros(self.class_num, device="cuda")
        # self.n_samples_repeated_per_class = torch.zeros(self.class_num, device="cuda")
        # self.running_class_entropy_seen = []
        # self.running_class_entropy_repeated = []
        self.temperature = temperature / 10
        self.class_rebalancing = class_rebalancing
        self.loss_type = loss_type
        if self.loss_type == 'poly_loss':
            self.polyloss = Poly1CrossEntropyLoss(num_classes=self.class_num,
                        reduction='none')
        self.buffer_size = buffer_size
        self.buffer = torch.zeros(self.buffer_size, device='cuda')

    def forward(self, x, rendition_mask=None):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            outputs = forward_and_adapt(self, x, self.model, self.optimizer, rendition_mask)

        return outputs

    def forward_only(self, x, rendition_mask=None):
        if rendition_mask is None:
            outputs = self.model(x) / self.temperature
        else:
            outputs = self.model(x)[:, rendition_mask] / self.temperature

        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt(self, x, model, optimizer, rendition_mask=None):
    """Forward and adapt model on batch of data.
    Measure entropy of the model prediction, take gradients, and update params.
    """
    # forward
    if rendition_mask is None:
        outputs = model(x) / self.temperature
    else:
        outputs = model(x)[:, rendition_mask] / self.temperature
    pseudo_labels = outputs.argmax(1).detach()

    # adapt
    if self.loss_type == 'cross_entropy':
        cost = nn.functional.cross_entropy(outputs, pseudo_labels, reduction='none')
    elif self.loss_type == 'poly_loss':
        cost = self.polyloss(outputs, pseudo_labels)
    else:
        cost = softmax_entropy(outputs)        

    # Compute number of times to take each selected sample into account in the loss
    # MOOB Wang et al. 2016
    if self.class_rebalancing:
        w_max = self.weight_per_class.max()
        # if len(pseudo_labels) > 1:
        nb_repeat = torch.poisson(w_max/self.weight_per_class[pseudo_labels])
        nb_repeat[nb_repeat == 0] = 1
        sample_weights = nb_repeat / (nb_repeat.sum() + self.buffer[0:self.buffer_size-len(nb_repeat)].sum()) * len(nb_repeat)
        self.buffer = torch.cat([nb_repeat, self.buffer[0:self.buffer_size-len(nb_repeat)]])
        # else:
            # sample_weights = -85.17 * self.weight_per_class[pseudo_labels] + 3.40
            # sample_weights = -torch.log((1 - self.time_decay) * self.class_num * self.weight_per_class[pseudo_labels])
            # sample_weights = 1
            # sample_weights = torch.tanh((w_max/self.weight_per_class[pseudo_labels]) - 1)
    else:
        sample_weights = torch.ones_like(pseudo_labels)
    
    loss = (cost * sample_weights).mean(0)
    loss.backward()

    optimizer.step()
    optimizer.zero_grad()
    
    self.weight_per_class = self.time_decay * self.weight_per_class + (1 - self.time_decay) * torch.bincount(pseudo_labels, minlength=self.class_num)

    return outputs


def collect_params(model):
    """Collect the affine scale + shift parameters from norm layers.
    Walk the model's modules and collect all normalization parameters.
    Return the parameters and their names.
    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def configure_model(model):
    # Replace BatchNorm2D layers by BatchRenorm2D
    replace_mods = find_bns(model, 0.95)
    for (parent, name, child) in replace_mods:
        setattr(parent, name, child) 
    """Configure model for use with tent."""
    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what tent updates
    model.requires_grad_(False)
    # configure norm for tent updates: enable grad + force batch statisics
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
        if isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
            m.requires_grad_(True)
    return model


def check_model(model):
    """Check model for compatability with tent."""
    is_training = model.training
    assert is_training, "tent needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "tent needs params to update: " \
                           "check which require grad"
    assert not has_all_params, "tent should not update all params: " \
                               "check which require grad"
    has_bn = any([isinstance(m, nn.BatchNorm2d) for m in model.modules()])
    assert has_bn, "tent needs normalization for its optimization"
