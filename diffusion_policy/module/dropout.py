import torch
from torch import nn


class DropoutFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, training, p):
        if training:
            mask = torch.empty_like(x).bernoulli_(1 - p) / (1 - p)
            ctx.save_for_backward(mask)
            return x * mask
        else:
            mask = None
            ctx.save_for_backward(mask)
            return x

    @staticmethod
    def backward(ctx, grad_output):
        mask = ctx.saved_tensors
        return grad_output * mask


class ManualDropout(nn.Module):
    def __init__(self, p):
        super(ManualDropout, self).__init__()
        self.p = p

    def forward(self, x):
        return DropoutFunc.apply(x, self.training, self.p)
