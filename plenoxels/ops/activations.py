import torch
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd

__all__ = (
    "trunc_exp",
)


class TruncatedExponential(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32) # cast to float32
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, g):
        x = ctx.saved_tensors[0]
        return g * torch.exp(x.clamp(-15, 15))


trunc_exp = TruncatedExponential.apply
