import torch
from torch import nn


class ForwardHook(nn.Module):
    """Class to pass intermediate results to Hook.

    Parameters
    ----------
    nn : _type_
        _description_
    """

    def __init__(self):
        super().__init__()

    def forward(self, **kwargs) -> None:
        """Hooks will catch kwargs.

        Returns
        -------
        None
        """
        del kwargs


class MyLayerNorm(nn.LayerNorm):
    """LayerNorm with hooks."""

    def __init__(
        self,
        normalized_shape,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        device=None,
        dtype=None,
        collapsed=False,
    ):
        super().__init__(
            normalized_shape,
            eps=eps,
            elementwise_affine=elementwise_affine,
            bias=bias,
            device=device,
            dtype=dtype,
        )
        self.mean = None
        self.var = None
        self.for_hook = ForwardHook()
        self.collapsed = collapsed

    # pylint: disable=arguments-renamed
    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        """LN with intervention.

        Parameters
        ----------
        hidden_states : torch.FloatTensor
            _description_

        Returns
        -------
        torch.FloatTensor
            _description_
        """
        if self.mean is not None:
            mean = self.mean
        else:
            mean = hidden_states.mean(dim=-1, keepdim=True)

        if self.var is not None:
            var = self.var
        else:
            var = hidden_states.var(dim=-1, unbiased=False, keepdim=True)
        self.for_hook(
            mean=mean.detach().to("cpu"),
            var=var.detach().to("cpu"),
        )
        self.mean = None
        self.var = None
        # return norm
        if self.collapsed:
            norm = hidden_states / torch.sqrt(var + self.eps)
            return norm

        norm = (hidden_states - mean) / torch.sqrt(var + self.eps)
        return norm * self.weight + self.bias
