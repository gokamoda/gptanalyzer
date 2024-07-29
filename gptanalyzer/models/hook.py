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
