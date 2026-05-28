import typing

import numpy as np
import torch


def to_tensor(x: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(x)


def to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.cpu().numpy()


def framework(x):
    return "torch"


ArrayLike = typing.Union[np.ndarray, torch.Tensor]
