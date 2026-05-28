import typing

import jax
import numpy as np


def to_tensor(x) -> jax.numpy.ndarray:
    return jax.numpy.asarray(x)


def to_numpy(x: jax.numpy.ndarray) -> np.ndarray:
    return np.asarray(x)


def framework(x):
    return "jax"


ArrayLike = typing.Union[np.ndarray, jax.numpy.ndarray]
