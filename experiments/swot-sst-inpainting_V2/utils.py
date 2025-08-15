r"""CIFAR experiment helpers"""
import inox # type: ignore
import inox.nn as nn # type: ignore
import jax # type: ignore
from jax import Array # type: ignore
from inox.tree import Partial as Partial # type: ignore

from typing import Dict, Sequence, Optional
from pathlib import Path
import os


# isort: split
from priors.diffusion import Denoiser
from priors.image import flatten, unflatten
from priors.common import distribute, sample_any
from priors.nn import UNet

if 'SCRATCH' in os.environ:
    SCRATCH = os.environ['SCRATCH']
    PATH = Path(SCRATCH) / 'priors/cifar'
else:
    PATH = Path('.')

PATH.mkdir(parents=True, exist_ok=True)

def measure(A: Array, x: Array, H: int, W: int, C: int) -> Array:
    x_unflat = unflatten(x, H, W,)
    return flatten(A * x_unflat) 

def sample(
    model: nn.Module,
    y: Array,
    A: Array,
    key: Array,
    shard: bool = False,
    **kwargs,
) -> Array:
    if shard:
        y, A = distribute((y, A))

    B, H, W, C = y.shape
    D = H * W * C

    x = sample_any(
        model=model,
        shape=flatten(y).shape,
        shard=shard,
        A=inox.tree.Partial(measure, A, H=H, W=W, C=C),
        y=flatten(y),
        cov_y=1e-3**2,
        key=key,
        **kwargs,
    )

    return unflatten(x, H, W)

def make_model(
    key: Array,
    hid_channels: Sequence[int] = (64, 128, 256),
    hid_blocks: Sequence[int] = (3, 3, 3),
    kernel_size: Sequence[int] = (3, 3),
    emb_features: int = 256,
    heads: Dict[int, int] = {2: 1},
    dropout: Optional[float] = None,
    **absorb,
) -> Denoiser:
    init_key, dropout_key = jax.random.split(key)
    return Denoiser(
        network=FlatUNet(
            in_channels=3,
            out_channels=3,
            hid_channels=hid_channels,
            hid_blocks=hid_blocks,
            kernel_size=kernel_size,
            emb_features=emb_features,
            heads=heads,
            dropout=dropout,
            init_key=init_key,
            dropout_key=dropout_key,
        ),
        emb_features=emb_features,
    )

class FlatUNet(UNet):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hid_channels: Sequence[int],
        hid_blocks: Sequence[int],
        kernel_size: Sequence[int],
        emb_features: int,
        heads: Dict[int, int],
        dropout: Optional[float],
        init_key: Array,
        dropout_key: Array,  # You may route this later
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            hid_channels=hid_channels,
            hid_blocks=hid_blocks,
            kernel_size=kernel_size,
            emb_features=emb_features,
            heads=heads,
            dropout=dropout,
            key=init_key,
        )
        self.dropout_key = dropout_key  # Store it if needed later
        # Split and pass to child modules here

    def __call__(self, x: Array, t: Array, key: Array = None) -> Array:
        x = unflatten(x, width=128, height=128)
        x = super().__call__(x, t, key)
        x = flatten(x)
        return x


