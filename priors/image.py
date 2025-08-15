r"""Image helpers"""

import dm_pix as pix # type: ignore
import jax # type: ignore
import jax.numpy as jnp # type: ignore
import numpy as np # type: ignore

from einops import rearrange # type: ignore
from jax import Array # type: ignore
from pathlib import Path
from PIL import Image # type: ignore
from typing import *


def flatten(x: Array) -> Array:
    return rearrange(x, '... H W C -> ... (H W C)')


def unflatten(x: Array, height: int, width: int) -> Array:
    return rearrange(x, '... (H W C) -> ... H W C', H=height, W=width)


def from_pil(img: Image.Image) -> Array:
    x = np.asarray(img)
    x = x * (4 / 256) - 2

    return x


def to_pil(
    x: np.ndarray,
    pad: int = 0,
    background: int = 255,
    zoom: int = 1,
    file: Optional[Union[str, Path]] = None,
) -> Image.Image:
    """
    Convert a batched grid of images into a single PIL Image.

    x: Array of shape (M, N, H, W, C)
       where:
         M: rows in grid
         N: cols in grid
         H: height
         W: width
         C: channels (1 or 3)
    """
    x = np.asarray(x)
    # Scale to uint8
    x = np.clip((x + 2) * (256 / 4), 0, 255)
    x = np.rint(x).astype(np.uint8)
    # Pad
    x = np.pad(
        x,
        pad_width=((0,0), (0,0), (pad,pad), (pad,pad), (0,0)),
        constant_values=background
    )
    # Rearrange grid to single large image
    x = rearrange(x, 'M N H W C -> (M H) (N W) C')
    # Handle single-channel (grayscale) or multi-channel
    if x.shape[-1] == 1:
        img = Image.fromarray(x.squeeze(-1), mode='L')
    elif x.shape[-1] == 3:
        img = Image.fromarray(x, mode='RGB')
    else:
        raise ValueError(f"Unsupported number of channels: {x.shape[-1]}")
    # Resize (zoom)
    if zoom > 1:
        img = img.resize(
            (zoom * img.width, zoom * img.height),
            Image.NEAREST
        )
    if file is not None:
        img.save(file)
    return img

def collate(
    images: List[List[Image.Image]],
    pad: int = 0,
    background: int = 255,
    file: Optional[Union[str, Path]] = None,
) -> Image.Image:
    M, N = len(images), max(map(len, images))
    W, H = None, None
    for i in range(M):
        for j in range(N):
            try:
                W, H = images[i][j].size
            except IndexError:
                continue
        if W is not None and H is not None:
            break  # Exit outer loop if size found

    if W is None or H is None:
        raise ValueError("No valid images found to determine canvas size.")
    canvas = Image.new(
        'RGB',
        size=(
            N * (W + pad) + pad,
            M * (H + pad) + pad,
        ),
        color=background,
    )
    for i in range(M):
        for j in range(N):
            offset = (
                j * (W + pad) + pad,
                i * (H + pad) + pad,
            )
            try:
                canvas.paste(images[i][j], offset)
            except IndexError:
                continue

    if file is not None:
        canvas.save(file)

    return canvas


def random_flip(x: Array, key: Array, axis: int = -2) -> Array:
    return jnp.where(
        jax.random.bernoulli(key),
        x,
        jnp.flip(x, axis=axis),
    )


def random_hue(x: Array, key: Array, delta: float = 1e-2) -> Array:
    x = (x + 2) / 4
    x = pix.random_hue(key, x, delta)
    x = x * 4 - 2

    return x


def random_saturation(x: Array, key: Array, lower: float = 0.95, upper: float = 1.05) -> Array:
    x = (x + 2) / 4
    x = pix.random_saturation(key, x, lower, upper)
    x = x * 4 - 2

    return x


def random_shake(x: Array, key: Array, delta: int = 1, mode: str = 'reflect') -> Array:
    i = jax.random.randint(key, shape=(3,), minval=0, maxval=2 * delta + 1)
    i = i.at[-1].set(0)

    return jax.lax.dynamic_slice(
        jnp.pad(
            x,
            pad_width=((delta, delta), (delta, delta), (0, 0)),
            mode=mode,
        ),
        start_indices=i,
        slice_sizes=x.shape,
    )


def psnr(a: Array, b: Array) -> Array:
    return pix.psnr((a + 2) / 4, (b + 2) / 4)


def ssim(a: Array, b: Array) -> Array:
    return pix.ssim((a + 2) / 4, (b + 2) / 4)
