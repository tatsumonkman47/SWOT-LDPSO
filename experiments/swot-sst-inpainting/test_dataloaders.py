from jax.scipy.sparse.linalg import cg
# Core Libraries
import inox                  # Custom library (likely for modeling and random utilities)
import inox.nn as nn         # Neural network components
import jax                   # JAX for high-performance computing
import numpy as np
import optax                 # Optimizers for JAX
import wandb                 # Weights and Biases for experiment tracking
import zarr

# Data handling
from datasets import Array3D, Features #load_from_disk

# Workflow management
from dawgz import job, schedule

from functools import partial
from tqdm import trange
from typing import *
from utils import *          # Assumed utility functions (augmentations, flatten, sampling, etc.)


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import jax.numpy as jnp
import numpy as np
import zarr
from glob import glob
from pathlib import Path

class PrecomputedJAXDataset:
    def __init__(self, source, format="zarr"):
        self.format = format
        self.source = Path(source)
        self.splits = {}
        for split in ["train", "val", "test"]:
            split_path = self.source / split
            print("splict_path",split_path)
            if not split_path.exists():
                continue
            if format == "npz":
                paths = sorted(glob(str(split_path / "sample_*.npz")))
                self.splits[split] = {
                    "type": "npz",
                    "paths": paths,
                    "length": len(paths)
                }
            elif format == "zarr":
                z = zarr.open_group(str(split_path), mode="r")
                print("z", z)

                # Compatibility with zarr v2 and v3
                keys = list(z.keys())
                if not keys:
                    raise ValueError(f"No datasets found in {split_path}")
                first_array = z[keys[0]]
                length = first_array.shape[0]

                self.splits[split] = {
                    "type": "zarr",
                    "zarr": z,
                    "length": length
                }
            else:
                raise ValueError(f"Unsupported format: {format}")
    def __getitem__(self, split):
        if split not in self.splits:
            raise KeyError(f"Split '{split}' not found. Available: {list(self.splits.keys())}")
        return self._get_split_dataset(split)
    def _get_split_dataset(self, split):
        info = self.splits[split]
        if info["type"] == "npz":
            return _NPZSubDataset(info["paths"])
        elif info["type"] == "zarr":
            return _ZarrSubDataset(info["zarr"])
        else:
            raise ValueError("Unknown dataset type")

class _NPZSubDataset:
    def __init__(self, paths):
        self.paths = paths
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        arrays = np.load(self.paths[idx])
        return {k: jnp.array(v) for k, v in arrays.items()}

class _ZarrSubDataset:
    def __init__(self, zarr_group):
        self.zarr = zarr_group
        # Zarr v2+v3 compatibility: Get one dataset to infer length
        keys = list(zarr_group.keys())
        if not keys:
            raise ValueError("Zarr group contains no arrays")
        self.keys = keys  # Save keys for __getitem__
        self.length = zarr_group[keys[0]].shape[0]
    def __len__(self):
        return self.length
    def __getitem__(self, idx):
        return {k: jnp.array(self.zarr[k][idx]) for k in self.keys}

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#src = "/home/tm3076/vast/priors_precomputed_datasets/precomputed_data_sst/sst_crho_0.3"
src = "/home/tm3076/scratch/priors_precomputed_datasets/precomputed_data_sst/sst_crho_0.4"
dataset = PrecomputedJAXDataset(src,format="zarr")
y_test = dataset["train"][0]["y"]
A_test = dataset["train"][0]["A"]
print(A_test)

def run_cg_on_sample(split="train", idx=0):
    # Load A and y from your Zarr/NPZ dataset
    dataset = PrecomputedJAXDataset(src,format="zarr")
    sample = dataset[split][idx]

    A = sample["A"]
    y = sample["y"]

    print("Loaded shapes:", A.shape, y.shape)
    print("Loaded dtypes:", A.dtype, y.dtype)

    # Ensure float32 everywhere
    A = A#.astype(jnp.float32)
    y = y#.astype(jnp.float32)
    b = jnp.ravel(A * y)

    def apply_A(x):
        x = x.reshape(y.shape)
        return jnp.ravel(A * x)

    x0 = jnp.zeros_like(b)
    sol, info = cg(apply_A, b, x0=x0)

    print("CG info:", info)
    print("Result shape:", sol.shape)

run_cg_on_sample()



from jax.scipy.sparse.linalg import cg
from functools import partial
import jax

# We'll test with a full batch (e.g. 16384 samples)
split = "train"
batch_size = 16384

# Load A and y as a batch
dataset = PrecomputedJAXDataset(src, format="zarr")
trainset = dataset[split]

y_batch = dataset["train"][:batch_size]["y"]
A_batch = dataset["train"][:batch_size]["A"]

print("A_batch.shape:", A_batch.shape)  # (16384, H, W, C)
print("y_batch.shape:", y_batch.shape)  # (16384, H, W, C)

# Get shape info
B, H, W, C = y_batch.shape
D = H * W * C

# Flatten y
y_flat = y_batch.reshape(B, -1)

# Define flatten/unflatten like in your utils.py
def flatten(x): return x.reshape((x.shape[0], -1))
def unflatten(x, H, W, C): return x.reshape((x.shape[0], H, W, C))

# Define the problematic measure
def measure(x,A, H, W, C):
    return flatten(A * unflatten(x, H, W, C))

# Now create the jitted partial with full A_batch baked in
measure_partial = inox.tree.Partial(measure, A_batch, H=H, W=W, C=C)

# Try calling cg with this (this should trigger the overflow or JAX crash)
print("Launching CG with batched partial...")
x0 = jnp.zeros_like(y_flat)


from jax import jit
from jax.scipy.sparse.linalg import cg

# Move A_batch outside the closure
def apply_A_single(x, A_sample, B_b, H, W, C):
    return jnp.ravel(A_sample * x.reshape(B_b,H, W, C))

def run_single_sample_cg(y, A, B_b, H, W, C):
    y_flat = jnp.ravel(y) 
    A_op = lambda x: apply_A_single(x, A, B_b, H, W, C)
    x0 = jnp.zeros_like(y_flat)
    sol, info = cg(A_op, y_flat, x0=x0)
    return sol, info

# Run this in a loop for a small batch
for i in range(0,16384,1000):  # or however many
    i_end = min(i+1000,16384)
    B_b = i_end - i
    print(f"Running CG for sample {i}")
    sol, info = run_single_sample_cg(y_batch[i:i_end], A_batch[i:i_end], B_b, H, W, C)
    print(f"Sample {i} CG info: {info}")

