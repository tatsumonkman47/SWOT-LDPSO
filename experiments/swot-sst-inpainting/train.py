#!/usr/bin/env python

# Core Libraries
import inox                  # Custom library (likely for modeling and random utilities)
import inox.nn as nn         # Neural network components
import jax                   # JAX for high-performance computing
import numpy as np
import optax                 # Optimizers for JAX
import wandb                 # Weights and Biases for experiment tracking

# Data handling
from datasets import Array3D, Features #load_from_disk
import sys
sys.path.append('./')
import JAXdata_loaders_seasonal

# Workflow management
from dawgz import job, schedule

# Utils
from functools import partial
from tqdm import trange
from typing import *
from utils import *          # Assumed utility functions (augmentations, flatten, sampling, etc.)

# Configuration dictionary defining hyperparameters and architecture
CONFIG = {
    # Data corruption level (percentage of masked pixels)
    #'corruption': 75,
    # Model architecture
    'hid_channels': (128, 256, 384),
    'hid_blocks': (5, 5, 5),
    'kernel_size': (3, 3),
    'emb_features': 256,
    'heads': {1: 4},
    'dropout': 0.1,
    # Diffusion sampling
    'sampler': 'ddpm',
    'sde': {'a': 1e-3, 'b': 1e2},
    'heuristic': None,
    'discrete': 256,
    'maxiter': 1,
    # Training settings
    'epochs': 256,
    'batch_size': 256,
    'scheduler': 'constant',
    'lr_init': 2e-4,
    'lr_end': 1e-6,
    'lr_warmup': 0.0,
    'optimizer': 'adam',
    'weight_decay': None,
    'clip': 1.0,
    'ema_decay': 0.9999,
}
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
standards = {
            "mean_ssh": 0.0, "std_ssh": 0.0453692672359483,
            "mean_sst": 15.956900367755182, "std_sst": 5.987649544923141
            }
DATA_CONFIG = {
    "data_dir": "/home/tm3076/scratch/pytorch_learning_tiles"  ,
    "N_t": 1,
    "infields": ["zarr_llc4320_SST_tiles_4km"],
    "in_mask_list": ["cloud_rho"],
    "in_transform_list": ["std_global_mean_sst_norm"],
    "standards":standards,
    "flatten": False,
    "return_meta_data": False,
    "cloud_rho": 0.5,
}

PATCH_COORDS = f"{DATA_CONFIG['data_dir']}/zarred_UVSST_x_y_coordinates_noland_nonan.npy",
T_RANGE = range(5, 360, 5)
SPLIT_FRACTIONS = {"train": 0.75, "val":0.15, "test":0.1}
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def generate_custom(model, dataset, rng, batch_size, **kwargs):
    """
    Generate denoised samples for the entire dataset using the provided model.
    Returns a new Hugging_face_wrapper holding only {'x'}.
    """
    x_list = []
    loader = dataset.iter(batch_size=batch_size, drop_last_batch=True)
    for batch in loader:
        y, A = batch['y'], batch['A']
        x = sample(model, y, A, rng.split(), **kwargs)
        x = np.asarray(x)
        x_list.append(x)
    # Concatenate all outputs
    x_full = np.concatenate(x_list, axis=0)
    # Create a dataset-like object that yields {'x'} entries
    class GeneratedDataset:
        def __init__(self, x_data):
            self.x_data = x_data
        def __len__(self):
            return len(self.x_data)
        def __getitem__(self, idx):
            if isinstance(idx, slice) or isinstance(idx, np.ndarray) or isinstance(idx, list):
                idx = np.arange(len(self))[idx] if isinstance(idx, slice) else idx
                x = np.stack([self[i]['x'] for i in idx])
                return {'x': x}
            return {'x': self.x_data[idx]}
    return JAXdata_loaders_seasonal.Hugging_face_wrapper(GeneratedDataset(x_full))

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
                self.splits[split] = {
                    "type": "zarr",
                    "zarr": z,
                    "length": len(next(iter(z.values())))
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
        self.length = len(next(iter(zarr_group.values())))
    def __len__(self):
        return self.length
    def __getitem__(self, idx):
        return {k: jnp.array(self.zarr[k][idx]) for k in self.zarr}
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

""" Old method
def generate(model, dataset, rng, batch_size, **kwargs):
    ```
    Generate denoised samples for the entire dataset using the provided model.
    This function applies the diffusion sampler to reconstruct full images.
    ```
    def transform(batch):
        y, A = batch['y'], batch['A']
        x = sample(model, y, A, rng.split(), **kwargs)
        x = np.asarray(x)
        return {'x': x}
    
    # Define output data structure
    types = {'x': Array3D(shape=(32, 32, 3), dtype='float32')}

    return dataset.map(
        transform,
        features=Features(types),
        remove_columns=['y', 'A'],
        keep_in_memory=True,
        batched=True,
        batch_size=batch_size,
        drop_last_batch=True,
    )
"""

def train(runid: int, lap: int, src: str):
    """
    Main training loop for a single training 'lap' (iteration).
    Each lap can be seen as one cycle of training, optionally starting from a prior checkpoint.
    """
    # Initialize Weights & Biases
    run = wandb.init(
        project='priors-SST-mask',
        id=runid,
        resume='allow',
        dir=PATH,
        config=CONFIG,
    )
    runpath = PATH / f'runs/{run.name}_{run.id}'
    runpath.mkdir(parents=True, exist_ok=True)
    config = run.config

    # Enable partitioning for reproducible RNG across shards
    jax.config.update('jax_threefry_partitionable', True)
    mesh = jax.sharding.Mesh(jax.devices(), 'i')
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    distributed = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('i'))

    # Initialize PRNG with lap-specific seed
    seed = hash((runpath, lap)) % 2**16
    rng = inox.random.PRNG(seed)

    # Create the SDE object (Variance Exploding SDE)
    sde = VESDE(**CONFIG.get('sde'))
    
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Load HuggingFace-formatted LLC4320 dataset
    # Load custom
    """
    dataset = JAXdata_loaders_seasonal.JAXLLC4320_HFformated_dataset(
            patch_coords=f"{config['data_dir']}/zarred_UVSST_x_y_coordinates_noland_nonan.npy",
            t_range=range(5, 360, 5),
            split_fractions={"train": 0.75, "val":0.15, "test":0.1},
            config=DATA_CONFIG,  
            )
    """
    dataset = PrecomputedJAXDataset(src,format="zarr")
    
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    trainset_yA = dataset['train']
    testset_yA = dataset['test']

    # Validation data (fixed samples)
    y_eval, A_eval = testset_yA[:16]['y'], testset_yA[:16]['A']
    y_eval, A_eval = jax.device_put((y_eval, A_eval), distributed)

    # If lap >0, load previous checkpoint, else fit prior Gaussian model
    if lap > 0:
        previous = load_module(runpath / f'checkpoint_{lap - 1}.pkl')
    else:
        # Fit Gaussian prior from first 16k training samples
        y_fit, A_fit = trainset_yA[:16384]['y'], trainset_yA[:16384]['A']
        y_fit, A_fit = jax.device_put((y_fit, A_fit), distributed)
        mu_x, cov_x = fit_moments(
            features=128 * 128 * 1, # The dimensionality of the latent variable x
            rank=320, # This is the low-rank dimension of your approximate posterior or prior covariance matrix
            shard=True,
            A=inox.Partial(measure, A_fit),
            y=flatten(y_fit),
            cov_y=1e-3**2,
            sampler='ddim',
            sde=sde,
            steps=256,
            maxiter=None,
            key=rng.split(),
        )
        del y_fit, A_fit
        previous = GaussianDenoiser(mu_x, cov_x)

    # Prepare the previous model for sampling new training targets
    static, arrays = previous.partition()
    arrays = jax.device_put(arrays, replicated)
    previous = static(arrays)

    # Generate synthetic training and testing data (denoised reconstructions)
    trainset = generate(
        model=previous,
        dataset=trainset_yA,
        rng=rng,
        batch_size=config.batch_size,
        shard=True,
        sampler=config.sampler,
        sde=sde,
        steps=config.discrete,
        maxiter=config.maxiter,
    )
    testset = generate(
        model=previous,
        dataset=testset_yA,
        rng=rng,
        batch_size=config.batch_size,
        shard=True,
        sampler=config.sampler,
        sde=sde,
        steps=config.discrete,
        maxiter=config.maxiter,
    )

    # Fit low-rank covariance (PPCA) on generated training data
    x_fit = trainset[:16384]['x']
    x_fit = flatten(x_fit)
    mu_x, cov_x = ppca(x_fit, rank=320, key=rng.split())
    del x_fit

    # Initialize model
    if lap > 0:
        model = previous
    else:
        model = make_model(key=rng.split(), **CONFIG)

    # Set model's prior mean
    model.mu_x = mu_x

    # Configure model's covariance heuristic
    if config.heuristic == 'zeros':
        model.cov_x = jnp.zeros_like(mu_x)
    elif config.heuristic == 'ones':
        model.cov_x = jnp.ones_like(mu_x)
    elif config.heuristic == 'cov_t':
        model.cov_x = jnp.ones_like(mu_x) * 1e6
    elif config.heuristic == 'cov_x':
        model.cov_x = cov_x

    model.train(True)

    # Partition model parameters
    static, params, others = model.partition(nn.Parameter)

    # Define denoising loss
    objective = DenoiserLoss(sde=sde)

    # Build optimizer
    steps = config.epochs * len(trainset_yA) // config.batch_size
    optimizer = Adam(steps=steps, **config)
    opt_state = optimizer.init(params)

    # Exponential moving average for parameter stabilization
    ema = EMA(decay=config.ema_decay)
    avrg = params

    # Put everything onto devices
    avrg, params, others, opt_state = jax.device_put((avrg, params, others, opt_state), replicated)

    # Data augmentation function (random flips, hue, saturation)
    @jax.jit
    @jax.vmap
    def augment(x, key):
        keys = jax.random.split(key, 3)
        x = random_flip(x, keys[0], axis=-2)
        x = random_hue(x, keys[1], delta=1e-2)
        x = random_saturation(x, keys[2], lower=0.95, upper=1.05)
        return x

    # Loss computation
    @jax.jit
    def ell(params, others, x, key):
        keys = jax.random.split(key, 3)
        z = jax.random.normal(keys[0], shape=x.shape)
        t = jax.random.beta(keys[1], a=3, b=3, shape=x.shape[:1])
        return objective(static(params, others), x, z, t, key=keys[2])

    # Single SGD update step
    @jax.jit
    def sgd_step(avrg, params, others, opt_state, x, key):
        loss, grads = jax.value_and_grad(ell)(params, others, x, key)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        avrg = ema(avrg, params)
        return loss, avrg, params, opt_state

    # Training loop over epochs
    for epoch in (bar := trange(config.epochs, ncols=88)):
        # Shuffle training set per epoch
        loader = trainset.shuffle(seed=seed + lap * config.epochs + epoch).iter(
            batch_size=config.batch_size, drop_last_batch=True
        )

        losses = []
        for batch in prefetch(loader):
            x = batch['x']
            x = jax.device_put(x, distributed)
            #x = augment(x, rng.split(len(x)))
            x = flatten(x)
            loss, avrg, params, opt_state = sgd_step(avrg, params, others, opt_state, x, key=rng.split())
            losses.append(loss)
        loss_train = np.stack(losses).mean()

        # Validation evaluation
        loader = testset.iter(batch_size=config.batch_size, drop_last_batch=True)
        losses = []
        for batch in prefetch(loader):
            x = batch['x']
            x = jax.device_put(x, distributed)
            x = flatten(x)
            loss = ell(avrg, others, x, key=rng.split())
            losses.append(loss)
        loss_val = np.stack(losses).mean()
        bar.set_postfix(loss=loss_train, loss_val=loss_val)

        # Every 16 epochs, sample validation images and log to wandb
        if (epoch + 1) % 16 == 0:
            model = static(avrg, others)
            model.train(False)
            x = sample(
                model=model,
                y=y_eval,
                A=A_eval,
                key=rng.split(),
                shard=True,
                sampler=config.sampler,
                steps=config.discrete,
                maxiter=config.maxiter,
            )
            x = x.reshape(4, 4, 128, 128, 1)
            run.log({
                'loss': loss_train,
                'loss_val': loss_val,
                'samples': wandb.Image(to_pil(x, zoom=4)),
            })
        else:
            run.log({
                'loss': loss_train,
                'loss_val': loss_val,
            })

    # Save checkpoint
    model = static(avrg, others)
    model.train(False)
    dump_module(model, runpath / f'checkpoint_{lap}.pkl')


if __name__ == '__main__':
    wandb.login()
    runid = wandb.util.generate_id()
    jobs = []
    src = "/home/tm3076/scratch/priors_precomputed_datasets/precomputed_data_sst/sst_crho_0.4"

    # Schedule multiple laps as Slurm jobs
    for lap in range(32):
        jobs.append(
            job(
                partial(train, runid=runid, lap=lap, src=src),
                name=f'train_{lap}',
                cpus=4,
                gpus=4,
                ram='128',
                time='1-00:00:00',
           )
        )
        if len(jobs) > 1:
            jobs[-1].after(jobs[-2])

    schedule(
        *jobs,
        name=f'Training {runid}',
        backend='slurm',
        export='ALL',
        env=['export WANDB_SILENT=true'],
        dry_run=False,
        singularity=(
                """singularity exec --nv  
                --overlay /scratch/tm3076/test_python_env/my_conda.ext3:ro 
                /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif 
                /bin/bash -c 'source /ext3/env.sh; conda activate priors; python'"""
            )
        )
     
