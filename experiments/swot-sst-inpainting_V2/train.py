#!/usr/bin/env python

# Core Libraries
import inox                  # type: ignore # Custom library (likely for modeling and random utilities)
import inox.nn as nn         # type: ignore # type: ignore # Neural network components
from inox import random as inox_random # type: ignore
import jax                   # type: ignore # JAX for high-performance computing
import numpy as np # type: ignore
import optax                 # type: ignore # Optimizers for JAX
import wandb                 # Weights and Biases for experiment tracking
from datasets import Dataset, Features, Array3D
import jax.numpy as jnp # type: ignore # JAX's numpy for array operations

# Workflow management
from dawgz import job, schedule # type: ignore

from priors.diffusion import VESDE, DenoiserLoss, GaussianDenoiser
from priors.data import prefetch
from priors.image import random_flip, random_hue, random_saturation
from priors.common import dump_module, ppca, fit_moments, load_module
from priors.optim import Adam, EMA

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
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import jax.numpy as jnp # type: ignore
import numpy as np # type: ignore
import zarr # type: ignore
import glob
from pathlib import Path
import time

def load_npz_as_hf_dataset(npz_dir):
    npz_files = sorted(glob.glob(f"{npz_dir}/sample_*.npz"))
    data_list = []
    for fname in npz_files:
        arrays = np.load(fname)
        # Convert arrays to regular numpy arrays if needed
        data_list.append({k: np.array(v) for k, v in arrays.items()})
    return Dataset.from_list(data_list)

def load_zarr_as_hf_dataset(zarr_path):
    z = zarr.open_group(zarr_path, mode="r")
    keys = list(z.keys())
    #length = z[keys[0]].shape[0]
    data_dict = {k: np.array(z[k][:]) for k in keys}
    return Dataset.from_dict(data_dict)

def load_dataset(split_path, format):
    if format == "npz":
        return load_npz_as_hf_dataset(split_path)
    elif format == "zarr":
        return load_zarr_as_hf_dataset(split_path)
    else:
        raise ValueError(f"Unsupported format: {format}")

def generate(model, dataset, rng, batch_size, **kwargs):
    def transform(batch):
        y, A = jnp.array(batch['y']), jnp.array(batch['A'])
        x = sample(model, y, A, rng.split(), **kwargs)
        x = jnp.array(x)
        return {'x': x}
    types = {'x': Array3D(shape=(128, 128, 2), dtype='float32')}
    return dataset.map(
        transform,
        features=Features(types),
        remove_columns=['y', 'A'],
        keep_in_memory=True,
        batched=True,
        batch_size=batch_size,
        drop_last_batch=True,
    )
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def train(runid: int, lap: int, src: str):
    """
    Main training loop for a single training 'lap' (iteration).
    Each lap can be seen as one cycle of training, optionally starting from a prior checkpoint.
    """
    # Initialize Weights & Biases
    start_time = time.time()
    run = wandb.init( # type: ignore
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
    t0 = time.time()
    trainset_yA = load_dataset(f"{src}/train", format="zarr")
    #valset = load_dataset(src / "val", format="zarr")
    testset_yA = load_dataset(f"{src}/test", format="zarr")
    print(f"trainset_yA keys: {trainset_yA.column_names}")
    jax.debug.print(f"[{time.strftime('%X')}] Loaded dataset in {time.time() - t0:.2f} seconds")
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Validation data (fixed samples)
    y_eval, A_eval = jnp.array(testset_yA[:16]['y']), jnp.array(testset_yA[:16]['A'])
    y_eval, A_eval = jax.device_put((y_eval, A_eval), distributed)
    B, H, W, C = y_eval.shape
    D = H * W * C

    # If lap >0, load previous checkpoint, else fit prior Gaussian model
    t1 = time.time()
    if lap > 0:
        previous = load_module(runpath / f'checkpoint_{lap - 1}.pkl')
        jax.debug.print(f"[{time.strftime('%X')}] Loaded previous checkpoint in {time.time() - t1:.2f} seconds")
    else:
        # Shuffle the training dataset for moment fitting
        shuffle_seed = hash((runid, "moment_fitting")) % 2**16
        shuffled_trainset = trainset_yA.shuffle(shuffle_seed)  # Add shuffle method
        # Now take first N samples (which are actually shuffled)
        y_fit, A_fit = jnp.array(shuffled_trainset[:6144]['y']), jnp.array(shuffled_trainset[:6144]['A'])
        y_fit, A_fit = jax.device_put((y_fit, A_fit), distributed)
        B, H, W, C = y_fit.shape
        D = H * W * C
        t1a = time.time()
        with inox_random.set_rng(init=inox_random.PRNG(rng.split())):
            mu_x, cov_x = fit_moments(
                features=D, # The dimensionality of the latent variable x
                rank=320, # This is the low-rank dimension of your approximate posterior or prior covariance matrix
                shard=True,
                A=inox.tree.Partial(measure, A_fit, H=H, W=W, C=C),
                y=flatten(y_fit),
                cov_y=1e-3**2,
                sampler='ddim',
                sde=sde,
                steps=256,
                maxiter=None,
                key=rng.split(),
            )
        jax.debug.print(f"[{time.strftime('%X')}] fit_moments completed in {time.time() - t1a:.2f} seconds")
        del y_fit, A_fit
        previous = GaussianDenoiser(mu_x, cov_x)
        jax.debug.print(f"[{time.strftime('%X')}] GaussianDenoiser created in {time.time() - t1:.2f} seconds")

    # Prepare the previous model for sampling new training targets
    t2 = time.time()
    static, arrays = previous.partition()
    arrays = jax.device_put(arrays, replicated)
    previous = static(arrays)
    print(f"[{time.strftime('%X')}] Model partitioned and moved to device in {time.time() - t2:.2f} seconds")

    # Generate synthetic training and testing data (denoised reconstructions)
    t3 = time.time()
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
    print(f"[{time.strftime('%X')}] Generated trainset in {time.time() - t3:.2f} seconds")
    t3b = time.time()
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
    print(f"[{time.strftime('%X')}] Generated testset in {time.time() - t3b:.2f} seconds")

    # Fit low-rank covariance (PPCA) on generated training data
    t4 = time.time()
    x_fit = jnp.array(trainset[:16384]['x'])
    x_fit = flatten(x_fit)
    mu_x, cov_x = ppca(x_fit, rank=320, key=rng.split())
    del x_fit
    print(f"[{time.strftime('%X')}] PPCA fit in {time.time() - t4:.2f} seconds")

    # Initialize model
    t5 = time.time()
    if lap > 0:
        model = previous
    else:
        with inox_random.set_rng(init=inox_random.PRNG(rng.split()), dropout=inox_random.PRNG(rng.split())):
            model = make_model(key=rng.split(), in_channels=C, out_channels=C, **CONFIG) 
    print(f"[{time.strftime('%X')}] Model initialized in {time.time() - t5:.2f} seconds")

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

    print(f"[{time.strftime('%X')}] Setup complete, entering training loop. Total setup time: {time.time() - start_time:.2f} seconds")

    # Training loop over epochs
    for epoch in (bar := trange(config.epochs, ncols=88)):
        epoch_start = time.time()
        # Shuffle training set per epoch
        loader = trainset.shuffle(seed=seed + lap * config.epochs + epoch).iter(
            batch_size=config.batch_size, drop_last_batch=True
        )

        losses = []
        for batch in prefetch(loader):
            x = batch['x']
            assert batch['x'] is not None, "Batch x is None!"
            jax.debug.print("Batch x shape:", batch['x'].shape)
            x = jax.device_put(x, distributed)
            #x = augment(x, rng.split(len(x)))
            x = flatten(x)
            with inox_random.set_rng(init=inox_random.PRNG(rng.split()), dropout=inox_random.PRNG(rng.split())):
                loss, avrg, params, opt_state = sgd_step(avrg, params, others, opt_state, x, key=rng.split())
            losses.append(loss)
        loss_train = np.stack(losses).mean()

        # Validation evaluation
        val_start = time.time()
        loader = testset.iter(batch_size=config.batch_size, drop_last_batch=True)
        losses = []
        for batch in prefetch(loader):
            x = batch['x']
            x = jax.device_put(x, distributed)
            x = flatten(x)
            loss = ell(avrg, others, x, key=rng.split())
            losses.append(loss)
        loss_val = np.stack(losses).mean()
        val_time = time.time() - val_start
        bar.set_postfix(loss=loss_train, loss_val=loss_val)

        # Every 16 epochs, sample validation images and log to wandb
        if (epoch + 1) % 16 == 0:
            sample_start = time.time()
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
            num = x.shape[0]
            cols = int(np.sqrt(num))
            rows = num // cols
            x = x.reshape(rows, cols, H, W, C)
            run.log({
                'loss': loss_train,
                'loss_val': loss_val,
                'samples': wandb.Image(to_pil(x, zoom=4)),
                'epoch_time': time.time() - epoch_start,
                'val_time': val_time,
                'sample_time': time.time() - sample_start,
            })
            jax.debug.print(f"[{time.strftime('%X')}] Epoch {epoch+1}: train_loss={loss_train:.4f}, val_loss={loss_val:.4f}, epoch_time={time.time() - epoch_start:.2f}s, val_time={val_time:.2f}s, sample_time={time.time() - sample_start:.2f}s")
        else:
            run.log({
                'loss': loss_train,
                'loss_val': loss_val,
                'epoch_time': time.time() - epoch_start,
                'val_time': val_time,
            })
            jax.debug.print(f"[{time.strftime('%X')}] Epoch {epoch+1}: train_loss={loss_train:.4f}, val_loss={loss_val:.4f}, epoch_time={time.time() - epoch_start:.2f}s, val_time={val_time:.2f}s")

    # Save checkpoint
    t_save = time.time()
    model = static(avrg, others)
    model.train(False)
    dump_module(model, runpath / f'checkpoint_{lap}.pkl')
    print(f"[{time.strftime('%X')}] Saved checkpoint in {time.time() - t_save:.2f} seconds")


if __name__ == '__main__':
    wandb.login() # type: ignore
    runid = wandb.util.generate_id() # type: ignore
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
                ram='128GB',
                time='1-00:00:00',
                partition='h200',
                wrap='\"hostname && sleep infinity\"'
           )
        )
        if len(jobs) > 1:
            jobs[-1].after(jobs[-2], status="any")
    
    dry_run = True
    if dry_run:
        jobs = jobs[:1]
    schedule(
        *jobs,
        name=f'Training {runid}',
        backend='slurm',
        debug=True,
        export='ALL',
        env=['export WANDB_SILENT=true'],
        dry_run=False,
        singularity=(
                """singularity exec --nv \
                --bind /opt/slurm:/opt/slurm \
                --bind /var/run/munge:/var/run/munge \
                --overlay /scratch/tm3076/singularity_container/EDIT_JAX-cuDNN9.8-overlay-15GB-500K.ext3:ro \
                /share/apps/images/cuda12.8.1-cudnn9.8.0-ubuntu24.04.2.sif \
                /bin/bash -c 'export PATH="/opt/slurm/bin:$PATH" && source /ext3/env.sh &&  {python_command}'"""
            ) 
        )
