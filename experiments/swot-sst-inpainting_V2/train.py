#!/usr/bin/env python
# Core Libraries
import inox                  # type: ignore # Custom library (likely for modeling and random utilities)
import inox.nn as nn         # type: ignore # type: ignore # Neural network components
from inox import random as inox_random # type: ignore
import jax                   # type: ignore # JAX for high-performance computing
import numpy as np # type: ignore
import optax                 # type: ignore # Optimizers for JAX
import wandb                 # Weights and Biases for experiment tracking
import jax.numpy as jnp # type: ignore # JAX's numpy for array operations
import sys
import os
import pickle

# Workflow management
from dawgz import job, schedule # type: ignore
from priors.diffusion import VESDE, DenoiserLoss, GaussianDenoiser
from priors.data import prefetch
from priors.image import random_flip, random_hue, random_saturation, to_pil, flatten
from priors.common import dump_module, ppca, fit_moments, load_module
from priors.optim import Adam, EMA

from functools import partial
from tqdm import trange
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from utils import make_model, sample, measure, PATH          # Assumed utility functions (augmentations, flatten, sampling, etc.)
import zarr # type: ignore
import time

# Configuration dictionary defining hyperparameters and architecture

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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
    'checkpoint_layers': (),
    # Fit moments for prior Gaussian model
    'cov_y': 1e-4**2, # From 1e-3**2, Expected observation noise covariance, should match the actual noise level in the data
    # Diffusion sampling
    'sampler': 'ddpm',
    'sde': {'a': 1e-4, 'b': 1e2}, # Variance Exploding SDE parameters. 'a' is the noise level, 'b' is the diffusion coefficient.
    'heuristic': None,
    'discrete': 256,
    'maxiter': 10,
    # Generation settings
    'generation_batch_size': 128,
    # Training settings
    'epochs': 2,
    'batch_size': 304,
    'scheduler': 'constant',
    'lr_init': 2e-4,
    'lr_end': 1e-6,
    'lr_warmup': 0.0,
    'optimizer': 'adam',
    'weight_decay': None,
    'clip': 1.0,
    'ema_decay': 0.9999,
}

def zarr_batch_iterator(array, batch_size, indices=None, drop_last_batch=True):
    N = array.shape[0]
    if indices is None:
        indices = np.arange(N)
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        if drop_last_batch and (end - start) < batch_size:
            break
        yield array[indices[start:end]]

def zarr_generate(model, dataset, rng, batch_size, shape, num_gpus, **kwargs):
    """Generate outputs for a dataset (Zarr or dict of arrays) in batches."""
    # Force eval mode during generation
    original_training = getattr(model, 'training', True)
    # Set eval mode with proper RNG context
    with inox_random.set_rng(
        init=inox_random.PRNG(rng.split()),
        dropout=inox_random.PRNG(rng.split()),
    ):
        model.train(False)
    N = dataset['y'].shape[0]
    xs = []
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        current_batch_size = end - start
        y_batch = dataset['y'][start:end]
        A_batch = dataset['A'][start:end]
        # Pad to make divisible by num_gpus if needed
        if current_batch_size % num_gpus != 0:
            pad_size = num_gpus - (current_batch_size % num_gpus)
            y_pad = np.repeat(y_batch[-1:], pad_size, axis=0)
            A_pad = np.repeat(A_batch[-1:], pad_size, axis=0)
            y_batch = np.concatenate([y_batch, y_pad], axis=0)
            A_batch = np.concatenate([A_batch, A_pad], axis=0)
        # Fresh RNG context for each batch
        batch_rng = rng.split()
        with inox_random.set_rng(
            init=inox_random.PRNG(batch_rng),
            dropout=inox_random.PRNG(jax.random.split(batch_rng)[1]),
        ):
            x_batch = sample(model, y_batch, A_batch, jax.random.split(batch_rng)[0], **kwargs)
        # Remove padding from output
        if current_batch_size % num_gpus != 0:
            x_batch = x_batch[:current_batch_size]
        xs.append(np.asarray(x_batch))
    xs = np.concatenate(xs, axis=0)
    # Restore original training mode with RNG context
    with inox_random.set_rng(
        init=inox_random.PRNG(rng.split()),
        dropout=inox_random.PRNG(rng.split()),
    ):
        model.train(original_training)
    return {'x': xs}

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def train(runid: int, lap: int, src: str):
    """
    Main training loop for a single training 'lap' (iteration).
    Each lap can be seen as one cycle of training, optionally starting from a prior checkpoint.
    """
    # Force early logging before ANY JAX operations
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

    print(f"TRAIN DEBUG: Starting lap {lap}, PID={os.getpid()}", flush=True)
    sys.stdout.flush()
    # Set JAX compilation flags to be more verbose and lazy
    os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=1'
    os.environ['JAX_TRACEBACK_FILTERING'] = 'off'
    jax.config.update('jax_compilation_cache_dir', '/tmp')
    jax.config.update('jax_persistent_cache_min_entry_size_bytes', -1)
    print(f"TRAIN DEBUG: JAX config set", flush=True)
    print(f"Starting Lap {lap} with runid {runid}")
    print(f"Source directory: {src}")

    # Enable partitioning for reproducible RNG across shards
    jax.config.update('jax_threefry_partitionable', True)
    jax.config.update('jax_enable_x64', False)  # Use float32 everywhere
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
    trainset_yA = zarr.open_group(f"{src}/train", mode="r")
    testset_yA = zarr.open_group(f"{src}/train", mode="r")
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Validation data (fixed samples)
    y_eval, A_eval = testset_yA['y'][:16], testset_yA['A'][:16]
    y_eval, A_eval = jax.device_put((y_eval, A_eval), distributed)
    B, H, W, C = y_eval.shape
    D = H * W * C
    jax.debug.print(f"[{time.strftime('%X')}] Loaded dataset in {time.time() - t0:.2f} seconds")

    # If lap >0, load previous checkpoint, else fit prior Gaussian model
    t1 = time.time()
    
    if lap > 0:
        checkpoint_path = runpath / f'checkpoint_{lap - 1}.pkl'
        # Load the model with proper RNG context
        with inox_random.set_rng(
            init=inox_random.PRNG(rng.split()),
            dropout=inox_random.PRNG(rng.split()),
        ):
            previous = load_module(checkpoint_path)
            # Ensure the model is in eval mode initially
            previous.train(False)
        jax.debug.print(f"[{time.strftime('%X')}] Model loaded successfully")
    else:
        y_fit, A_fit = trainset_yA['y'][:16384], trainset_yA['A'][:16384]
        y_fit, A_fit = jax.device_put((y_fit, A_fit), distributed)
        jax.debug.print(f"[{time.strftime('%X')}] Loaded fitting dataset in {time.time() - t0:.2f} seconds")
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
                cov_y=CONFIG.get('cov_y', 1e-3**2), # Expected observation noise covariance
                sampler='ddim',
                sde=sde,
                steps=256,
                maxiter=CONFIG.get('maxiter',10), # Increased for robustness
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
    num_gpus = len(jax.devices())
    trainset = zarr_generate(
        model=previous,
        dataset=trainset_yA,
        rng=rng,
        batch_size=config.generation_batch_size,
        shape=(H, W, C),
        num_gpus=num_gpus,
        shard=True,
        sampler=config.sampler,
        sde=sde,
        steps=config.discrete,
        maxiter=config.maxiter,
    )
    print(f"[{time.strftime('%X')}] Generated trainset in {time.time() - t3:.2f} seconds")
    t3b = time.time()
    testset = zarr_generate(
        model=previous,
        dataset=testset_yA,
        rng=rng,
        batch_size=config.generation_batch_size,
        shape = (H, W, C),
        num_gpus=num_gpus,
        shard=True,
        sampler=config.sampler,
        sde=sde,
        steps=config.discrete,
        maxiter=config.maxiter,
    )
    jax.debug.print(f"[{time.strftime('%X')}] Generated testset in {time.time() - t3b:.2f} seconds")

    # Fit low-rank covariance (PPCA) on generated training data
    t4 = time.time()
    x_fit = trainset['x'][:16384]
    x_fit = flatten(x_fit)
    mu_x, cov_x = ppca(x_fit, rank=320, key=rng.split())
    del x_fit
    print(f"[{time.strftime('%X')}] PPCA fit in {time.time() - t4:.2f} seconds")

    # Initialize model
    t5 = time.time()

    if lap > 0:
        model = previous
    else:
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
    
    # Check memory usage
    for i, device in enumerate(jax.devices()):
        memory_info = device.memory_stats()
        print(f"GPU {i}: {memory_info['bytes_in_use'] / 1e9:.1f}GB used")
    print(f"[{time.strftime('%X')}] Setup complete, entering training loop. Total setup time: {time.time() - start_time:.2f} seconds")

    # Training loop over epochs
    for epoch in (bar := trange(config.epochs, ncols=88)):
        epoch_start = time.time()
        # Shuffle training set per epoch
        N = trainset['x'].shape[0]  # or whatever your dataset size is
        shuffle_seed = seed + lap * config.epochs + epoch
        indices = np.random.RandomState(shuffle_seed).permutation(N)
        losses = []
        #for batch in prefetch(loader):
        for x_batch in prefetch(zarr_batch_iterator(trainset['x'], config.batch_size, indices=indices, drop_last_batch=True)):
            assert x_batch is not None, "x_batch is None!"
            x_batch = jax.device_put(x_batch, distributed)
            x_batch = flatten(x_batch)
            #with inox_random.set_rng(init=inox_random.PRNG(rng.split()), dropout=inox_random.PRNG(rng.split())):
            loss, avrg, params, opt_state = sgd_step(avrg, params, others, opt_state, x_batch, key=rng.split())
            losses.append(loss)
        loss_train = np.stack(losses).mean()

        # Validation evaluation
        val_start = time.time()
        losses = []
        for x_batch in prefetch(zarr_batch_iterator(testset['x'], config.batch_size, drop_last_batch=True)):
            x_batch = jax.device_put(x_batch, distributed)
            x_batch = flatten(x_batch)
            loss = ell(avrg, others, x_batch, key=rng.split())
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
            model.train(True)  # Restore training mode if continuing to train
            num = x.shape[0]
            cols = int(np.sqrt(num))
            rows = num // cols
            x = x.reshape(rows, cols, H, W, C)
            
            # Fix: Handle the list returned by to_pil
            pil_images = to_pil(x, zoom=4)
            log_dict = {
                'loss': loss_train,
                'loss_val': loss_val,
                'epoch_time': time.time() - epoch_start,
                'val_time': val_time,
                'sample_time': time.time() - sample_start,
            }
            # Handle single image or multiple channels
            if isinstance(pil_images, list):
                # Log each channel separately
                for i, img in enumerate(pil_images):
                    log_dict[f'samples_channel_{i}'] = wandb.Image(img)
            else:
                # Single image case
                log_dict['samples'] = wandb.Image(pil_images)
            run.log(log_dict)
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
    """
    # Save only parameters, not the full model
    static_part, params_part = model.partition()
    checkpoint_data = {
        'params': params_part,
        'others': others,
        'mu_x': model.mu_x,  # Save the prior mean
        'cov_x': getattr(model, 'cov_x', None),  # Save covariance if it exists
        'lap': lap,
        'config': dict(CONFIG),  # Save config for reconstruction
    }
    with open(runpath / f'checkpoint_{lap}.pkl', 'wb') as f:
        pickle.dump(checkpoint_data, f)
    """
    dump_module(model, runpath / f'checkpoint_{lap}.pkl')
    
    print(f"[{time.strftime('%X')}] Saved checkpoint in {time.time() - t_save:.2f} seconds")

if __name__ == '__main__':
    wandb.login() # type: ignore
    runid = wandb.util.generate_id() # type: ignore
    jobs = []
    src = "/home/tm3076/scratch/priors_precomputed_datasets/precomputed_data_sst/sst_crho_0.4"
    #src = "/home/tm3076/scratch/priors_precomputed_datasets/precomputed_data_sshsst/sshsst_swot_nadir_crho_0.3"

    # Schedule multiple laps as Slurm jobs
    for lap in range(0,32):
        jobs.append(
            job(
                partial(train, runid=runid, lap=lap, src=src),
                name=f'train_{lap}',
                cpus=4,
                gpus=4,
                ram='128GB',
                time='1-00:00:00',
                partition='h200',
                #delay="00:05:00",
                #wrap='\"hostname && sleep infinity\"'
           )
        )
        if len(jobs) > 1:
            jobs[-1].after(jobs[-2], status="success")
    
    dry_run = False
    dry_run_scheduler = True
    if dry_run:
        jobs = jobs[:1]
    if dry_run_scheduler:
        jobs = jobs[:2]

    # Add debug prints to see what DAWGZ is actually doing
    print(f"DAWGZ DEBUG: Created {len(jobs)} jobs")
    for i, job in enumerate(jobs):
        print(f"DAWGZ DEBUG: Job {i}: {job}")
        if hasattr(job, 'dependencies'):
            print(f"DAWGZ DEBUG: Job {i} dependencies: {job.dependencies}")

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
                /bin/bash -c 'export PATH="/opt/slurm/bin:$PATH" && unset XLA_FLAGS && unset CUDA_CACHE_PATH && source /ext3/env.sh &&  {python_command}'"""
            ) 
        )

