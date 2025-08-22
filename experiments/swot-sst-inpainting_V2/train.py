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
    """Generate outputs for a dataset (Zarr or dict of arrays) in batches.
    Returns a dict of arrays, similar to HuggingFace Dataset output."""
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
            # Repeat last samples to pad
            y_pad = np.repeat(y_batch[-1:], pad_size, axis=0)
            A_pad = np.repeat(A_batch[-1:], pad_size, axis=0)
            y_batch = np.concatenate([y_batch, y_pad], axis=0)
            A_batch = np.concatenate([A_batch, A_pad], axis=0)
        x_batch = sample(model, y_batch, A_batch, rng.split(), **kwargs)
        # Remove padding from output
        if current_batch_size % num_gpus != 0:
            x_batch = x_batch[:current_batch_size]
        xs.append(np.asarray(x_batch))
    xs = np.concatenate(xs, axis=0)
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
    print(f"Starting lap {lap} with runid {runid}")
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
        print(f"TRAIN DEBUG: Lap {lap} - loading checkpoint with optimized approach", flush=True)
        # Create a fresh model template (this will compile fast since it's empty)
        with inox_random.set_rng(init=inox_random.PRNG(rng.split()), dropout=inox_random.PRNG(rng.split())):
            model_template = make_model(key=rng.split(), in_channels=C, out_channels=C, **CONFIG)
        print(f"TRAIN DEBUG: Model template created, now loading parameters", flush=True)

        # Load the saved model (this should be faster now)
        with inox_random.set_rng(init=inox_random.PRNG(rng.split()), dropout=inox_random.PRNG(rng.split())):
            saved_model = load_module(runpath / f'checkpoint_{lap - 1}.pkl')
        # Copy parameters from saved model to template
        static_template, _ = model_template.partition()
        _, params_saved = saved_model.partition()
        previous = static_template(params_saved)
        print(f"TRAIN DEBUG: Checkpoint loaded and compiled successfully", flush=True)
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
            with inox_random.set_rng(init=inox_random.PRNG(rng.split()), dropout=inox_random.PRNG(rng.split())):
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


# Debugging and minimal test functions below 
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''
def test_minimal(runid: int, lap: int, src: str):
    print(f"MINIMAL TEST: lap {lap}, runid {runid}", flush=True)
    import time
    try:
        #!/usr/bin/env python
        print("IMPORT: Starting train.py import", flush=True)
        # Core Libraries
        print("IMPORT: Importing inox...", flush=True)
        import inox                  # type: ignore # Custom library (likely for modeling and random utilities)
        print("IMPORT: Importing inox.nn as nn...", flush=True)
        import inox.nn as nn         # type: ignore # type: ignore # Neural network components
        print("IMPORT: Importing inox_random", flush=True)
        from inox import random as inox_random # type: ignore
        print("IMPORT: Importing jax", flush=True)
        import jax                   # type: ignore # JAX for high-performance computing
        print("IMPORT: Importing numpy", flush=True)
        import numpy as np # type: ignore
        print("IMPORT: Importing optax", flush=True)
        import optax                 # type: ignore # Optimizers for JAX
        print("IMPORT: Importing wandb", flush=True)
        import wandb                 # Weights and Biases for experiment tracking
        print("IMPORT: Importing jax.numpy as jnp", flush=True)
        import jax.numpy as jnp # type: ignore # JAX's numpy for array operations
        # Workflow management
        print("IMPORT: Importing dawgz", flush=True)
        from dawgz import job, schedule # type: ignore
        print("IMPORT: Importing priors", flush=True)
        from priors.diffusion import VESDE, DenoiserLoss, GaussianDenoiser
        from priors.data import prefetch
        from priors.image import random_flip, random_hue, random_saturation, to_pil
        from priors.common import dump_module, ppca, fit_moments, load_module
        from priors.optim import Adam, EMA
        print("IMPORT: Importing partial", flush=True)
        from functools import partial
        print("IMPORT: Importing tqdm", flush=True)
        from tqdm import trange
        print("IMPORT: Importing typing", flush=True)
        from typing import Dict, List, Tuple, Optional, Union, Any, Callable
        print("IMPORT: Importing utils", flush=True)
        from utils import make_model, sample, measure, PATH          # Assumed utility functions (augmentations, flatten, sampling, etc.)
        print("IMPORT: Importing zarr", flush=True)
        import zarr # type: ignore
        print("IMPORT: Importing time", flush=True)
        import time
    except Exception as e:
        print(f"TEST ERROR: {e}", flush=True)
        raise
    time.sleep(5)
    print(f"MINIMAL TEST COMPLETED: lap {lap}", flush=True)
    return f"success_{lap}"

def train_minimal_jax(runid: int, lap: int, src: str):
    print(f"TRAIN JAX: Starting lap {lap}", flush=True)
    # Import JAX inside function
    import jax
    import jax.numpy as jnp
    print("TRAIN JAX: JAX imported", flush=True)
    # Simple JAX operation
    x = jnp.array([1, 2, 3])
    y = jnp.sum(x)
    print(f"TRAIN JAX: Simple operation result: {y}", flush=True)
    return f"success_{lap}"

def train_step1(runid: int, lap: int, src: str):
    print(f"TRAIN STEP1: Starting lap {lap}", flush=True)
    # Test basic imports first
    import time
    import wandb
    print("TRAIN STEP1: Basic imports OK", flush=True)
    # Test WANDB init
    run = wandb.init(
        project='priors-SST-mask',
        id=runid,
        resume='allow',
        dir="/scratch/tm3076/project/LDP_OEM/Tatsu/experiments/swot-sst-inpainting_V2",  # Use hardcoded path first
    )
    print("TRAIN STEP1: WANDB init OK", flush=True)
    return f"success_{lap}"

def train_step2(runid: int, lap: int, src: str):
    print(f"TRAIN STEP2: Starting lap {lap}", flush=True)
    import time
    import wandb
    import jax
    import jax.numpy as jnp
    # Test JAX config
    jax.config.update('jax_threefry_partitionable', True)
    jax.config.update('jax_enable_x64', False)
    print("TRAIN STEP2: JAX config OK", flush=True)
    # Test JAX devices (this might be the issue!)
    devices = jax.devices()
    print(f"TRAIN STEP2: Found {len(devices)} devices", flush=True)
    return f"success_{lap}"

def train_step3(runid: int, lap: int, src: str):
    print(f"TRAIN STEP3: Starting lap {lap}", flush=True)
    # Import each custom module individually to isolate the issue
    try:
        print("Importing inox...", flush=True)
        import inox
        print("Importing inox.nn...", flush=True)
        import inox.nn as nn
        print("Importing inox random...", flush=True)
        from inox import random as inox_random
        print("Importing priors.diffusion...", flush=True)
        from priors.diffusion import VESDE, DenoiserLoss, GaussianDenoiser
        print("Importing priors.data...", flush=True)
        from priors.data import prefetch
        print("Importing priors.image...", flush=True)
        from priors.image import random_flip, random_hue, random_saturation, to_pil, flatten
        print("Importing priors.common...", flush=True)
        from priors.common import dump_module, ppca, fit_moments, load_module
        print("Importing priors.optim...", flush=True)
        from priors.optim import Adam, EMA
        print("Importing utils...", flush=True)
        from utils import make_model, sample, measure, PATH  # This might be the issue!
        print("All imports successful!", flush=True)
    except Exception as e:
        print(f"Import failed: {e}", flush=True)
        raise
    return f"success_{lap}"


def train_step4(runid: int, lap: int, src: str):
    print(f"TRAIN STEP4: Starting lap {lap}", flush=True)
    
    # Do all the imports (we know these work)
    import time
    import wandb
    import jax
    import jax.numpy as jnp
    from utils import PATH
    
    print("STEP4: Starting WANDB init...", flush=True)
    run = wandb.init(
        project='priors-SST-mask',
        id=runid,
        resume='allow',
        dir=PATH,
        config={'test': True},  # Simplified config first
    )
    print("STEP4: WANDB init complete", flush=True)
    
    print("STEP4: Setting up JAX config...", flush=True)
    jax.config.update('jax_threefry_partitionable', True)
    jax.config.update('jax_enable_x64', False)
    print("STEP4: JAX config complete", flush=True)
    
    print("STEP4: Setting up mesh and sharding...", flush=True)
    mesh = jax.sharding.Mesh(jax.devices(), 'i')
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    distributed = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('i'))
    print("STEP4: Mesh and sharding complete", flush=True)
    
    print("STEP4: Testing dataset loading...", flush=True)
    import zarr
    trainset_yA = zarr.open_group(f"{src}/train", mode="r")
    print(f"STEP4: Dataset opened, shape: {trainset_yA['y'].shape}", flush=True)
    
    print("STEP4: Testing small data transfer to GPU...", flush=True)
    y_small = trainset_yA['y'][:4]  # Just 2 samples first
    A_small = trainset_yA['A'][:4]
    print(f"STEP4: Small data loaded from zarr", flush=True)
    
    y_gpu, A_gpu = jax.device_put((y_small, A_small), distributed)
    print(f"STEP4: Small data transferred to GPU successfully", flush=True)
    
    return f"success_{lap}"

def train_step5(runid: int, lap: int, src: str):
    print(f"TRAIN STEP5: Starting lap {lap}", flush=True)
    
    # All the setup we know works
    import time
    import wandb
    import jax
    import jax.numpy as jnp
    from utils import PATH
    from priors.diffusion import VESDE
    from priors.image import flatten
    from inox import random as inox_random
    
    # Copy the global CONFIG locally to avoid pickle issues
    CONFIG = {
        'sde': {'a': 1e-4, 'b': 1e2},
        'cov_y': 1e-4**2,
    }
    
    print("STEP5: WANDB init...", flush=True)
    run = wandb.init(
        project='priors-SST-mask',
        id=runid,
        resume='allow',
        dir=PATH,
        config=CONFIG,  # Use local CONFIG
    )
    print("STEP5: WANDB init complete", flush=True)
    
    print("STEP5: JAX setup...", flush=True)
    jax.config.update('jax_threefry_partitionable', True)
    jax.config.update('jax_enable_x64', False)
    mesh = jax.sharding.Mesh(jax.devices(), 'i')
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    distributed = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('i'))
    print("STEP5: JAX setup complete", flush=True)
    
    print("STEP5: Loading dataset...", flush=True)
    import zarr
    trainset_yA = zarr.open_group(f"{src}/train", mode="r")
    testset_yA = zarr.open_group(f"{src}/train", mode="r")
    print("STEP5: Dataset loaded", flush=True)
    
    print("STEP5: Testing LARGE data transfer (16 samples)...", flush=True)
    y_eval, A_eval = testset_yA['y'][:16], testset_yA['A'][:16]
    y_eval, A_eval = jax.device_put((y_eval, A_eval), distributed)
    print("STEP5: Large eval data transferred successfully", flush=True)
    
    print(f"STEP5: Testing VERY LARGE data transfer (16384 samples for lap {lap})...", flush=True)
    if lap == 0:  # Only test large transfer for lap 0
        y_fit, A_fit = trainset_yA['y'][:16384], trainset_yA['A'][:16384]
        print("STEP5: Large data loaded from zarr", flush=True)
        
        y_fit, A_fit = jax.device_put((y_fit, A_fit), distributed)
        print("STEP5: VERY large data transferred successfully", flush=True)
        
        print("STEP5: Testing flatten operation...", flush=True)
        y_flat = flatten(y_fit)
        print(f"STEP5: Flatten successful, shape: {y_flat.shape}", flush=True)
    else:
        print("STEP5: Skipping large data transfer for lap > 0", flush=True)
    
    return f"success_{lap}"

def train_step6(runid: int, lap: int, src: str):
    print(f"TRAIN STEP6: Starting lap {lap}", flush=True)
    
    # All previous setup that works...
    import time
    import wandb
    import jax
    import jax.numpy as jnp
    from utils import PATH, measure
    from priors.diffusion import VESDE
    from priors.image import flatten
    from priors.common import fit_moments
    from inox import random as inox_random
    import inox
    
    CONFIG = {
        'sde': {'a': 1e-4, 'b': 1e2},
        'cov_y': 1e-4**2,
        'maxiter': 10,
    }
    
    # Setup (we know this works)
    run = wandb.init(project='priors-SST-mask', id=runid, resume='allow', dir=PATH, config=CONFIG)
    jax.config.update('jax_threefry_partitionable', True)
    jax.config.update('jax_enable_x64', False)
    mesh = jax.sharding.Mesh(jax.devices(), 'i')
    distributed = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('i'))
    
    runpath = PATH / f'runs/{run.name}_{run.id}'
    runpath.mkdir(parents=True, exist_ok=True)
    seed = hash((runpath, lap)) % 2**16
    rng = inox.random.PRNG(seed)
    sde = VESDE(**CONFIG['sde'])
    
    print("STEP6: Loading dataset...", flush=True)
    import zarr
    trainset_yA = zarr.open_group(f"{src}/train", mode="r")
    
    if lap == 0:  # Only test fit_moments for lap 0
        print("STEP6: Loading data for fit_moments...", flush=True)
        y_fit, A_fit = trainset_yA['y'][:1024], trainset_yA['A'][:1024]  # Smaller dataset first
        y_fit, A_fit = jax.device_put((y_fit, A_fit), distributed)
        B, H, W, C = y_fit.shape
        D = H * W * C
        print(f"STEP6: Data loaded, shape: {y_fit.shape}, D: {D}", flush=True)
        
        print("STEP6: Testing fit_moments (this might hang)...", flush=True)
        with inox_random.set_rng(init=inox_random.PRNG(rng.split())):
            mu_x, cov_x = fit_moments(
                features=D,
                rank=32,  # Much smaller rank for testing
                shard=True,
                A=inox.tree.Partial(measure, A_fit, H=H, W=W, C=C),
                y=flatten(y_fit),
                cov_y=CONFIG['cov_y'],
                sampler='ddim',
                sde=sde,
                steps=32,  # Much fewer steps for testing
                maxiter=2,  # Much fewer iterations
                key=rng.split(),
            )
        print("STEP6: fit_moments completed successfully!", flush=True)
    else:
        print("STEP6: Skipping fit_moments for lap > 0", flush=True)
    
    return f"success_{lap}"

def train_step7(runid: int, lap: int, src: str):
    """Test with exact same structure as train() but local CONFIG"""
    print(f"TRAIN STEP7: Starting lap {lap}", flush=True)
    
    # Move ALL global references to local (this is the key test)
    CONFIG = {
        'hid_channels': (128, 256, 384),
        'hid_blocks': (5, 5, 5),
        'kernel_size': (3, 3),
        'emb_features': 256,
        'heads': {1: 4},
        'dropout': 0.1,
        'checkpoint_layers': (),
        'cov_y': 1e-4**2,
        'sampler': 'ddpm',
        'sde': {'a': 1e-4, 'b': 1e2},
        'heuristic': None,
        'discrete': 256,
        'maxiter': 10,
        'generation_batch_size': 128,
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
    
    # Import everything locally (avoid global import state)
    import time
    import wandb
    import jax
    import jax.numpy as jnp
    from utils import PATH
    from priors.diffusion import VESDE
    from priors.image import flatten
    from inox import random as inox_random
    import inox
    import zarr
    
    print("STEP7: Starting EXACT train() logic...", flush=True)
    
    # Copy the EXACT logic from train() with local variables
    start_time = time.time()
    run = wandb.init(
        project='priors-SST-mask',
        id=runid,
        resume='allow',
        dir=PATH,
        config=CONFIG,  # Local CONFIG
    )
    runpath = PATH / f'runs/{run.name}_{run.id}'
    runpath.mkdir(parents=True, exist_ok=True)
    config = run.config

    print("STEP7: WANDB setup complete", flush=True)

    # Enable partitioning for reproducible RNG across shards
    jax.config.update('jax_threefry_partitionable', True)
    jax.config.update('jax_enable_x64', False)
    mesh = jax.sharding.Mesh(jax.devices(), 'i')
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    distributed = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('i'))

    print("STEP7: JAX setup complete", flush=True)

    # Initialize PRNG with lap-specific seed
    seed = hash((runpath, lap)) % 2**16
    rng = inox.random.PRNG(seed)

    # Create the SDE object (Variance Exploding SDE)
    sde = VESDE(**CONFIG.get('sde'))
    
    print("STEP7: Loading dataset...", flush=True)
    trainset_yA = zarr.open_group(f"{src}/train", mode="r")
    testset_yA = zarr.open_group(f"{src}/train", mode="r")
    
    print("STEP7: Testing eval data transfer...", flush=True)
    y_eval, A_eval = testset_yA['y'][:16], testset_yA['A'][:16]
    y_eval, A_eval = jax.device_put((y_eval, A_eval), distributed)
    B, H, W, C = y_eval.shape
    D = H * W * C
    print(f"STEP7: Eval data loaded successfully, shape: {y_eval.shape}", flush=True)
    
    if lap > 0:
        print("STEP7: Would load checkpoint here (lap > 0)", flush=True)
        # Skip checkpoint loading for now
    else:
        print("STEP7: Testing lap 0 logic (fit_moments)...", flush=True)
        # Use smaller dataset for testing
        y_fit, A_fit = trainset_yA['y'][:1024], trainset_yA['A'][:1024]
        y_fit, A_fit = jax.device_put((y_fit, A_fit), distributed)
        print("STEP7: Fit data loaded", flush=True)
        # Skip fit_moments for now since we tested it works
        print("STEP7: Skipping fit_moments (we know it works)", flush=True)
    
    print("STEP7: Test completed successfully!", flush=True)
    return f"success_{lap}"
    
def train_step8(runid: int, lap: int, src: str):
    """Test checkpoint loading specifically"""
    print(f"TRAIN STEP8: Starting lap {lap}", flush=True)
    
    if lap == 0:
        print("STEP8: Lap 0, no checkpoint needed", flush=True)
        return f"success_{lap}"
    
    # Test checkpoint loading
    from utils import PATH
    from priors.common import load_module
    import time
    
    print("STEP8: Looking for checkpoint...", flush=True)
    
    # Try to find the checkpoint from previous lap
    checkpoint_path = PATH / f'runs/*/checkpoint_{lap - 1}.pkl'
    import glob
    checkpoints = glob.glob(str(checkpoint_path))
    
    if not checkpoints:
        print(f"STEP8: No checkpoint found at {checkpoint_path}", flush=True)
        return f"no_checkpoint_{lap}"
    
    checkpoint_file = checkpoints[0]
    print(f"STEP8: Found checkpoint: {checkpoint_file}", flush=True)
    
    print("STEP8: Loading checkpoint (this might hang)...", flush=True)
    start_time = time.time()
    previous = load_module(checkpoint_file)
    load_time = time.time() - start_time
    print(f"STEP8: Checkpoint loaded in {load_time:.2f} seconds!", flush=True)
    
    return f"success_{lap}"

'''