#!/usr/bin/env python

# Core Libraries
import inox                  # Custom library (likely for modeling and random utilities)
import inox.nn as nn         # Neural network components
import jax                   # JAX for high-performance computing
import numpy as np
import optax                 # Optimizers for JAX
import wandb                 # Weights and Biases for experiment tracking

# Hydra and configuration
import hydra
from omegaconf import DictConfig, OmegaConf
import os
from pathlib import Path

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
        if isinstance(idx, slice):
            # Handle slice: return dict with arrays for multiple samples
            indices = range(*idx.indices(len(self)))
            result = {}
            for i, file_idx in enumerate(indices):
                arrays = np.load(self.paths[file_idx])
                if i == 0:
                    # Initialize result dict with empty lists
                    result = {k: [] for k in arrays.keys()}
                for k, v in arrays.items():
                    result[k].append(jnp.array(v))
            # Stack all arrays
            return {k: jnp.stack(v) for k, v in result.items()}
        else:
            # Handle single index
            arrays = np.load(self.paths[idx])
            return {k: jnp.array(v) for k, v in arrays.items()}

class _ZarrSubDataset:
    def __init__(self, zarr_group):
        self.zarr = zarr_group
        keys = list(zarr_group.keys())
        if not keys:
            raise ValueError("Zarr group contains no arrays")
        self.keys = keys
        self.length = zarr_group[keys[0]].shape[0]
        self._indices = None  # For shuffling
    def __len__(self):
        return self.length
    def shuffle(self, seed):
        """Shuffle the dataset indices."""
        rng = np.random.RandomState(seed)
        self._indices = rng.permutation(self.length)
        return self
    def __getitem__(self, idx):
        # Apply shuffled indices if they exist
        if self._indices is not None:
            if isinstance(idx, slice):
                slice_indices = self._indices[idx]
                return {k: jnp.array(self.zarr[k][slice_indices]) for k in self.keys}
            else:
                actual_idx = self._indices[idx]
                return {k: jnp.array(self.zarr[k][actual_idx]) for k in self.keys}
        else:
            # Original behavior
            if isinstance(idx, slice):
                return {k: jnp.array(self.zarr[k][idx]) for k in self.keys}
            else:
                return {k: jnp.array(self.zarr[k][idx]) for k in self.keys}

# Add this class after your existing dataset classes
class SimpleDataset:
    """Simple dataset class that mimics the interface expected by the training code."""
    def __init__(self, data):
        self.data = data
        self.length = len(list(data.values())[0])

    def __len__(self):
        return self.length

    def __getitem__(self, key):
        if isinstance(key, slice):
            # Return dict with sliced arrays for operations like trainset[:10384]['x']
            return {k: v[key] for k, v in self.data.items()}
        else:
            # Return specific key for operations like trainset['x']
            return self.data[key]

    def shuffle(self, seed):
        """Return a new shuffled dataset."""
        rng = np.random.RandomState(seed)
        indices = rng.permutation(self.length)
        shuffled_data = {k: v[indices] for k, v in self.data.items()}
        return SimpleDataset(shuffled_data)

    def iter(self, batch_size, drop_last_batch=True):
        """Iterate over the dataset in batches."""
        for i in range(0, self.length, batch_size):
            if drop_last_batch and i + batch_size > self.length:
                break
            batch = {k: v[i:i+batch_size] for k, v in self.data.items()}
            yield batch

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def generate(model, dataset, rng, batch_size, **kwargs):
    def transform(batch):
        y, A = batch['y'], batch['A']
        x = sample(model, y, A, rng.split(), **kwargs)
        x = np.asarray(x)
        return {'x': x}

    # Process the dataset in batches and collect results
    results = []
    for i in range(0, len(dataset), batch_size):
        # Get batch indices
        batch_indices = list(range(i, min(i + batch_size, len(dataset))))
        if len(batch_indices) < batch_size:
            break  # Drop last incomplete batch

        # Create batch by collecting items
        batch = {k: [] for k in dataset[0].keys()}
        for idx in batch_indices:
            item = dataset[idx]
            for k, v in item.items():
                batch[k].append(v)

        # Stack the batch arrays
        batch = {k: jnp.stack(v) for k, v in batch.items()}

        # Apply transform
        transformed_batch = transform(batch)
        results.append(transformed_batch)

    # Combine all results
    all_data = {}
    for key in results[0].keys():
        all_data[key] = jnp.concatenate([batch[key] for batch in results])

    # Return a SimpleDataset that supports the operations used later
    return SimpleDataset(all_data)

def setup_paths(cfg: DictConfig):
    """Setup paths based on configuration."""
    if cfg.paths.scratch_env_var in os.environ:
        SCRATCH = os.environ[cfg.paths.scratch_env_var]
        PATH = Path(SCRATCH) / 'priors/cifar'
    else:
        PATH = Path(cfg.paths.default_path)

    PATH.mkdir(parents=True, exist_ok=True)
    return PATH

def train(runid: str, lap: int, cfg: DictConfig):
    """
    Main training loop for a single training 'lap' (iteration).
    Each lap can be seen as one cycle of training, optionally starting from a prior checkpoint.
    """
    # Setup paths
    PATH = setup_paths(cfg)

    # Convert config sections to dictionaries for compatibility with existing code
    config_dict = {
        **OmegaConf.to_container(cfg.model, resolve=True),
        **OmegaConf.to_container(cfg.diffusion, resolve=True),
        **OmegaConf.to_container(cfg.training, resolve=True),
    }

    # Initialize Weights & Biases
    run = wandb.init(
        project=cfg.experiment.project_name,
        id=runid,
        resume='allow',
        dir=PATH,
        config=config_dict,
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
    sde = VESDE(**cfg.diffusion.sde)

    # Load dataset
    dataset = PrecomputedJAXDataset(cfg.data.source, format=cfg.data.format)

    trainset_yA = dataset['train']
    testset_yA = dataset['test']

    # Validation data (fixed samples)
    y_eval, A_eval = testset_yA[:16]['y'], testset_yA[:16]['A']
    y_eval, A_eval = jax.device_put((y_eval, A_eval), distributed)
    B, H, W, C = y_eval.shape
    D = H * W * C

    # If lap >0, load previous checkpoint, else fit prior Gaussian model
    if lap > 0:
        previous = load_module(runpath / f'checkpoint_{lap - 1}.pkl')
    else:
        # Shuffle the training dataset for moment fitting
        shuffle_seed = hash((runid, "moment_fitting")) % 2**16
        shuffled_trainset = trainset_yA.shuffle(shuffle_seed)
        # Now take first N samples (which are actually shuffled)
        y_fit, A_fit = shuffled_trainset[:cfg.data.moment_fitting_samples]['y'], shuffled_trainset[:cfg.data.moment_fitting_samples]['A']
        y_fit, A_fit = jax.device_put((y_fit, A_fit), distributed)
        B, H, W, C = y_fit.shape
        D = H * W * C
        mu_x, cov_x = fit_moments(
            features=D,
            rank=320,
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
        batch_size=cfg.training.batch_size,
        shard=True,
        sampler=cfg.diffusion.sampler,
        sde=sde,
        steps=cfg.diffusion.discrete,
        maxiter=cfg.diffusion.maxiter,
    )
    testset = generate(
        model=previous,
        dataset=testset_yA,
        rng=rng,
        batch_size=cfg.training.batch_size,
        shard=True,
        sampler=cfg.diffusion.sampler,
        sde=sde,
        steps=cfg.diffusion.discrete,
        maxiter=cfg.diffusion.maxiter,
    )

    # Fit low-rank covariance (PPCA) on generated training data
    x_fit = trainset[:cfg.data.ppca_samples]['x']
    x_fit = flatten(x_fit)
    mu_x, cov_x = ppca(x_fit, rank=320, key=rng.split())
    del x_fit

    # Initialize model
    if lap > 0:
        model = previous
    else:
       model = make_model(key=rng.split(), in_channels=C, out_channels=C, **config_dict)

    # Set model's prior mean
    model.mu_x = mu_x

    # Configure model's covariance heuristic
    if cfg.diffusion.heuristic == 'zeros':
        model.cov_x = jnp.zeros_like(mu_x)
    elif cfg.diffusion.heuristic == 'ones':
        model.cov_x = jnp.ones_like(mu_x)
    elif cfg.diffusion.heuristic == 'cov_t':
        model.cov_x = jnp.ones_like(mu_x) * 1e6
    elif cfg.diffusion.heuristic == 'cov_x':
        model.cov_x = cov_x

    model.train(True)

    # Partition model parameters
    static, params, others = model.partition(nn.Parameter)

    # Define denoising loss
    objective = DenoiserLoss(sde=sde)

    # Build optimizer
    steps = cfg.training.epochs * len(trainset_yA) // cfg.training.batch_size
    optimizer = Adam(steps=steps, **OmegaConf.to_container(cfg.training, resolve=True))
    opt_state = optimizer.init(params)

    # Exponential moving average for parameter stabilization
    ema = EMA(decay=cfg.training.ema_decay)
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
    for epoch in (bar := trange(cfg.training.epochs, ncols=88)):
        # Shuffle training set per epoch
        loader = trainset.shuffle(seed=seed + lap * cfg.training.epochs + epoch).iter(
            batch_size=cfg.training.batch_size, drop_last_batch=True
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
        loader = testset.iter(batch_size=cfg.training.batch_size, drop_last_batch=True)
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
                sampler=cfg.diffusion.sampler,
                steps=cfg.diffusion.discrete,
                maxiter=cfg.diffusion.maxiter,
            )
            num = x.shape[0]
            cols = int(np.sqrt(num))
            rows = num // cols
            x = x.reshape(rows, cols, H, W, C)
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

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main function that sets up and schedules training jobs."""

    print(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    wandb.login()
    runid = wandb.util.generate_id()
    jobs = []

    # Schedule multiple laps as Slurm jobs
    for lap in range(cfg.experiment.num_laps):
        jobs.append(
            job(
                partial(train, runid=runid, lap=lap, cfg=cfg),
                name=f'train_{lap}',
                cpus=cfg.slurm.cpus,
                gpus=cfg.slurm.gpus,
                ram=cfg.slurm.ram,
                time=cfg.slurm.time,
                partition=cfg.slurm.partition,
                wrap='\"hostname && sleep infinity\"'
           )
        )
        if len(jobs) > 1:
            jobs[-1].after(jobs[-2], status="any")

    singularity_cmd = f"""singularity exec --nv \\
                --bind /opt/slurm:/opt/slurm \\
                --bind /var/run/munge:/var/run/munge \\
                --overlay {cfg.slurm.overlay_path}:ro \\
                {cfg.slurm.singularity_image} \\
                /bin/bash -c 'export PATH="/opt/slurm/bin:$PATH" && source /ext3/env.sh &&  {{python_command}}'"""

    schedule(
        *jobs,
        name=f'Training {runid}',
        backend='slurm',
        debug=cfg.debug.debug_mode,
        export='ALL',
        env=['export WANDB_SILENT=true'],
        dry_run=cfg.debug.dry_run,
        singularity=singularity_cmd
    )

if __name__ == '__main__':
    main()
