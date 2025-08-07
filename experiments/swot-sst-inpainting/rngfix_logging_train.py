#!/usr/bin/env python

# Core Libraries
import inox                  # Custom library (likely for modeling and random utilities)
import inox.nn as nn         # Neural network components
import jax                   # JAX for high-performance computing
import numpy as np
import optax                 # Optimizers for JAX
import wandb                 # Weights and Biases for experiment tracking
import time                  # For timing debug info
import psutil                # For memory monitoring
import gc                    # For garbage collection

# Data handling
from datasets import Array3D, Features #load_from_disk

# Workflow management
from dawgz import job, schedule

from functools import partial
from tqdm import trange
from typing import *
from utils import *          # Assumed utility functions (augmentations, flatten, sampling, etc.)

# Prefetch function for overlapping computation and data loading
def prefetch(iterator, num_batches=2):
    """Prefetch batches to overlap computation and data loading"""
    import threading
    import queue
    
    def producer(it, q):
        try:
            for item in it:
                q.put(item)
        except Exception as e:
            print(f"[PREFETCH] Error in producer: {e}")
            q.put(e)
        finally:
            q.put(StopIteration)
    
    q = queue.Queue(maxsize=num_batches)
    thread = threading.Thread(target=producer, args=(iterator, q))
    thread.daemon = True
    thread.start()
    
    while True:
        item = q.get()
        if isinstance(item, StopIteration):
            break
        elif isinstance(item, Exception):
            raise item
        else:
            yield item

# Add memory monitoring function
def log_memory_usage(stage=""):
    """Log current memory usage for debugging"""
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / 1024 / 1024
    print(f"[MEMORY {stage}] RSS: {memory_mb:.1f}MB")
    
    # Also log GPU memory if available
    try:
        import GPUtil
        gpus = GPUtil.getGPUs()
        for i, gpu in enumerate(gpus):
            print(f"[GPU {i} {stage}] Memory: {gpu.memoryUsed}MB/{gpu.memoryTotal}MB ({gpu.memoryUtil*100:.1f}%)")
    except:
        pass

def log_jax_memory():
    """Log JAX-specific memory usage"""
    try:
        from jax.lib import xla_bridge
        backend = xla_bridge.get_backend()
        for i, device in enumerate(backend.local_devices()):
            try:
                memory_stats = device.memory_stats()
                print(f"[JAX GPU {i}] Memory stats: {memory_stats}")
            except:
                print(f"[JAX GPU {i}] Memory stats not available")
    except Exception as e:
        print(f"[JAX] Could not get memory stats: {e}")

# Timeout handler for detecting hangs
class TimeoutHandler:
    def __init__(self, timeout=600):  # 10 minutes default
        self.timeout = timeout
        self.old_handler = None
        
    def __enter__(self):
        import signal
        def timeout_handler(signum, frame):
            print(f"[TIMEOUT] Operation timed out after {self.timeout}s")
            print(f"[TIMEOUT] Current frame: {frame.f_code.co_filename}:{frame.f_lineno}")
            raise TimeoutError(f"Operation timed out after {self.timeout}s")
        
        self.old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(self.timeout)
        return self
        
    def __exit__(self, *args):
        import signal
        signal.alarm(0)
        if self.old_handler is not None:
            signal.signal(signal.SIGALRM, self.old_handler)

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
    'batch_size': 1056,
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
import jax.numpy as jnp
import numpy as np
import zarr
from glob import glob
from pathlib import Path

class PrecomputedJAXDataset:
    def __init__(self, source, format="zarr"):
        print(f"[DATASET] Initializing dataset from {source} with format {format}")
        self.format = format
        self.source = Path(source)
        self.splits = {}
        for split in ["train", "val", "test"]:
            split_path = self.source / split
            print(f"[DATASET] Checking split_path: {split_path}")
            if not split_path.exists():
                print(f"[DATASET] Split {split} not found, skipping")
                continue
            if format == "npz":
                paths = sorted(glob(str(split_path / "sample_*.npz")))
                self.splits[split] = {
                    "type": "npz",
                    "paths": paths,
                    "length": len(paths)
                }
                print(f"[DATASET] NPZ split {split}: {len(paths)} files")
            elif format == "zarr":
                z = zarr.open_group(str(split_path), mode="r")
                print(f"[DATASET] Zarr group {split}: {z}")

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
                print(f"[DATASET] Zarr split {split}: {length} samples, keys: {keys}")
            else:
                raise ValueError(f"Unsupported format: {format}")
                
    def __getitem__(self, split):
        if split not in self.splits:
            raise KeyError(f"Split '{split}' not found. Available: {list(self.splits.keys())}")
        print(f"[DATASET] Getting split dataset: {split}")
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
        print(f"[NPZ_DATASET] Initialized with {len(paths)} files")
        
    def __len__(self):
        return len(self.paths)
        
    def __getitem__(self, idx):
        if isinstance(idx, slice):
            print(f"[NPZ_DATASET] Getting slice: {idx}")
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
        print(f"[ZARR_DATASET] Initialized with {self.length} samples, keys: {keys}")
        
    def __len__(self):
        return self.length
        
    def shuffle(self, seed):
        """Shuffle the dataset indices."""
        print(f"[ZARR_DATASET] Shuffling with seed {seed}")
        start_time = time.time()
        rng = np.random.RandomState(seed)
        self._indices = rng.permutation(self.length)
        print(f"[ZARR_DATASET] Shuffle completed in {time.time() - start_time:.2f}s")
        return self
        
    def __getitem__(self, idx):
        # Apply shuffled indices if they exist
        if self._indices is not None:
            if isinstance(idx, slice):
                slice_indices = self._indices[idx]
                result = {k: jnp.array(self.zarr[k][slice_indices]) for k in self.keys}
                return result
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
        print(f"[SIMPLE_DATASET] Initialized with {self.length} samples")
        log_memory_usage("SimpleDataset.__init__")
    
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
        print(f"[SIMPLE_DATASET] Shuffling with seed {seed}")
        start_time = time.time()
        rng = np.random.RandomState(seed)
        indices = rng.permutation(self.length)
        shuffled_data = {k: v[indices] for k, v in self.data.items()}
        result = SimpleDataset(shuffled_data)
        print(f"[SIMPLE_DATASET] Shuffle completed in {time.time() - start_time:.2f}s")
        return result
    
    def iter(self, batch_size, drop_last_batch=True):
        """Iterate over the dataset in batches."""
        print(f"[SIMPLE_DATASET] Starting iteration with batch_size={batch_size}, drop_last_batch={drop_last_batch}")
        batch_count = 0
        for i in range(0, self.length, batch_size):
            if drop_last_batch and i + batch_size > self.length:
                print(f"[SIMPLE_DATASET] Dropping last incomplete batch at index {i}")
                break
            batch = {k: v[i:i+batch_size] for k, v in self.data.items()}
            batch_count += 1
            if batch_count % 10 == 0:  # Log every 10th batch
                print(f"[SIMPLE_DATASET] Yielding batch {batch_count} (indices {i}:{i+batch_size})")
            yield batch
        print(f"[SIMPLE_DATASET] Iteration completed, yielded {batch_count} batches")

# Chunked Dataset class for managing large generated datasets
class ChunkedDataset:
    """Dataset that stores data in chunks to manage memory efficiently"""
    def __init__(self, chunks, total_length):
        self.chunks = chunks  # List of data chunks
        self.total_length = total_length
        self.chunk_sizes = [len(list(chunk.values())[0]) for chunk in chunks]
        self.cumulative_sizes = np.cumsum([0] + self.chunk_sizes)
        print(f"[CHUNKED_DATASET] Initialized with {len(chunks)} chunks, total length: {total_length}")
        
    def __len__(self):
        return self.total_length
    
    def _find_chunk_and_index(self, idx):
        """Find which chunk contains the given index"""
        if idx >= self.total_length:
            raise IndexError(f"Index {idx} out of bounds for dataset of size {self.total_length}")
        
        chunk_idx = np.searchsorted(self.cumulative_sizes[1:], idx, side='right')
        local_idx = idx - self.cumulative_sizes[chunk_idx]
        return chunk_idx, local_idx
    
    def __getitem__(self, key):
        if isinstance(key, slice):
            # Handle slice access
            start, stop, step = key.indices(self.total_length)
            if step != 1:
                raise NotImplementedError("Step slicing not supported")
            
            # Collect data across chunks
            result = None
            for i in range(start, stop):
                chunk_idx, local_idx = self._find_chunk_and_index(i)
                if result is None:
                    result = {k: [] for k in self.chunks[chunk_idx].keys()}
                
                for k in result.keys():
                    result[k].append(self.chunks[chunk_idx][k][local_idx])
            
            # Stack the collected data
            if result:
                return {k: jnp.stack(v) for k, v in result.items()}
            else:
                return {}
        else:
            # Handle single key access (like dataset['x'])
            if isinstance(key, str):
                # Return concatenated data for specific key
                return jnp.concatenate([chunk[key] for chunk in self.chunks])
            else:
                # Handle single index access
                chunk_idx, local_idx = self._find_chunk_and_index(key)
                return {k: v[local_idx] for k, v in self.chunks[chunk_idx].items()}
    
    def shuffle(self, seed):
        """Return a shuffled version using SimpleDataset for simplicity"""
        print(f"[CHUNKED_DATASET] Converting to SimpleDataset for shuffling")
        all_data = {}
        for key in self.chunks[0].keys():
            all_data[key] = jnp.concatenate([chunk[key] for chunk in self.chunks])
        return SimpleDataset(all_data).shuffle(seed)
    
    def iter(self, batch_size, drop_last_batch=True):
        """Iterate over the dataset in batches"""
        print(f"[CHUNKED_DATASET] Starting iteration with batch_size={batch_size}")
        batch_count = 0
        
        for i in range(0, self.total_length, batch_size):
            if drop_last_batch and i + batch_size > self.total_length:
                break
            
            # Collect batch data across chunks
            batch_data = {k: [] for k in self.chunks[0].keys()}
            
            for j in range(i, min(i + batch_size, self.total_length)):
                chunk_idx, local_idx = self._find_chunk_and_index(j)
                for k in batch_data.keys():
                    batch_data[k].append(self.chunks[chunk_idx][k][local_idx])
            
            # Stack the batch
            batch = {k: jnp.stack(v) for k, v in batch_data.items()}
            batch_count += 1
            
            if batch_count % 10 == 0:
                print(f"[CHUNKED_DATASET] Yielding batch {batch_count}")
            
            yield batch
        
        print(f"[CHUNKED_DATASET] Iteration completed, yielded {batch_count} batches")

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def generate(model, dataset, rng, batch_size, chunk_size=None, **kwargs):
    """Generate data in chunks to manage memory efficiently"""
    print(f"[GENERATE] Starting chunked generation for dataset with {len(dataset)} samples, batch_size={batch_size}")

    # Default chunk size to process 4 batches at a time
    if chunk_size is None:
        chunk_size = batch_size * 4

    print(f"[GENERATE] Using chunk_size={chunk_size}")
    log_memory_usage("generate_start")

    def transform(batch, key):
        y, A = batch['y'], batch['A']
        print(f"[GENERATE] Transforming batch with shapes y: {y.shape}, A: {A.shape}")
        start_time = time.time()
        x = sample(model, y, A, key=key, **kwargs)
        x = np.asarray(x)
        print(f"[GENERATE] Transform completed in {time.time() - start_time:.2f}s, output shape: {x.shape}")
        return {'x': x}

    chunks = []
    total_processed = 0
    chunk_count = 0

    for chunk_start in range(0, len(dataset), chunk_size):
        chunk_count += 1
        chunk_end = min(chunk_start + chunk_size, len(dataset))
        chunk_length = chunk_end - chunk_start

        print(f"[GENERATE] Processing chunk {chunk_count} (indices {chunk_start}:{chunk_end}, length: {chunk_length})")
        chunk_start_time = time.time()

        chunk_results = []

        for i in range(chunk_start, chunk_end, batch_size):
            batch_end = min(i + batch_size, chunk_end)
            actual_batch_size = batch_end - i

            if actual_batch_size < batch_size:
                print(f"[GENERATE] Skipping incomplete batch at end of chunk (size: {actual_batch_size})")
                break

            print(f"[GENERATE] Processing batch within chunk (indices {i}:{batch_end})")
            batch_process_start = time.time()

            # Create batch
            batch = {k: [] for k in dataset[0].keys()}
            for idx in range(i, batch_end):
                item = dataset[idx]
                for k, v in item.items():
                    batch[k].append(v)
            batch = {k: jnp.stack(v) for k, v in batch.items()}

            # ⚠️ Split RNG per batch
            rng, batch_key = jax.random.split(rng)
            transformed_batch = transform(batch, key=batch_key)
            chunk_results.append(transformed_batch)

            print(f"[GENERATE] Batch processed in {time.time() - batch_process_start:.2f}s")
            del batch

        if chunk_results:
            print(f"[GENERATE] Combining {len(chunk_results)} batches in chunk {chunk_count}")
            chunk_data = {}
            for key in chunk_results[0].keys():
                chunk_data[key] = jnp.concatenate([batch[key] for batch in chunk_results])
            chunks.append(chunk_data)
            total_processed += len(chunk_data[list(chunk_data.keys())[0]])

            print(f"[GENERATE] Chunk {chunk_count} completed in {time.time() - chunk_start_time:.2f}s")
            print(f"[GENERATE] Chunk {chunk_count} contains {len(chunk_data[list(chunk_data.keys())[0]])} samples")

        del chunk_results
        log_memory_usage(f"generate_chunk_{chunk_count}")
        print(f"[GENERATE] Running garbage collection after chunk {chunk_count}")
        gc.collect()

    print(f"[GENERATE] All chunks processed. Total samples: {total_processed}")
    log_memory_usage("generate_end")
    result_dataset = ChunkedDataset(chunks, total_processed)
    print(f"[GENERATE] Generation completed, returning chunked dataset with {len(result_dataset)} samples")
    return result_dataset


@partial(jax.jit, static_argnums=(0, 1))  # 0=sde, 1=A (the measure function)
def run_fit_moments(sde, A, y, key):
    return fit_moments(
        features=y.shape[1],
        rank=320,
        shard=True,
        A=A,
        y=y,
        cov_y=1e-3**2,
        sampler='ddim',
        sde=sde,
        steps=256,
        maxiter=None,
        key=key
    )

def train(runid: int, lap: int, src: str):
    """
    Main training loop for a single training 'lap' (iteration).
    Each lap can be seen as one cycle of training, optionally starting from a prior checkpoint.
    """
    print(f"[TRAIN] Starting training lap {lap} for run {runid}")
    log_memory_usage("train_start")
    
    # Initialize Weights & Biases
    print(f"[TRAIN] Initializing wandb")
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
    print(f"[TRAIN] Run path: {runpath}")

    # Enable partitioning for reproducible RNG across shards
    print(f"[TRAIN] Setting up JAX configuration")
    jax.config.update('jax_threefry_partitionable', True)
    mesh = jax.sharding.Mesh(jax.devices(), 'i')
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    distributed = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('i'))
    print(f"[TRAIN] JAX devices: {jax.devices()}")

    # Initialize PRNG with lap-specific seed
    seed = hash((runpath, lap)) % 2**16
    rng = jax.random.PRNGKey(seed)
    rng, key = jax.random.split(rng)
    print(f"[TRAIN] Using seed: {seed}")

    # Create the SDE object (Variance Exploding SDE)
    print(f"[TRAIN] Creating SDE object")
    sde = VESDE(**CONFIG.get('sde'))
    
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Load HuggingFace-formatted LLC4320 dataset
    print(f"[TRAIN] Loading dataset from {src}")
    dataset = PrecomputedJAXDataset(src, format="zarr")
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    print(f"[TRAIN] Getting train and test datasets")
    trainset_yA = dataset['train']
    testset_yA = dataset['test']
    print(f"[TRAIN] Dataset sizes - train: {len(trainset_yA)}, test: {len(testset_yA)}")

    # Validation data (fixed samples)
    print(f"[TRAIN] Loading validation data")
    y_eval, A_eval = testset_yA[:16]['y'], testset_yA[:16]['A']
    y_eval, A_eval = jax.device_put((y_eval, A_eval), distributed)
    B, H, W, C = y_eval.shape
    D = H * W * C
    print(f"[TRAIN] Validation data shape: B={B}, H={H}, W={W}, C={C}, D={D}")

    # If lap >0, load previous checkpoint, else fit prior Gaussian model
    if lap > 0:
        print(f"[TRAIN] Loading previous checkpoint from lap {lap-1}")
        previous = load_module(runpath / f'checkpoint_{lap - 1}.pkl')
    else:
        print(f"[TRAIN] First lap - fitting moments")
        # Shuffle the training dataset for moment fitting
        shuffle_seed = hash((runid, "moment_fitting")) % 2**16
        print(f"[TRAIN] Shuffling training data with seed {shuffle_seed}")
        shuffled_trainset = trainset_yA.shuffle(shuffle_seed)
        """ 
        # Now take first N samples (which are actually shuffled)
        print(f"[TRAIN] Taking 6144 samples for moment fitting")
        y_dummy, A_dummy = shuffled_trainset[:8]['y'], shuffled_trainset[:8]['A']
        y_dummy, A_dummy = jax.device_put((y_dummy, A_dummy), distributed)
        B, H, W, C = y_dummy.shape
        D = H * W * C
        
        print(f"[TRAIN] Starting moment fitting")
        start_time = time.time()
        # Dummy warmup to compile JIT safely
        print(f"[TRAIN] JIT warming fit_moments with dummy data...")
        dummy_y = jnp.zeros((16, D))
        dummy_y  = jax.device_put(dummy_y, distributed)
        dummy_A = inox.tree.Partial(measure, 
                                    jax.device_put(jnp.zeros((16, H, W, C)), distributed), 
                                    H=H, W=W, C=C)
        _ = run_fit_moments(sde, dummy_A, dummy_y, key)
        print(f"[TRAIN] Warmup completed ✔️")
        """
        print(f"[TRAIN] Starting moment fitting with real data..")
        
        # Now run actual fit_moments
        start_time = time.time()
        rng, fit_key = jax.random.split(rng)
        print(f"[TRAIN] Taking 6144 samples for moment fitting")
        y_fit, A_fit = shuffled_trainset[:6000]['y'], shuffled_trainset[:6000]['A']
        y_fit, A_fit = jax.device_put((y_fit, A_fit), distributed)
        B, H, W, C = y_fit.shape
        print(f"[TRAIN] Moment fitting data shape: B={B}, H={H}, W={W}, C={C}, D={D}")
        mu_x, cov_x = run_fit_moments(sde, inox.tree.Partial(measure, A_fit, H=H, W=W, C=C), flatten(y_fit), key)
        print(f"[TRAIN] Moment fitting completed in {time.time() - start_time:.2f}s")
        del y_fit, A_fit
        previous = GaussianDenoiser(mu_x, cov_x)
        log_memory_usage("after_moment_fitting")

    # Prepare the previous model for sampling new training targets
    print(f"[TRAIN] Preparing previous model for sampling")
    static, arrays = previous.partition()
    arrays = jax.device_put(arrays, replicated)
    previous = static(arrays)

    # Generate synthetic training and testing data (denoised reconstructions)
    print(f"[TRAIN] Generating synthetic training data")
    trainset = generate(
        model=previous,
        dataset=trainset_yA,
        rng=rng,
        batch_size=config.batch_size,
        chunk_size=config.batch_size * 4,  # Process 4 batches per chunk
        shard=True,
        sampler=config.sampler,
        sde=sde,
        steps=config.discrete,
        maxiter=config.maxiter,
    )
    
    print(f"[TRAIN] Generating synthetic test data")
    testset = generate(
        model=previous,
        dataset=testset_yA,
        rng=rng,
        batch_size=config.batch_size,
        chunk_size=config.batch_size * 2,  # Smaller chunks for test set
        shard=True,
        sampler=config.sampler,
        sde=sde,
        steps=config.discrete,
        maxiter=config.maxiter,
    )
    
    log_memory_usage("after_generation")

    # Fit low-rank covariance (PPCA) on generated training data
    print(f"[TRAIN] Fitting PPCA on generated training data")
    x_fit = trainset[:16384]['x']
    x_fit = flatten(x_fit)
    mu_x, cov_x = ppca(x_fit, rank=320, key=key)
    del x_fit
    log_memory_usage("after_ppca")

    # Initialize model
    if lap > 0:
        print(f"[TRAIN] Using previous model from lap {lap-1}")
        model = previous
    else:
        print(f"[TRAIN] Creating new model")
        model = make_model(key=key, in_channels=C, out_channels=C, **CONFIG)
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
    print(f"[TRAIN] Using heuristic: {config.heuristic}")

    model.train(True)

    # Partition model parameters
    print(f"[TRAIN] Partitioning model parameters")
    static, params, others = model.partition(nn.Parameter)

    # Define denoising loss
    objective = DenoiserLoss(sde=sde)

    # Build optimizer
    steps = config.epochs * len(trainset_yA) // config.batch_size
    print(f"[TRAIN] Total training steps: {steps}")
    optimizer = Adam(steps=steps, **config)
    opt_state = optimizer.init(params)

    # Exponential moving average for parameter stabilization
    ema = EMA(decay=config.ema_decay)
    avrg = params

    # Put everything onto devices
    print(f"[TRAIN] Moving parameters to devices")
    avrg, params, others, opt_state = jax.device_put((avrg, params, others, opt_state), replicated)
    log_memory_usage("after_device_put")

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

    print(f"[TRAIN] Starting training loop for {config.epochs} epochs")
    
    # Training loop over epochs
    for epoch in (bar := trange(config.epochs, ncols=88)):
        epoch_start_time = time.time()
        print(f"[TRAIN] Starting epoch {epoch+1}/{config.epochs}")
        log_memory_usage(f"epoch_{epoch}_start")
        
        # Shuffle training set per epoch
        print(f"[TRAIN] Shuffling training set for epoch {epoch}")
        shuffle_start_time = time.time()
        loader = trainset.shuffle(seed=seed + lap * config.epochs + epoch).iter(
            batch_size=config.batch_size, drop_last_batch=True
        )
        print(f"[TRAIN] Shuffle completed in {time.time() - shuffle_start_time:.2f}s")

        losses = []
        batch_count = 0
        
        print(f"[TRAIN] Starting batch processing for epoch {epoch}")
        batch_loop_start = time.time()
        
        try:
            for batch in prefetch(loader):
                batch_count += 1
                batch_start_time = time.time()
                
                if batch_count % 10 == 0:  # Log every 10 batches
                    print(f"[TRAIN] Processing batch {batch_count} in epoch {epoch}")
                    log_memory_usage(f"epoch_{epoch}_batch_{batch_count}")
                    log_jax_memory()
                
                # Use timeout handler for batch processing to detect hangs
                with TimeoutHandler(300):  # 5 minute timeout per batch
                    x = batch['x']
                    device_put_time = time.time()
                    x = jax.device_put(x, distributed)
                    
                    if batch_count % 10 == 0:
                        print(f"[TRAIN] Device put completed in {time.time() - device_put_time:.3f}s")
                    
                    #x = augment(x, rng.split(len(x)))
                    flatten_time = time.time()
                    x = flatten(x)
                    
                    sgd_time = time.time()
                    loss, avrg, params, opt_state = sgd_step(avrg, params, others, opt_state, x, key=key)
                    
                    if batch_count % 10 == 0:
                        print(f"[TRAIN] SGD step completed in {time.time() - sgd_time:.3f}s")
                        print(f"[TRAIN] Batch {batch_count} total time: {time.time() - batch_start_time:.3f}s")
                
                losses.append(loss)
                
                # Check for potential hang conditions
                batch_time = time.time() - batch_start_time
                if batch_time > 120:  # 2 minutes per batch is concerning
                    print(f"[TRAIN] WARNING: Batch {batch_count} took {batch_time:.1f}s")
                    log_jax_memory()
                    
        except TimeoutError as e:
            print(f"[TRAIN] Batch processing timed out at batch {batch_count}: {e}")
            log_memory_usage("timeout")
            log_jax_memory()
            raise
        
        print(f"[TRAIN] Epoch {epoch} batch processing completed in {time.time() - batch_loop_start:.2f}s")
        loss_train = np.stack(losses).mean()

        # Validation evaluation
        print(f"[TRAIN] Starting validation for epoch {epoch}")
        val_start_time = time.time()
        loader = testset.iter(batch_size=config.batch_size, drop_last_batch=True)
        losses = []
        val_batch_count = 0
        
        try:
            with TimeoutHandler(600):  # 10 minute timeout for entire validation
                for batch in prefetch(loader):
                    val_batch_count += 1
                    if val_batch_count % 5 == 0:
                        print(f"[TRAIN] Processing validation batch {val_batch_count}")
                    
                    x = batch['x']
                    x = jax.device_put(x, distributed)
                    x = flatten(x)
                    loss = ell(avrg, others, x, key=key)
                    losses.append(loss)
                    
        except TimeoutError as e:
            print(f"[TRAIN] Validation timed out at batch {val_batch_count}: {e}")
            log_memory_usage("validation_timeout")
            log_jax_memory()
            raise
            
        loss_val = np.stack(losses).mean()
        print(f"[TRAIN] Validation completed in {time.time() - val_start_time:.2f}s")
        
        bar.set_postfix(loss=loss_train, loss_val=loss_val)

        # Every 16 epochs, sample validation images and log to wandb
        if (epoch + 1) % 16 == 0:
            print(f"[TRAIN] Generating samples for epoch {epoch}")
            sample_start_time = time.time()
            
            model = static(avrg, others)
            model.train(False)
            x = sample(
                model=model,
                y=y_eval,
                A=A_eval,
                key=key,
                shard=True,
                sampler=config.sampler,
                steps=config.discrete,
                maxiter=config.maxiter,
            )
            num = x.shape[0]
            cols = int(np.sqrt(num))
            rows = num // cols
            x = x.reshape(rows, cols, H, W, C)
            
            print(f"[TRAIN] Sampling completed in {time.time() - sample_start_time:.2f}s")
            
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

        epoch_time = time.time() - epoch_start_time
        print(f"[TRAIN] Epoch {epoch+1} completed in {epoch_time:.2f}s")
        log_memory_usage(f"epoch_{epoch}_end")
        
        # Force garbage collection every few epochs
        if (epoch + 1) % 4 == 0:
            print(f"[TRAIN] Running garbage collection after epoch {epoch+1}")
            gc.collect()

    # Save checkpoint
    print(f"[TRAIN] Saving checkpoint for lap {lap}")
    model = static(avrg, others)
    model.train(False)
    dump_module(model, runpath / f'checkpoint_{lap}.pkl')
    print(f"[TRAIN] Training lap {lap} completed successfully")
    log_memory_usage("train_end")


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
                ram='128GB',
                time='1-00:00:00',
                partition='h200',
                wrap='\"hostname && sleep infinity\"'
           )
        )
        if len(jobs) > 1:
            jobs[-1].after(jobs[-2], status="success")

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
