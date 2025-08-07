import numpy as np
import xarray as xr
import zarr
from datetime import date, timedelta
import os
from functools import partial
import dask
dask.config.set(scheduler='synchronous')
import sys
if os.path.exists('/home.ufs/tm3076/swot_SUM03/SWOT_project/SWOT-data-analysis/src'):
    sys.path.append('/home.ufs/tm3076/swot_SUM03/SWOT_project/SWOT-data-analysis/src')
else: 
    sys.path.append('/home/tm3076/projects/NYU_SWOT_project/SWOT-data-analysis/src')
import interp_utils  # This assumes it's JAX-agnostic
import jax.numpy as jnp



class ConcatDatasetTime:
    """
    Concat my datasets along all of the time steps I want
    """
    def __init__(self, datasets):
        self.datasets = datasets
        self.cumulative_sizes = np.cumsum([len(ds) for ds in datasets])
    def __len__(self):
        return self.cumulative_sizes[-1]
    def __getitem__(self, idx):
        # Find which sub-dataset this index falls into
        dataset_idx = np.searchsorted(self.cumulative_sizes, idx, side='right')
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

class Hugging_face_wrapper:
    """
    A wrapper to make sure my dataset behaves similarly to a HuggingFace dataset,
    and supports splitting into train/val/test subsets.
    """
    def __init__(self, dataset, split_fractions=None, seed=None, indices=None):
        self.dataset = dataset
        # If indices are provided, this is a subset (train/val/test)
        if indices is not None:
            self.indices = np.array(indices)
            self.subsets = None
            return
        # Otherwise, this is the main wrapper, and we split here
        N = len(dataset)
        all_indices = np.arange(N)
        if split_fractions is None:
            split_fractions = {"train": 0.8, "val": 0.1, "test": 0.1}
        if abs(sum(split_fractions.values()) - 1.0) > 1e-5:
            raise ValueError("split_fractions must sum to 1.0")
        # Shuffle indices reproducibly
        rng = np.random.RandomState(seed)
        rng.shuffle(all_indices)
        n_train = int(N * split_fractions["train"])
        n_val = int(N * split_fractions["val"])
        train_idx = all_indices[:n_train]
        val_idx = all_indices[n_train:n_train + n_val]
        test_idx = all_indices[n_train + n_val:]
        # Store subsets as Hugging_face_wrapper instances
        self.subsets = {
            "train": Hugging_face_wrapper(dataset, indices=train_idx),
            "val": Hugging_face_wrapper(dataset, indices=val_idx),
            "test": Hugging_face_wrapper(dataset, indices=test_idx)
        }
        
    def __getitem__(self, idx):
        # If indexing by split name
        if self.subsets and isinstance(idx, str):
            if idx not in self.subsets:
                raise KeyError(f"Invalid split '{idx}'. Must be one of {list(self.subsets.keys())}")
            return self.subsets[idx]
        # Handle batching
        if isinstance(idx, slice) or isinstance(idx, np.ndarray) or isinstance(idx, list):
            if isinstance(idx, slice):
                idx = np.arange(len(self))[idx]
            items = [self[i] for i in idx]
            y = np.stack([x['y'] for x in items])
            A = np.stack([x['A'] for x in items])
            return {'y': y, 'A': A}
        else:
            true_idx = self.indices[idx] if hasattr(self, "indices") else idx
            invar, mask = self.dataset[true_idx]
            return {'y': invar, 'A': mask}
            
    def __len__(self):
        return len(self.indices) if hasattr(self, "indices") else len(self.dataset)
        
    def shuffle(self, seed=None):
        indices = np.arange(len(self))
        rng = np.random.RandomState(seed)
        rng.shuffle(indices)
        #self.shuffled_indices = indices
        self.indices = indices
        return self
        
    def iter(self, batch_size, drop_last_batch=True):
        indices = getattr(self, 'indices', np.arange(len(self)))
        num_batches = len(indices) // batch_size
        for i in range(num_batches):
            idx = indices[i * batch_size:(i + 1) * batch_size]
            batch = [self[k] for k in idx]
            y_batch = np.stack([b['y'] for b in batch])
            A_batch = np.stack([b['A'] for b in batch])
            yield {'y': y_batch, 'A': A_batch}


def standardize(x, mean=None, std=1.0):
    if mean is not None:
        x = x - mean
    return x / std

def standardize_samplewise(x, std=1.0):
    return (x - np.mean(x)) / std

def no_transform(x):
    return x


class JAXLLC4320Dataset:
    def __init__(self, data_dir, mid_timestep, N_t, patch_coords, 
                 infields, in_mask_list, in_transform_list,
                 SST_quality_level=1, sst_only=False, sst_cloud_mask=False,
                 N=128, L_x=512e3, L_y=512e3, flatten=False, return_meta_data=True,
                 standards=None, apply_mask=True, cloud_rho=0.7):

        self.data_dir = data_dir
        self.mid_timestep = mid_timestep # I use mid_timestep to sample from the timeseries
        self.N_t = N_t
        self.patch_coords = patch_coords
        self.infields = infields
        self.in_mask_list = in_mask_list
        self.apply_mask = apply_mask
        self.in_transform_list = in_transform_list
        self.SST_quality_level = SST_quality_level
        self.N = N
        self.L_x = L_x
        self.L_y = L_y
        self.flatten = flatten
        self.return_meta_data = return_meta_data
        self.cloud_rho = cloud_rho
        if standards is None:
            standards = {
                "mean_ssh": 0.0, "std_ssh": 1.0,
                "mean_sst": 0.0, "std_sst": 1.0
            }
        self.transforms = {
            "std_ssh_norm": partial(standardize, std=standards["std_ssh"]),
            "std_sst_norm": partial(standardize, std=standards["std_sst"]),
            "std_mean_ssh_norm": partial(standardize_samplewise, std=standards["std_ssh"]),
            "std_mean_sst_norm": partial(standardize_samplewise, std=standards["std_sst"]),
            "std_global_mean_ssh_norm": partial(standardize, mean=standards["mean_ssh"], std=standards["std_ssh"]),
            "std_global_mean_sst_norm": partial(standardize, mean=standards["mean_sst"], std=standards["std_sst"]),
            "no_transform": no_transform,
        }
        self.worker_generic_swath0 = xr.open_zarr(f"{self.data_dir}/SWOT_swaths_488/hawaii_c488_p015.zarr")
        self.worker_generic_swath1 = xr.open_zarr(f"{self.data_dir}/SWOT_swaths_488/hawaii_c488_p028.zarr")
        self.cloud_catalog = xr.open_zarr(f"{self.data_dir}/HRS_SST_tiles/agg_cloud_percentages/catalog.zarr").compute()

    def __len__(self):
        return self.patch_coords.shape[0]

    def __getitem__(self, idx):
        try:
            return self._load_patch(idx)
        except Exception as e:
            print(f"[Warning] Failed to load patch {str(int(self.patch_coords[idx, 2])).zfill(3)}: {e} — falling back to patch 065")
            return self._load_patch(patch_id="065")

    def _load_patch(self, idx=None, patch_id=None):
        if patch_id is None:
            patch_id = str(int(self.patch_coords[idx, 2])).zfill(3)
            coords = self.patch_coords[idx]
        else:
            coords = None
        if idx is not None: # Create a deterministic RNG using idx
            rng = np.random.RandomState(idx)
        else:
            rng = np.random.RandomState(42)  # Fallback for manual patch_id
        invars, masks = self._load_patch_fields(patch_id, self.infields, self.in_transform_list, self.in_mask_list, rng)
        invar = np.stack(invars, axis=-1)
        mask = np.stack(masks, axis=-1)
        if self.flatten:
            invar = invar.reshape(-1)
            mask = mask.reshape(-1)
        if self.return_meta_data:
            return invar, mask, {
                "patch_ID": patch_id,
                "mid_timestep": self.mid_timestep,
                "patch_coords": coords,
                "latitude": self.latitude,
                "longitude": self.longitude
            }
        if self.N_t == 1:
            invar = invar[0]
        return invar, mask

    def _load_patch_fields(self, patch_id, fields, transform_keys, mask_keys, rng):
        variables = []
        masks = []
        for i, field in enumerate(fields):
            ds = xr.open_zarr(f"{self.data_dir}/{field}/{patch_id}.zarr").isel(
                time=slice(int(self.mid_timestep - self.N_t / 2), int(self.mid_timestep + self.N_t / 2))
            )
            self.latitude = ds.latitude.values
            self.longitude = ds.longitude.values
            var = ds[list(ds.data_vars.keys())[0]]
            var = self.transforms[transform_keys[i]](var)
            # Important: pass rng here
            mask = self.get_mask(mask_keys[i], patch_id, rng)
            if self.apply_mask:
                variables.append(var.values * mask)
            else:
                variables.append(var.values)
            masks.append(mask)
        return variables, masks

    def get_mask(self, mask_key, patch_ID, rng):
        if mask_key is None or "None" in str(mask_key):
            return 1.0
        elif "swot" in str(mask_key).lower():
            return self.get_random_swot_mask(rng)
        elif "cloud_tseries" in str(mask_key).lower():
            return self.get_cloud_mask_timeseries(rng, patch_ID)
        elif "cloud_rho" in str(mask_key).lower():
            return self.get_cloud_mask_rho(rng)
        else:
            raise ValueError(f"Unknown mask type: {mask_key}")

    def get_random_swot_mask(self, rng, version="random"):
        sw_corner = [-153, 29.0]
        ne_corner = [-148, 43.0]
        lon = rng.randint(sw_corner[0], ne_corner[0])
        lat = rng.randint(sw_corner[1], ne_corner[1])
        if version == "random":
            if rng.randint(2) == 0:
                m0 = interp_utils.grid_everything(self.worker_generic_swath0, lat, lon, n=self.N, L_x=self.L_x, L_y=self.L_y)
            else:
                m0 = interp_utils.grid_everything(self.worker_generic_swath1, lat, lon, n=self.N, L_x=self.L_x, L_y=self.L_y)
            mask = (m0.ssha.fillna(0)).values > 0
        elif version == "both":
            m0 = interp_utils.grid_everything(self.worker_generic_swath0, lat, lon, n=self.N, L_x=self.L_x, L_y=self.L_y)
            m1 = interp_utils.grid_everything(self.worker_generic_swath1, lat, lon, n=self.N, L_x=self.L_x, L_y=self.L_y)
            mask = (m0.ssha.fillna(0) + m1.ssha.fillna(0)).values > 0
        return mask.astype(np.float32)
    
    def get_cloud_mask_timeseries(self, rng, patch_ID):
        path = f"{self.data_dir}/HRS_SST_tiles/agg_cloud_masks/{patch_ID}.nc"
        cm = xr.open_dataset(path).sst_filtered_q5
        mid = rng.randint(int(self.N_t / 2), len(cm.time) - int(self.N_t / 2))
        cm = cm.isel(time=slice(mid - self.N_t // 2, mid + self.N_t // 2))
        cm = (cm * 0 + 1).where(cm > 0, other=0)
        return cm.values.astype(np.float32)

    def get_cloud_mask_rho(self, rng):
        cloud_catalog_rho = self.cloud_catalog.where(self.cloud_catalog.rho >= self.cloud_rho, drop=True)
        sample_N = cloud_catalog_rho.isel(i_time=rng.randint(len(cloud_catalog_rho.i_time)))
        sample_N_tstep = int(sample_N.patch_timestep)
        patch_id = str(int(sample_N.patch_id)).zfill(3)
        path = f"{self.data_dir}/HRS_SST_tiles/agg_cloud_masks/{patch_id}.nc"
        cm = ~np.isnan(xr.open_dataset(path).isel(time=sample_N_tstep).sst_filtered_q5)
        k_np_rot = rng.randint(4)
        return np.rot90(cm.values.astype(np.float32), k_np_rot)


def JAXLLC4320_HFformated_dataset(patch_coords, t_range, split_fractions, config, seed=None):
    """
    Create a wrapped dataset compatible with the HuggingFace-style API.
    """
    if isinstance(patch_coords, str):
        patch_coords = np.load(patch_coords)
    dataset_list = [
        JAXLLC4320Dataset(
            patch_coords=patch_coords,
            mid_timestep=mid_timestep,
            **config  # forward all config kwargs
        )
        for mid_timestep in t_range
    ]
    concatenated = ConcatDatasetTime(dataset_list)
    return Hugging_face_wrapper(concatenated,split_fractions=split_fractions, seed=seed)
        
