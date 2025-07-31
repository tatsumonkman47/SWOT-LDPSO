import numpy as np
import xarray as xr
import zarr
from datetime import date, timedelta
import os
from functools import partial
import dask
dask.config.set(scheduler='synchronous')

SWOT_DL_SRC_PATH = '/home/tm3076/projects/NYU_SWOT_project/Inpainting_Pytorch_gen/SWOT-inpainting-DL'
import sys
if os.path.exists('/home/tm3076/projects/NYU_SWOT_project/'):
    sys.path.append('/home/tm3076/projects/NYU_SWOT_project/Inpainting_Pytorch_gen/SWOT-inpainting-DL/src')
    sys.path.append('/home/tm3076/projects/NYU_SWOT_project/SWOT-data-analysis/src')
    SWOT_DL_SRC_PATH = '/home/tm3076/projects/NYU_SWOT_project/Inpainting_Pytorch_gen/SWOT-inpainting-DL'
else: 
    sys.path.append('/home.ufs/tm3076/swot_SUM03/SWOT_project/SWOT-inpainting-DL/src')
    SWOT_DL_SRC_PATH = '/home.ufs/tm3076/swot_SUM03/SWOT_project/SWOT-inpainting-DL'
import interp_utils
# This assumes it's JAX-agnostic
import jax.numpy as jnp
import jax



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


class JAXLLC4320Dataset:
    def __init__(self, data_dir, mid_timestep, N_t, patch_coords, 
                 infields, in_mask_list, in_transform_list,
                 SST_quality_level=1, sst_only=False, sst_cloud_mask=False,
                 N=128, L_x=512e3, L_y=512e3, flatten=False, return_meta_data=True,
                 standards=None, apply_mask=True, cloud_rho=0.7, regrid_SWOT=False, 
                 shared_field_cache=None,**config):

        self.data_dir = data_dir
        self.mid_timestep = mid_timestep
        self.N_t = N_t
        self.patch_coords = patch_coords
        self.infields = infields
        self.in_mask_list = in_mask_list
        self.in_transform_list = in_transform_list
        self.SST_quality_level = SST_quality_level
        self.N = N
        self.L_x = L_x
        self.L_y = L_y
        self.flatten = flatten
        self.return_meta_data = return_meta_data
        self.apply_mask = apply_mask
        self.cloud_rho = cloud_rho
        self.regrid_SWOT = regrid_SWOT
        if standards is None:
            standards = {
                "mean_ssh": 0.0, "std_ssh": 1.0,
                "mean_sst": 0.0, "std_sst": 1.0,
                "extra_mean_tuning": 0.0
            }
        self.worker_generic_swath0, self.worker_generic_swath1 = config["preloaded_worker_generic_swaths"]
        self.cloud_catalog_rho = config["preloaded_cloud_catalog_rho"]
        self.SST_mean_climatology = config["preloaded_climatology"]
        self.transforms = self._create_transforms(standards)
        # Lazy load SWOT swaths or numpy mask catalog
        if self.regrid_SWOT:
            self.swot_ds = [
                xr.open_zarr(fs.get_mapper(f"{self.data_dir}/SWOT_swaths_488/hawaii_c488_p015.zarr")),
                xr.open_zarr(fs.get_mapper(f"{self.data_dir}/SWOT_swaths_488/hawaii_c488_p028.zarr"))
                ]
        else:
            self.swot_npy = config["preloaded_swot_npy"]

        if not shared_field_cache: 
            self._field_cache = {}
            for fld in self.infields:
                ds = xr.open_zarr(f"{self.data_dir}/{fld}_allpatches.zarr")
                self._field_cache[fld] = ds
        else:
            self._field_cache = shared_field_cache
                
    def _create_transforms(self, standards):
        def jaxify_input(x):
            if isinstance(x, xr.DataArray):
                return jnp.asarray(x.values, dtype=jnp.float32)
            return jnp.asarray(x, dtype=jnp.float32) if isinstance(x, np.ndarray) else x
        def make_standardize(mean=None, std=1.0):
            if mean is not None:
                @jax.jit
                def fn(x):
                    x = jaxify_input(x)
                    return (x - mean) / std
            else:
                @jax.jit
                def fn(x):
                    x = jaxify_input(x)
                    return x / std
            return fn
        def make_std_samplewise(std=1.0):
            @jax.jit
            def fn(x):
                x = jaxify_input(x)
                return (x - jnp.mean(x)) / std
            return fn
        def make_seasonal_standardize(SST_mean_climatology, N_t, mid_timestep, std=5.0, extra_mean_tuning=0):
            clim_vals = jaxify_input(SST_mean_climatology.SST)#.values.astype(np.float32)
            clim_len = clim_vals.shape[0]
            # Precompute clipped indices and broadcasted climatology array
            time_start = mid_timestep - N_t // 2
            time_end = mid_timestep + N_t // 2 + N_t % 2
            time_indices = np.arange(time_start, time_end)
            clipped_indices = np.clip(time_indices, 0, clim_len - 1)
            clim_means = clim_vals[clipped_indices]  # shape: (T,)
            clim_means_broadcast = jnp.asarray(clim_means[:, None, None], dtype=jnp.float32)
            @jax.jit
            def transform(arr):
                """Assume arr is shape (T, H, W) or (H, W)"""
                arr = jnp.asarray(arr, dtype=jnp.float32)
                if arr.ndim == 3:
                    return (arr - clim_means_broadcast - extra_mean_tuning) / std
                elif arr.ndim == 2:
                    clim_val = clim_means[N_t // 2] if len(clim_means) > N_t // 2 else clim_means[0]
                    return (arr - clim_val - extra_mean_tuning) / std
                else:
                    return (arr - clim_means.mean() - extra_mean_tuning) / std
            return transform
        return {
            "std_ssh_norm": make_standardize(std=standards["std_ssh"]),
            "std_sst_norm": make_standardize(std=standards["std_sst"]),
            "std_mean_ssh_norm": make_std_samplewise(std=standards["std_ssh"]),
            "std_mean_sst_norm": make_std_samplewise(std=standards["std_sst"]),
            "std_global_mean_ssh_norm": make_standardize(mean=standards["mean_ssh"], std=standards["std_ssh"]),
            "std_global_mean_sst_norm": make_standardize(mean=standards["mean_sst"], std=standards["std_sst"]),
            "std_seasonal_mean_sst_norm": make_seasonal_standardize(self.SST_mean_climatology, self.N_t, self.mid_timestep, std=standards["std_sst"], extra_mean_tuning=standards["extra_mean_tuning"]),
            "no_transform": lambda x: jnp.array(x.values.astype(np.float32)) if isinstance(x, xr.DataArray) else jnp.array(x.astype(np.float32))
            }
    
    def __len__(self):
        return self.patch_coords.shape[0]

    def __getitem__(self, idx):
        try:
            return self._load_patch(idx)
        except Exception as e:
            print(f"[Warning] Failed to load patch {str(int(self.patch_coords[idx, 2])).zfill(3)}: {e} — falling back to patch 065")
            return self._load_patch(patch_id="065")

    def _load_patch(self, idx=None, patch_id=None):
        meta = {"patch_ID": pid, "patch_coords": coords, "mid_timestep": self.mid_timestep}
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
        if return_metadata:
            return invar, mask, meta
        return invar, mask
        

    def _load_patch_fields(self, patch_id, fields, transform_keys, mask_keys, rng):
        vars, masks = [], []
        for fld, tk, mask_key in zip(fields, transform_keys, mask_keys):
            # Use cached zarr opening
            ds = self._field_cache[fld].loc[{"patch":int(patch_id)}]
            d = ds.isel(time=slice(self.mid_timestep - self.N_t//2,
                                  self.mid_timestep + self.N_t//2 + self.N_t%2))
            if isinstance(d, xr.Dataset):
                d = next(iter(d.data_vars.values()))
            if isinstance(d.data, dask.array.Array):
                d_np = d.data.compute()
            else:
                d_np = np.asarray(d.data)
            # Apply transform directly
            ten = self.transforms[tk](d_np)
            mask = self._get_mask(mask_key, patch_id, rng)
            vars.append(ten * mask)
            masks.append(mask)
        return vars, masks

    def _get_mask(self, mask_key, patch_ID, rng):
        if (mask_key is None) or ("None" in mask_key):
            return jnp.array(shape, dtype=jnp.float32)
        elif "swot" in mask_key.lower():
            sampling = "all"
            version = "random"
            if "calval" in mask_key.lower():
                version = "calval"
            if "central" in mask_key.lower():
                sampling = "central"
            if "random" in mask_key.lower():
                sampling = "random"
            if "nadir" in mask_key.lower():
                result = (self._get_random_swot_mask(patch_ID, version, sampling, rng) + 
                         self._get_nadir_mask(patch_ID, rng)) > 0
            else:
                result = self._get_random_swot_mask(patch_ID, version, sampling, rng)
        elif "nadir" in mask_key.lower():
            result = self._get_nadir_mask(patch_ID, rng)
        elif "cloud_rho" in mask_key.lower():
            result = self._get_cloud_mask_rho(rng)
        else:
            raise ValueError(f"Unknown mask type: {mask_key}")
        #print(f"{mask_key} shape: {result.shape}")
        return result

    def _get_random_swot_mask(self, patch_ID, version, sampling, rng):
        """Optimized SWOT mask generation"""
        if self.regrid_SWOT:
            sw_corner, ne_corner = [-154.5, 35.3], [-147.5, 42.3]
            lat_max, lat_min, l_step, lon_i = 9000, 2000, 4, rng.randint(5)
            lon = rng.uniform(sw_corner[0], ne_corner[0])
            lat = rng.uniform(sw_corner[1], ne_corner[1])
            ds = rng.choice(_thread_local.swot_ds)
            m0 = interp_utils.grid_everything(
                self.swot_ds[0].ssha, lat=lat, lon=lon,
                n=self.N, L_x=self.L_x, L_y=self.L_y).values
            m1 = interp_utils.grid_everything(
                self.swot_ds[1].ssha, lat=lat, lon=lon,
                n=self.N, L_x=self.L_x, L_y=self.L_y).values
            m01 = np.stack([m0, m1])
        else:
            i_rand = rng.randint(64, 225-64)
            j_rand = rng.randint(128, 800-64)
            m01 = self.swot_npy[:, j_rand-64:j_rand+64, i_rand-64:i_rand+64]
        if rng.randint(2) < 1:
            m01 = m01[::-1, ...]
        if sampling == "central":
            mask = np.zeros([self.N_t] + list(m01.shape)[-2:], dtype=np.float32)
            mask[self.N_t//2, :, :] = m01[0]
        elif sampling == "all":
            if self.N_t > 1:
                mask_broadcast = np.broadcast_to(m01, (self.N_t//2 + self.N_t%2, 2, 128, 128))
                mask = mask_broadcast.reshape(self.N_t + self.N_t%2, 128, 128)[:self.N_t]
            else:
                mask = m01[rng.randint(2)]
                mask = mask.astype(np.float32)
        #print(f"SWOT mask shape {mask.shape}")
        return jnp.array(mask, dtype=jnp.float32)

    def _get_nadir_mask(self, patch_ID, rng, version="random", sample_time="1D",):
        """Optimized nadir mask generation"""
        try:
            rand_index = rng.randint(422)
            path = f"{self.data_dir}/copernicus_nadir_SSH_daily/{rand_index:03}.zarr"
            random_tile = xr.open_zarr(path).sla_filtered
        except Exception as e:
            fallback_path = f"{self.data_dir}/copernicus_nadir_SSH_daily/002.zarr"
            random_tile = xr.open_zarr(fallback_path).sla_filtered
        # Temporal downsampling + slicing
        time_len = len(random_tile.time)
        mid = rng.randint(self.N_t // 2, time_len - self.N_t // 2)
        sliced = random_tile.isel(time=slice(mid - self.N_t//2, mid + self.N_t//2 + self.N_t%2))
        if self.N_t <= 1:
            sliced = sliced.squeeze()
        # Optimized mask computation
        mask_values = np.where(sliced.values > 0, 1.0, 0.0).astype(np.float32)
        #print(f"Nadir mask shape {mask_values.shape}")
        return jnp.array(mask_values, dtype=jnp.float32)
    
    def _get_cloud_mask_timeseries(self, rng, patch_ID):
        path = f"{self.data_dir}/HRS_SST_tiles/agg_cloud_masks_zarr/{patch_ID}.zarr"
        cm = xr.open_dataset(path).sst_filtered_q5
        mid = rng.randint(int(self.N_t / 2), len(cm.time) - int(self.N_t / 2))
        cm = cm.isel(time=slice(mid - self.N_t // 2, mid + self.N_t // 2 + self.N_t%2))
        cm = (cm * 0 + 1).where(cm > 0, other=0)
        #print(f"Cloud mask shape {cm.shape}")
        return jnp.array(cm, dtype=jnp.float32)

    def _get_cloud_mask_rho(self, rng):
        sample_N = self.cloud_catalog_rho.isel(i_time=rng.randint(len(self.cloud_catalog_rho.i_time)))
        sample_N_tstep = int(sample_N.patch_timestep)
        patch_id = str(int(sample_N.patch_id)).zfill(3)
        path = f"{self.data_dir}/HRS_SST_tiles/agg_cloud_masks_zarr/{patch_id}.zarr"
        cm = ~np.isnan(xr.open_dataset(path).isel(time=sample_N_tstep).sst_filtered_q5)
        k_np_rot = rng.randint(4)
        #print(f"Cloud mask shape {cm.shape}")
        return jnp.array(np.rot90(cm.values, k_np_rot), dtype=jnp.float32)


def JAXLLC4320_HFformated_dataset(patch_coords, t_range, split_fractions, config, seed=None):
    """
    Create a wrapped dataset compatible with the HuggingFace-style API,
    with maximum preloading of Zarr/Numpy data for performance.
    """
    if isinstance(patch_coords, str):
        patch_coords = np.load(patch_coords)
    print(f"Preloading fields: {config['infields']}")
    dask.config.set(scheduler="synchronous")  # enable parallel persist
    # 1. Field-level .zarr store
    shared_field_cache = {
        fld: xr.open_zarr(f"{config['data_dir']}/{fld}_allpatches.zarr").persist()
        for fld in config["infields"]
    }
    config["shared_field_cache"] = shared_field_cache
    # 2. SWOT swaths (used for _get_random_swot_mask)
    print("Preloading SWOT swath tiles...")
    swath0 = xr.open_zarr(f"{config['data_dir']}/SWOT_swaths_488/hawaii_c488_p015.zarr").persist()
    swath1 = xr.open_zarr(f"{config['data_dir']}/SWOT_swaths_488/hawaii_c488_p028.zarr").persist()
    config["preloaded_swot_npy"] = np.load(f"{config['data_dir']}/swot_npy_mask_4km.npy", mmap_mode="r") * 1  # already optimized
    config["preloaded_worker_generic_swaths"] = (swath0, swath1)
    # 3. Cloud catalog
    print("Loading cloud catalog...")
    cloud_catalog = xr.open_zarr(f"{config['data_dir']}/catalog.zarr").compute()
    cloud_catalog = cloud_catalog.where(cloud_catalog.rho >= config["cloud_rho"], drop=True)
    config["preloaded_cloud_catalog_rho"] = cloud_catalog
    # 4. SST climatology (seasonal normalization)
    print("Loading climatology...")
    climatology = xr.open_dataset(f"{SWOT_DL_SRC_PATH}/data/SST_NP_daily_climatology.nc").compute()
    config["preloaded_climatology"] = climatology
    
    print("Preloading complete. Instantiating datasets...")
    dataset_list = [
        JAXLLC4320Dataset(
            patch_coords=patch_coords,
            mid_timestep=mid_timestep,
            **config
        )
        for mid_timestep in t_range
    ]
    concatenated = ConcatDatasetTime(dataset_list)
    return Hugging_face_wrapper(concatenated, split_fractions=split_fractions, seed=seed)        
