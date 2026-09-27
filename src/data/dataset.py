"""
Weather dataset abstraction for spatio-temporal modeling and sample extraction.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import xarray as xr


class WeatherDataset:
    """
    Dataset wrapper providing windowed spatio-temporal sample indexing from an xarray Dataset or Zarr store.
    """

    def __init__(
        self,
        ds: xr.Dataset,
        variables: Optional[List[str]] = None,
        input_timesteps: int = 1,
        target_timesteps: int = 1,
        lead_time_steps: int = 1,
    ):
        """
        Args:
            ds: Standardized xarray.Dataset.
            variables: List of variables to include in feature tensors.
            input_timesteps: Number of historical time steps used as input.
            target_timesteps: Number of forecast/target time steps to predict.
            lead_time_steps: Offset steps between input and target.
        """
        self.ds = ds
        self.variables = variables or list(ds.data_vars.keys())
        self.input_timesteps = input_timesteps
        self.target_timesteps = target_timesteps
        self.lead_time_steps = lead_time_steps

        self.times = ds["time"].values
        self.num_times = len(self.times)

        # Calculate number of valid window samples
        self.window_len = self.input_timesteps + self.lead_time_steps + self.target_timesteps - 1
        self.valid_samples = max(0, self.num_times - self.window_len + 1)

    def __len__(self) -> int:
        return self.valid_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Retrieve a single sample with input fields and target fields as numpy arrays.

        Returns:
            Dict containing:
                - 'input': numpy array of shape [input_timesteps, num_vars, lat, lon]
                - 'target': numpy array of shape [target_timesteps, num_vars, lat, lon]
                - 'input_times': array of datetime stamps for inputs
                - 'target_times': array of datetime stamps for targets
                - 'variables': list of variable names
        """
        if idx < 0 or idx >= self.valid_samples:
            raise IndexError(f"Index {idx} out of range for dataset with {self.valid_samples} samples.")

        in_start = idx
        in_end = in_start + self.input_timesteps

        target_start = in_end + self.lead_time_steps - 1
        target_end = target_start + self.target_timesteps

        # Extract variables
        input_arrays = []
        target_arrays = []

        for var in self.variables:
            in_slice = self.ds[var].isel(time=slice(in_start, in_end)).values
            tgt_slice = self.ds[var].isel(time=slice(target_start, target_end)).values

            input_arrays.append(in_slice)
            target_arrays.append(tgt_slice)

        # Stack into [timesteps, variables, lat, lon]
        # in_slice shape: (input_timesteps, lat, lon) -> stack vars on axis 1
        in_data = np.stack(input_arrays, axis=1)
        tgt_data = np.stack(target_arrays, axis=1)

        return {
            "input": in_data,
            "target": tgt_data,
            "input_times": self.times[in_start:in_end],
            "target_times": self.times[target_start:target_end],
            "variables": self.variables,
        }
