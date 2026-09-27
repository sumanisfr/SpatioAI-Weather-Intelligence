"""
Unit tests for SpatioAI Phase 1 Data Pipeline.
"""

from pathlib import Path
import numpy as np
import pytest
import xarray as xr

from src.data.base_loader import WeatherDataLoader
from src.data.dataset import WeatherDataset
from src.data.era5_loader import ERA5Loader
from src.data.nwp_loader import NWPLoader
from src.data.preprocessing import (
    convert_units,
    preprocess_dataset,
    standardize_coordinates,
    standardize_variables,
)
from src.data.quality import DatasetValidator, QualityControlError, validate_dataset
from src.data.regridding import Regridder
from src.data.synthetic import generate_synthetic_weather_dataset


@pytest.fixture
def sample_synthetic_dataset() -> xr.Dataset:
    """Fixture providing a realistic synthetic weather dataset."""
    return generate_synthetic_weather_dataset(
        start_date="2024-01-01",
        end_date="2024-01-03",
        freq="6h",
        lat_min=10.0,
        lat_max=25.0,
        lat_step=1.0,
        lon_min=75.0,
        lon_max=90.0,
        lon_step=1.0,
        seed=42,
    )


def test_synthetic_dataset(sample_synthetic_dataset: xr.Dataset):
    """Test synthetic dataset dimensions, coordinates, and variables."""
    ds = sample_synthetic_dataset
    assert "time" in ds.coords
    assert "latitude" in ds.coords
    assert "longitude" in ds.coords

    expected_vars = ["precipitation", "temperature", "u10", "v10", "surface_pressure"]
    for v in expected_vars:
        assert v in ds.data_vars
        assert ds[v].shape == (len(ds.time), len(ds.latitude), len(ds.longitude))
        assert not np.isnan(ds[v].values).any()


def test_coordinate_standardization():
    """Test renaming of non-standard coordinate names and sorting."""
    raw_ds = xr.Dataset(
        data_vars={"precipitation": (("time", "lat", "lon"), np.ones((2, 3, 4)))},
        coords={
            "time": ["2024-01-01", "2024-01-02"],
            "lat": [20.0, 10.0, 15.0],  # unsorted
            "lon": [80.0, 81.0, 82.0, 83.0],
        },
    )
    std_ds = standardize_coordinates(raw_ds)
    assert "latitude" in std_ds.coords
    assert "longitude" in std_ds.coords
    assert "lat" not in std_ds.coords
    # Verify latitude is sorted ascending
    assert np.all(np.diff(std_ds.latitude.values) > 0)


def test_variable_standardization():
    """Test mapping of ERA5 / model short-names to internal names."""
    raw_ds = xr.Dataset(
        data_vars={
            "tp": (("time", "latitude", "longitude"), np.zeros((1, 2, 2))),
            "2t": (("time", "latitude", "longitude"), np.full((1, 2, 2), 290.0)),
            "10u": (("time", "latitude", "longitude"), np.ones((1, 2, 2))),
            "10v": (("time", "latitude", "longitude"), np.ones((1, 2, 2))),
            "sp": (("time", "latitude", "longitude"), np.full((1, 2, 2), 1013.25)),
        },
        coords={
            "time": ["2024-01-01"],
            "latitude": [10.0, 11.0],
            "longitude": [70.0, 71.0],
        },
    )
    std_ds = standardize_variables(raw_ds)
    assert "precipitation" in std_ds.data_vars
    assert "temperature" in std_ds.data_vars
    assert "u10" in std_ds.data_vars
    assert "v10" in std_ds.data_vars
    assert "surface_pressure" in std_ds.data_vars


def test_unit_conversion():
    """Test conversion of physical units (e.g., m -> mm, Celsius -> K, Pa -> hPa)."""
    raw_ds = xr.Dataset(
        data_vars={
            "precipitation": (("time", "latitude", "longitude"), np.array([[[0.025]]])),
            "temperature": (("time", "latitude", "longitude"), np.array([[[25.0]]])),
            "surface_pressure": (("time", "latitude", "longitude"), np.array([[[101325.0]]])),
        },
        coords={"time": ["2024-01-01"], "latitude": [10.0], "longitude": [70.0]},
    )
    raw_ds["precipitation"].attrs["units"] = "m"
    raw_ds["temperature"].attrs["units"] = "degC"
    raw_ds["surface_pressure"].attrs["units"] = "Pa"

    converted = convert_units(raw_ds)
    assert np.isclose(converted["precipitation"].values[0, 0, 0], 25.0)
    assert converted["precipitation"].attrs["units"] == "mm"

    assert np.isclose(converted["temperature"].values[0, 0, 0], 298.15)
    assert converted["temperature"].attrs["units"] == "K"

    assert np.isclose(converted["surface_pressure"].values[0, 0, 0], 1013.25)
    assert converted["surface_pressure"].attrs["units"] == "hPa"


def test_quality_checks(sample_synthetic_dataset: xr.Dataset):
    """Test dataset validation passes on valid data and detects violations."""
    # 1. Valid dataset
    is_valid, errors = validate_dataset(sample_synthetic_dataset, raise_on_failure=False)
    assert is_valid
    assert len(errors) == 0

    # 2. Dataset with excessive NaNs
    corrupted_ds = sample_synthetic_dataset.copy(deep=True)
    corrupted_ds["precipitation"].values[:] = np.nan
    is_valid, errors = validate_dataset(corrupted_ds, raise_on_failure=False)
    assert not is_valid
    assert any("NaN" in err for err in errors)

    with pytest.raises(QualityControlError):
        validate_dataset(corrupted_ds, raise_on_failure=True)


def test_region_crop(sample_synthetic_dataset: xr.Dataset):
    """Test geographic bounding box cropping."""
    loader = ERA5Loader()
    cropped = loader.crop_region(
        sample_synthetic_dataset,
        lat_min=12.0,
        lat_max=20.0,
        lon_min=78.0,
        lon_max=85.0,
    )
    assert cropped.latitude.min().item() >= 12.0
    assert cropped.latitude.max().item() <= 20.0
    assert cropped.longitude.min().item() >= 78.0
    assert cropped.longitude.max().item() <= 85.0


def test_regridding(sample_synthetic_dataset: xr.Dataset):
    """Test spatial interpolation regridder."""
    regridder = Regridder(
        lat_bounds=(10.0, 25.0),
        lon_bounds=(75.0, 90.0),
        lat_step=0.25,
        lon_step=0.25,
        method="linear",
    )
    regridded = regridder.regrid(sample_synthetic_dataset)
    assert len(regridded.latitude) == 61
    assert len(regridded.longitude) == 61
    assert "precipitation" in regridded.data_vars


def test_zarr_write_read(sample_synthetic_dataset: xr.Dataset, tmp_path: Path):
    """Test writing xarray Dataset to Zarr and reopening lazily."""
    zarr_file = tmp_path / "test_weather.zarr"
    WeatherDataLoader.save_to_zarr(
        sample_synthetic_dataset,
        zarr_file,
        chunks={"time": 2, "latitude": 5, "longitude": 5},
    )
    assert zarr_file.exists()

    reopened_ds = WeatherDataLoader.load_from_zarr(zarr_file)
    assert "precipitation" in reopened_ds.data_vars
    assert len(reopened_ds.time) == len(sample_synthetic_dataset.time)
    assert np.allclose(
        reopened_ds["precipitation"].values,
        sample_synthetic_dataset["precipitation"].values,
    )


def test_weather_dataset_samples(sample_synthetic_dataset: xr.Dataset):
    """Test WeatherDataset windowed sample indexing."""
    dataset = WeatherDataset(
        ds=sample_synthetic_dataset,
        variables=["precipitation", "temperature"],
        input_timesteps=2,
        target_timesteps=1,
        lead_time_steps=1,
    )
    assert len(dataset) > 0
    sample = dataset[0]
    assert "input" in sample
    assert "target" in sample
    # Shape check: [timesteps, variables, lat, lon]
    assert sample["input"].shape[0] == 2
    assert sample["input"].shape[1] == 2
    assert sample["target"].shape[0] == 1
    assert sample["target"].shape[1] == 2
