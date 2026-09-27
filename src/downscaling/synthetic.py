"""
Realistic synthetic meteorological paired dataset generator for Phase 5 downscaling validation.
Generates physically inspired high-resolution weather fields (Gaussian multi-cell precipitation systems,
elongated squall lines/rain bands, localized convective peaks, spatial gradients, background rain)
and downsamples them with area-averaging / point-sampling to low resolution.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from src.downscaling.grid import calculate_grid_spacing, create_target_grid
from src.downscaling.interpolation import interpolate_field
from src.utils.logger import get_logger

logger = get_logger("SyntheticDownscaling")


def generate_synthetic_storm_field(
    lats: np.ndarray,
    lons: np.ndarray,
    num_cells: int = 3,
    has_rainband: bool = True,
    peak_intensity_range: Tuple[float, float] = (50.0, 120.0),
    background_rain: float = 2.0,
    noise_std: float = 1.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate a realistic synthetic 2D high-resolution precipitation field in mm.

    Features:
        - Multi-cell convective rain cores with Gaussian intensity profiles
        - Elongated oriented rainbands (simulating monsoonal troughs or tropical squall lines)
        - Background stratiform rainfall
        - Smooth spatial gradients and realistic convective-scale noise
        - Non-negative precipitation guarantee
    """
    rng = np.random.default_rng(seed)
    n_lat = len(lats)
    n_lon = len(lons)

    min_lat, max_lat = float(np.min(lats)), float(np.max(lats))
    min_lon, max_lon = float(np.min(lons)), float(np.max(lons))

    # Meshgrid coordinates
    mesh_lat, mesh_lon = np.meshgrid(lats, lons, indexing="ij")

    field = np.zeros((n_lat, n_lon), dtype=np.float64)

    # 1. Base stratiform rain field with mild smooth spatial gradients
    grad_x = np.sin((mesh_lon - min_lon) / max(max_lon - min_lon, 1e-3) * np.pi)
    grad_y = np.cos((mesh_lat - min_lat) / max(max_lat - min_lat, 1e-3) * np.pi)
    base_field = background_rain * (0.5 + 0.5 * grad_x * grad_y)
    field += base_field

    # 2. Convective precipitation cells
    for _ in range(num_cells):
        c_lat = rng.uniform(min_lat + 0.2 * (max_lat - min_lat), max_lat - 0.2 * (max_lat - min_lat))
        c_lon = rng.uniform(min_lon + 0.2 * (max_lon - min_lon), max_lon - 0.2 * (max_lon - min_lon))
        peak_val = rng.uniform(*peak_intensity_range)

        # Spatial scale in degrees (e.g., 0.15 to 0.45 deg ~ 15-50 km)
        sigma_lat = rng.uniform(0.12, 0.35)
        sigma_lon = rng.uniform(0.12, 0.35)

        # Gaussian convective core
        dist_sq = ((mesh_lat - c_lat) / sigma_lat) ** 2 + ((mesh_lon - c_lon) / sigma_lon) ** 2
        cell = peak_val * np.exp(-0.5 * dist_sq)
        field += cell

    # 3. Elongated rain band / squall line
    if has_rainband:
        band_angle = rng.uniform(-np.pi / 4, np.pi / 4)
        band_center_lat = (min_lat + max_lat) / 2.0 + rng.uniform(-0.5, 0.5)
        band_center_lon = (min_lon + max_lon) / 2.0 + rng.uniform(-0.5, 0.5)
        band_peak = rng.uniform(30.0, 70.0)

        # Coordinate rotation
        rot_u = (mesh_lat - band_center_lat) * np.cos(band_angle) - (mesh_lon - band_center_lon) * np.sin(band_angle)
        rot_v = (mesh_lat - band_center_lat) * np.sin(band_angle) + (mesh_lon - band_center_lon) * np.cos(band_angle)

        band_sigma_u = rng.uniform(0.10, 0.22)  # Narrow cross-track
        band_sigma_v = rng.uniform(0.60, 1.50)  # Elongated along-track

        band = band_peak * np.exp(-0.5 * ((rot_u / band_sigma_u) ** 2 + (rot_v / band_sigma_v) ** 2))
        field += band

    # 4. Spatially correlated noise
    raw_noise = rng.normal(0, noise_std, size=(n_lat, n_lon))
    smooth_noise = gaussian_filter(raw_noise, sigma=1.0)
    field += smooth_noise

    # Ensure non-negativity
    field = np.clip(field, 0.0, None)
    return field


def downsample_field_to_coarse(
    high_res_field: np.ndarray,
    high_lats: np.ndarray,
    high_lons: np.ndarray,
    coarse_res_km: float = 12.0,
    method: str = "area_average",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Degrade a high-resolution field to low-resolution (e.g. ~12 km).

    Args:
        high_res_field: 2D array [n_lat, n_lon].
        high_lats: 1D array of fine latitudes.
        high_lons: 1D array of fine longitudes.
        coarse_res_km: Target coarse resolution in km (e.g. 12 km).
        method: 'area_average' or 'subsample'.

    Returns:
        Tuple[coarse_field, coarse_lats, coarse_lons]
    """
    lat_bounds = (float(np.min(high_lats)), float(np.max(high_lats)))
    lon_bounds = (float(np.min(high_lons)), float(np.max(high_lons)))

    coarse_lats, coarse_lons = create_target_grid(lat_bounds, lon_bounds, target_res_km=coarse_res_km)

    if method == "area_average":
        # First apply spatial smoothing kernel corresponding to coarse pixel footprint
        high_spacing = calculate_grid_spacing(high_lats, high_lons)
        scale_ratio = max(1.0, coarse_res_km / max(high_spacing["effective_res_km"], 0.1))
        sigma_pixels = scale_ratio / 2.0
        smoothed = gaussian_filter(high_res_field, sigma=sigma_pixels, mode="nearest")
        coarse_field = interpolate_field(smoothed, high_lats, high_lons, coarse_lats, coarse_lons, method="bilinear")
    else:
        coarse_field = interpolate_field(high_res_field, high_lats, high_lons, coarse_lats, coarse_lons, method="bilinear")

    coarse_field = np.clip(coarse_field, 0.0, None)
    return coarse_field, coarse_lats, coarse_lons


def generate_paired_synthetic_dataset(
    num_events: int = 20,
    timesteps_per_event: int = 5,
    crop_size_deg: float = 4.0,
    high_res_km: float = 5.0,
    low_res_km: float = 12.0,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Generate a paired downscaling dataset with tracks and timestamps.

    Returns:
        List of sample dicts containing low-resolution input and high-resolution target.
    """
    rng = np.random.default_rng(seed)
    samples: List[Dict[str, Any]] = []

    base_time = pd.to_datetime("2024-07-01 00:00:00")

    for ev_idx in range(num_events):
        track_id = f"TRK_SYNTH_{ev_idx + 1:03d}"
        origin_lat = rng.uniform(12.0, 24.0)
        origin_lon = rng.uniform(72.0, 88.0)
        drift_lat = rng.uniform(-0.15, 0.15)
        drift_lon = rng.uniform(-0.15, 0.25)

        for step in range(timesteps_per_event):
            event_id = f"EV_SYNTH_{ev_idx + 1:03d}_{step + 1:02d}"
            curr_lat = origin_lat + step * drift_lat
            curr_lon = origin_lon + step * drift_lon
            curr_time = base_time + pd.Timedelta(hours=ev_idx * 24 + step * 6)

            lat_bounds = (curr_lat - crop_size_deg / 2.0, curr_lat + crop_size_deg / 2.0)
            lon_bounds = (curr_lon - crop_size_deg / 2.0, curr_lon + crop_size_deg / 2.0)

            # For batch collation in PyTorch, standard square event crops use fixed target grid shape (e.g. 90x90)
            target_shape = (
                max(16, int(np.round(crop_size_deg * 111.139 / high_res_km))),
                max(16, int(np.round(crop_size_deg * 111.139 / high_res_km))),
            )
            high_lats, high_lons = create_target_grid(lat_bounds, lon_bounds, target_shape=target_shape)

            # Evolving intensity and cell count
            peak_val = 60.0 + 20.0 * np.sin(step * 0.8) + rng.normal(0, 5.0)
            high_field = generate_synthetic_storm_field(
                lats=high_lats,
                lons=high_lons,
                num_cells=rng.integers(2, 5),
                has_rainband=True,
                peak_intensity_range=(max(30.0, peak_val - 15.0), peak_val + 25.0),
                seed=int(rng.integers(0, 1000000)),
            )

            coarse_field, coarse_lats, coarse_lons = downsample_field_to_coarse(
                high_res_field=high_field,
                high_lats=high_lats,
                high_lons=high_lons,
                coarse_res_km=low_res_km,
                method="area_average",
            )

            # High-res grid metrics & low-res grid metrics
            high_metrics = calculate_grid_spacing(high_lats, high_lons)
            low_metrics = calculate_grid_spacing(coarse_lats, coarse_lons)

            sample = {
                "event_id": event_id,
                "track_id": track_id,
                "step_in_track": step + 1,
                "timestamp": str(curr_time),
                "low_res": coarse_field.astype(np.float32),
                "high_res": high_field.astype(np.float32),
                "low_lats": coarse_lats.astype(np.float32),
                "low_lons": coarse_lons.astype(np.float32),
                "high_lats": high_lats.astype(np.float32),
                "high_lons": high_lons.astype(np.float32),
                "low_res_km": low_metrics["effective_res_km"],
                "high_res_km": high_metrics["effective_res_km"],
                "centroid_lat": float(curr_lat),
                "centroid_lon": float(curr_lon),
                "max_intensity_target": float(np.max(high_field)),
                "max_intensity_low": float(np.max(coarse_field)),
            }
            samples.append(sample)

    logger.info(f"Generated {len(samples)} synthetic paired downscaling samples across {num_events} tracks.")
    return samples
