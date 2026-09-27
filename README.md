# SpatioAI

**AI-Driven Spatio-Temporal Tracking of Extreme Weather Anomalies in Medium-Range Forecasts**

---

## 1. Overview
SpatioAI is an AI-powered meteorological intelligence framework designed to detect, track, and downscale extreme weather events (such as intense precipitation anomalies over India and the Bay of Bengal) from medium-range Numerical Weather Prediction (NWP) models and reanalysis datasets.

> [!NOTE]
> **Scientific Scope & Distinction**:
> - **Reanalysis (ERA5)**: Blended model-observation historical baseline, not direct ground-truth station observations.
> - **NWP Forecasts (NCUM / NEPS-G / GFS)**: Deterministic and ensemble physics-based numerical simulations.
> - **Baseline Tracker**: Classical deterministic multi-object trajectory association using geodesic distances and multi-factor matching costs. Serves as a transparent baseline benchmark for later Spatio-Temporal GNN models.
> - This project is an experimental research system and is not an operational weather forecasting agency service.

---

## 2. Implemented Modules

### Phase 1: Data Ingestion & Preprocessing
- **Synthetic Weather Dataset Generator**: Zero-dependency generation of multi-variable spatio-temporal datasets for testing without downloading multi-gigabyte files.
- **Config-Driven Architecture**: Fully parameterized via `configs/data.yaml` (geographic bounding box, variable selection, chunk sizes, target resolutions).
- **Abstract & Modular Data Loaders**: `WeatherDataLoader` base class with specialized `ERA5Loader` and `NWPLoader` implementations.
- **Standardization & Quality Control**: Normalizes coordinates, renames aliases to canonical variables, performs documented unit conversions (e.g., $m \to mm$, $^\circ C \to K$, $Pa \to hPa$), and runs rigorous sanity/NaN checks.
- **Regridding Engine**: Modular spatial interpolation engine (e.g. 12 km resolution grid).
- **Chunked Zarr Storage**: Lazy multi-dimensional loading and writing using `xarray` + `Dask` + `Zarr`.

### Phase 2: Climatology & Extreme Weather Anomaly Detection
- **Time-Aware Climatological Baseline**: Multi-variable baseline computation (`mean`, `std`, `median`, `p90`, `p95`, `p99`) with time groupings (`dayofyear_hour`, `month_hour`, `month_day_hour`) and leap year handling.
- **Anomaly Engine**:
  - Absolute Anomaly: $\Delta = \text{actual} - \mu_{\text{clim}}$
  - Standardized Anomaly: $z = \frac{\text{actual} - \mu_{\text{clim}}}{\sigma_{\text{clim}} + \epsilon}$ (with zero/low-variance regularization).
- **Extreme Event Thresholding**: Binary candidate extreme mask generation based on percentile exceedance (e.g., $> p95$, $> p99$) or standardized $z$-score thresholds.
- **Connected Component Segmentation**:
  - $2\text{D}$ spatial clustering using 8-connectivity / 4-connectivity.
  - Spherical latitude-dependent physical cell area computation ($\text{Area} \approx R^2 \cos(\phi) \Delta\phi \Delta\lambda$).
  - Spatial feature extraction: `centroid_lat`, `centroid_lon` (intensity-weighted), bounding boxes (`min_lat`, `max_lat`, `min_lon`, `max_lon`), `area_km2`, `max_intensity`, `mean_intensity`.
  - Noise filtering via `minimum_cells` and `minimum_area_km2`.
  - Export to Parquet and CSV in `data/events/`.

### Phase 3: Temporal Extreme Weather Event Tracking
- **Geodesic Dynamics**: Haversine great-circle distance, initial compass bearing angles ($0^\circ$ to $360^\circ$), propagation speeds in km/h, and bounding-box Intersection-over-Union (IoU).
- **Multi-Factor Data Association**:
  - Distance cost, bounding-box overlap cost, area ratio change penalty, and intensity difference.
  - Hard kinematic gates (`max_centroid_distance_km`, `max_speed_kmh`, `max_area_change_ratio`).
  - Global optimal assignment using the Hungarian algorithm (`scipy.optimize.linear_sum_assignment`) with Greedy fallback.
- **Track Lifecycle Management**:
  - Automatic track creation, multi-timestep trajectory aggregation, and temporary gap bridging (`max_missed_steps`).
  - Optional 2D constant-velocity kinematic Kalman filter motion predictor.
  - Trajectory metrics: total displacement (km), cumulative distance (km), mean/max speed (km/h), net bearing ($^\circ$), duration (hours), max intensity, and max area.
  - Export to `precipitation_event_tracks.parquet` and `precipitation_track_summary.parquet`.

---

## 3. Project Structure

```text
SpatioAI/
├── .venv/                         # Python 3.11+ Virtual Environment
├── configs/
│   └── data.yaml                  # Data, region, climatology, event, & tracking config
├── data/
│   ├── raw/                       # Raw input NetCDF/GRIB datasets
│   ├── interim/                   # Intermediate staged data
│   ├── processed/                 # Standardized clean datasets
│   ├── climatology/               # Multi-variable climatology Zarr stores
│   ├── events/                    # Event tables & track tables (Parquet/CSV)
│   └── zarr/                      # Processed weather Zarr stores
├── src/
│   ├── data/
│   │   ├── base_loader.py         # Abstract base weather data loader
│   │   ├── era5_loader.py         # ERA5 reanalysis data loader
│   │   ├── nwp_loader.py          # NWP forecast data loader
│   │   ├── preprocessing.py       # Coordinate & variable standardization, unit conversion
│   │   ├── quality.py             # Physical bounds validation and QC checks
│   │   ├── regridding.py          # Spatial interpolation & grid harmonization
│   │   ├── synthetic.py           # Synthetic dataset generator & extreme event injection
│   │   └── dataset.py             # Spatio-temporal windowed dataset abstraction
│   ├── climatology/
│   │   ├── statistics.py          # Time-aware grouping & statistical metrics
│   │   ├── baseline.py            # ClimatologyBaseline fit, save, load, and align
│   │   └── anomaly.py             # Absolute, standardized, and percentile anomalies
│   ├── events/
│   │   ├── detection.py           # Binary extreme threshold detection
│   │   ├── segmentation.py        # Connected components, spatial metrics, & filtering
│   │   ├── association.py         # Data association, cost metrics, & Hungarian matching
│   │   └── tracking.py            # EventTrack lifecycle, trajectories, & tracker
│   ├── gnn/                       # (Phase 4)
│   ├── downscaling/               # (Phases 5-7)
│   ├── physics/                   # (Phase 8)
│   ├── ensemble/                  # (Phase 9)
│   ├── risk/                      # (Phase 9)
│   └── utils/
│       ├── logger.py              # Centralized logging
│       └── geodesics.py           # Haversine distance, bearing, speed, and IoU
├── notebooks/
│   ├── 02_anomaly_detection.ipynb # Anomaly detection & segmentation walkthrough
│   └── 03_event_tracking.ipynb    # Temporal event tracking & trajectory visualization
├── scripts/
│   ├── prepare_data.py            # Data preparation CLI
│   ├── build_climatology.py       # Climatological baseline generator CLI
│   ├── run_anomaly_detection.py   # Anomaly detection & event extraction CLI
│   ├── run_event_tracking.py      # Temporal event tracking CLI
│   ├── validate_event_pipeline.py # Anomaly detection validation runner
│   └── validate_tracking_pipeline.py # Tracking pipeline validation runner
├── tests/
│   ├── test_data_pipeline.py      # Unit tests for Phase 1
│   ├── test_climatology_and_events.py # Unit tests for Phase 2
│   └── test_tracking.py           # Unit tests for Phase 3
├── requirements.txt
├── .gitignore
├── Dockerfile
└── README.md
```

### Phase 6: Conditional Diffusion Downscaling

Phase 6 adds a conditional diffusion model for statistical precipitation downscaling from approximately 12 km to approximately 5 km. It reuses the Phase 5 paired synthetic dataset, track-based split, geographic interpolation, U-Net comparison path, and extreme-event metrics.

The model receives the bilinearly interpolated coarse precipitation field at every denoising step. A configurable linear or cosine beta schedule drives forward diffusion, while a sinusoidal timestep embedding supplies diffusion-time information to a compact conditional U-Net. Training predicts injected Gaussian noise with an MSE objective and an optional percentile-weighted extreme component. Precipitation is trained in `log1p` space and mapped back with a numerically safe inverse transform.

Multiple conditional samples can be generated with fixed seeds, and their ensemble mean, standard deviation, and quantiles can be inspected alongside Nearest, Bilinear, Bicubic, and Phase 5 U-Net baselines. The lightweight validation script checks forward diffusion, conditioning, tiny loss reduction, reverse sampling, non-negativity, and finite outputs.

> Phase 6 implements statistical conditional generative downscaling. Physical consistency and operational meteorological validation are not established by this phase.

Run the focused Phase 6 tests and synthetic validation with:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_diffusion.py -q
.venv\Scripts\python.exe scripts/validate_diffusion_pipeline.py
```

### Phase 7: Physics-Informed Downscaling

Phase 7 adds physically motivated regularization to precipitation downscaling. These constraints are differentiable terms, not a complete atmospheric simulator and not a guarantee of physical correctness.

The active constraints are:

- non-negativity penalty on physical precipitation, without using clipping as the training loss;
- area-weighted coarse-to-fine consistency, using spatial average intensity because the synthetic fields represent precipitation intensity over a timestep;
- normalized area-weighted domain-integrated precipitation intensity consistency as a mass-related proxy, not atmospheric mass conservation;
- gradient-structure matching using finite differences, which preserves legitimate sharp features rather than forcing gradients toward zero;
- smooth percentile-based extreme-area and intensity structure matching.

The diffusion trainer estimates clean `x0` from the predicted noise, applies the inverse `log1p` transform, and computes physics losses in physical precipitation space. Latitude-dependent relative cell areas use `cos(latitude)`. The optional moisture proxy is disabled by default and is not a moisture-conservation law; no humidity fields are fabricated for the current synthetic dataset.

The controlled ablation script runs standard diffusion, extreme-aware diffusion, and physics-informed diffusion, saving measured histories and comparison outputs under `data/outputs/downscaling/`. The configured physics weight and component weights are experimental hyperparameters, not scientifically optimal values.

> The implemented constraints are physically motivated regularization terms. They do not constitute a complete numerical atmospheric model or guarantee physical correctness.

Run Phase 7 validation with:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_physics.py -q
.venv\Scripts\python.exe scripts/evaluate_physics_diffusion.py --epochs 1 --device cpu
```

### Phase 8: Ensemble Uncertainty and Risk

Phase 8 converts conditional diffusion samples into descriptive uncertainty diagnostics and deterministic, machine-readable risk fields. It reuses the Phase 6 sampler, Phase 7 latitude-dependent area weighting, Phase 2/3 event metadata conventions, and existing synthetic paired precipitation data.

For each condition, the ensemble layer calculates mean, median, standard deviation, quantiles, coefficient of variation, and empirical threshold-exceedance frequency. The value `P(P > T)` is explicitly the fraction of generated conditional samples exceeding threshold `T`; it is not automatically a calibrated real-world forecast probability. Conditional sample spread is reported as a generative uncertainty diagnostic, not complete model uncertainty.

The risk layer supports configurable fixed thresholds and climatological percentile thresholds when a climatology field is supplied. It calculates a threshold-ratio severity index, latitude/longitude cell-area-weighted affected area, probability masks, and an explainable experimental score:

`risk_score = exceedance_probability * severity_index * affected_area_km2 / area_scale_km2`

Event summaries preserve `event_id`, `track_id`, timestamp, centroid, bounding box, probability, predicted intensity, uncertainty, affected area, and score. Brier score, reliability bins, and empirical CRPS are available when independent observations or targets exist. No automatic recalibration is performed.

> Phase 8 converts conditional ensemble predictions into uncertainty diagnostics and empirical threshold-exceedance estimates. These outputs are not official meteorological warnings and require validation/calibration against independent real-world observations before operational use.

Run the synthetic Phase 8 validation with:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_ensemble.py tests/test_risk.py -q
.venv\Scripts\python.exe scripts/validate_risk_pipeline.py
```

### Phase 9: FastAPI Backend

Phase 9 exposes the existing ML pipeline through a versioned FastAPI integration layer. Routes remain thin and delegate to model, downscaling, ensemble, risk, event, and tracking services. Models are loaded at application startup from configured checkpoints; unavailable checkpoints are reported explicitly and are never silently replaced.

Available endpoints include:

- `GET /health`, `GET /version`, and `GET /docs`
- `GET /api/v1/events` and `GET /api/v1/events/{event_id}`
- `GET /api/v1/tracks` and `GET /api/v1/tracks/{track_id}`
- `POST /api/v1/tracking/predict`
- `POST /api/v1/downscaling/predict`
- `POST /api/v1/ensemble/predict`
- `POST /api/v1/risk/analyze`

Responses use Pydantic schemas, request IDs are returned in `X-Request-ID`, configured CORS origins are applied for future frontend integration, and risk masks are returned as GeoJSON-compatible `FeatureCollection` objects using `[longitude, latitude]` coordinate order. API development uses deterministic synthetic demo data and does not retrain models during requests.

Start the local API with:

```powershell
.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

> The Phase 9 API exposes the existing ML pipeline for inference and integration. It does not establish operational meteorological forecasting or official warning capability.

### Phase 10: React/TypeScript/Vite/Tailwind Dashboard

Phase 10 adds the geospatial research dashboard under `frontend/`. It consumes the existing FastAPI contracts without moving ML logic into the browser. The interface includes event selection and filtering, Leaflet map footprints and trajectories, model readiness, track intensity timelines, empirical risk summaries, loading/error states, and explicit synthetic/demo labeling.

The current event response contains metadata but not the coarse precipitation arrays required by the downscaling and ensemble POST endpoints. The dashboard therefore leaves those panels input-gated rather than fabricating fields. The physics-informed diffusion checkpoint is also shown as unavailable when the backend registry reports it missing.

Run the backend:

```powershell
.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Run the frontend in a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Set `VITE_API_BASE_URL=http://localhost:8000` in `frontend/.env.local` when using a non-default API host. Frontend checks are `npm run lint`, `npm test`, and `npm run build` from `frontend/`.

---

## 4. Execution Commands

### 1. Run Complete Test Suite
```powershell
.venv\Scripts\python.exe -m pytest tests/ -v
```

### 2. Ingest and Process Data (Phase 1)
```powershell
.venv\Scripts\python.exe scripts/prepare_data.py --config configs/data.yaml
```

### 3. Build Climatology Baseline (Phase 2)
```powershell
.venv\Scripts\python.exe scripts/build_climatology.py --config configs/data.yaml
```

### 4. Run Anomaly Detection & Extract Candidate Events (Phase 2)
```powershell
.venv\Scripts\python.exe scripts/run_anomaly_detection.py --config configs/data.yaml
```

### 5. Run Temporal Event Tracking (Phase 3)
```powershell
.venv\Scripts\python.exe scripts/run_event_tracking.py --config configs/data.yaml
```

### 6. Run Synthetic Tracking Validation
```powershell
.venv\Scripts\python.exe scripts/validate_tracking_pipeline.py
```
