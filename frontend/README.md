# SpatioAI Dashboard

React + TypeScript + Vite + Tailwind dashboard for the existing SpatioAI FastAPI backend.

## Run

From the repository root, start the backend:

```powershell
.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

In a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Set `VITE_API_BASE_URL` in `.env.local` when the API is not running at `http://localhost:8000`.

## Integration

The dashboard uses the existing endpoints for health, events, tracks, GNN tracking, downscaling, ensembles, and risk. Risk is automatically requested for the selected event. Downscaling and ensemble panels stay input-gated because event responses currently expose metadata but not the coarse precipitation array required by those POST endpoints.

The interface labels the current source as research/synthetic validation. It does not claim calibrated operational probability or official warning capability.
