# 🚀 SpatioAI — Production Deployment Guide

This guide covers complete step-by-step instructions for deploying the **SpatioAI** weather intelligence system to production using **Render** (FastAPI Backend) and **Vercel** (React + TypeScript Frontend).

---

## 🏗️ Architecture Overview

```
                        ┌────────────────────────────────────────┐
                        │              Client Browser            │
                        └───────────────────┬────────────────────┘
                                            │
                    ┌───────────────────────┴────────────────────────┐
                    │                                                │
                    ▼                                                ▼
       ┌─────────────────────────┐                      ┌─────────────────────────┐
       │   Frontend UI (Vercel)  │                      │   API Backend (Render)  │
       │   React 19 + TypeScript │ ── REST / JSON API ──▶   FastAPI + PyTorch     │
       │   Vite + Glassmorphism  │                      │   Synthetic Demo Engine │
       └─────────────────────────┘                      └─────────────────────────┘
```

---

## 1. ⚙️ Deploying the FastAPI Backend to Render

Render hosts the Python FastAPI application with automatic SSL, zero-downtime deployment, and environment configuration.

### Method A: Automated via `render.yaml` (Recommended)
1. Log into your [Render Dashboard](https://dashboard.render.com/).
2. Click **New +** ➔ **Blueprint**.
3. Connect your repository: `https://github.com/sumanisfr/SpatioAI-Weather-Intelligence`.
4. Render will automatically detect [`render.yaml`](file:///render.yaml) and configure:
   - **Environment**: `Python 3.11.9`
   - **Build Command**: `pip install -r requirements-prod.txt`
   - **Start Command**: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
   - **Health Check Path**: `/health`
5. Click **Apply**.

### Method B: Manual Web Service Setup
1. On Render, click **New +** ➔ **Web Service**.
2. Select your repository `SpatioAI-Weather-Intelligence`.
3. Configure the settings:
   - **Name**: `spatioai-backend`
   - **Region**: `Oregon (US West)` or closest to your users
   - **Branch**: `main`
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements-prod.txt`
   - **Start Command**: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
   - **Health Check Path**: `/health`
4. Add **Environment Variables**:
   | Key | Value | Description |
   |---|---|---|
   | `DEMO_MODE` | `true` | Enables deterministic synthetic demo engine |
   | `INFERENCE_DEVICE` | `cpu` | Uses CPU for cloud inference |
   | `CORS_ORIGINS` | `http://localhost:5173,https://*.vercel.app` | Allowed origins (or comma-separated Vercel URL) |
5. Click **Create Web Service**.
6. Once deployed, copy your backend URL (e.g., `https://spatioai-backend.onrender.com`).

---

## 2. ⚡ Deploying the Frontend to Vercel

Vercel provides edge delivery, fast CDN caching, and automated builds for Vite + React.

### Step-by-Step Setup
1. Log into your [Vercel Dashboard](https://vercel.com/).
2. Click **Add New...** ➔ **Project**.
3. Import your GitHub repository: `sumanisfr/SpatioAI-Weather-Intelligence`.
4. In the configuration screen:
   - **Framework Preset**: `Vite`
   - **Root Directory**: `frontend` *(Click Edit and choose `frontend`)*
   - **Build Command**: `npm run build`
   - **Output Directory**: `dist`
   - **Install Command**: `npm install`
5. Expand **Environment Variables** and add:
   | Key | Value | Notes |
   |---|---|---|
   | `VITE_API_BASE_URL` | `https://spatioai-backend.onrender.com` | **Set to your Render backend URL** |
   | `VITE_OPENWEATHER_API_KEY` | `your_optional_api_key` | Optional: for live station observation tiles |
6. Click **Deploy**.
7. Once finished, Vercel will provide your live URL (e.g., `https://spatioai-weather-intelligence.vercel.app`).

---

## 3. 🔐 Security & Secret Protection Best Practices

1. **Never Commit Secrets**:
   - Both root [`.gitignore`](file:///.gitignore) and [`frontend/.gitignore`](file:///frontend/.gitignore) strictly ignore `.env` and `.env.*` files.
   - Only `.env.example` templates are tracked.
2. **Frontend Environment Exposure**:
   - In Vite, only variables prefixed with `VITE_` are bundled to the client.
   - Do **NOT** store private API tokens or database secrets in `VITE_` variables.
3. **CORS Tightening**:
   - Once your Vercel URL is live (e.g. `https://spatioai-weather.vercel.app`), update `CORS_ORIGINS` in your Render Web Service settings to match:
     `https://spatioai-weather.vercel.app,http://localhost:5173`

---

## 4. 🧪 Production Verification Checklist

Run these quick checks on your live deployment:

| Check | URL / Action | Expected Result |
|---|---|---|
| **Backend Health** | `GET https://your-backend.onrender.com/health` | `{"status": "ok", "service": "SpatioAI", ...}` |
| **API Version** | `GET https://your-backend.onrender.com/version` | `{"service": "SpatioAI", "version": "0.9.0", ...}` |
| **Events Endpoint** | `GET https://your-backend.onrender.com/api/v1/events` | JSON list of tracked extreme weather events |
| **Tracks Endpoint** | `GET https://your-backend.onrender.com/api/v1/tracks` | Trajectory paths and centroids |
| **Risk Analysis** | `POST https://your-backend.onrender.com/api/v1/risk/analyze` | Empirical exceedance metrics & GeoJSON |
| **Frontend Map** | Visit Vercel App URL | Interactive Leaflet map with glowing markers and sidebar |
| **Event Selection** | Click an event in sidebar | Map flies to centroid, displays radar inspection, and live charts |

---

## 5. 🛠️ Local Development & Testing

```bash
# 1. Start Backend locally
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload

# 2. Start Frontend locally
cd frontend
npm run dev
```
