# HeatVision — Urban Heat Island Mapper

A full-stack geospatial web application that maps Land Surface Temperature (LST) and vegetation health (NDVI) across cities, identifies Urban Heat Island (UHI) hotspots, and predicts heat-risk zones using a Random Forest model. Features a dark-themed Leaflet dashboard with draw-analyze capabilities, real-time weather fetching for any Earth location, and environmental justice analysis.

![Tech Stack](https://img.shields.io/badge/FastAPI-009688?logo=fastapi) ![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python) ![PostgreSQL](https://img.shields.io/badge/PostgreSQL-4169E1?logo=postgresql) ![Leaflet](https://img.shields.io/badge/Leaflet-199900?logo=leaflet)

**Live demo:** https://heatmapper-jt16.onrender.com

## Features

- **Interactive Heat Map Dashboard** — Multi-layer map (temperature, risk zones, hotspots), ward search, export (GeoJSON/CSV), bookmarks
- **Draw-and-Analyze** — Draw a rectangle anywhere on Earth; analyzes using city data or real-time Open-Meteo weather
- **ML Risk Prediction** — RandomForestRegressor predicts LST from NDVI features; risk categories (low/medium/high)
- **What-If Simulator** — Adjust green cover, cool roofs, and traffic reduction sliders to see predicted cooling impact
- **Time Machine** — Seasonal slider applies monthly LST offsets to visualize heat variation across the year
- **Per-Cell AI Analysis** — Click any cell for detailed LST/NDVI comparison vs city average, risk factors, recommendations, and similar cells
- **Equity Analysis** — Correlates heat with socio-economic factors (income, green cover, population density)
- **Keyboard Shortcuts** — `1` heat · `2` risk · `3` hotspots · `D` draw · `S` search · `F` fullscreen · `E` toggle panel · `?` shortcuts
- **Global Mode** — Uses free Open-Meteo API for any location on Earth

**Supported cities:** Bangalore · Mumbai · Delhi · Chennai · Kolkata (seeded with synthetic data on startup, no external download needed)

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.12, FastAPI, Uvicorn |
| Database | PostgreSQL + asyncpg via SQLAlchemy 2.0 (async) |
| Geospatial | GeoPandas, Shapely, Rasterio, SciPy |
| ML | scikit-learn (RandomForestRegressor), NumPy, Pandas |
| Frontend | Vanilla JS, Leaflet.js, Leaflet.draw, Three.js |
| Task Queue | Celery + Redis (optional) |
| HTTP Client | httpx |

## Getting Started

### Prerequisites

- Python 3.12
- PostgreSQL 14+
- Redis 6+ (optional, only for Celery pipeline)

### Local Setup

```bash
git clone <repo-url> HeatMapper
cd HeatMapper
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Configure environment variables in `.env`:

```
DB_USER=postgres
DB_PASSWORD=postgres
DB_HOST=localhost
DB_PORT=5432
DB_NAME=heatmapper
```

Create the database:

```bash
createdb heatmapper
```

### Run

```bash
uvicorn main:app --reload --port 8000
```

Open http://localhost:8000. The app auto-creates tables and seeds 5 cities with mock data on first startup.

### Docker (local)

```bash
docker compose up -d
```

Open http://localhost:8000.

### Celery (optional)

```bash
redis-server                                    # Terminal 1
celery -A app.celery_app worker --beat --loglevel=info   # Terminal 2
```

## Deploy to Render (free)

1. Push to GitHub
2. Go to https://dashboard.render.com → New Web Service → Connect repo
3. Fill:

| Field | Value |
|---|---|
| Name | `heatmapper` |
| Runtime | **Python 3** |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `uvicorn main:app --host 0.0.0.0 --port \$PORT` |
| Plan | **Free** |

4. Add PostgreSQL (New → PostgreSQL → Free plan)
5. Set environment variables in web service:

```
DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME (from Postgres)
PYTHONUNBUFFERED=1
```

6. Deploy

**Note:** Free tier sleeps after 15 min idle; wakes on first request (~30s). PostgreSQL expires after 90 days.

## API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Frontend dashboard |
| GET | `/api/cities` | List available cities |
| GET | `/api/wards?city=` | Ward boundaries |
| GET | `/api/heat-map?city=` | LST + NDVI readings by ward |
| GET | `/api/predictions?city=` | ML risk predictions |
| GET | `/api/equity?city=` | Equity & environmental justice metrics |
| GET | `/api/stats?city=` | Dashboard summary statistics |
| GET | `/api/wards/{id}` | Single ward detail |
| POST | `/api/analyze-area` | Analyze a drawn area (city or global) |
| POST | `/api/refresh` | Trigger Celery refresh pipeline |
| GET | `/api/tasks/{id}` | Poll Celery task status |
| GET | `/api/health` | Health check |

## Project Structure

```
HeatMapper/
├── main.py                     # FastAPI entry point
├── Dockerfile                  # Docker image
├── docker-compose.yml          # Local dev with DB + Redis
├── .python-version             # Python version (3.12)
├── requirements.txt
├── .env
├── app/
│   ├── config.py               # Pydantic settings
│   ├── database.py             # Async SQLAlchemy engine
│   ├── celery_app.py           # Celery config & beat schedule
│   ├── seed.py                 # City seeding pipeline
│   ├── api/
│   │   ├── routes.py           # All API route definitions
│   │   └── schemas.py          # Pydantic request/response models
│   ├── models/                 # SQLAlchemy ORM models
│   │   ├── ward.py, lst.py, ndvi.py, prediction.py, equity.py
│   ├── analysis/
│   │   ├── area_analysis.py    # Draw-analyze (city mode)
│   │   ├── global_analysis.py  # Draw-analyze (global/Open-Meteo mode)
│   │   ├── equity.py           # Environmental justice analysis
│   │   └── joiner.py           # Spatial joins
│   ├── ml/
│   │   ├── features.py         # Feature engineering
│   │   └── model.py            # RandomForest training & prediction
│   ├── raster/
│   │   ├── ingestion.py        # GeoTIFF raster loading & zonal stats
│   │   └── sample_data.py      # Mock MODIS-like data generation
│   ├── tasks/
│   │   └── refresh.py          # Celery tasks: ingest → train → refresh
│   └── frontend/
│       └── index.html          # Single-page Leaflet dashboard (~2300 lines)
├── scripts/
│   ├── ingest_raster.py
│   ├── run_analysis.py
│   ├── run_equity.py
│   └── train_model.py
├── data/
│   ├── sample/                 # Pre-generated sample rasters & GeoJSON
│   └── models/                 # Saved ML models
└── fly.toml                    # Fly.io deployment config
```

## CLI Scripts

```bash
python scripts/ingest_raster.py --sample    # Generate & ingest sample rasters
python scripts/train_model.py               # Train ML model from DB data
python scripts/run_equity.py                # Run equity/environmental justice analysis
```

All scripts support `--dry-run` for testing without a database.

## Configuration

All settings in `app/config.py`, overridable via `.env`:

| Setting | Default | Description |
|---|---|---|
| `DB_HOST` | localhost | PostgreSQL host |
| `DB_PORT` | 5432 | PostgreSQL port |
| `DB_USER` | postgres | PostgreSQL user |
| `DB_PASSWORD` | postgres | PostgreSQL password |
| `DB_NAME` | heatmapper | Database name |
| `REDIS_HOST` | localhost | Redis host |
| `REDIS_PORT` | 6379 | Redis port |
| `REFRESH_SCHEDULE_HOUR` | 2 | Nightly Celery refresh hour |

**Note:** Geometry is stored as GeoJSON text — no PostGIS extension required. No external API keys needed (Open-Meteo is free, no registration).
