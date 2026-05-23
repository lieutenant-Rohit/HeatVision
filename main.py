"""
Urban Heat Island Mapper — Entry point.

Start the FastAPI dev server with:
    uvicorn main:app --reload

Open http://localhost:8000 to view the Leaflet.js map dashboard.
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router as api_router
from app.database import init_db


async def _auto_seed():
    """Seed any cities not yet in the database."""
    from app.seed import seed_city

    ALL_CITIES = ["Bangalore", "Mumbai", "Delhi", "Chennai", "Kolkata"]

    from sqlalchemy import select, func
    from app.database import async_session_factory
    from app.models.ward import Ward

    async with async_session_factory() as db:
        result = await db.execute(select(Ward.city).distinct())
        existing = {r[0] for r in result.fetchall()}

    missing = [c for c in ALL_CITIES if c not in existing]
    if not missing:
        async with async_session_factory() as db:
            result = await db.execute(select(Ward.city).distinct())
            existing_list = [r[0] for r in result.fetchall()]
            count = (await db.execute(select(func.count(Ward.id)))).scalar()
        print(f"✓ Database seeded ({count} wards across {len(existing_list)} cities: {', '.join(existing_list)})")
        return

    print(f"Seeding {len(missing)}/{len(ALL_CITIES)} cities: {', '.join(missing)}")
    for city in missing:
        await seed_city(city, "20240515")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """On startup: create tables, seed demo cities."""
    try:
        await init_db()
        await _auto_seed()
        app.state.db_available = True
        print("✓ HeatMapper ready — open http://localhost:8000")
    except Exception as exc:
        app.state.db_available = False
        print(f"⚠ DB unavailable ({exc}). API will serve frontend but DB endpoints will fail.")
    yield


app = FastAPI(title="Urban Heat Island Mapper", version="0.1.0", lifespan=lifespan)

# ── CORS (allow the frontend to call the API) ──────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── API routes ──────────────────────────────────────────────────────────
app.include_router(api_router)

# ── Serve frontend static files ────────────────────────────────────────
frontend_dir = Path(__file__).resolve().parent / "app" / "frontend"
app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="frontend")


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the Leaflet.js map dashboard."""
    html_path = frontend_dir / "index.html"
    return HTMLResponse(content=html_path.read_text(), status_code=200)


@app.get("/api/health")
async def health():
    return {"status": "ok", "app": "Urban Heat Island Mapper"}
