# VANTA

VANTA is an AI-powered real-time waste classification and disposal guidance system. This repository currently contains the project foundation only: a Next.js frontend, FastAPI backend, local MySQL infrastructure, and a placeholder for future model training code.

## Repository layout

```text
frontend/  Next.js App Router application
backend/   FastAPI application, database configuration, and tests
training/  Placeholder for future PyTorch/MobileNetV3-Small work
```

## Prerequisites

- Node.js 20.9 or newer and npm
- Python 3.12
- Docker with Docker Compose

## 1. Configure and start MySQL

From the repository root:

```bash
cp .env.example .env
docker compose up -d mysql
docker compose ps
```

The example values are for local development only. Change them before using any shared environment. MySQL is exposed on `127.0.0.1:3308` by default and persists data in the `mysql_data` Docker volume. The nonstandard host port avoids conflicts with other local MySQL services; the container still listens on port `3306` internally.

## 2. Run the backend

From the repository root:

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
cp .env.example .env
alembic upgrade head
uvicorn app.main:app --reload --port 8000
```

The API is available at `http://localhost:8000`, interactive docs at `http://localhost:8000/docs`, and health status at `http://localhost:8000/api/v1/health`.

Run backend tests:

```bash
cd backend
source .venv/bin/activate
pytest
```

Create a future database migration after importing new models into `backend/app/db/base.py`:

```bash
cd backend
source .venv/bin/activate
alembic revision --autogenerate -m "describe change"
alembic upgrade head
```

## 3. Run the frontend

In a second terminal, from the repository root:

```bash
cd frontend
npm install
cp .env.example .env.local
npm run dev
```

Open `http://localhost:3000`.

Frontend checks:

```bash
cd frontend
npm run lint
npm run type-check
npm run build
```

## Stop local infrastructure

```bash
docker compose down
```

To also delete the local database volume, run `docker compose down --volumes`. This permanently removes local database data.
