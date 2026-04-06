# Outfit Scan API (FastAPI)

Local or hosted server that accepts an outfit photo and returns bounding boxes for visible clothing items.

## Run locally

```bash
cd outfit-scan-backend
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env and set OPENAI_API_KEY (required for real vision; otherwise mock data is returned)
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

- API: `http://127.0.0.1:8000`
- Interactive docs: `http://127.0.0.1:8000/docs`
- Health: `GET /health`

## Security (production)

- Set secrets via environment or host dashboard (**never** commit `.env`).
- **`CORS_ORIGINS`**: comma-separated web origins; avoid `*` in production.
- **Rate limits**: `RATE_LIMIT_IP` and `RATE_LIMIT_USER` (slowapi format, e.g. `60/minute`). Clients may send `X-Client-User-ID` (Firebase UID) for fair per-user bucketing — this is **not** verified auth.
- **Upload cap**: `MAX_UPLOAD_BYTES` (default ~12 MiB).

See repo root `SECURITY.md` for the full threat model.

## iOS app

Point the app base URL to your server (Info.plist `OUTFIT_SCAN_BASE_URL`, UserDefaults `OutfitScanBaseURL`, or the default production URL in code). Physical devices must use a reachable host/LAN IP, not `localhost`.
