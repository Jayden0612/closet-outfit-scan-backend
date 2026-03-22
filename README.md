# Outfit Scan Backend (Mock)

Local FastAPI server that accepts an outfit photo and returns mock detected clothing items.

## Run

```bash
cd outfit-scan-backend
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Server will be at `http://192.168.0.34:8000`. Docs: `http://192.168.0.34:8000/docs`.

## iOS

- **Simulator**: use `http://192.168.0.34:8000` as the API base URL.
- **Physical device**: use your Mac's LAN IP (e.g. `http://192.168.1.100:8000`) instead of localhost, since the device cannot reach the host machine's localhost.
