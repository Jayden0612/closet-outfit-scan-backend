"""
Outfit Scan API.

Accepts an outfit photo and returns rough bounding boxes for the main visible
outfit pieces on the person (top, bottoms, shoes, outerwear). Does not guess
hidden clothing.

Response format (per item):
  - id: unique string for this item
  - category: "top" | "bottom" | "shoes" | "outerwear"
  - confidence: float in [0, 1]
  - x, y, width, height: normalized coordinates in [0, 1] (left, top, width, height)
"""

import base64
import json
import os
import re
import uuid
from typing import List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()

try:
    import cv2
    import numpy as np
    _HAS_OPENCV = True
except ImportError:
    _HAS_OPENCV = False

from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from verify_subscription import router as subscription_router
from promo_routes import router as promo_router
from feedback_routes import router as feedback_router

try:
    from openai import OpenAI

    _HAS_OPENAI = True
except ImportError:
    _HAS_OPENAI = False

# Lazy OpenAI client — only construct when OPENAI_API_KEY is set (avoids import-time side effects).
_openai_client: Optional["OpenAI"] = None


def _get_openai_client() -> Optional["OpenAI"]:
    """Return a configured OpenAI client or None if the SDK/key is unavailable."""
    global _openai_client
    if not _HAS_OPENAI or not os.getenv("OPENAI_API_KEY"):
        return None
    if _openai_client is None:
        _openai_client = OpenAI()
    return _openai_client


# OWASP-friendly defaults: tune via env without code changes.
# RATE_LIMIT_IP: requests per minute per client IP (e.g. shared NAT).
# RATE_LIMIT_USER: requests per minute per X-Client-User-ID when present, else per-IP anonymous bucket.
RATE_LIMIT_IP = os.getenv("RATE_LIMIT_IP", "60/minute")
RATE_LIMIT_USER = os.getenv("RATE_LIMIT_USER", "120/minute")
# Maximum upload size for the scan image (bytes). Prevents memory exhaustion / DoS.
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(12 * 1024 * 1024)))
ALLOWED_IMAGE_CONTENT_TYPES = frozenset(
    {
        "image/jpeg",
        "image/png",
        "image/webp",
    }
)


def _rate_limit_key_ip(request: Request) -> str:
    """Primary bucket: client IP (from proxy headers if configured)."""
    return get_remote_address(request)


def _rate_limit_key_user_or_anon(request: Request) -> str:
    """
    Secondary bucket: Firebase UID from X-Client-User-ID when syntactically valid,
    otherwise anonymous key scoped by IP.

    Security note: this header is NOT authenticated here. It is used only for fair rate-limit
    distribution; abuse prevention still relies on the IP limit and production WAF / API gateway.
    """
    raw = (request.headers.get("X-Client-User-ID") or "").strip()
    if raw and len(raw) <= 128 and re.match(r"^[A-Za-z0-9_-]{1,128}$", raw):
        return f"uid:{raw}"
    return f"anon:{get_remote_address(request)}"


def _cors_origins() -> List[str]:
    """Comma-separated allowlist; default '*' for local dev only. Set CORS_ORIGINS in production."""
    raw = (os.getenv("CORS_ORIGINS") or "*").strip()
    if raw == "*":
        return ["*"]
    return [o.strip() for o in raw.split(",") if o.strip()]

ALLOWED_CATEGORIES: List[str] = ["top", "bottom", "shoes", "outerwear"]

CONFIDENCE_THRESHOLD: float = 0.5
MAX_BOX_AREA: float = 0.45
MIN_BOX_AREA: float = 0.015
OVERLAP_THRESHOLD: float = 0.45
TOP_CENTER_Y_MAX: float = 0.65
BOTTOM_CENTER_Y_MIN: float = 0.3
SHOES_CENTER_Y_MIN: float = 0.5
OUTERWEAR_CENTER_Y_MIN: float = 0.1
OUTERWEAR_CENTER_Y_MAX: float = 0.85


class ClothingRegion(BaseModel):
    """Validated response item — bounds reduce oversized / malicious model output."""

    model_config = ConfigDict(str_strip_whitespace=True)

    id: str = Field(max_length=128)
    category: str = Field(max_length=32)
    label: str = Field(default="", max_length=200)
    confidence: float = Field(ge=0.0, le=1.0)
    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)
    width: float = Field(ge=0.0, le=1.0)
    height: float = Field(ge=0.0, le=1.0)


class ScanOutfitResponse(BaseModel):
    scanId: str = Field(max_length=128)
    items: List[ClothingRegion]


# --- Price detection (scan review pre-fill) ---
MAX_DETECT_PRICE_B64_CHARS = int(os.getenv("MAX_DETECT_PRICE_B64_CHARS", str(18 * 1024 * 1024)))  # ~13MB raw image


class DetectPriceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    image_base64: str = Field(..., min_length=1, max_length=MAX_DETECT_PRICE_B64_CHARS)
    brand: str = Field(default="", max_length=200)
    category: str = Field(..., max_length=64)
    name: str = Field(..., max_length=300)


class DetectPriceResponse(BaseModel):
    detected_name: Optional[str] = None
    price: Optional[float] = None
    confidence: str = "low"
    source: Optional[str] = None


def _detect_price_error_response() -> DetectPriceResponse:
    return DetectPriceResponse(detected_name=None, price=None, confidence="low", source=None)


_DETECT_PRICE_SYSTEM = """You are a fashion retail pricing assistant. Given an image of a clothing item and its metadata, identify the specific product as precisely as possible (model name, colorway, style) and return its typical retail price in USD.

Use web search if you can find a reliable match. If you can find the item on the brand's website or a major retailer, use that price. If you can only estimate based on brand and category, provide your best estimate but flag it as estimated.

Respond ONLY in JSON, no markdown:
{
  "detected_name": "Levi's 501 Original Fit Jeans - Medium Wash",
  "price": 79.50,
  "confidence": "high",
  "source": "levis.com"
}

confidence must be one of: "high", "medium", "low".
source may be a hostname like "levis.com", the string "estimated", or null.

If you cannot determine a price at all, return price as null and confidence as "low"."""


def _call_detect_price_model(image_bytes: bytes, brand: str, category: str, name: str) -> dict:
    """Returns a dict with keys detected_name, price, confidence, source."""
    client = _get_openai_client()
    if client is None:
        return {"detected_name": None, "price": None, "confidence": "low", "source": None}

    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    meta = f"Brand: {brand or '(unknown)'}\nCategory: {category}\nName / label: {name}"

    try:
        completion = client.chat.completions.create(
            model=os.getenv("DETECT_PRICE_MODEL", "gpt-4o"),
            messages=[
                {"role": "system", "content": _DETECT_PRICE_SYSTEM},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}",
                                "detail": "high",
                            },
                        },
                        {"type": "text", "text": meta},
                    ],
                },
            ],
            response_format={"type": "json_object"},
            max_tokens=512,
        )
    except Exception as e:
        print(f"[DetectPrice] OpenAI error: {e}")
        return {"detected_name": None, "price": None, "confidence": "low", "source": None}

    raw_content = completion.choices[0].message.content
    try:
        parsed = json.loads(raw_content or "{}")
    except Exception as e:
        print(f"[DetectPrice] JSON parse error: {e}")
        return {"detected_name": None, "price": None, "confidence": "low", "source": None}

    def _norm_conf(v: object) -> str:
        s = str(v or "").strip().lower()
        if s in ("high", "medium", "low"):
            return s
        return "low"

    price_raw = parsed.get("price")
    price: Optional[float]
    try:
        if price_raw is None:
            price = None
        else:
            price = float(price_raw)
            if price < 0 or price > 1_000_000:
                price = None
    except (TypeError, ValueError):
        price = None

    src = parsed.get("source")
    source: Optional[str]
    if src is None or src == "":
        source = None
    else:
        source = str(src)[:200]

    dn = parsed.get("detected_name")
    detected_name: Optional[str] = None
    if dn is not None and str(dn).strip():
        detected_name = str(dn).strip()[:300]

    return {
        "detected_name": detected_name,
        "price": price,
        "confidence": _norm_conf(parsed.get("confidence")),
        "source": source,
    }


_DETECT_ITEM_SYSTEM = """You are a fashion retail pricing expert with deep knowledge of clothing brands and their current retail prices.

Given an image of a clothing item, identify it as precisely as possible and return ONLY raw JSON with no markdown:
{
  "brand": "Rusty",
  "color": "Navy",
  "category": "top",
  "price": 49.99,
  "confidence": "high",
  "source": "rusty.com"
}

Rules:
- brand: exact brand name as shown on the item (logo, tag, text), or null if not visible
- color: primary color as a simple word ("Black", "Navy", "White", "Cream", "Olive", etc.), never null
- category: one of exactly: "top", "bottom", "shoes", "outerwear", "accessory" — never null
- price: the CURRENT retail price in USD from the brand's official website or major retailer (e.g. ASOS, Nordstrom, SSENSE). Be as precise as possible — do not round to nearest $10. If the brand is visible, use your knowledge of that brand's actual price range for that garment type. A graphic tee from a surf brand is typically $35-55. A basic tee from H&M is $10-20. A Nike hoodie is $55-75. Match the specific garment type and brand tier carefully.
- confidence: "high" if you can identify the specific brand AND garment type with certainty, "medium" if brand is visible but exact style is unclear, "low" if guessing
- source: brand's website domain if known (e.g. "rusty.com"), "estimated" if using brand knowledge, null if unknown

Always return all six keys. Never return markdown or explanation. Prioritize accuracy over confidence — it is better to return a precise price with medium confidence than a rounded guess with high confidence."""

_DETECT_ITEM_ALLOWED_CATEGORIES = frozenset({"top", "bottom", "shoes", "outerwear", "accessory"})


def _detect_item_error_dict() -> dict:
    return {"brand": None, "color": None, "category": None, "price": None, "confidence": "low", "source": None}


def _call_detect_item_model(image_bytes: bytes) -> dict:
    """Returns brand, color, category, price, confidence, source for add-item auto-fill."""
    client = _get_openai_client()
    if client is None:
        return _detect_item_error_dict()

    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    try:
        completion = client.chat.completions.create(
            model=os.getenv("DETECT_PRICE_MODEL", "gpt-4o"),
            messages=[
                {"role": "system", "content": _DETECT_ITEM_SYSTEM},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}",
                                "detail": "high",
                            },
                        },
                        {"type": "text", "text": "Identify this clothing item."},
                    ],
                },
            ],
            response_format={"type": "json_object"},
            max_tokens=300,
            temperature=0.2,
        )
    except Exception as e:
        print(f"[DetectItem] OpenAI error: {e}")
        return _detect_item_error_dict()

    raw_content = completion.choices[0].message.content or "{}"
    try:
        parsed = json.loads(raw_content)
    except Exception as e:
        print(f"[DetectItem] JSON parse error: {e}")
        return _detect_item_error_dict()

    print(f"[DetectItem] result: {parsed}")

    price_raw = parsed.get("price")
    try:
        price = float(price_raw) if price_raw is not None else None
        if price is not None and (price < 0 or price > 1_000_000):
            price = None
    except (TypeError, ValueError):
        price = None

    conf = str(parsed.get("confidence") or "low").lower()
    if conf not in ("high", "medium", "low"):
        conf = "low"

    cat_raw = parsed.get("category")
    category: Optional[str] = None
    if cat_raw is not None:
        c = str(cat_raw).strip().lower()
        if c in _DETECT_ITEM_ALLOWED_CATEGORIES:
            category = c

    brand_raw = parsed.get("brand")
    brand: Optional[str] = None
    if brand_raw is not None:
        b = str(brand_raw).strip()[:200]
        brand = b if b else None

    color_raw = parsed.get("color")
    color: Optional[str] = None
    if color_raw is not None:
        col = str(color_raw).strip()[:120]
        color = col if col else None

    src = parsed.get("source")
    source: Optional[str]
    if src is None or str(src).strip() == "":
        source = None
    else:
        source = str(src).strip()[:200]

    return {
        "brand": brand,
        "color": color,
        "category": category,
        "price": price,
        "confidence": conf,
        "source": source,
    }


limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Outfit Scan API")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.include_router(subscription_router)
app.include_router(promo_router)
app.include_router(feedback_router)

_cors = _cors_origins()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors,
    # Browsers: credentials + wildcard origin is invalid; keep credentials off when using "*".
    allow_credentials=False if _cors == ["*"] else True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.middleware("http")
async def reject_unknown_scan_query_params(request: Request, call_next):
    """Reject unexpected query keys (OWASP: strict input surface for public endpoints)."""
    if request.url.path == "/scanOutfit" and request.query_params:
        for key in request.query_params.keys():
            if key != "full_image":
                return JSONResponse(
                    status_code=400,
                    content={
                        "detail": "Unexpected query parameter.",
                        "allowed_query_parameters": ["full_image"],
                    },
                )
    return await call_next(request)


def _build_prompt() -> str:
    return """
You are a fashion AI. Look at this photo of a person wearing an outfit.
Identify each individual clothing item that is clearly visible (jacket, jeans, shoes, etc).

For each item return a JSON object with:
- "category": one of exactly these values: "top", "bottom", "shoes", "outerwear"
- "label": short human-readable garment name (e.g. "Denim jacket", "Black cargo pants", "White sneakers")
- "confidence": float 0-1
- "x", "y", "width", "height": normalized 0-1 bounding box (x=left, y=top)

Rules:
- Focus ONLY on clothing. Do NOT box the face, head, hair, hands, or bare skin.
- Each box must tightly wrap just that garment.
- For "top": box the shirt/hoodie/jacket torso area (roughly y=0.1 to y=0.5)
- For "bottom": box the pants/jeans area (roughly y=0.45 to y=0.85)  
- For "shoes": box the feet and ankle area (roughly y=0.72 to y=1.0), make sure the full shoe is included including the top of the shoe
- If unsure, omit the item rather than guess.

Output JSON only (no markdown):
{
  "items": [
    {"category": "top", "label": "Denim jacket", "confidence": 0.92, "x": 0.15, "y": 0.12, "width": 0.65, "height": 0.35},
    {"category": "bottom", "label": "Black cargo pants", "confidence": 0.88, "x": 0.2, "y": 0.48, "width": 0.55, "height": 0.35},
    {"category": "shoes", "label": "White sneakers", "confidence": 0.85, "x": 0.25, "y": 0.84, "width": 0.45, "height": 0.14}
  ]
}
"""


def _call_model(image_bytes: bytes) -> List[dict]:
    client = _get_openai_client()
    if client is None:
        print("[OutfitScanBackend] WARNING: No API key found, returning mock data")
        return [
            {"category": "top", "label": "Top", "confidence": 0.9, "x": 0.15, "y": 0.12, "width": 0.65, "height": 0.35},
        ]

    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    prompt = _build_prompt()

    try:
        completion = client.chat.completions.create(
            model=os.getenv("OUTFIT_SCAN_MODEL", "gpt-4o"),
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise fashion vision assistant. Focus on clothing items only. Never box the face or head.",
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}",
                                "detail": "high",
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                },
            ],
            response_format={"type": "json_object"},
            max_tokens=1024,
        )
    except Exception as e:
        print(f"[OutfitScanBackend] OpenAI error: {e}")
        raise HTTPException(status_code=502, detail=f"Vision API error: {e!s}")

    raw_content = completion.choices[0].message.content
    try:
        parsed = json.loads(raw_content or "{}")
        items = parsed.get("items", [])
        if not isinstance(items, list):
            return []
        passthrough = []
        for item in items:
            if not isinstance(item, dict):
                continue
            passthrough.append({
                "category": item.get("category", ""),
                "label": str(item.get("label", "") or ""),
                "confidence": item.get("confidence", 0.0),
                "x": item.get("x", 0.0),
                "y": item.get("y", 0.0),
                "width": item.get("width", 0.0),
                "height": item.get("height", 0.0),
            })
        return passthrough
    except Exception as e:
        print(f"[OutfitScanBackend] Failed to parse model JSON: {e}")
        return []


def _detect_person_and_crop(image_bytes: bytes) -> Optional[Tuple[bytes, float, float, float, float]]:
    if not _HAS_OPENCV:
        return None
    try:
        buf = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            return None
        h, w = img.shape[:2]
        if w < 32 or h < 32:
            return None
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        found, _ = hog.detectMultiScale(img, winStride=(8, 8), padding=(16, 16), scale=1.05)
        if found is None or len(found) == 0:
            return None
        best = max(found, key=lambda r: r[2] * r[3])
        x, y, bw, bh = best
        pad = 0.03
        x1 = max(0, int(x - pad * w))
        y1 = max(0, int(y - pad * h))
        x2 = min(w, int(x + bw + pad * w))
        y2 = min(h, int(y + bh + pad * h))
        if x2 <= x1 or y2 <= y1:
            return None
        crop = img[y1:y2, x1:x2]
        _, enc = cv2.imencode(".jpg", crop)
        crop_bytes = enc.tobytes()
        px = x1 / w
        py = y1 / h
        pw = (x2 - x1) / w
        ph = (y2 - y1) / h
        return (crop_bytes, px, py, pw, ph)
    except Exception as e:
        print(f"[OutfitScan] person detection failed: {e}")
        return None


def _parse_raw_item(item: dict) -> Optional[Tuple[str, float, float, float, float, float, str]]:
    try:
        category = str(item.get("category", "")).strip().lower()
        confidence = float(item.get("confidence", 0.0))
        bbox = item.get("boundingBox")
        if bbox is not None:
            x, y = float(bbox.get("x", 0)), float(bbox.get("y", 0))
            w, h = float(bbox.get("width", 0)), float(bbox.get("height", 0))
        else:
            x = float(item.get("x", 0))
            y = float(item.get("y", 0))
            w = float(item.get("width", 0))
            h = float(item.get("height", 0))
        if not (0.0 <= x < 1.0 and 0.0 <= y < 1.0 and w > 0 and h > 0 and x + w <= 1.0 + 1e-3 and y + h <= 1.0 + 1e-3):
            return None
        raw_label = str(item.get("label", "") or "")
        label = raw_label.strip()[:200]
        return (category, confidence, x, y, w, h, label)
    except (TypeError, ValueError):
        return None


def _iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1 = a[0], a[1]
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx1, by1 = b[0], b[1]
    bx2, by2 = b[0] + b[2], b[1] + b[3]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = a[2] * a[3]
    area_b = b[2] * b[3]
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _filter_and_validate_items(raw_items: List[dict]) -> Tuple[List[ClothingRegion], List[str]]:
    log_lines: List[str] = []
    candidates: List[Tuple[str, float, float, float, float, float, str]] = []
    seen_key: set = set()

    for i, item in enumerate(raw_items):
        parsed = _parse_raw_item(item)
        if parsed is None:
            log_lines.append(f"item {i}: rejected: invalid bbox or parse error")
            continue
        category, confidence, x, y, w, h, label = parsed
        if category not in ALLOWED_CATEGORIES:
            log_lines.append(f"item {i} ({category!r}): rejected: category not allowed")
            continue
        if confidence < CONFIDENCE_THRESHOLD:
            log_lines.append(f"item {i} ({category}): rejected: confidence {confidence:.2f} < {CONFIDENCE_THRESHOLD}")
            continue
        key = (category, round(x, 3), round(y, 3))
        if key in seen_key:
            log_lines.append(f"item {i} ({category}): rejected: duplicate")
            continue
        seen_key.add(key)
        candidates.append((category, confidence, x, y, w, h, label))

    step2 = [(c, cf, x, y, w, h, l) for c, cf, x, y, w, h, l in candidates if w * h <= MAX_BOX_AREA]

    step3: List[Tuple[str, float, float, float, float, float, str]] = []
    for cat, conf, x, y, w, h, label in step2:
        center_y = y + h / 2
        if cat == "top" and center_y > TOP_CENTER_Y_MAX:
            log_lines.append(f"{cat}: rejected: center_y {center_y:.2f} too low for top")
            continue
        if cat == "bottom" and center_y < BOTTOM_CENTER_Y_MIN:
            log_lines.append(f"{cat}: rejected: center_y {center_y:.2f} too high for bottom")
            continue
        if cat == "shoes" and center_y < SHOES_CENTER_Y_MIN:
            log_lines.append(f"{cat}: rejected: center_y {center_y:.2f} too high for shoes")
            continue
        step3.append((cat, conf, x, y, w, h, label))

    step4 = [(c, cf, x, y, w, h, l) for c, cf, x, y, w, h, l in step3 if w * h >= MIN_BOX_AREA]

    step4_sorted = sorted(step4, key=lambda t: -t[1])
    kept: List[Tuple[str, float, float, float, float, float, str]] = []
    for cat, conf, x, y, w, h, label in step4_sorted:
        box = (x, y, w, h)
        overlaps = any(_iou(box, (kx, ky, kw, kh)) > OVERLAP_THRESHOLD for _, _, kx, ky, kw, kh, _ in kept)
        if not overlaps:
            kept.append((cat, conf, x, y, w, h, label))

    regions = [
        ClothingRegion(id=str(uuid.uuid4()), category=cat, label=label, confidence=conf, x=x, y=y, width=w, height=h)
        for cat, conf, x, y, w, h, label in kept
    ]
    return regions, log_lines


def _normalize_mime(ct: Optional[str]) -> Optional[str]:
    if not ct:
        return None
    return ct.split(";")[0].strip().lower()


async def _read_upload_with_limit(upload: UploadFile) -> bytes:
    """Read multipart body in chunks with a hard cap (OWASP: limit resource exhaustion)."""
    chunks: List[bytes] = []
    total = 0
    chunk_size = 64 * 1024
    while True:
        chunk = await upload.read(chunk_size)
        if not chunk:
            break
        total += len(chunk)
        if total > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Image exceeds maximum size of {MAX_UPLOAD_BYTES} bytes",
            )
        chunks.append(chunk)
    return b"".join(chunks)


@app.get("/health")
async def health():
    """Liveness probe for load balancers (Railway, etc.)."""
    return {"status": "ok"}


@app.post("/detect-price", response_model=DetectPriceResponse)
@limiter.limit(RATE_LIMIT_IP, key_func=_rate_limit_key_ip)
@limiter.limit(RATE_LIMIT_USER, key_func=_rate_limit_key_user_or_anon)
async def detect_price(request: Request, body: DetectPriceRequest) -> DetectPriceResponse:
    """
    AI retail price hint for a single garment image + metadata.
    On any failure returns 200 with null price and confidence low (client stays frictionless).
    """
    try:
        # Padding may be present; use standard decode
        raw = base64.b64decode(body.image_base64)
    except Exception:
        return _detect_price_error_response()
    if not raw or len(raw) > MAX_UPLOAD_BYTES:
        return _detect_price_error_response()
    try:
        d = _call_detect_price_model(raw, body.brand.strip(), body.category.strip(), body.name.strip())
        conf = str(d.get("confidence") or "low").lower()
        if conf not in ("high", "medium", "low"):
            conf = "low"
        return DetectPriceResponse(
            detected_name=d.get("detected_name"),
            price=d.get("price"),
            confidence=conf,
            source=d.get("source"),
        )
    except Exception as e:
        print(f"[DetectPrice] handler error: {e}")
        return _detect_price_error_response()


@app.post("/detect-item")
@limiter.limit(RATE_LIMIT_IP, key_func=_rate_limit_key_ip)
@limiter.limit(RATE_LIMIT_USER, key_func=_rate_limit_key_user_or_anon)
async def detect_item(request: Request, body: DetectPriceRequest) -> JSONResponse:
    """
    Full garment field hints for manual add-item flow (brand, color, category, price).
    Same JSON request shape as /detect-price. On failure returns 200 with null fields.
    """
    try:
        raw = base64.b64decode(body.image_base64)
    except Exception:
        return JSONResponse(_detect_item_error_dict())
    if not raw or len(raw) > MAX_UPLOAD_BYTES:
        return JSONResponse(_detect_item_error_dict())
    try:
        d = _call_detect_item_model(raw)
        return JSONResponse(d)
    except Exception as e:
        print(f"[DetectItem] handler error: {e}")
        return JSONResponse(_detect_item_error_dict())


@app.post("/scanOutfit", response_model=ScanOutfitResponse)
@limiter.limit(RATE_LIMIT_IP, key_func=_rate_limit_key_ip)
@limiter.limit(RATE_LIMIT_USER, key_func=_rate_limit_key_user_or_anon)
async def scan_outfit(
    request: Request,
    image: UploadFile = File(..., alias="image"),
    full_image: bool = Query(False),
):
    # Filename length (metadata only; still bounded).
    if image.filename is not None and len(image.filename) > 255:
        raise HTTPException(status_code=400, detail="Filename too long")

    mime = _normalize_mime(image.content_type)
    if mime not in ALLOWED_IMAGE_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported Content-Type. Allowed: {sorted(ALLOWED_IMAGE_CONTENT_TYPES)}",
        )

    image_bytes = await _read_upload_with_limit(image)
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image upload")

    if full_image:
        person_result = _detect_person_and_crop(image_bytes)
        if person_result is not None:
            crop_bytes, px, py, pw, ph = person_result
            raw_items = _call_model(crop_bytes)
            mapped = []
            for item in raw_items:
                p = _parse_raw_item(item)
                if p is None:
                    continue
                cat, conf, cx, cy, cw, ch, label = p
                mapped.append({"category": cat, "label": label, "confidence": conf,
                                "x": px + cx * pw, "y": py + cy * ph,
                                "width": cw * pw, "height": ch * ph})
            raw_items = mapped
        else:
            raw_items = _call_model(image_bytes)
    else:
        raw_items = _call_model(image_bytes)

    print(f"[OutfitScan] raw model output: {json.dumps(raw_items, indent=2)}")
    print(f"[OutfitScan] raw items with labels: {[(i.get('category'), i.get('label', 'NO LABEL')) for i in raw_items]}")

    cleaned_items, rejection_log = _filter_and_validate_items(raw_items)
    for line in rejection_log:
        print(f"[OutfitScan] {line}")
    print(f"[OutfitScan] filtered items: {[(r.category, r.x, r.y, r.width, r.height) for r in cleaned_items]}")

    return ScanOutfitResponse(scanId=str(uuid.uuid4()), items=cleaned_items)


# --- Wishlist AI Take ---

class WishlistAITakeWardrobeItem(BaseModel):
    name: str = Field(default="", max_length=300)
    category: str = Field(default="", max_length=64)
    color: str = Field(default="", max_length=120)
    brand: str = Field(default="", max_length=200)


class WishlistAITakeRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    itemName: str = Field(..., max_length=300)
    brand: str = Field(default="", max_length=200)
    price: Optional[float] = None
    cooldownDate: Optional[str] = None
    wardrobe: List[WishlistAITakeWardrobeItem] = []


class WishlistAITakeResponse(BaseModel):
    aiTake: str


_WISHLIST_AI_TAKE_SYSTEM = """You are a concise personal stylist giving honest buy-or-wait advice. Given a wishlist item and the user's existing wardrobe, respond in three flowing sentences with no headers or bullet points:
1. A direct buy or wait recommendation and why (be honest, not just encouraging).
2. If a price is provided, estimate cost-per-wear at 20, 30, and 50 wears (e.g. "$4, $2.67, $2 per wear").
3. Name 2-3 specific wardrobe items this pairs well with, or note if nothing matches well.
Keep the total response under 80 words."""


@app.post("/wishlist-ai-take", response_model=WishlistAITakeResponse)
@limiter.limit(RATE_LIMIT_IP, key_func=_rate_limit_key_ip)
@limiter.limit(RATE_LIMIT_USER, key_func=_rate_limit_key_user_or_anon)
async def wishlist_ai_take(request: Request, body: WishlistAITakeRequest) -> WishlistAITakeResponse:
    client = _get_openai_client()
    if client is None:
        return WishlistAITakeResponse(aiTake="AI unavailable — missing API key.")

    wardrobe_lines = "\n".join(
        f"- {w.name} ({w.category}, {w.color}, {w.brand})" for w in body.wardrobe[:50]
    ) or "No wardrobe items provided."

    price_line = f"Price: ${body.price:.2f}" if body.price is not None else "Price: not provided"
    cooldown_line = f"Cool-down date: {body.cooldownDate}" if body.cooldownDate else ""

    user_message = f"""Wishlist item: {body.itemName}
Brand: {body.brand or "unknown"}
{price_line}
{cooldown_line}

User's wardrobe:
{wardrobe_lines}"""

    try:
        completion = client.chat.completions.create(
            model=os.getenv("OUTFIT_SCAN_MODEL", "gpt-4o"),
            messages=[
                {"role": "system", "content": _WISHLIST_AI_TAKE_SYSTEM},
                {"role": "user", "content": user_message},
            ],
            max_tokens=160,
            temperature=0.7,
        )
        result = (completion.choices[0].message.content or "").strip()
        return WishlistAITakeResponse(aiTake=result if result else "No take available.")
    except Exception as e:
        print(f"[WishlistAITake] OpenAI error: {e}")
        return WishlistAITakeResponse(aiTake="Couldn't generate a take right now. Try again shortly.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
