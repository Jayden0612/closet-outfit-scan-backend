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
from datetime import datetime, timezone
import urllib.error
import urllib.parse
import urllib.request
from typing import List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()

# Backend secrets (Railway / `.env` — never hardcode):
# OPENAI_API_KEY, ANTHROPIC_API_KEY (style-search vision), SERPAPI_API_KEY (style-search images)

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
SHOES_CENTER_Y_MIN: float = 0.45  # lowered from 0.5 to catch more shoes
OUTERWEAR_CENTER_Y_MIN: float = 0.1
OUTERWEAR_CENTER_Y_MAX: float = 0.85


class ClothingRegion(BaseModel):
    """Validated response item — bounds reduce oversized / malicious model output."""

    model_config = ConfigDict(str_strip_whitespace=True)

    id: str = Field(max_length=128)
    category: str = Field(max_length=32)
    label: str = Field(default="", max_length=200)
    color: str = Field(default="", max_length=80)
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


class WishlistWardrobeItem(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    name: str = Field(default="", max_length=300)
    category: str = Field(default="", max_length=80)
    color: str = Field(default="", max_length=80)
    brand: str = Field(default="", max_length=200)


class WishlistAITakeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    itemName: str = Field(..., max_length=300)
    brand: str = Field(default="", max_length=200)
    price: Optional[float] = None
    cooldownDate: Optional[str] = Field(default=None, max_length=80)
    wardrobe: List[WishlistWardrobeItem] = Field(default_factory=list)


class WishlistAITakeResponse(BaseModel):
    aiTake: str


class RedeemPromoRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    code: str = Field(..., min_length=1, max_length=64)
    userId: str = Field(..., min_length=1, max_length=128)


# --- Style search (image inspiration via SerpApi Google Images) ---
STYLE_SEARCH_CLAUDE_MODEL = "claude-sonnet-4-20250514"
MAX_STYLE_SEARCH_B64_CHARS = int(os.getenv("MAX_STYLE_SEARCH_B64_CHARS", str(18 * 1024 * 1024)))

_STYLE_SEARCH_VISION_PROMPT = (
    "Identify the clothing item in this image. Return only a short descriptive label like "
    "'oversized yellow linen shirt' or 'white leather sneakers'. Be specific about color, "
    "fit, and material if visible."
)


class StyleSearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    user_id: str = Field(..., min_length=1, max_length=128)
    query: Optional[str] = Field(default=None, max_length=300)
    image_base64: Optional[str] = Field(default=None, max_length=MAX_STYLE_SEARCH_B64_CHARS)


class StyleSearchResultItem(BaseModel):
    image_url: str
    source_url: str
    title: str
    item_identified: str


class StyleSearchResponse(BaseModel):
    results: List[StyleSearchResultItem]


def _detect_price_error_response() -> DetectPriceResponse:
    return DetectPriceResponse(detected_name=None, price=None, confidence="low", source=None)


def _wishlist_ai_take_error_response() -> WishlistAITakeResponse:
    return WishlistAITakeResponse(aiTake="")


def _style_search_strip_data_uri(b64: str) -> Tuple[str, str]:
    """Return (raw base64 payload, media_type) from a plain or data-URI string."""
    raw = b64.strip()
    if raw.lower().startswith("data:") and "," in raw:
        header, _, payload = raw.partition(",")
        media = "image/jpeg"
        if ";" in header:
            type_part = header[5:].split(";", 1)[0].strip().lower()
            if type_part:
                media = type_part
        return payload.strip(), media
    return raw, "image/jpeg"


def _call_style_search_vision(image_bytes: bytes, media_type: str) -> str:
    api_key = (os.getenv("ANTHROPIC_API_KEY") or "").strip()
    if not api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY is not configured")

    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    payload = {
        "model": STYLE_SEARCH_CLAUDE_MODEL,
        "max_tokens": 120,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_b64,
                        },
                    },
                    {"type": "text", "text": _STYLE_SEARCH_VISION_PROMPT},
                ],
            }
        ],
    }
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        raise HTTPException(status_code=500, detail=f"Claude Vision error: {err_body}") from e
    except urllib.error.URLError as e:
        raise HTTPException(status_code=500, detail=f"Claude Vision error: {e!s}") from e

    for block in body.get("content") or []:
        if isinstance(block, dict) and block.get("type") == "text":
            label = str(block.get("text") or "").strip()
            if label:
                return label[:300]
    raise HTTPException(status_code=500, detail="Claude Vision returned no label")


class StyleSearchError(Exception):
    """Raised when the SerpApi image search cannot return results (missing key, HTTP/network error)."""


def _serpapi_style_image_search(search_query: str) -> dict:
    api_key = (os.environ.get("SERPAPI_API_KEY") or "").strip()
    print(
        f"[StyleSearch] SERPAPI_API_KEY from os.environ: "
        f"set={bool(api_key)} len={len(api_key)}"
    )
    if not api_key:
        raise StyleSearchError("SERPAPI_API_KEY is not configured")

    params = urllib.parse.urlencode(
        {
            "engine": "google_images",
            "q": search_query,
            "api_key": api_key,
            "num": 20,
            "safe": "active",
        }
    )
    url = f"https://serpapi.com/search.json?{params}"
    # Log full URL with API key redacted (never print the real key).
    log_url = re.sub(r"([&?])api_key=[^&]*", r"\1api_key=***", url)
    print(f"[StyleSearch] SerpApi URL: {log_url}")
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            print(f"[StyleSearch] SerpApi response status: {resp.status}")
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        print(f"[StyleSearch] SerpApi response status: {e.code}")
        print(f"[StyleSearch] SerpApi error body: {err_body[:800]}")
        raise StyleSearchError(f"SerpApi HTTP {e.code}: {err_body[:300]}") from e
    except urllib.error.URLError as e:
        print(f"[StyleSearch] SerpApi request failed (no HTTP status): {e!s}")
        raise StyleSearchError(f"SerpApi request failed: {e!s}") from e


def _parse_serpapi_image_results(data: dict, item_identified: str) -> List[StyleSearchResultItem]:
    items = data.get("images_results") or []
    results: List[StyleSearchResultItem] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        image_url = str(item.get("original") or item.get("thumbnail") or "").strip()
        if not image_url:
            continue
        source_url = str(item.get("link") or item.get("source") or image_url).strip()
        title = str(item.get("title") or "").strip()[:500]
        results.append(
            StyleSearchResultItem(
                image_url=image_url,
                source_url=source_url,
                title=title,
                item_identified=item_identified,
            )
        )
    return results


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


_DETECT_ITEM_SYSTEM = """You are a clothing item scanner for a wardrobe app. The user will send you a photo of a SINGLE clothing item laid flat or held up.

FIRST — check if the image is scannable:
- If the image contains a person wearing multiple items (mirror selfie, full outfit photo, gym photo, etc.) return: {"error": "full_outfit", "message": "Please photograph one item at a time, laid flat or held up separately."}
- If the image contains multiple clothing items visible at once return: {"error": "multiple_items", "message": "Please photograph one item at a time."}
- If the image is blurry, dark, or the clothing item is not clearly visible return: {"error": "unclear_image", "message": "Please take a clearer photo with good lighting."}
- If there is no clothing item in the image at all return: {"error": "no_item", "message": "No clothing item detected. Please photograph a single clothing item."}

ONLY if the image shows a single clearly visible clothing item, return ONLY raw JSON with no markdown:
{
  "brand": "Rusty",
  "color": "Navy",
  "category": "top",
  "price": 49.99,
  "confidence": "high",
  "source": "rusty.com"
}

Rules for valid clothing item scan:
- brand: exact brand name as shown on the item (logo, tag, text), or null if not visible
- color: primary color as a simple word ("Black", "Navy", "White", "Cream", "Olive", etc.), never null
- category: MUST be exactly one of these five strings only: "top", "bottom", "shoes", "outerwear", "accessory". Never use subtypes like "pants", "sneakers", or "jacket" — map mentally to the parent bucket (e.g. pants → "bottom", sneakers → "shoes", jacket → "outerwear"). Never null. Focus on the SINGLE most prominent item only.
- price: the CURRENT retail price in USD from the brand's official website or major retailer (e.g. ASOS, Nordstrom, SSENSE). Be as precise as possible — do not round to nearest $10. If the brand is visible, use your knowledge of that brand's actual price range for that garment type. A graphic tee from a surf brand is typically $35-55. A basic tee from H&M is $10-20. A Nike hoodie is $55-75. Match the specific garment type and brand tier carefully.
- confidence: "high" if you can identify the specific brand AND garment type with certainty, "medium" if brand is visible but exact style is unclear, "low" if guessing
- source: brand's website domain if known (e.g. "rusty.com"), "estimated" if using brand knowledge, null if unknown

Always return all six keys for valid items. Never return markdown or explanation."""

_DETECT_ITEM_ALLOWED_CATEGORIES = frozenset({"top", "bottom", "shoes", "outerwear", "accessory"})

_DETECT_ITEM_CATEGORY_SYNONYMS: dict[str, str] = {}
for _parent, _labels in (
    ("top", ("top", "shirt", "tshirt", "t-shirt", "blouse", "sweater", "hoodie")),
    ("bottom", ("bottom", "pants", "jeans", "shorts", "skirt", "trousers")),
    ("shoes", ("shoes", "sneakers", "boots", "heels", "sandals", "footwear")),
    ("outerwear", ("outerwear", "coat", "jacket", "blazer")),
    ("accessory", ("accessory", "accessories", "bag", "hat", "belt", "watch")),
):
    for _label in _labels:
        _DETECT_ITEM_CATEGORY_SYNONYMS[_label] = _parent


def _normalize_detect_item_category(raw: object) -> Optional[str]:
    if raw is None:
        return None
    c = str(raw).strip().lower()
    if c in _DETECT_ITEM_ALLOWED_CATEGORIES:
        return c
    return _DETECT_ITEM_CATEGORY_SYNONYMS.get(c)

_SCAN_TAG_SYSTEM = """You are a fashion tag reader. The user will send you a photo of a clothing tag. Extract all readable information and return ONLY a JSON object with these fields:
{
  brand: string or null,
  size: string or null,
  fabric: string or null (e.g. '100% Cotton'),
  care_instructions: string or null (describe wash symbols in plain English),
  country_of_origin: string or null,
  styling_notes: string (2-3 sentences of styling tips based on the fabric and brand),
  item_name: string or null (best-guess product name from brand/tag clues),
  item_category: string or null (one of: top, bottom, shoes, outerwear, accessory, unknown),
  item_color: string or null (best guess from visible garment/tag context),
  match_confidence: string or null (high, medium, low),
  lookup_source: string or null (domain if known, otherwise inferred)
}
Return only valid JSON, no markdown, no preamble."""


def _detect_item_error_dict() -> dict:
    return {"brand": None, "color": None, "category": None, "price": None, "confidence": "low", "source": None}


def _scan_tag_error() -> HTTPException:
    return HTTPException(status_code=422, detail="Could not read tag")


def _scan_tag_optional_str(value: object, max_len: int = 400) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text[:max_len]


def _call_scan_tag_model(image_bytes: bytes, mime: str) -> dict:
    client = _get_openai_client()
    if client is None:
        raise _scan_tag_error()

    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    try:
        completion = client.chat.completions.create(
            model=os.getenv("SCAN_TAG_MODEL", "gpt-4o"),
            messages=[
                {"role": "system", "content": _SCAN_TAG_SYSTEM},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime};base64,{image_b64}",
                                "detail": "high",
                            },
                        },
                        {"type": "text", "text": "Read this clothing tag."},
                    ],
                },
            ],
            response_format={"type": "json_object"},
            max_tokens=420,
            temperature=0.1,
        )
    except Exception as e:
        print(f"[ScanTag] OpenAI error: {e}")
        raise _scan_tag_error()

    raw_content = completion.choices[0].message.content or "{}"
    try:
        parsed = json.loads(raw_content)
    except Exception as e:
        print(f"[ScanTag] JSON parse error: {e}")
        raise _scan_tag_error()

    if not isinstance(parsed, dict):
        raise _scan_tag_error()

    item_category_raw = _scan_tag_optional_str(parsed.get("item_category"), max_len=40)
    item_category: Optional[str]
    if item_category_raw is None:
        item_category = None
    else:
        normalized_category = item_category_raw.lower()
        allowed_categories = {"top", "bottom", "shoes", "outerwear", "accessory", "unknown"}
        item_category = normalized_category if normalized_category in allowed_categories else "unknown"

    confidence_raw = _scan_tag_optional_str(parsed.get("match_confidence"), max_len=20)
    match_confidence: Optional[str]
    if confidence_raw is None:
        match_confidence = None
    else:
        normalized_confidence = confidence_raw.lower()
        match_confidence = normalized_confidence if normalized_confidence in {"high", "medium", "low"} else "low"

    return {
        "brand": _scan_tag_optional_str(parsed.get("brand"), max_len=200),
        "size": _scan_tag_optional_str(parsed.get("size"), max_len=120),
        "fabric": _scan_tag_optional_str(parsed.get("fabric"), max_len=240),
        "care_instructions": _scan_tag_optional_str(parsed.get("care_instructions"), max_len=400),
        "country_of_origin": _scan_tag_optional_str(parsed.get("country_of_origin"), max_len=120),
        "styling_notes": _scan_tag_optional_str(parsed.get("styling_notes"), max_len=500),
        "item_name": _scan_tag_optional_str(parsed.get("item_name"), max_len=240),
        "item_category": item_category,
        "item_color": _scan_tag_optional_str(parsed.get("item_color"), max_len=80),
        "match_confidence": match_confidence,
        "lookup_source": _scan_tag_optional_str(parsed.get("lookup_source"), max_len=120),
    }


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

    # Handle error responses from the model (unscannable images).
    if isinstance(parsed, dict) and "error" in parsed:
        error_code = parsed.get("error", "unknown")
        error_message = parsed.get("message", "Could not scan this image.")
        print(f"[DetectItem] Model rejected image: {error_code} — {error_message}")
        return {
            "brand": None,
            "color": None,
            "category": None,
            "price": None,
            "confidence": "low",
            "source": None,
            "error": error_code,
            "error_message": error_message,
        }

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

    category = _normalize_detect_item_category(parsed.get("category"))

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


def _ensure_firebase_admin_initialized() -> None:
    """Initialize firebase_admin once for Firestore writes."""
    import firebase_admin
    from firebase_admin import credentials

    if firebase_admin._apps:
        return

    cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
    if cred_path and os.path.isfile(cred_path):
        firebase_admin.initialize_app(credentials.Certificate(cred_path))
        return

    json_str = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON", "").strip()
    if json_str:
        info = json.loads(json_str)
        firebase_admin.initialize_app(credentials.Certificate(info))
        return

    # Fall back to ADC in Railway/cloud environments.
    firebase_admin.initialize_app()


limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Outfit Scan API")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.include_router(subscription_router)
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
You are a precise fashion AI analyzing a photo of a person wearing an outfit.
Your job is to identify each INDIVIDUAL clothing item and return tight bounding boxes around ONLY the clothing — never the person's face, head, hair, or skin.

CRITICAL RULES:
- A "top" box should wrap ONLY the shirt/hoodie/jacket fabric — start below the chin (y > 0.12) and end at the waist
- A "bottom" box should wrap ONLY the pants/jeans/shorts — start at the waist and end above the ankles
- A "shoes" box should wrap ONLY the shoes/feet — start just above the ankle and go to the bottom of the image. ALWAYS include shoes if feet are visible, even partially.
- NEVER let a bounding box include the person's face or head
- Each box must be TIGHT around just that specific garment — do not use large loose boxes
- If an item is partially cut off by the image edge, still include it with confidence 0.6+

For each item return:
- "category": exactly one of: "top", "bottom", "shoes", "outerwear"
- "label": specific garment name (e.g. "White graphic t-shirt", "Blue straight-leg jeans", "White New Balance sneakers")
- "color": primary color as one word (e.g. "White", "Blue", "Black")
- "confidence": float 0.0-1.0
- "x", "y", "width", "height": normalized 0-1 bounding box (x=left edge, y=top edge)

Typical vertical positions for a full-body standing photo:
- Top/shirt: y=0.15 to y=0.50, avoid including face above y=0.12
- Bottom/pants: y=0.45 to y=0.85
- Shoes: y=0.78 to y=1.0 (go all the way to 1.0 to capture full shoe)
- Outerwear/jacket: y=0.10 to y=0.65

Output JSON only, no markdown:
{
  "items": [
    {"category": "top", "label": "White graphic t-shirt", "color": "White", "confidence": 0.95, "x": 0.20, "y": 0.15, "width": 0.55, "height": 0.30},
    {"category": "bottom", "label": "Blue straight-leg jeans", "color": "Blue", "confidence": 0.92, "x": 0.22, "y": 0.48, "width": 0.50, "height": 0.38},
    {"category": "shoes", "label": "White sneakers", "color": "White", "confidence": 0.88, "x": 0.25, "y": 0.82, "width": 0.45, "height": 0.18}
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
                "color": str(item.get("color", "") or ""),
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


def _parse_raw_item(item: dict) -> Optional[Tuple[str, float, float, float, float, float, str, str]]:
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
        raw_color = str(item.get("color", "") or "")
        color = raw_color.strip()[:80]
        return (category, confidence, x, y, w, h, label, color)
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
    candidates: List[Tuple[str, float, float, float, float, float, str, str]] = []
    seen_key: set = set()

    for i, item in enumerate(raw_items):
        parsed = _parse_raw_item(item)
        if parsed is None:
            log_lines.append(f"item {i}: rejected: invalid bbox or parse error")
            continue
        category, confidence, x, y, w, h, label, color = parsed
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
        candidates.append((category, confidence, x, y, w, h, label, color))

    step2 = [(c, cf, x, y, w, h, l, col) for c, cf, x, y, w, h, l, col in candidates if w * h <= MAX_BOX_AREA]

    step3: List[Tuple[str, float, float, float, float, float, str, str]] = []
    for cat, conf, x, y, w, h, label, color in step2:
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
        step3.append((cat, conf, x, y, w, h, label, color))

    step4 = [(c, cf, x, y, w, h, l, col) for c, cf, x, y, w, h, l, col in step3 if w * h >= MIN_BOX_AREA]

    step4_sorted = sorted(step4, key=lambda t: -t[1])
    kept: List[Tuple[str, float, float, float, float, float, str, str]] = []
    for cat, conf, x, y, w, h, label, color in step4_sorted:
        box = (x, y, w, h)
        overlaps = any(_iou(box, (kx, ky, kw, kh)) > OVERLAP_THRESHOLD for _, _, kx, ky, kw, kh, _, _ in kept)
        if not overlaps:
            kept.append((cat, conf, x, y, w, h, label, color))

    regions = [
        ClothingRegion(id=str(uuid.uuid4()), category=cat, label=label, color=color, confidence=conf, x=x, y=y, width=w, height=h)
        for cat, conf, x, y, w, h, label, color in kept
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


@app.post("/redeem-promo")
async def redeem_promo(body: RedeemPromoRequest):
    """
    Redeem a promo code and grant account-level Pro in Firestore.
    """
    try:
        _ensure_firebase_admin_initialized()
        from firebase_admin import firestore

        if body.code.strip().lower() != "famandfri":
            raise HTTPException(status_code=400, detail="Invalid or expired promo code.")

        db = firestore.client()
        user_ref = db.collection("users").document(body.userId)
        user_ref.set(
            {
                # Current app entitlement fields:
                "isPremium": True,
                "promoUnlocked": True,
                # Promo metadata:
                "promoCode": "Famandfri",
                "promoRedeemedAt": datetime.now(timezone.utc),
            },
            merge=True,
        )
        return {"success": True, "message": "Code applied! Enjoy Pro access."}
    except HTTPException:
        raise
    except Exception as e:
        print(f"[Promo] Failed to apply code for user {body.userId}: {e}")
        raise HTTPException(status_code=500, detail="Failed to apply code.")


@app.post("/style-search", response_model=StyleSearchResponse)
@limiter.limit(RATE_LIMIT_IP, key_func=_rate_limit_key_ip)
@limiter.limit(RATE_LIMIT_USER, key_func=_rate_limit_key_user_or_anon)
async def style_search(request: Request, body: StyleSearchRequest) -> StyleSearchResponse:
    print(f"[StyleSearch] POST /style-search hit user_id={body.user_id[:12]}...")
    try:
        query_text = (body.query or "").strip()
        image_b64_raw = (body.image_base64 or "").strip()

        if not query_text and not image_b64_raw:
            raise HTTPException(
                status_code=400,
                detail="Provide at least one of query or image_base64",
            )

        if image_b64_raw:
            b64_payload, media_type = _style_search_strip_data_uri(image_b64_raw)
            try:
                image_bytes = base64.b64decode(b64_payload, validate=False)
            except Exception as e:
                raise HTTPException(status_code=400, detail="Invalid image_base64") from e
            if not image_bytes or len(image_bytes) > MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=400, detail="Invalid or oversized image")
            item_identified = _call_style_search_vision(image_bytes, media_type)
        else:
            item_identified = query_text

        search_query = f"{item_identified} outfit ideas how to style"
        print(f"[StyleSearch] SerpApi search query: {search_query!r}")
        try:
            serp_data = _serpapi_style_image_search(search_query)
            results = _parse_serpapi_image_results(serp_data, item_identified)
        except StyleSearchError as e:
            # SerpApi is the only image search path — no Google fallback. On failure, return
            # an empty result set so the feature degrades gracefully instead of erroring out.
            print(f"[StyleSearch] SerpApi failed — returning empty results: {e}")
            return StyleSearchResponse(results=[])
        print(f"[StyleSearch] Returning {len(results)} image results")
        return StyleSearchResponse(results=results)
    except HTTPException:
        raise
    except Exception as e:
        print(f"[StyleSearch] handler error: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/wishlist-ai-take", response_model=WishlistAITakeResponse)
@limiter.limit(RATE_LIMIT_IP, key_func=_rate_limit_key_ip)
@limiter.limit(RATE_LIMIT_USER, key_func=_rate_limit_key_user_or_anon)
async def wishlist_ai_take(request: Request, body: WishlistAITakeRequest) -> WishlistAITakeResponse:
    client = _get_openai_client()
    if client is None:
        return _wishlist_ai_take_error_response()

    try:
        wardrobe_lines: List[str] = []
        for idx, piece in enumerate(body.wardrobe[:60], start=1):
            wardrobe_lines.append(
                f"{idx}. {piece.name or 'Item'} | category: {piece.category or 'Unknown'} | color: {piece.color or 'Unknown'} | brand: {piece.brand or ''}"
            )
        wardrobe_text = "\n".join(wardrobe_lines) if wardrobe_lines else "No wardrobe items provided."
        price_text = "not provided" if body.price is None else f"${body.price:.2f}"
        cooldown_text = body.cooldownDate or "not provided"
        user_message = (
            f"Wishlist item name: {body.itemName}\n"
            f"Brand: {body.brand or ''}\n"
            f"Price: {price_text}\n"
            f"Cooldown date: {cooldown_text}\n"
            f"Wardrobe context:\n{wardrobe_text}"
        )

        completion = client.chat.completions.create(
            model=os.getenv("WISHLIST_AI_TAKE_MODEL", "gpt-4o"),
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a concise personal stylist. Given a wishlist item and the user's existing wardrobe, "
                        "respond in 3 short sections with NO headers or bullet points — just flowing sentences: "
                        "1. Buy or wait recommendation and why (be direct and honest) "
                        "2. If price is provided, estimate cost-per-wear assuming 20, 30, and 50 wears "
                        "3. Name 2-3 specific wardrobe items this pairs well with, or note if nothing matches well. "
                        "Keep the total response under 80 words."
                    ),
                },
                {"role": "user", "content": user_message},
            ],
            max_tokens=220,
        )
        ai_take = (completion.choices[0].message.content or "").strip()
        if not ai_take:
            return _wishlist_ai_take_error_response()
        return WishlistAITakeResponse(aiTake=ai_take[:800])
    except Exception as e:
        print(f"[WishlistAITake] handler error: {e}")
        return _wishlist_ai_take_error_response()


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
                cat, conf, cx, cy, cw, ch, label, color = p
                mapped.append({"category": cat, "label": label, "color": color, "confidence": conf,
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


@app.post("/scan-tag")
@limiter.limit(RATE_LIMIT_IP, key_func=_rate_limit_key_ip)
@limiter.limit(RATE_LIMIT_USER, key_func=_rate_limit_key_user_or_anon)
async def scan_tag(
    request: Request,
    image: UploadFile = File(..., alias="image"),
):
    if image.filename is not None and len(image.filename) > 255:
        raise HTTPException(status_code=400, detail="Filename too long")

    mime = _normalize_mime(image.content_type)
    allowed = {"image/jpeg", "image/png"}
    if mime not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported Content-Type. Allowed: {sorted(allowed)}",
        )

    image_bytes = await _read_upload_with_limit(image)
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image upload")

    try:
        payload = _call_scan_tag_model(image_bytes, mime)
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ScanTag] handler error: {e}")
        raise _scan_tag_error()

    if not isinstance(payload, dict):
        raise _scan_tag_error()
    return JSONResponse(payload)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)