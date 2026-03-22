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

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

try:
    from openai import OpenAI
    _openai_client: "OpenAI | None" = OpenAI()
except Exception:
    _openai_client = None

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
    id: str
    category: str
    label: str = ""
    confidence: float
    x: float
    y: float
    width: float
    height: float


class ScanOutfitResponse(BaseModel):
    scanId: str
    items: List[ClothingRegion]


app = FastAPI(title="Outfit Scan API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
    if _openai_client is None or not os.getenv("OPENAI_API_KEY"):
        print("[OutfitScanBackend] WARNING: No API key found, returning mock data")
        return [
            {"category": "top", "label": "Top", "confidence": 0.9, "x": 0.15, "y": 0.12, "width": 0.65, "height": 0.35},
        ]

    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    prompt = _build_prompt()

    try:
        completion = _openai_client.chat.completions.create(
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
        return (category, confidence, x, y, w, h, item.get("label", ""))
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


@app.post("/scanOutfit", response_model=ScanOutfitResponse)
async def scan_outfit(
    image: UploadFile = File(..., alias="image"),
    full_image: bool = Query(False),
):
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    image_bytes = await image.read()
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
