"""
POST /redeem-promo — validate promo codes and set users/{uid}.promoUnlocked in Firestore.

Requires Firebase Admin (service account on server). Authorization: Bearer <Firebase ID token>.
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional

from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

router = APIRouter(tags=["promo"])

VALID_CODES: frozenset[str] = frozenset(
    {
        "CLOSET2024",
        "VIPACCESS",
        "JAYDENVIP",
        "CLOSETAMBSDR",  # ClosetAmbsdr (case-insensitive via _normalize_code)
    }
)


def _ensure_firebase() -> None:
    """Initialize firebase_admin once; uses GOOGLE_APPLICATION_CREDENTIALS, FIREBASE_SERVICE_ACCOUNT_JSON, or ADC."""
    import firebase_admin
    from firebase_admin import credentials

    if firebase_admin._apps:
        return

    cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
    if cred_path and os.path.isfile(cred_path):
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred)
        return

    json_str = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON", "").strip()
    if json_str:
        info = json.loads(json_str)
        cred = credentials.Certificate(info)
        firebase_admin.initialize_app(cred)
        return

    try:
        firebase_admin.initialize_app()
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="Firebase Admin not configured. Set GOOGLE_APPLICATION_CREDENTIALS or FIREBASE_SERVICE_ACCOUNT_JSON.",
        ) from exc


class RedeemPromoBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str = Field(..., min_length=1, max_length=64)
    userId: str = Field(..., min_length=1, max_length=128)


def _normalize_code(raw: str) -> str:
    return raw.strip().upper()


@router.post("/redeem-promo")
async def redeem_promo(
    body: RedeemPromoBody,
    authorization: Optional[str] = Header(default=None),
) -> Any:
    """
    Redeem a promo code for the authenticated user. userId must match the ID token uid.
    """
    try:
        _ensure_firebase()
    except HTTPException:
        raise
    except ImportError as e:
        raise HTTPException(status_code=503, detail="firebase-admin not installed") from e

    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")

    token_str = authorization[7:].strip()
    if not token_str:
        raise HTTPException(status_code=401, detail="Missing ID token")

    from firebase_admin import auth as firebase_auth

    try:
        decoded = firebase_auth.verify_id_token(token_str)
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid ID token")

    token_uid = decoded.get("uid")
    if not token_uid or token_uid != body.userId:
        raise HTTPException(status_code=403, detail="userId does not match authenticated user")

    from firebase_admin import firestore

    db = firestore.client()
    ref = db.collection("users").document(body.userId)
    snap = ref.get()
    data = snap.to_dict() if snap.exists else {}

    if data.get("promoUnlocked") is True:
        return {"success": True}

    code_norm = _normalize_code(body.code)
    if code_norm not in VALID_CODES:
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": "Invalid promo code"},
        )

    ref.set({"promoUnlocked": True}, merge=True)
    return {"success": True}
