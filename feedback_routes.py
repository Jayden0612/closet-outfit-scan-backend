"""
POST /submit-feedback — store user feedback in Firestore collection `feedback`.
"""

from __future__ import annotations

import json
import os
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

router = APIRouter(tags=["feedback"])


def _ensure_firebase() -> None:
    """Initialize firebase_admin once (same pattern as promo_routes)."""
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


class SubmitFeedbackBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    userId: str = Field(default="", max_length=128)
    category: str = Field(default="", max_length=120)
    message: str = Field(default="", max_length=8000)


@router.post("/submit-feedback")
async def submit_feedback(body: SubmitFeedbackBody) -> Any:
    msg = body.message.strip()
    if not msg:
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": "Message cannot be empty"},
        )

    try:
        _ensure_firebase()
    except HTTPException:
        raise
    except ImportError as e:
        raise HTTPException(status_code=503, detail="firebase-admin not installed") from e

    from firebase_admin import firestore
    from google.cloud.firestore import SERVER_TIMESTAMP

    db = firestore.client()
    ref = db.collection("feedback").document()
    ref.set(
        {
            "userId": body.userId.strip()[:128],
            "category": body.category.strip()[:120],
            "message": msg[:8000],
            "createdAt": SERVER_TIMESTAMP,
        }
    )
    return {"success": True}
