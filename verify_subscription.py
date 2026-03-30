"""
POST /verify-subscription — server-side subscription validation (StoreKit 2 + App Store Server API).

Set ENTITLEMENT_VERIFY_BASE_URL in the iOS app (same host as outfit scan or dedicated service).

Environment:
  SKIP_APPLE_VERIFICATION=1  — returns active=true without calling Apple (local dev only).
  APP_STORE_ISSUER_ID, APP_STORE_KEY_ID, APP_STORE_PRIVATE_KEY (PEM) — for production validation
    via Apple's App Store Server API (see https://developer.apple.com/documentation/appstoreserverapi).

OWASP: Never trust the client; this endpoint must validate with Apple before returning active=true.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field

router = APIRouter(tags=["subscription"])


class VerifySubscriptionBody(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    transaction_id: str = Field(..., min_length=1, max_length=64)
    original_transaction_id: str = Field(..., min_length=1, max_length=64)
    product_id: str = Field(..., min_length=1, max_length=256)
    environment: str = Field(default="production", max_length=32)


def _verify_with_apple(body: VerifySubscriptionBody) -> dict[str, Any]:
    """
    Production: use app-store-server-library + App Store Connect API key to query subscription status.
    This stub documents the contract; deploy with real credentials on Railway.
    """
    issuer = os.getenv("APP_STORE_ISSUER_ID")
    key_id = os.getenv("APP_STORE_KEY_ID")
    p8 = os.getenv("APP_STORE_PRIVATE_KEY")
    if issuer and key_id and p8:
        try:
            from appstoreserverlibrary.api_client import AppStoreServerAPIClient  # type: ignore
            from appstoreserverlibrary.models import Environment  # type: ignore
        except ImportError:
            raise HTTPException(
                status_code=503,
                detail="Server missing app-store-server-library; pip install app-store-server-library",
            )
        env = Environment.SANDBOX if body.environment.lower() == "sandbox" else Environment.PRODUCTION
        # Example integration point (uncomment and wire bundle id / app id as required):
        # client = AppStoreServerAPIClient(...)
        # status = client.get_subscription_statuses(original_transaction_id=body.original_transaction_id)
        # map status → active, expires_at
        raise HTTPException(
            status_code=501,
            detail="App Store Server API wiring not completed — set SKIP_APPLE_VERIFICATION=1 for dev or finish integration.",
        )
    raise HTTPException(
        status_code=503,
        detail="Apple API credentials not configured (APP_STORE_ISSUER_ID, APP_STORE_KEY_ID, APP_STORE_PRIVATE_KEY).",
    )


@router.post("/verify-subscription")
async def verify_subscription(
    body: VerifySubscriptionBody,
    authorization: Optional[str] = Header(default=None),
) -> dict[str, Any]:
    """
    Validates the subscription transaction server-side.

    - Authorization: optional `Bearer <Firebase ID token>` (verify in production with Firebase Admin).
    - Request body: transaction identifiers from StoreKit (already verified on device; re-validated here).
    """
    if authorization and authorization.startswith("Bearer "):
        # TODO: verify JWT with Firebase Admin SDK in production.
        pass

    if os.getenv("SKIP_APPLE_VERIFICATION", "").lower() in ("1", "true", "yes"):
        return {"active": True, "status": "active", "expires_at": None, "reason": None}

    return _verify_with_apple(body)
