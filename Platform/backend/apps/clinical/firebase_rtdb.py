"""
Server-side Firebase RTDB vitals (same paths as medical_assistant.fetch_latest_firebase_vitals).
Used by the platform API so browsers only talk to Django (Vercel + Render), not RTDB directly.
"""

from __future__ import annotations

import json
import urllib.request
from urllib.parse import quote

from django.conf import settings


def fetch_firebase_vitals(firebase_user_id: str) -> dict:
    base = (getattr(settings, "FIREBASE_RTDB_URL", None) or "").strip().rstrip("/")
    if not base:
        return {"ok": False, "message": "FIREBASE_RTDB_URL is not configured on the server."}

    user_id = str(firebase_user_id or "").strip()
    if not user_id:
        return {"ok": False, "message": "Missing Firebase user id."}

    url_direct = f"{base}/users/{quote(user_id, safe='')}/vitals.json"
    url_users = f"{base}/users.json"
    try:

        def _read_json(url: str):
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=4) as resp:
                payload = resp.read().decode("utf-8", errors="ignore")
            return json.loads(payload) if payload else None

        direct = _read_json(url_direct)
        if isinstance(direct, dict) and direct:
            hr = direct.get("HeartRate")
            spo2 = direct.get("BloodOxygen")
            return {
                "ok": True,
                "heart_rate": int(hr) if hr is not None else None,
                "blood_oxygen": int(spo2) if spo2 is not None else None,
                "raw": direct,
                "source": f"users/{user_id}/vitals",
            }

        users_payload = _read_json(url_users)
        candidate = None
        if isinstance(users_payload, list):
            idx = int(user_id) if user_id.isdigit() else None
            if idx is not None and len(users_payload) > idx and isinstance(users_payload[idx], dict):
                candidate = (users_payload[idx] or {}).get("vitals")
        elif isinstance(users_payload, dict):
            user_entry = users_payload.get(user_id)
            if isinstance(user_entry, dict):
                candidate = user_entry.get("vitals")

        if isinstance(candidate, dict) and candidate:
            hr = candidate.get("HeartRate")
            spo2 = candidate.get("BloodOxygen")
            return {
                "ok": True,
                "heart_rate": int(hr) if hr is not None else None,
                "blood_oxygen": int(spo2) if spo2 is not None else None,
                "raw": candidate,
                "source": "users fallback",
            }

        return {
            "ok": False,
            "message": f"No vitals found at users/{user_id}/vitals or users fallback.",
            "raw": {"direct": direct, "users": users_payload},
        }
    except Exception as e:
        return {"ok": False, "message": f"Firebase vitals fetch failed: {e}"}
