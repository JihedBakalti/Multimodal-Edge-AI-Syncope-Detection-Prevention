from django.conf import settings
from django.utils import timezone
from datetime import timedelta
import logging
from rest_framework import status
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.accounts.models import PatientProfile
from apps.clinical.models import (
    AlertEventLog,
    MonitoringIncident,
    ModelTriggerLog,
    VitalSignalLog,
    VoiceSafetyCheckLog,
)
from .models import ProcessedIngestEvent

logger = logging.getLogger(__name__)


class OrchestratorEventIngestView(APIView):
    permission_classes = [AllowAny]

    @staticmethod
    def _resolve_patient(user_id: str):
        normalized = str(user_id).strip()
        if not normalized:
            return None

        # Preferred canonical match: ESP/Firebase user_id mirrored on patient profile.
        patient = PatientProfile.objects.filter(firebase_user_id=normalized).first()
        if patient:
            return patient

        # Compatibility fallback for deployments that send Django PK-style IDs.
        if normalized.isdigit():
            patient = PatientProfile.objects.filter(id=int(normalized)).first()
            if patient:
                return patient
            patient = PatientProfile.objects.filter(user_id=int(normalized)).first()
            if patient:
                return patient

        return None

    def post(self, request):
        token = request.headers.get("X-Platform-Ingest-Token", "")
        if token != settings.PLATFORM_INGEST_TOKEN:
            logger.warning("Ingest rejected: invalid token")
            return Response({"detail": "Unauthorized"}, status=status.HTTP_401_UNAUTHORIZED)

        payload = request.data
        user_id = payload.get("user_id")
        if not user_id:
            logger.warning("Ingest rejected: missing user_id")
            return Response({"detail": "user_id is required"}, status=status.HTTP_400_BAD_REQUEST)

        patient = self._resolve_patient(user_id)
        if patient is None:
            logger.warning("Ingest rejected: unknown user_id=%s", user_id)
            return Response({"detail": "Unknown patient mapping"}, status=status.HTTP_404_NOT_FOUND)

        logged_at = payload.get("timestamp")
        if logged_at:
            try:
                dt = timezone.datetime.fromisoformat(logged_at.replace("Z", "+00:00"))
            except ValueError:
                logger.warning("Ingest rejected: invalid timestamp format")
                return Response({"detail": "Invalid timestamp format"}, status=status.HTTP_400_BAD_REQUEST)
        else:
            dt = timezone.now()

        # Replay protection window for static-token MVP.
        now = timezone.now()
        replay_window = timedelta(seconds=getattr(settings, "PLATFORM_INGEST_REPLAY_WINDOW_SEC", 300))
        if dt < now - replay_window or dt > now + timedelta(seconds=60):
            logger.warning("Ingest rejected: timestamp outside allowed replay window")
            return Response({"detail": "Timestamp outside allowed window"}, status=status.HTTP_400_BAD_REQUEST)

        idempotency_key = str(payload.get("idempotency_key", "")).strip()
        if not idempotency_key:
            logger.warning("Ingest rejected: missing idempotency_key")
            return Response({"detail": "idempotency_key is required"}, status=status.HTTP_400_BAD_REQUEST)
        if ProcessedIngestEvent.objects.filter(idempotency_key=idempotency_key).exists():
            logger.info("Ingest deduplicated for key=%s", idempotency_key)
            return Response({"ok": True, "deduplicated": True})

        event_type = str(payload.get("event_type", "orchestrator_state"))

        if event_type == "voice_safety_check":
            vs = payload.get("voice") if isinstance(payload.get("voice"), dict) else {}
            try:
                attempt = int(vs.get("attempt", 1))
            except (TypeError, ValueError):
                attempt = 1
            VoiceSafetyCheckLog.objects.create(
                patient=patient,
                check_kind=str(payload.get("check_kind", "unknown"))[:32],
                orchestrator_state=str(payload.get("orchestrator_state", ""))[:32],
                attempt=max(1, min(attempt, 255)),
                prob_normal=float(vs.get("prob_normal", 0.0) or 0.0),
                prob_dysarthric=float(vs.get("prob_dysarthric", 0.0) or 0.0),
                pred_label=str(vs.get("pred_label", ""))[:128],
                transcript_preview=str(vs.get("transcript_preview", ""))[:2000],
                response_class=str(vs.get("response_class", ""))[:32],
                heart_rate=int(payload["heart_rate"]) if payload.get("heart_rate") is not None else None,
                anomaly_value=float(payload.get("anomaly_value", 0.0) or 0.0),
                dl_risk_score=float(payload["dl_risk_score"]) if payload.get("dl_risk_score") is not None else None,
                session_id=str(payload.get("session_id", ""))[:128],
                extra=payload.get("voice_extra") if isinstance(payload.get("voice_extra"), dict) else {},
                logged_at=dt,
            )
            ProcessedIngestEvent.objects.create(idempotency_key=idempotency_key)
            logger.info("Voice safety ingest for user_id=%s attempt=%s", user_id, attempt)
            return Response({"ok": True, "deduplicated": False})

        heart_rate = payload.get("heart_rate")
        if heart_rate is not None:
            VitalSignalLog.objects.create(
                patient=patient,
                heart_rate=int(heart_rate),
                blood_oxygen=int(payload.get("blood_oxygen")) if payload.get("blood_oxygen") is not None else None,
                anomaly_value=float(payload.get("anomaly_value", 0.0)),
                source=str(payload.get("source", "orchestrator")),
                logged_at=dt,
            )

        ModelTriggerLog.objects.create(
            patient=patient,
            model_name=str(payload.get("model_name", "intersense_orchestrator")),
            threshold=float(payload.get("threshold", 0.0)),
            score=float(payload.get("dl_risk_score", payload.get("score", 0.0))),
            state=str(payload.get("state", "unknown")),
            action=",".join(payload.get("actions", [])) if isinstance(payload.get("actions"), list) else str(payload.get("actions", "")),
            metadata=payload,
            logged_at=dt,
        )

        MonitoringIncident.objects.create(
            patient=patient,
            state=str(payload.get("state", "unknown")),
            message=str(payload.get("message", "")),
            payload=payload,
            session_id=str(payload.get("session_id", "")),
            logged_at=dt,
        )

        alert_payload = payload.get("alert") if isinstance(payload.get("alert"), dict) else {}
        should_log_alert = bool(alert_payload) or str(payload.get("state", "")).lower() == "critical_emergency"
        if should_log_alert:
            AlertEventLog.objects.create(
                patient=patient,
                alert_type=str(alert_payload.get("type", "critical_emergency_alert")),
                channel=str(alert_payload.get("channel", "whatsapp")),
                status=str(alert_payload.get("status", "triggered")),
                details=alert_payload or {"message": payload.get("message", ""), "actions": payload.get("actions", [])},
                session_id=str(payload.get("session_id", "")),
                logged_at=dt,
            )

        ProcessedIngestEvent.objects.create(idempotency_key=idempotency_key)
        logger.info("Ingest accepted for user_id=%s state=%s", user_id, payload.get("state", "unknown"))
        return Response({"ok": True, "deduplicated": False})
