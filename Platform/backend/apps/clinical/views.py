from rest_framework import generics
from rest_framework.permissions import IsAuthenticated

from apps.accounts.models import DoctorPatientAssignment, PatientProfile
from .models import AlertEventLog, MonitoringIncident, ModelTriggerLog, VitalSignalLog, VoiceSafetyCheckLog
from .serializers import (
    AlertEventLogSerializer,
    MonitoringIncidentSerializer,
    ModelTriggerLogSerializer,
    VitalSignalLogSerializer,
    VoiceSafetyCheckLogSerializer,
)


def _allowed_patient_ids(user):
    if user.role == "patient":
        return [user.patient_profile.id]
    if user.role == "doctor":
        return list(
            DoctorPatientAssignment.objects.filter(doctor__user=user, active=True).values_list("patient_id", flat=True)
        )
    return []


class ScopedListAPIView(generics.ListAPIView):
    permission_classes = [IsAuthenticated]
    patient_field = "patient_id"

    def get_queryset(self):
        ids = _allowed_patient_ids(self.request.user)
        qs = self.queryset.filter(**{f"{self.patient_field}__in": ids})
        pid = self.request.query_params.get("patient_id") or self.request.query_params.get("patient")
        if pid and str(pid).isdigit():
            i = int(pid)
            if i in ids:
                qs = qs.filter(**{self.patient_field: i})
        return qs.order_by("-logged_at")


class VitalSignalLogListView(ScopedListAPIView):
    queryset = VitalSignalLog.objects.all()
    serializer_class = VitalSignalLogSerializer


class VoiceSafetyCheckLogListView(ScopedListAPIView):
    queryset = VoiceSafetyCheckLog.objects.all()
    serializer_class = VoiceSafetyCheckLogSerializer


class ModelTriggerLogListView(ScopedListAPIView):
    queryset = ModelTriggerLog.objects.all()
    serializer_class = ModelTriggerLogSerializer


class MonitoringIncidentListView(ScopedListAPIView):
    queryset = MonitoringIncident.objects.all()
    serializer_class = MonitoringIncidentSerializer


class AlertEventLogListView(ScopedListAPIView):
    queryset = AlertEventLog.objects.all()
    serializer_class = AlertEventLogSerializer


class PatientSummaryView(generics.RetrieveAPIView):
    permission_classes = [IsAuthenticated]
    serializer_class = None
    lookup_url_kwarg = "patient_id"

    def get(self, request, *args, **kwargs):
        from rest_framework import status
        from rest_framework.response import Response

        patient_id = kwargs.get("patient_id")
        if int(patient_id) not in _allowed_patient_ids(request.user):
            return Response({"detail": "Forbidden"}, status=status.HTTP_403_FORBIDDEN)
        patient = PatientProfile.objects.get(id=patient_id)
        latest_vital = patient.vital_logs.order_by("-logged_at").first()
        latest_incident = patient.incidents.order_by("-logged_at").first()
        latest_voice = patient.voice_safety_logs.order_by("-logged_at").first()
        payload = {
            "patient_id": patient.id,
            "user_id": patient.firebase_user_id,
            "latest_heart_rate": latest_vital.heart_rate if latest_vital else None,
            "latest_blood_oxygen": latest_vital.blood_oxygen if latest_vital else None,
            "latest_state": latest_incident.state if latest_incident else "normal",
            "vitals_count": patient.vital_logs.count(),
            "triggers_count": patient.trigger_logs.count(),
            "incidents_count": patient.incidents.count(),
            "alerts_count": patient.alert_logs.count(),
            "voice_checks_count": patient.voice_safety_logs.count(),
            "latest_voice_prob_dysarthric": float(latest_voice.prob_dysarthric) if latest_voice else None,
            "latest_voice_response_class": latest_voice.response_class if latest_voice else None,
        }
        return Response(payload)
