from django.urls import path
from .views import (
    AlertEventLogListView,
    FirebaseLiveVitalsView,
    ModelTriggerLogListView,
    MonitoringIncidentListView,
    PatientSummaryView,
    VitalSignalLogListView,
    VoiceSafetyCheckLogListView,
)

urlpatterns = [
    path("vitals/", VitalSignalLogListView.as_view(), name="clinical-vitals"),
    path("firebase-live-vitals/", FirebaseLiveVitalsView.as_view(), name="clinical-firebase-live-vitals"),
    path("voice-checks/", VoiceSafetyCheckLogListView.as_view(), name="clinical-voice-checks"),
    path("triggers/", ModelTriggerLogListView.as_view(), name="clinical-triggers"),
    path("incidents/", MonitoringIncidentListView.as_view(), name="clinical-incidents"),
    path("alerts/", AlertEventLogListView.as_view(), name="clinical-alerts"),
    path("patients/<int:patient_id>/summary/", PatientSummaryView.as_view(), name="clinical-patient-summary"),
]
