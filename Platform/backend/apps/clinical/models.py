from django.conf import settings
from django.db import models

from apps.accounts.models import PatientProfile


class VitalSignalLog(models.Model):
    patient = models.ForeignKey(PatientProfile, on_delete=models.CASCADE, related_name="vital_logs")
    heart_rate = models.IntegerField()
    blood_oxygen = models.IntegerField(null=True, blank=True)
    anomaly_value = models.FloatField(default=0.0)
    source = models.CharField(max_length=64, default="orchestrator")
    logged_at = models.DateTimeField()
    created_at = models.DateTimeField(auto_now_add=True)
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.SET_NULL, related_name="+"
    )


class ModelTriggerLog(models.Model):
    patient = models.ForeignKey(PatientProfile, on_delete=models.CASCADE, related_name="trigger_logs")
    model_name = models.CharField(max_length=128)
    threshold = models.FloatField(default=0.0)
    score = models.FloatField(default=0.0)
    state = models.CharField(max_length=64)
    action = models.CharField(max_length=128, blank=True)
    metadata = models.JSONField(default=dict, blank=True)
    logged_at = models.DateTimeField()
    created_at = models.DateTimeField(auto_now_add=True)


class MonitoringIncident(models.Model):
    patient = models.ForeignKey(PatientProfile, on_delete=models.CASCADE, related_name="incidents")
    state = models.CharField(max_length=64)
    message = models.TextField(blank=True)
    payload = models.JSONField(default=dict, blank=True)
    session_id = models.CharField(max_length=64, blank=True)
    logged_at = models.DateTimeField()
    created_at = models.DateTimeField(auto_now_add=True)


class VoiceSafetyCheckLog(models.Model):
    """Per-attempt voice dysarthria risk scores during orchestrator verbal safety checks."""

    patient = models.ForeignKey(PatientProfile, on_delete=models.CASCADE, related_name="voice_safety_logs")
    check_kind = models.CharField(
        max_length=32,
        default="no_human",
        help_text="no_human (no camera person) or human_proactive (warning path).",
    )
    orchestrator_state = models.CharField(max_length=32, blank=True, default="")
    attempt = models.PositiveSmallIntegerField(default=1)
    prob_normal = models.FloatField(default=0.0)
    prob_dysarthric = models.FloatField(default=0.0)
    pred_label = models.CharField(max_length=128, blank=True, default="")
    transcript_preview = models.TextField(blank=True, default="")
    response_class = models.CharField(max_length=32, blank=True, default="")
    heart_rate = models.IntegerField(null=True, blank=True)
    anomaly_value = models.FloatField(default=0.0)
    dl_risk_score = models.FloatField(null=True, blank=True)
    session_id = models.CharField(max_length=128, blank=True, default="")
    extra = models.JSONField(default=dict, blank=True)
    logged_at = models.DateTimeField()
    created_at = models.DateTimeField(auto_now_add=True)


class AlertEventLog(models.Model):
    patient = models.ForeignKey(PatientProfile, on_delete=models.CASCADE, related_name="alert_logs")
    alert_type = models.CharField(max_length=64, default="clinical_alert")
    channel = models.CharField(max_length=64, default="whatsapp")
    status = models.CharField(max_length=32, default="unknown")
    details = models.JSONField(default=dict, blank=True)
    session_id = models.CharField(max_length=64, blank=True)
    logged_at = models.DateTimeField()
    created_at = models.DateTimeField(auto_now_add=True)
