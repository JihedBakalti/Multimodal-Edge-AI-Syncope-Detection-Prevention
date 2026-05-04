from django.conf import settings
from django.db import models
from apps.accounts.models import PatientProfile


class DoctorReport(models.Model):
    patient = models.ForeignKey(PatientProfile, on_delete=models.CASCADE, related_name="reports")
    doctor = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="generated_reports")
    period_start = models.DateTimeField()
    period_end = models.DateTimeField()
    summary = models.TextField()
    risk_notes = models.TextField(blank=True)
    recommendations = models.TextField(blank=True)
    source_log_ids = models.JSONField(default=list, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
