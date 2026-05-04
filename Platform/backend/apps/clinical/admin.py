from django.contrib import admin
from .models import AlertEventLog, VitalSignalLog, ModelTriggerLog, MonitoringIncident, VoiceSafetyCheckLog

admin.site.register(VitalSignalLog)
admin.site.register(ModelTriggerLog)
admin.site.register(MonitoringIncident)
admin.site.register(AlertEventLog)
admin.site.register(VoiceSafetyCheckLog)
