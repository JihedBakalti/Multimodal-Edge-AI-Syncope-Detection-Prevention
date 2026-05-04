from rest_framework import serializers
from .models import AlertEventLog, MonitoringIncident, ModelTriggerLog, VitalSignalLog, VoiceSafetyCheckLog


class VitalSignalLogSerializer(serializers.ModelSerializer):
    class Meta:
        model = VitalSignalLog
        fields = "__all__"


class VoiceSafetyCheckLogSerializer(serializers.ModelSerializer):
    class Meta:
        model = VoiceSafetyCheckLog
        fields = "__all__"


class ModelTriggerLogSerializer(serializers.ModelSerializer):
    class Meta:
        model = ModelTriggerLog
        fields = "__all__"


class MonitoringIncidentSerializer(serializers.ModelSerializer):
    class Meta:
        model = MonitoringIncident
        fields = "__all__"


class AlertEventLogSerializer(serializers.ModelSerializer):
    class Meta:
        model = AlertEventLog
        fields = "__all__"
