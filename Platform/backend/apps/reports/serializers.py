from rest_framework import serializers
from .models import DoctorReport


class DoctorReportSerializer(serializers.ModelSerializer):
    class Meta:
        model = DoctorReport
        fields = "__all__"
        read_only_fields = ["doctor", "created_at"]
