from rest_framework import serializers
from django.contrib.auth import get_user_model
from django.db import transaction
from .models import DoctorProfile, PatientProfile, DoctorPatientAssignment


User = get_user_model()


class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ["id", "email", "username", "role"]


class DoctorProfileSerializer(serializers.ModelSerializer):
    user = UserSerializer(read_only=True)

    class Meta:
        model = DoctorProfile
        fields = ["id", "user", "specialty", "license_id"]


class PatientProfileSerializer(serializers.ModelSerializer):
    user = UserSerializer(read_only=True)
    user_id = serializers.CharField(source="firebase_user_id")

    class Meta:
        model = PatientProfile
        fields = ["id", "user", "user_id", "firebase_user_id", "date_of_birth", "emergency_contact"]


class DoctorPatientAssignmentSerializer(serializers.ModelSerializer):
    doctor = DoctorProfileSerializer(read_only=True)
    patient = PatientProfileSerializer(read_only=True)

    class Meta:
        model = DoctorPatientAssignment
        fields = ["id", "doctor", "patient", "assigned_at", "active"]


class DoctorCreatePatientAccountSerializer(serializers.Serializer):
    email = serializers.EmailField()
    username = serializers.CharField(max_length=150)
    password = serializers.CharField(min_length=8, write_only=True)
    user_id = serializers.CharField(max_length=64)
    emergency_contact = serializers.CharField(max_length=120, required=False, allow_blank=True)
    date_of_birth = serializers.DateField(required=False)

    def validate_user_id(self, value):
        if PatientProfile.objects.filter(firebase_user_id=value).exists():
            raise serializers.ValidationError("This user_id is already linked to another patient.")
        return value

    def validate_email(self, value):
        if User.objects.filter(email=value).exists():
            raise serializers.ValidationError("A user with this email already exists.")
        return value

    def create(self, validated_data):
        doctor_user = self.context["request"].user
        with transaction.atomic():
            patient_user = User.objects.create_user(
                email=validated_data["email"],
                username=validated_data["username"],
                password=validated_data["password"],
                role="patient",
            )
            patient_profile = PatientProfile.objects.create(
                user=patient_user,
                firebase_user_id=validated_data["user_id"],
                emergency_contact=validated_data.get("emergency_contact", ""),
                date_of_birth=validated_data.get("date_of_birth"),
            )
            doctor_profile = DoctorProfile.objects.get(user=doctor_user)
            DoctorPatientAssignment.objects.create(
                doctor=doctor_profile,
                patient=patient_profile,
                active=True,
            )
        return patient_profile


class DoctorUpdatePatientAccountSerializer(serializers.Serializer):
    email = serializers.EmailField(required=False)
    username = serializers.CharField(max_length=150, required=False)
    user_id = serializers.CharField(max_length=64, required=False)
    emergency_contact = serializers.CharField(max_length=120, required=False, allow_blank=True)
    date_of_birth = serializers.DateField(required=False, allow_null=True)

    def validate(self, attrs):
        patient = self.context["patient"]
        email = attrs.get("email")
        if email and User.objects.filter(email=email).exclude(id=patient.user_id).exists():
            raise serializers.ValidationError({"email": "A user with this email already exists."})

        user_id = attrs.get("user_id")
        if user_id and PatientProfile.objects.filter(firebase_user_id=user_id).exclude(id=patient.id).exists():
            raise serializers.ValidationError({"user_id": "This user_id is already linked to another patient."})
        return attrs

    def update(self, patient, validated_data):
        user = patient.user
        with transaction.atomic():
            if "email" in validated_data:
                user.email = validated_data["email"]
            if "username" in validated_data:
                user.username = validated_data["username"]
            user.save(update_fields=["email", "username"])

            if "user_id" in validated_data:
                patient.firebase_user_id = validated_data["user_id"]
            if "emergency_contact" in validated_data:
                patient.emergency_contact = validated_data["emergency_contact"]
            if "date_of_birth" in validated_data:
                patient.date_of_birth = validated_data["date_of_birth"]
            patient.save(update_fields=["firebase_user_id", "emergency_contact", "date_of_birth"])
        return patient
