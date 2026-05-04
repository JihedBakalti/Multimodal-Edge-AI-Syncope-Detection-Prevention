from django.contrib.auth.models import AbstractUser
from django.db import models


class UserRole(models.TextChoices):
    PATIENT = "patient", "Patient"
    DOCTOR = "doctor", "Doctor"


class User(AbstractUser):
    email = models.EmailField(unique=True)
    role = models.CharField(max_length=16, choices=UserRole.choices)
    USERNAME_FIELD = "email"
    REQUIRED_FIELDS = ["username"]

    def __str__(self) -> str:
        return f"{self.email} ({self.role})"


class DoctorProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name="doctor_profile")
    specialty = models.CharField(max_length=128, blank=True)
    license_id = models.CharField(max_length=64, blank=True)

    def __str__(self) -> str:
        return f"DoctorProfile<{self.user.email}>"


class PatientProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name="patient_profile")
    firebase_user_id = models.CharField(max_length=64, unique=True, db_index=True)
    date_of_birth = models.DateField(null=True, blank=True)
    emergency_contact = models.CharField(max_length=120, blank=True)

    def __str__(self) -> str:
        return f"PatientProfile<{self.user.email}>"


class DoctorPatientAssignment(models.Model):
    doctor = models.ForeignKey(DoctorProfile, on_delete=models.CASCADE, related_name="assignments")
    patient = models.ForeignKey(PatientProfile, on_delete=models.CASCADE, related_name="assignments")
    assigned_at = models.DateTimeField(auto_now_add=True)
    active = models.BooleanField(default=True)

    class Meta:
        unique_together = ("doctor", "patient")

    def __str__(self) -> str:
        return f"{self.doctor.user.email} -> {self.patient.user.email}"
