import base64
from django.contrib.auth import authenticate
from rest_framework import status
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import DoctorPatientAssignment, PatientProfile
from .serializers import (
    DoctorCreatePatientAccountSerializer,
    DoctorUpdatePatientAccountSerializer,
    DoctorPatientAssignmentSerializer,
    PatientProfileSerializer,
    UserSerializer,
)


class LoginView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        email = str(request.data.get("email", "")).strip().lower()
        password = str(request.data.get("password", ""))
        if not email or not password:
            return Response({"detail": "email and password are required."}, status=status.HTTP_400_BAD_REQUEST)
        user = authenticate(request, username=email, password=password)
        if user is None:
            return Response({"detail": "Invalid credentials."}, status=status.HTTP_401_UNAUTHORIZED)
        basic_token = base64.b64encode(f"{email}:{password}".encode("utf-8")).decode("utf-8")
        return Response({"token": basic_token, "user": UserSerializer(user).data})


class LogoutView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        # Basic auth is stateless. Frontend clears token locally.
        return Response({"ok": True})


class MeView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        return Response(UserSerializer(request.user).data)


class MyAssignmentsView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        if request.user.role == "doctor":
            qs = DoctorPatientAssignment.objects.filter(doctor__user=request.user, active=True)
        elif request.user.role == "patient":
            qs = DoctorPatientAssignment.objects.filter(patient__user=request.user, active=True)
        else:
            qs = DoctorPatientAssignment.objects.none()
        return Response(DoctorPatientAssignmentSerializer(qs, many=True).data)


class DoctorCreatePatientAccountView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        # Doctor onboarding remains backoffice-only: no API to create doctor accounts.
        if request.user.role != "doctor":
            return Response({"detail": "Only doctors can create patient accounts."}, status=status.HTTP_403_FORBIDDEN)

        serializer = DoctorCreatePatientAccountSerializer(data=request.data, context={"request": request})
        serializer.is_valid(raise_exception=True)
        patient = serializer.save()
        return Response(
            {
                "message": "Patient account created and linked successfully.",
                "patient": PatientProfileSerializer(patient).data,
            },
            status=status.HTTP_201_CREATED,
        )


class DoctorUpdatePatientAccountView(APIView):
    permission_classes = [IsAuthenticated]

    def patch(self, request, patient_id):
        if request.user.role != "doctor":
            return Response({"detail": "Only doctors can update patient accounts."}, status=status.HTTP_403_FORBIDDEN)

        assignment = DoctorPatientAssignment.objects.filter(
            doctor__user=request.user,
            patient_id=patient_id,
            active=True,
        ).first()
        if not assignment:
            return Response({"detail": "Patient is not assigned to this doctor."}, status=status.HTTP_403_FORBIDDEN)

        patient = PatientProfile.objects.get(id=patient_id)
        serializer = DoctorUpdatePatientAccountSerializer(
            patient,
            data=request.data,
            partial=True,
            context={"request": request, "patient": patient},
        )
        serializer.is_valid(raise_exception=True)
        updated = serializer.save()
        return Response(
            {
                "message": "Patient account updated successfully.",
                "patient": PatientProfileSerializer(updated).data,
            }
        )
