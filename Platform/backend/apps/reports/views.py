from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from rest_framework import generics
from rest_framework.exceptions import PermissionDenied
from rest_framework.permissions import IsAuthenticated
from django.http import FileResponse

from apps.accounts.models import DoctorPatientAssignment
from .models import DoctorReport
from .serializers import DoctorReportSerializer


class DoctorReportListCreateView(generics.ListCreateAPIView):
    permission_classes = [IsAuthenticated]
    serializer_class = DoctorReportSerializer

    def get_queryset(self):
        qs = DoctorReport.objects.all().order_by("-created_at")
        if self.request.user.role == "doctor":
            return qs.filter(doctor=self.request.user)
        if self.request.user.role == "patient":
            return qs.filter(patient__user=self.request.user)
        return qs.none()

    def perform_create(self, serializer):
        patient = serializer.validated_data["patient"]
        allowed = DoctorPatientAssignment.objects.filter(
            doctor__user=self.request.user, patient=patient, active=True
        ).exists()
        if not allowed:
            raise PermissionDenied("Doctor not assigned to this patient.")
        serializer.save(doctor=self.request.user)


class DoctorReportPdfView(generics.RetrieveAPIView):
    permission_classes = [IsAuthenticated]
    queryset = DoctorReport.objects.all()
    serializer_class = DoctorReportSerializer

    def get(self, request, *args, **kwargs):
        report = self.get_object()
        buf = BytesIO()
        pdf = canvas.Canvas(buf, pagesize=A4)
        pdf.drawString(40, 800, f"Doctor Report #{report.id}")
        pdf.drawString(40, 780, f"Patient: {report.patient.user.email}")
        pdf.drawString(40, 760, f"Doctor: {report.doctor.email}")
        pdf.drawString(40, 740, f"Period: {report.period_start} -> {report.period_end}")
        pdf.drawString(40, 710, "Summary:")
        pdf.drawString(60, 690, report.summary[:120])
        pdf.drawString(40, 660, "Risk notes:")
        pdf.drawString(60, 640, report.risk_notes[:120])
        pdf.drawString(40, 610, "Recommendations:")
        pdf.drawString(60, 590, report.recommendations[:120])
        pdf.showPage()
        pdf.save()
        buf.seek(0)
        return FileResponse(buf, as_attachment=True, filename=f"doctor_report_{report.id}.pdf")
