from django.urls import path
from .views import DoctorReportListCreateView, DoctorReportPdfView


urlpatterns = [
    path("", DoctorReportListCreateView.as_view(), name="reports-list-create"),
    path("<int:pk>/pdf/", DoctorReportPdfView.as_view(), name="reports-pdf"),
]
