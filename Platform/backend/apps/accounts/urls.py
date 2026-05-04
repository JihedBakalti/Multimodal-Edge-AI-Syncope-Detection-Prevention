from django.urls import path
from .views import DoctorCreatePatientAccountView, DoctorUpdatePatientAccountView, LoginView, LogoutView, MeView, MyAssignmentsView


urlpatterns = [
    path("login/", LoginView.as_view(), name="accounts-login"),
    path("logout/", LogoutView.as_view(), name="accounts-logout"),
    path("me/", MeView.as_view(), name="accounts-me"),
    path("assignments/", MyAssignmentsView.as_view(), name="accounts-assignments"),
    path("patients/create/", DoctorCreatePatientAccountView.as_view(), name="accounts-create-patient"),
    path("patients/<int:patient_id>/", DoctorUpdatePatientAccountView.as_view(), name="accounts-update-patient"),
]
