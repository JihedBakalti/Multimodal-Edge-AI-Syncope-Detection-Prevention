from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as DjangoUserAdmin
from .models import User, DoctorProfile, PatientProfile, DoctorPatientAssignment


@admin.register(User)
class UserAdmin(DjangoUserAdmin):
    fieldsets = DjangoUserAdmin.fieldsets + (("Platform", {"fields": ("role",)}),)
    list_display = ("email", "username", "role", "is_staff")


admin.site.register(DoctorProfile)
admin.site.register(PatientProfile)
admin.site.register(DoctorPatientAssignment)
