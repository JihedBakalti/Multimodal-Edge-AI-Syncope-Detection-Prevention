from django.contrib import admin
from django.urls import include, path


urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/accounts/", include("apps.accounts.urls")),
    path("api/clinical/", include("apps.clinical.urls")),
    path("api/ingestion/", include("apps.ingestion.urls")),
    path("api/messaging/", include("apps.messaging.urls")),
    path("api/reports/", include("apps.reports.urls")),
]
