from django.contrib import admin
from django.http import JsonResponse
from django.urls import include, path


def root_health(_request):
    """Render/browser hits to `/` are not an error — API lives under `/api/`."""
    return JsonResponse(
        {
            "ok": True,
            "service": "intersense-platform-api",
            "api": "/api/",
            "admin": "/admin/",
        }
    )


urlpatterns = [
    path("", root_health),
    path("admin/", admin.site.urls),
    path("api/accounts/", include("apps.accounts.urls")),
    path("api/clinical/", include("apps.clinical.urls")),
    path("api/ingestion/", include("apps.ingestion.urls")),
    path("api/messaging/", include("apps.messaging.urls")),
    path("api/reports/", include("apps.reports.urls")),
]
