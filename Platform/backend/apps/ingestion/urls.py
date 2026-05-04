from django.urls import path
from .views import OrchestratorEventIngestView


urlpatterns = [
    path("orchestrator-event/", OrchestratorEventIngestView.as_view(), name="ingest-orchestrator-event"),
]
