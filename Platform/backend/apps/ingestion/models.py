from django.db import models


class ProcessedIngestEvent(models.Model):
    idempotency_key = models.CharField(max_length=128, unique=True, db_index=True)
    received_at = models.DateTimeField(auto_now_add=True)

    def __str__(self) -> str:
        return self.idempotency_key
