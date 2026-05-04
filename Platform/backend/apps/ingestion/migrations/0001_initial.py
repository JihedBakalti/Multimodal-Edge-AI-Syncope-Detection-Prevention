from django.db import migrations, models


class Migration(migrations.Migration):
    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="ProcessedIngestEvent",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("idempotency_key", models.CharField(db_index=True, max_length=128, unique=True)),
                ("received_at", models.DateTimeField(auto_now_add=True)),
            ],
        ),
    ]
