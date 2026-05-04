from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):
    dependencies = [
        ("accounts", "0002_rename_external_patient_id_user_id"),
        ("clinical", "0002_vitalsignallog_blood_oxygen"),
    ]

    operations = [
        migrations.CreateModel(
            name="AlertEventLog",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("alert_type", models.CharField(default="clinical_alert", max_length=64)),
                ("channel", models.CharField(default="whatsapp", max_length=64)),
                ("status", models.CharField(default="unknown", max_length=32)),
                ("details", models.JSONField(blank=True, default=dict)),
                ("session_id", models.CharField(blank=True, max_length=64)),
                ("logged_at", models.DateTimeField()),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                (
                    "patient",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="alert_logs",
                        to="accounts.patientprofile",
                    ),
                ),
            ],
        ),
    ]
