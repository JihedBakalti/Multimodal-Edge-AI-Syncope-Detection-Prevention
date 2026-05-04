# Generated manually for VoiceSafetyCheckLog

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ("clinical", "0003_alerteventlog"),
    ]

    operations = [
        migrations.CreateModel(
            name="VoiceSafetyCheckLog",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("check_kind", models.CharField(default="no_human", help_text="no_human (no camera person) or human_proactive (warning path).", max_length=32)),
                ("orchestrator_state", models.CharField(blank=True, default="", max_length=32)),
                ("attempt", models.PositiveSmallIntegerField(default=1)),
                ("prob_normal", models.FloatField(default=0.0)),
                ("prob_dysarthric", models.FloatField(default=0.0)),
                ("pred_label", models.CharField(blank=True, default="", max_length=128)),
                ("transcript_preview", models.TextField(blank=True, default="")),
                ("response_class", models.CharField(blank=True, default="", max_length=32)),
                ("heart_rate", models.IntegerField(blank=True, null=True)),
                ("anomaly_value", models.FloatField(default=0.0)),
                ("dl_risk_score", models.FloatField(blank=True, null=True)),
                ("session_id", models.CharField(blank=True, default="", max_length=128)),
                ("extra", models.JSONField(blank=True, default=dict)),
                ("logged_at", models.DateTimeField()),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                (
                    "patient",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="voice_safety_logs",
                        to="accounts.patientprofile",
                    ),
                ),
            ],
        ),
    ]
