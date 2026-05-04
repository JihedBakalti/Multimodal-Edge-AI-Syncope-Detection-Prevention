from django.db import migrations, models
import django.db.models.deletion


def populate_conversation_doctor(apps, schema_editor):
    Conversation = apps.get_model("messaging", "Conversation")
    DoctorPatientAssignment = apps.get_model("accounts", "DoctorPatientAssignment")
    DoctorProfile = apps.get_model("accounts", "DoctorProfile")

    fallback_doctor = DoctorProfile.objects.order_by("id").first()
    for conversation in Conversation.objects.filter(doctor__isnull=True):
        assignment = (
            DoctorPatientAssignment.objects.filter(patient_id=conversation.patient_id, active=True)
            .order_by("assigned_at")
            .first()
        )
        if assignment:
            conversation.doctor_id = assignment.doctor_id
        elif fallback_doctor:
            conversation.doctor_id = fallback_doctor.id
        conversation.save(update_fields=["doctor"])


class Migration(migrations.Migration):
    dependencies = [
        ("accounts", "0002_rename_external_patient_id_user_id"),
        ("messaging", "0001_initial"),
    ]

    operations = [
        migrations.AddField(
            model_name="conversation",
            name="doctor",
            field=models.ForeignKey(
                null=True,
                blank=True,
                on_delete=django.db.models.deletion.CASCADE,
                related_name="conversations",
                to="accounts.doctorprofile",
            ),
        ),
        migrations.AddField(
            model_name="message",
            name="delivered_at",
            field=models.DateTimeField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="message",
            name="read_at",
            field=models.DateTimeField(blank=True, null=True),
        ),
        migrations.RunPython(populate_conversation_doctor, migrations.RunPython.noop),
        migrations.AlterField(
            model_name="conversation",
            name="doctor",
            field=models.ForeignKey(
                on_delete=django.db.models.deletion.CASCADE,
                related_name="conversations",
                to="accounts.doctorprofile",
            ),
        ),
        migrations.AlterUniqueTogether(
            name="conversation",
            unique_together={("doctor", "patient")},
        ),
    ]
