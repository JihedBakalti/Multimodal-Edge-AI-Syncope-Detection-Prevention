from django.db import migrations


class Migration(migrations.Migration):
    dependencies = [
        ("accounts", "0001_initial"),
    ]

    operations = [
        migrations.RenameField(
            model_name="patientprofile",
            old_name="external_patient_id",
            new_name="firebase_user_id",
        ),
    ]
