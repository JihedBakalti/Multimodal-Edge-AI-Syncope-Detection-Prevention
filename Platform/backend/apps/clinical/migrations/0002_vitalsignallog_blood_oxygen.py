from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("clinical", "0001_initial"),
    ]

    operations = [
        migrations.AddField(
            model_name="vitalsignallog",
            name="blood_oxygen",
            field=models.IntegerField(blank=True, null=True),
        ),
    ]
