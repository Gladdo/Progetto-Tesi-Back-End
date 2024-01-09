# Generated by Django 4.2.6 on 2023-11-26 21:02

import diffusers_api.models
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('diffusers_api', '0004_actionimage_subject_face_mask'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='loramodel',
            name='model',
        ),
        migrations.AddField(
            model_name='loramodel',
            name='model_path',
            field=models.FileField(null=True, upload_to=diffusers_api.models.LoraModel.model_path),
        ),
        migrations.AddField(
            model_name='loramodel',
            name='model_trigger',
            field=models.CharField(max_length=30, null=True),
        ),
    ]
