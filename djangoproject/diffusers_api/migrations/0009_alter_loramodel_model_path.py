# Generated by Django 4.2.6 on 2023-12-05 20:22

import diffusers_api.models
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('diffusers_api', '0008_alter_actionimage_shot_type_and_more'),
    ]

    operations = [
        migrations.AlterField(
            model_name='loramodel',
            name='model_path',
            field=models.FileField(upload_to=diffusers_api.models.LoraModel.model_path),
        ),
    ]
