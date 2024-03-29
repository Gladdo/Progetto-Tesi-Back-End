# Generated by Django 4.2.6 on 2024-01-22 13:49

import diffusers_api.models
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('diffusers_api', '0013_remove_poiimage_shot_type_alter_loramodel_model_path'),
    ]

    operations = [
        migrations.RenameField(
            model_name='poiimage',
            old_name='description',
            new_name='prompt_description',
        ),
        migrations.AddField(
            model_name='poiimage',
            name='user_description',
            field=models.TextField(default='default description'),
            preserve_default=False,
        ),
        migrations.AlterField(
            model_name='loramodel',
            name='model_path',
            field=models.FileField(upload_to=diffusers_api.models.LoraModel.model_path),
        ),
    ]
