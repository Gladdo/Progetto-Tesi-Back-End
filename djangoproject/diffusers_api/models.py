from django.db import models
from enum import Enum
import os
from django.core.validators import int_list_validator

# Create your models here.

# Gladdo: modelli per gestire le immagini
class POI(models.Model):
    name = models.CharField(max_length=30, unique=True)  

    # Gladdo: override del metodo __str__ che viene chiamato quando un'istanza di POI viene interpretata come stringa
    def __str__(self):
        return self.name

class POIImage(models.Model):

    poi = models.ForeignKey(POI, on_delete=models.CASCADE)

    name = models.CharField(max_length=30, unique=True)  
    prompt_description = models.CharField(max_length=65)
    user_description = models.TextField()

    # Gladdo: funzione che ritorna il path dei file; la current directory da cui viene elaborato il path è specificato nella variabile
    # MEDIA_URL in settings.py
    def poi_image_path(instance, filename):
        return 'point_of_interest/' + instance.poi.name + '/' + filename
  
    image = models.ImageField(upload_to=poi_image_path)

"""
# Gladdo: modello per storage delle immagini di un'utente
class UserImage
"""

class Action(models.Model):
    name = models.CharField(max_length=30, unique=True)
    description = models.CharField(max_length=65)

# Gladdo: override del metodo __str__ che viene chiamato quando un'istanza di Action viene interpretata come stringa
    def __str__(self):
        return self.name

class ActionImage(models.Model):

    action = models.ForeignKey(Action, on_delete=models.CASCADE)

    # Gladdo: Enum per i differenti tipi di shots:
    class ActionShotTypes(models.TextChoices):
        CLOSE_SHOT = 'CLS', 'close_shot'
        MEDIUM_SHOT = 'MES', 'medium_shot'
        FULL_SHOT = 'FUS', 'full_shot'

    # Gladdo: (TODO) mettere defaults
    shot_type = models.CharField(max_length=3, choices=ActionShotTypes.choices)

    # Gladdo: funzione che ritorna il path dei file; la current directory da cui viene elaborato il path è specificato nella variabile
    # MEDIA_URL in settings.py
    def image_path(instance, filename):
        return 'subject_action/' + instance.action.name + '/' + instance.shot_type + '/' + filename

    pose_image = models.ImageField(upload_to=image_path)
    mask_image = models.ImageField(upload_to=image_path)
    mask_image_refined = models.ImageField(upload_to=image_path)
    subject_face_mask = models.ImageField(upload_to=image_path)

    class Meta:
        unique_together = ('action','shot_type')

    def save(self, *args, **kwargs):
        try:
            old_obj = ActionImage.objects.get(action = self.action, shot_type = self.shot_type)
            if(old_obj.pose_image != self.pose_image):
                os.remove(old_obj.pose_image.path)
            if(old_obj.mask_image != self.mask_image):
                os.remove(old_obj.mask_image.path)
            if(old_obj.mask_image_refined != self.mask_image_refined):
                os.remove(old_obj.mask_image_refined.path)
            if(old_obj.subject_face_mask != self.subject_face_mask):
                os.remove(old_obj.subject_face_mask.path)
        except: pass
        super(ActionImage, self).save(*args, **kwargs)

class LoraModel(models.Model):
    
    name = models.CharField(max_length=30, unique=True)

    def model_path(instance, filename):
        return 'lora_models/raw_lora/' + filename

    model_path = models.FileField(upload_to=model_path)
    model_trigger = models.CharField(max_length=30)
    created_at = models.DateTimeField(auto_now_add=True)

class DispatchedLoraCodes(models.Model):

    code = models.CharField(max_length=8, unique=True)


# -----------------------------------------------------------------------------------
#                              MOBILE FRONT END DATA
# -----------------------------------------------------------------------------------

class FECity(models.Model):
    city_name = models.CharField(max_length=30, unique=True)

    def __str__(self):
        return self.city_name

class FEArea(models.Model):
    city = models.ForeignKey(FECity, on_delete=models.CASCADE)
    area_number = models.IntegerField()
    canvas_x_pos = models.FloatField()
    canvas_y_pos = models.FloatField()

    def area_image_path(instance, filename):
        return 'poi_maps/' + instance.city.city_name + '/' + filename

    area_image = models.ImageField(upload_to=area_image_path)
    connected_areas = models.CharField(validators=[int_list_validator], max_length=30)

class FEPoiMarker(models.Model):
    city = models.ForeignKey(FECity, on_delete=models.CASCADE)
    area_number = models.IntegerField()
    x_pos = models.IntegerField()
    y_pos = models.IntegerField()

    poi = models.ForeignKey(POI, on_delete=models.CASCADE)

    def overview_image_path(instance, filename):
        return 'poi_maps/' + instance.city.city_name + '/overviews/' + filename

    overview_image = models.ImageField(upload_to=overview_image_path)






