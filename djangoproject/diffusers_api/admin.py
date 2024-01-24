from django.contrib import admin
from .models import POI, POIImage, Action, ActionImage, LoraModel, DispatchedLoraCodes
# Register your models here.

class POIAdmin(admin.ModelAdmin):
    list_display = ["name"]

class POIImageAdmin(admin.ModelAdmin):
    list_display = ["poi", "name", "image", "user_description", "prompt_description"]

class ActionAdmin(admin.ModelAdmin):
    list_display = ["name", "description"]

class ActionImageAdmin(admin.ModelAdmin):
    list_display = ["action", "shot_type", "pose_image", "mask_image", "mask_image_refined", "subject_face_mask" ]

class LoraModelAdmin(admin.ModelAdmin):
    list_display = ["name", "model_path", "model_trigger", "created_at"]

class DispatchedLoraCodesAdmin(admin.ModelAdmin):
    list_display = ["code"]

admin.site.register(DispatchedLoraCodes, DispatchedLoraCodesAdmin)
admin.site.register(LoraModel, LoraModelAdmin)
admin.site.register(POI, POIAdmin)
admin.site.register(POIImage, POIImageAdmin)
admin.site.register(Action, ActionAdmin)
admin.site.register(ActionImage, ActionImageAdmin)