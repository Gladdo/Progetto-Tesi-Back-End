from django.urls import path
from . import views

urlpatterns = [
    path('generate_image', views.GenerateImage, name="GenerateImage"),
    path('db_summary', views.DBSummary, name="DBSummary"),
    path('get_actions', views.GetActions, name="GetActions"),
    path('lora_training', views.LoraTraining, name="LoraTraining"),
    path('post_lora_model', views.PostLoraModel, name="PostLoraModel"),
    path('get_poi_images', views.GetPoiImages, name="GetPoiImages"),
    path('check_lora_code', views.CheckLoraCode, name="CheckLoraCode"),
    path('get_maps_data', views.GetMapsData, name="GetMapsData")

]