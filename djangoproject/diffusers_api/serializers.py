from rest_framework import serializers

from .models import POI, POIImage, Action, ActionImage, LoraModel

class POISerializer(serializers.ModelSerializer):
    class Meta:
        model = POI
        fields = ['name']

class POIImageSerializer(serializers.ModelSerializer):
    poi = serializers.StringRelatedField()
    class Meta:
        model = POIImage
        fields = ['poi', 'name', 'image']

class ActionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Action
        fields = ['name']

class LoraModelSerializer(serializers.ModelSerializer):
    class Meta:
        model = LoraModel
        fields = ['name']

