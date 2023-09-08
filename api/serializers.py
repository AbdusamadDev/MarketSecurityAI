from rest_framework import serializers
from django.core.exceptions import ValidationError

from api.models import Camera, Criminals
from api.utils import is_valid_character

ERROR_MESSAGE = "Please provide valid characters: leters, numbers and underscore only"


class CameraSerializer(serializers.ModelSerializer):
    class Meta:
        model = Camera
        fields = ["name", "url", "longitude", "latitude"]

    def validate_name(self, value):
        if is_valid_character(value):
            return value

        raise ValidationError(ERROR_MESSAGE)


class CriminalsPOSTAndPUTSerializer(serializers.Serializer):
    folder = serializers.FileField()
    first_name = serializers.CharField()
    last_name = serializers.CharField()
    age = serializers.IntegerField()
    description = serializers.CharField()

    def validate_first_name(self, value):
        if is_valid_character(value):   
            return value

        raise ValidationError(ERROR_MESSAGE)


class CriminalsSerializer(serializers.ModelSerializer):
    class Meta:
        model = Criminals
        fields = "__all__"
