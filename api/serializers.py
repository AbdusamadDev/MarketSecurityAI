from rest_framework import serializers
from django.core.exceptions import ValidationError

from api.models import Camera, Criminals
from api.utils import is_valid_character


class CameraSerializer(serializers.ModelSerializer):
    class Meta:
        model = Camera
        fields = ["name", "url", "longitude", "latitude"]

    def validate_name(self, value):
        is_valid_character(value, "name")
        return value


class CriminalsSerializer(serializers.Serializer):
    folder = serializers.FileField(required=True)
    first_name = serializers.CharField(required=True)
    last_name = serializers.CharField(required=True)
    age = serializers.IntegerField(required=True)
    description = serializers.CharField(required=True)

    def validate(self, attrs):
        first_name = is_valid_character(attrs.get("first_name"))
        last_name = is_valid_character(attrs.get("last_name"))
        print(first_name, last_name)
        if first_name and last_name:
            return attrs
        else:
            raise ValidationError("Validation failed for field: first_name, last_name")
