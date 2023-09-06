from rest_framework import serializers
from django.core.exceptions import ValidationError
from django.contrib.auth.models import User

from api.models import Camera, Criminals
from api.utils import allowed_characters


class CameraSerializer(serializers.ModelSerializer):
    class Meta:
        model = Camera
        fields = ["name", "url", "longitude", "latitude"]

    def validate_name(self, value):
        self.check_allowed_characters(value, "name")
        return value

    def check_allowed_characters(self, value, field_name):
        for letter in value:
            if letter not in allowed_characters:
                raise ValidationError(
                    f"Field {field_name}: Only letters, capital letters, digits, and underscore are allowed."
                )


class CriminalsSerializer(serializers.ModelSerializer):
    class Meta:
        model = Criminals
        fields = ["first_name", "last_name", "age", "description"]

    def validate(self, attrs):
        self.check_allowed_characters(attrs.get("first_name"), "first_name")
        self.check_allowed_characters(attrs.get("last_name"), "last_name")
        return attrs

    def check_allowed_characters(self, value, field_name):
        for letter in value:
            if letter not in allowed_characters:
                raise ValidationError(
                    f"Field {field_name}: Only letters, capital letters, digits, and underscore are allowed."
                )

