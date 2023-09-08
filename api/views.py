from django.conf import settings
from rest_framework.viewsets import ModelViewSet
from rest_framework.response import Response
from rest_framework import status

from api.models import Camera, Criminals
from api.filters import CameraFilter, CriminalsFilter
from api.pagination import CameraPagination, CriminalsPagination
from api.utils import extract_zip, remove_directory, move_extracted
from api.serializers import (
    CameraSerializer,
    CriminalsSerializer,
    CriminalsPOSTAndPUTSerializer,
)

import os


class CameraAPIView(ModelViewSet):
    model = Camera
    serializer_class = CameraSerializer
    queryset = Camera.objects.all().order_by("name")
    lookup_field = "name"
    filterset_class = CameraFilter
    pagination_class = CameraPagination


class CriminalsAPIView(ModelViewSet):
    model = Criminals
    serializer_class = CriminalsSerializer
    queryset = Criminals.objects.all()
    lookup_field = "pk"
    filterset_class = CriminalsFilter
    pagination_class = CriminalsPagination

    def create(self, request, *args, **kwargs):
        serializer = CriminalsPOSTAndPUTSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        folder = serializer.validated_data["folder"]
        extractor = extract_zip(
            zip_path=folder,
            save_as_name=serializer.validated_data["first_name"],
            extract_to=os.path.join(settings.MEDIA_ROOT, "media/"),
        )
        if extractor:
            return self.perform_create(serializer.data)
        else:
            return Response(
                data={"msg": "The user is already added to database"},
                status=status.HTTP_200_OK,
            )

    def perform_create(self, serializer):
        del serializer["folder"]
        model = self.model(**serializer)
        model.save()
        return Response(
            data={"msg": "Criminal details added to database"},
            status=status.HTTP_201_CREATED,
        )

    def update(self, request, *args, **kwargs):
        serializer = CriminalsPOSTAndPUTSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        is_updated = move_extracted(
            zip_path=serializer.validated_data.get("folder"),
            move_to="media/" + serializer.validated_data["first_name"] + "/",
        )
        if is_updated[-1]:
            serializer_class = self.serializer_class(data=request.data)
            serializer_class.is_valid(raise_exception=True)
            self.perform_update(serializer_class)
            return Response(
                data={"msg": is_updated[0]},
                status=status.HTTP_200_OK,
            )
        else:
            return Response(
                data={"msg": is_updated[0]}, status=status.HTTP_400_BAD_REQUEST
            )

    def destroy(self, request, *args, **kwargs):
        wanted = self.model.objects.get(pk=self.kwargs.get("pk"))
        if wanted is not None:
            path = settings.MEDIA_ROOT + "media/" + wanted.first_name
            if os.path.exists(path):
                remove_directory(path)
                self.perform_destroy(wanted)
                return Response(status=status.HTTP_204_NO_CONTENT)
            else:
                return Response(
                    data={"msg": "Error occured"},
                    status=status.HTTP_400_BAD_REQUEST,
                )
        else:
            return Response(
                data={"msg": "User not found"}, status=status.HTTP_404_NOT_FOUND
            )
