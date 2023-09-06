from rest_framework.viewsets import ModelViewSet

from api.models import Camera, Criminals
from api.filters import CameraFilter, CriminalsFilter
from api.pagination import CameraPagination, CriminalsPagination
from api.serializers import CameraSerializer, CriminalsSerializer


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
