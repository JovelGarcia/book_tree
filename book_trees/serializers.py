from rest_framework import serializers
from .models import EpubFile

class EpubSerializer(serializers.ModelSerializer):
    class Meta:
        model = EpubFile
        fields = '__all__'
