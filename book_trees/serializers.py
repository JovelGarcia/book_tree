from rest_framework import serializers
from .models import EpubFile,  Character, Relationship

class EpubSerializer(serializers.ModelSerializer):
    class Meta:
        model = EpubFile
        fields = '__all__'

class CharacterSerializer(serializers.ModelSerializer):
    class Meta:
        model = Character
        fields = ['id', 'name', 'aliases', 'mention_count', 'first_appearance_chapter']


class RelationshipSerializer(serializers.ModelSerializer):
    character_1_name = serializers.CharField(source='character_1.name', read_only=True)
    character_2_name = serializers.CharField(source='character_2.name', read_only=True)

    class Meta:
        model = Relationship
        fields = [
            'id',
            'character_1', 'character_1_name',
            'character_2', 'character_2_name',
            'relationship_type', 'confidence', 'evidence'
        ]