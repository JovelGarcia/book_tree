from rest_framework import serializers
from .models import EpubFile, Character, Relationship, LabeledSentence

class LabeledSentenceSerializer(serializers.ModelSerializer):
    class Meta:
        model = LabeledSentence
        fields = [
            'id',
            'epub',
            'text',
            'entities',
            'source',
            'labeled_by',
            'labeled_at',
            'is_reviewed',
        ]
        read_only_fields = ['labeled_at']

    def validate_entities(self, value):
        """Ensure entities are valid spaCy-compatible spans."""
        if not isinstance(value, list):
            raise serializers.ValidationError("Entities must be a list.")
        for ent in value:
            if not all(k in ent for k in ('start', 'end', 'label')):
                raise serializers.ValidationError(
                    "Each entity must have 'start', 'end', and 'label' keys."
                )
            if not isinstance(ent['start'], int) or not isinstance(ent['end'], int):
                raise serializers.ValidationError("'start' and 'end' must be integers.")
            if ent['start'] >= ent['end']:
                raise serializers.ValidationError("'start' must be less than 'end'.")
        return value

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