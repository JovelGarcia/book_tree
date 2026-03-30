from rest_framework import serializers
from .models import MediaRequest, Character, Relationship


class MediaRequestSerializer(serializers.ModelSerializer):
    media_type_display = serializers.CharField(source='get_media_type_display', read_only=True)
    status_display     = serializers.CharField(source='get_status_display',     read_only=True)

    class Meta:
        model  = MediaRequest
        fields = [
            'id', 'title', 'media_type', 'media_type_display',
            'wiki_slug', 'wiki_url', 'metadata_categories', 'resolution_strategy', 'chosen_category', 'strategy_reasoning',
            'status', 'status_display', 'error_message',
            'submitted_at', 'completed_at',
        ]
        read_only_fields = [
            'wiki_slug', 'wiki_url',
            'status', 'error_message',
            'submitted_at', 'completed_at',
        ]


class CharacterSerializer(serializers.ModelSerializer):
    class Meta:
        model  = Character
        fields = ['id', 'name', 'aliases', 'description', 'image_url', 'wiki_page', 'mention_count']


class RelationshipSerializer(serializers.ModelSerializer):
    source = serializers.IntegerField(source='character_1_id', read_only=True)
    target = serializers.IntegerField(source='character_2_id', read_only=True)
    source_name = serializers.CharField(source='character_1.name', read_only=True)
    target_name = serializers.CharField(source='character_2.name', read_only=True)

    class Meta:
        model  = Relationship
        fields = [
            'id', 'source', 'target', 'source_name', 'target_name',
            'relationship_type', 'relationship_subtype', 'confidence',
        ]