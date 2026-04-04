from rest_framework import serializers
from .models import MediaRequest, Character, Relationship


class MediaRequestSerializer(serializers.ModelSerializer):
    media_type_display = serializers.CharField(source='get_media_type_display', read_only=True)
    status_display     = serializers.CharField(source='get_status_display',     read_only=True)
    cached_id          = serializers.SerializerMethodField()

    class Meta:
        model  = MediaRequest
        fields = [
            'id', 'title', 'media_type', 'media_type_display',
            'wiki_slug', 'wiki_url',
            'release_year', 'creators', 'genres',
            'resolution_strategy', 'chosen_category', 'metadata_categories', 'strategy_reasoning',
            'status', 'status_display', 'error_message',
            'submitted_at', 'completed_at',
            'cached_id',
        ]
        read_only_fields = [
            'wiki_slug', 'wiki_url',
            'release_year', 'creators', 'genres',
            'status', 'error_message',
            'submitted_at', 'completed_at',
        ]

    def get_cached_id(self, obj) -> int | None:
        """
        For duplicate requests, parse the cached MediaRequest id out of
        error_message so the frontend can redirect without a second lookup.
        Returns None for all non-duplicate requests.
        """
        if obj.status != 'dup':
            return None
        match = re.search(r'#(\d+)', obj.error_message or '')
        return int(match.group(1)) if match else None

class CharacterSerializer(serializers.ModelSerializer):
    class Meta:
        model  = Character
        fields = ['id', 'name', 'aliases', 'description', 'image_url', 'wiki_page', 'mention_count']


class RelationshipSerializer(serializers.ModelSerializer):
    source      = serializers.IntegerField(source='source_id',    read_only=True)
    target      = serializers.IntegerField(source='target_id',    read_only=True)
    source_name = serializers.CharField(source='source.name',     read_only=True)
    target_name = serializers.CharField(source='target.name',     read_only=True)

    class Meta:
        model  = Relationship
        fields = [
            'id', 'source', 'target', 'source_name', 'target_name',
            'relationship_type', 'description',
        ]