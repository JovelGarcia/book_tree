from django.contrib import admin
from .models import MediaRequest, Character, Relationship


@admin.register(MediaRequest)
class MediaRequestAdmin(admin.ModelAdmin):
    list_display    = ['title', 'media_type', 'status', 'resolution_strategy', 'wiki_slug', 'submitted_at', 'completed_at']
    list_filter     = ['status', 'media_type', 'resolution_strategy', 'submitted_at']
    search_fields   = ['title', 'wiki_slug']
    readonly_fields = ['wiki_slug', 'wiki_url', 'error_message', 'submitted_at', 'completed_at']


@admin.register(Character)
class CharacterAdmin(admin.ModelAdmin):
    list_display    = ['name', 'media', 'mention_count', 'wiki_page']
    list_filter     = ['media']
    search_fields   = ['name', 'media__title']
    readonly_fields = ['media']


@admin.register(Relationship)
class RelationshipAdmin(admin.ModelAdmin):
    list_display    = ['source', 'target', 'relationship_type', 'media']
    list_filter     = ['media', 'relationship_type']
    search_fields   = ['source__name', 'target__name']  # also fixed the typo (was source__name twice)
    readonly_fields = ['media', 'source', 'target']