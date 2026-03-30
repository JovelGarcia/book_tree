from django.contrib import admin
from .models import MediaRequest, Character, Relationship


@admin.register(MediaRequest)
class MediaRequestAdmin(admin.ModelAdmin):
    list_display   = ['title', 'media_type', 'status', 'wiki_slug', 'submitted_at', 'completed_at']
    list_filter    = ['status', 'media_type', 'submitted_at']
    search_fields  = ['title', 'wiki_slug']
    readonly_fields = ['wiki_slug', 'wiki_url', 'status', 'error_message',
                       'submitted_at', 'completed_at']


@admin.register(Character)
class CharacterAdmin(admin.ModelAdmin):
    list_display  = ['name', 'media', 'mention_count', 'wiki_page']
    list_filter   = ['media']
    search_fields = ['name']
    readonly_fields = ['media']


@admin.register(Relationship)
class RelationshipAdmin(admin.ModelAdmin):
    list_display  = ['character_1', 'character_2', 'relationship_type', 'confidence']
    list_filter   = ['media', 'relationship_type']
    search_fields = ['character_1__name', 'character_2__name']
    readonly_fields = ['media', 'character_1', 'character_2']