from django.contrib import admin
from django.utils.html import format_html
from .models import EpubFile, Chapter, Character, Relationship
import json


# Register your models here.

@admin.register(EpubFile)
class EpubFileAdmin(admin.ModelAdmin):
    list_display = ['original_filename', 'status', 'uploaded_at', 'processed', 'id']
    list_filter =  ['status', 'processed', 'uploaded_at']
    search_fields = ['original_filename']
    readonly_fields = ['uploaded_at']


#ChapterAdmin
@admin.register(Chapter)
class ChapterAdmin(admin.ModelAdmin):
    list_display = ['epub', 'title', 'chapter_number', 'chunk_count']
    list_filter = ['epub']
    search_fields = ['chapter_number', 'title']
    readonly_fields = ['content', 'epub', 'formatted_chunks']

    fieldsets = (
        ('Basic Info', {
            'fields': ('epub', 'title', 'chapter_number')
        }),
        ('Content', {
            'fields': ('content',),
            'classes': ('collapse',)
        }),
        ('Annotated Chunks', {
            'fields': ('formatted_chunks',),
        }),
    )

    def chunk_count(self, obj):
        """Show number of chunks found"""
        return len(obj.annotated_sentences) if obj.annotated_sentences else 0

    chunk_count.short_description = 'Chunks'

    def formatted_chunks(self, obj):
        """Display chunks in a readable HTML format"""
        if not obj.annotated_sentences:
            return format_html('<p>No chunks yet</p>')

        html = '<div>'

        for i, chunk in enumerate(obj.annotated_sentences, 1):
            characters = chunk.get('characters_in_context', [])
            char_list = ', '.join(characters) if characters else 'None'
            center_chars = chunk.get('characters_in_sentence', [])
            context_text = chunk.get('context', chunk.get('center_sentence', 'No context'))

            html += f'''
                <div style="border: 2px solid; padding: 10px; margin-bottom: 15px;">
                    <p><strong>Chunk #{i}</strong></p>
                    <p><strong>Characters:</strong> {char_list}</p>
                    <p><strong>Center:</strong> {center_chars}</p>
                    <p style="background-color:; padding: 10px;">
                        {context_text}
                    </p>
                </div>
            '''

        html += '</div>'

        return format_html(html)

    formatted_chunks.short_description = 'Annotated Chunks (Readable)'


#CharacterAdmin
@admin.register(Character)
class CharacterAdmin(admin.ModelAdmin):
    list_display = ['name', 'aliases', 'mention_count', 'first_appearance_chapter']
    list_filter = ['epub']
    search_fields = ['name', 'aliases']
    readonly_fields = ['epub']


#RelationshipAdmin
@admin.register(Relationship)
class RelationshipAdmin(admin.ModelAdmin):
    list_display = ['character_1', 'character_2', 'relationship_type', 'confidence', 'evidence']
    list_filter = ['relationship_type', 'confidence']
    search_fields = ['character_1__name', 'character_2__name', 'relationship_type']
    readonly_fields = ['confidence']