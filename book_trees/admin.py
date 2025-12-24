from django.contrib import admin
from .models import EpubFile, Chapter, Character, Relationship

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
    list_display = ['epub', 'title', 'chapter_number', 'content', 'annotated_sentences']
    list_filter = ['epub']
    search_fields = ['chapter_number']
    readonly_fields = ['content', 'epub']


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
    search_fields = ['character_1', 'character_2', 'aliases']
    readonly_fields = ['confidence']
