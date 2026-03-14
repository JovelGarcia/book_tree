"""Defines URL patterns for book_trees."""

from django.urls import path
from . import views

urlpatterns = [
    path('api/epubs/', views.epub_list_api),
    path('api/epubs/upload/', views.epub_upload_api),        # upload before <int:id>
    path('api/epubs/<int:id>/', views.epub_api),

    # Characters
    path('api/epubs/<int:epub_id>/characters/', views.character_list_api),
    path('api/epubs/<int:epub_id>/characters/<int:character_id>/', views.character_detail_api),

    # Relationships
    path('api/epubs/<int:epub_id>/relationships/', views.relationship_list_api),
    path('api/epubs/<int:epub_id>/relationships/<int:relationship_id>/', views.relationship_detail_api),

    #Labels
    path('api/label/sentences/', views.labeled_sentence_list_api),
    path('api/label/sentences/export/', views.export_spacy_format),
]