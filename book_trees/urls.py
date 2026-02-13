"""Defines URL patterns for book_trees."""

from django.urls import path
from . import views

urlpatterns = [
    path('api/epubs/', views.epub_list_api),
]
