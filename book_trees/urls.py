from django.urls import path
from . import views

urlpatterns = [
    path('api/media/',                        views.media_list_api),
    path('api/media/submit/',                 views.media_submit_api),
    path('api/media/recent/',                 views.media_recent_api),
    path('api/media/<int:id>/',               views.media_detail_api),
    path('api/media/<int:id>/graph/',         views.media_graph_api),
    path('api/media/<int:id>/characters/',    views.character_list_api),
    path('api/media/<int:id>/relationships/', views.relationship_list_api),
    path('api/media/popular/',          views.media_popular_api,         name='media-popular'),
    path('api/media/<int:id>/view/',    views.media_increment_view_api,  name='media-increment-view'),
]