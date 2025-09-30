from django.urls import path
from . import views

urlpatterns = [
    path('', views.upload_document, name = 'upload_document'),
    path('query/', views.query_document, name = 'query_document'),
    path('api/query/', views.api_query, name = 'api_query'),
]
