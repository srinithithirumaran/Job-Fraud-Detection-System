from django.urls import path

from . import views

urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('dataset/', views.dataset, name='dataset'),
    path('algorithms/', views.algorithms, name='algorithms'),
    path('comparison/', views.comparison, name='comparison'),
    path('prediction/', views.prediction, name='prediction'),
]
