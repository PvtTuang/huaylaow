from django.urls import path
from lottery import views

urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('history/', views.history, name='history'),
    path('api/fetch/', views.fetch_now, name='fetch_now'),
    path('api/predict/', views.refresh_prediction, name='refresh_prediction'),
]
