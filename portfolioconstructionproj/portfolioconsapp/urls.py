"""
URL configuration for portfolioconstructionproj project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from . import views

app_name = 'portfolio'

urlpatterns = [
    path('index.html',views.index, name='home'),
    path('result/', views.buildDatasets, name='result'),
    path('download/<str:filename>', views.downloadfile, name='download'),
    path('visual_p1.html', views.generateVisualizationForP1, name='visual1'),
    path('visual_p2.html', views.generateVisualizationForP2, name='visual2'),
    path('model.html', views.model1, name='model'),
    path('model2.html', views.model2, name='model2'),
    path('end.html', views.recommendation, name='end'),
    path('abstract.html', views.abstract, name='abstract'),
]
