"""skintonedetection URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
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
from django.conf.urls import url
from django.conf.urls.static import static
from django.conf import settings

urlpatterns = [
    path('admin/', admin.site.urls),
    path("", views.home, name='home'),
    path("about", views.about, name='about'),
    path("contact", views.contact, name='contact'),
    path("captureimage", views.captureimage, name='captureimage'),
    path("disclaimer", views.disclaimer, name='disclaimer'),
    path("knowmore", views.knowmore, name='knowmore'),
    path("thanku", views.thanku, name='thanku'),
    path("result", views.result, name="result"),
    path("detected", views.detectedSkin, name='detected'),

]

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
