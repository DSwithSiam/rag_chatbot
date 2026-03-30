from django.contrib import admin
from django.urls import path, include
from drf_yasg import openapi
from rest_framework import permissions
from drf_yasg.views import get_schema_view


schema_view = get_schema_view(
   openapi.Info(
      title="Rag Chatbot API",
      default_version='v1',
      description="Rag Chatbot API Documentation",
      terms_of_service="",
      contact=openapi.Contact(email="siam.dev404@gmail.com", name="Md Masipul Islam Siam", role="Backend Developer"),
      license=openapi.License(name="MIT License"),
   ),
   public=True,
   permission_classes=(permissions.AllowAny,),
)


urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/v1/chatbot/', include('chatbot.urls')),
    path('api/v1/accounts/', include('users.urls')),

    path('swagger/', schema_view.with_ui('swagger', cache_timeout=0), name='schema-swagger-ui'),
]
