from django.urls import path
from .views import ChatView, ChatMessageView

urlpatterns = [
    path('chat/', ChatView.as_view(), name='chat'),
    path('messages/', ChatMessageView.as_view(), name='chatmessage'),
]