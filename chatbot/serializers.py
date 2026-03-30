from rest_framework import serializers
from django.contrib.auth import get_user_model

from chatbot.models import Chat, ChatMessage
User = get_user_model()




class ChatSerializer(serializers.ModelSerializer):
    class Meta:
        model = Chat
        fields = ['id', 'chat_title', 'timestamp']
        extra_kwargs = {
            'chat_title': {'required': False, 'default': 'New Chat'}
        }

class ChatMessageSerializer(serializers.ModelSerializer):
    class Meta:
        model = ChatMessage
        fields = ['id', 'sender', 'message', 'file', 'timestamp']
        extra_kwargs = {
            'file': {'required': False, 'allow_null': True}
        }

    