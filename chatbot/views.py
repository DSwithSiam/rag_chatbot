from rest_framework import generics, status

from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from yaml import serializer

from chatbot.models import Chat, ChatMessage
from chatbot.serializers import ChatMessageSerializer, ChatSerializer


class ChatView(generics.CreateAPIView):
    permission_classes = [IsAuthenticated]  

    def post(self, request):
        try:
            user = request.user
            chat = ChatSerializer(data=request.data)
            if chat.is_valid():
                chat_instance = chat.save(user=user)
                return Response({'chat_id': chat_instance.id}, status=status.HTTP_201_CREATED)
            return Response(chat.errors, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)

    def get(self, request):
        try:
            user = request.user
            chats = Chat.objects.filter(user=user).order_by('-timestamp')
            serializers = ChatSerializer(chats, many=True)
            return Response(serializers.data, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)
        

class ChatMessageView(generics.CreateAPIView):
    permission_classes = [IsAuthenticated]

    def post(self, request, chat_id):
        try:
            user = request.user
            chat = Chat.objects.get(id=chat_id, user=user)
            previous_chats = ChatMessage.objects.filter(chat=chat).order_by('timestamp')[:-10]
            previous_chat_history = ChatMessageSerializer(data=previous_chats, many=True)
            
            ai_bot = "AI Bot(previous_chat_history, request.data['message'])"
            ai_bot_response = "AI Bot response based on the message and previous chat history"
            
            user_message = ChatMessageSerializer(data={
                'chat': chat.id,
                'sender': 'user',
                'message': request.data['message'],
                'file': request.data.get('file', None)
            })
            if user_message.is_valid():
                user_message.save()
            else:
                return Response(user_message.errors, status=status.HTTP_400_BAD_REQUEST)
            
            bot_message = ChatMessageSerializer(data={
                'chat': chat.id,
                'sender': 'bot',
                'message': ai_bot_response,
            })
            if bot_message.is_valid():
                bot_message.save()
            else:                
                return Response(bot_message.errors, status=status.HTTP_400_BAD_REQUEST)

            return Response({'message': ai_bot_response}, status=status.HTTP_201_CREATED)
        except Chat.DoesNotExist:
            return Response({'error': 'Chat not found'}, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)