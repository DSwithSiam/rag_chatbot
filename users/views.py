
from rest_framework import status
from rest_framework.response import Response
from rest_framework.decorators import APIView
from rest_framework.permissions import AllowAny
from rest_framework_simplejwt.tokens import RefreshToken
from django.contrib.auth import authenticate
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi

from .serializers import *




class RegisterView(APIView):
    permission_classes = [AllowAny]
    
    @swagger_auto_schema(
        operation_description="User registration endpoint. Create a new user account with email and password.",
        request_body=openapi.Schema(
            type=openapi.TYPE_OBJECT,
            properties={
                'email': openapi.Schema(type=openapi.TYPE_STRING, description='User email number'),
                'password': openapi.Schema(type=openapi.TYPE_STRING, description='User password'),
                'password2': openapi.Schema(type=openapi.TYPE_STRING, description='Password confirmation'),
            },
            required=['email', 'password', 'password2'],
        ),
        responses={
            201: openapi.Response(
                description='User created successfully',
                schema=openapi.Schema(
                    type=openapi.TYPE_OBJECT,
                    properties={
                        'refresh_token': openapi.Schema(type=openapi.TYPE_STRING),
                        'access_token': openapi.Schema(type=openapi.TYPE_STRING),
                    }
                )
            ),
            400: 'Serializer validation errors'
        }
    )
    def post(self, request):
        serializer = RegisterSerializer(data=request.data)
        if serializer.is_valid():
            user = serializer.save()
            refresh = RefreshToken.for_user(user)
            return Response(
                data={
                    'refresh_token': str(refresh),
                    'access_token': str(refresh.access_token),
                },
                status=status.HTTP_201_CREATED
            )
        return Response(
            data=serializer.errors,
            status=status.HTTP_400_BAD_REQUEST
        )



class LoginView(APIView):
    permission_classes = [AllowAny]
    
    @swagger_auto_schema(
        operation_description="User login endpoint. Authenticate with email number and password to get JWT tokens.",
        request_body=openapi.Schema(
            type=openapi.TYPE_OBJECT,
            properties={
                'email': openapi.Schema(type=openapi.TYPE_STRING, description='User email number'),
                'password': openapi.Schema(type=openapi.TYPE_STRING, description='User password'),
            },
            required=['email', 'password'],
        ),
        responses={
            200: openapi.Response(
                description='Login successful',
                schema=openapi.Schema(
                    type=openapi.TYPE_OBJECT,
                    properties={
                        'refresh_token': openapi.Schema(type=openapi.TYPE_STRING),
                        'access_token': openapi.Schema(type=openapi.TYPE_STRING),
                    }
                )
            ),
            401: openapi.Response(
                description='Invalid credentials',
                schema=openapi.Schema(
                    type=openapi.TYPE_OBJECT,
                    properties={
                        'message': openapi.Schema(type=openapi.TYPE_STRING),
                    }
                )
            ),
            403: openapi.Response(
                description='Account not verified',
                schema=openapi.Schema(
                    type=openapi.TYPE_OBJECT,
                    properties={
                        'message': openapi.Schema(type=openapi.TYPE_STRING),
                    }
                )
            ),
        }
    )
    def post(self, request):
        email = request.data.get('email')
        password = request.data.get('password')
        user = authenticate(request, email=email, password=password)

        if user is not None:
            if not user.is_verified:
                return Response(
                    data={
                        'message': 'Account is not verified. Please verify your account before logging in.'
                    },
                    status=status.HTTP_403_FORBIDDEN
                )
            refresh = RefreshToken.for_user(user)
            return Response(
                data={
                    'refresh_token': str(refresh),
                    'access_token': str(refresh.access_token),
                },
                status=status.HTTP_200_OK
            )
        return Response(
            data={
                'message': 'Invalid email number or password. Please try again.'
            },
            status=status.HTTP_401_UNAUTHORIZED
        )