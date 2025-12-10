from django.urls import path
from django.urls import path
from .views import recommend_view, start_chat_view, chat_message_view

urlpatterns = [
    # 기존 recommend/ 주소는 그대로 둡니다.
    path('recommend/', recommend_view, name='recommend'),

    # 👇 이 한 줄을 추가해주세요.
    path('', recommend_view, name='home'),
    
    # Chat API
    path('api/chat/start/', start_chat_view, name='chat_start'),
    path('api/chat/message/', chat_message_view, name='chat_message'),
]