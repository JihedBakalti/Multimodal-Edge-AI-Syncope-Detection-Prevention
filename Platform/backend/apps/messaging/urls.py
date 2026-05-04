from django.urls import path
from .views import ConversationListCreateView, MessageListCreateView, MessageMarkReadView


urlpatterns = [
    path("conversations/", ConversationListCreateView.as_view(), name="messaging-conversations"),
    path("conversations/<int:conversation_id>/messages/", MessageListCreateView.as_view(), name="messaging-messages"),
    path("conversations/<int:conversation_id>/mark-read/", MessageMarkReadView.as_view(), name="messaging-mark-read"),
]
