from rest_framework import serializers
from .models import Conversation, Message


class MessageSerializer(serializers.ModelSerializer):
    sender_email = serializers.EmailField(source="sender.email", read_only=True)
    sender_username = serializers.CharField(source="sender.username", read_only=True)
    sender_role = serializers.CharField(source="sender.role", read_only=True)
    status = serializers.SerializerMethodField()

    class Meta:
        model = Message
        fields = [
            "id",
            "conversation",
            "sender",
            "sender_email",
            "sender_username",
            "sender_role",
            "body",
            "sent_at",
            "delivered_at",
            "read_at",
            "status",
        ]
        read_only_fields = ["conversation", "sender", "sent_at", "delivered_at", "read_at", "status"]

    def get_status(self, obj):
        user = getattr(self.context.get("request"), "user", None)
        if not user or not user.is_authenticated:
            return "sent"
        if obj.read_at:
            return "read"
        if obj.delivered_at:
            return "delivered"
        return "sent"


class ConversationSerializer(serializers.ModelSerializer):
    unread_count = serializers.SerializerMethodField()
    last_message = serializers.SerializerMethodField()
    counterpart = serializers.SerializerMethodField()

    class Meta:
        model = Conversation
        fields = ["id", "doctor", "patient", "created_at", "unread_count", "last_message", "counterpart"]

    def get_unread_count(self, obj):
        user = getattr(self.context.get("request"), "user", None)
        if not user or not user.is_authenticated:
            return 0
        return obj.messages.exclude(sender=user).filter(read_at__isnull=True).count()

    def get_last_message(self, obj):
        message = obj.messages.select_related("sender").order_by("-sent_at").first()
        if not message:
            return None
        return MessageSerializer(message, context=self.context).data

    def get_counterpart(self, obj):
        user = getattr(self.context.get("request"), "user", None)
        if not user or not user.is_authenticated:
            return None

        if user.role == "doctor":
            counterpart_user = obj.patient.user
            return {
                "id": obj.patient.id,
                "user_id": obj.patient.firebase_user_id,
                "email": counterpart_user.email,
                "username": counterpart_user.username,
                "role": counterpart_user.role,
            }

        counterpart_user = obj.doctor.user
        return {
            "id": obj.doctor.id,
            "email": counterpart_user.email,
            "username": counterpart_user.username,
            "role": counterpart_user.role,
        }
