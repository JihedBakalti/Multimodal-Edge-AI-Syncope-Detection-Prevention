from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer
from django.utils import timezone
from rest_framework import generics, status
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.permissions import IsAuthenticated

from apps.accounts.models import DoctorPatientAssignment, DoctorProfile
from .models import Conversation, Message
from .serializers import ConversationSerializer, MessageSerializer


def _allowed_patient_ids(user):
    if user.role == "patient":
        patient_profile = getattr(user, "patient_profile", None)
        return [patient_profile.id] if patient_profile else []
    if user.role == "doctor":
        return list(
            DoctorPatientAssignment.objects.filter(doctor__user=user, active=True).values_list("patient_id", flat=True)
        )
    return []


def _allowed_doctor_ids(user):
    if user.role == "doctor":
        doctor_profile, _ = DoctorProfile.objects.get_or_create(user=user)
        return [doctor_profile.id]
    if user.role == "patient":
        return list(
            DoctorPatientAssignment.objects.filter(patient__user=user, active=True).values_list("doctor_id", flat=True)
        )
    return []


def _conversation_queryset_for_user(user):
    return Conversation.objects.filter(
        patient_id__in=_allowed_patient_ids(user),
        doctor_id__in=_allowed_doctor_ids(user),
    ).select_related("patient__user", "doctor__user")


def _as_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _broadcast_to_conversation(conversation_id, payload):
    channel_layer = get_channel_layer()
    async_to_sync(channel_layer.group_send)(
        f"conversation_{conversation_id}",
        {"type": "chat_message", "payload": payload},
    )


class ConversationListCreateView(generics.ListCreateAPIView):
    permission_classes = [IsAuthenticated]
    serializer_class = ConversationSerializer

    def get_queryset(self):
        return _conversation_queryset_for_user(self.request.user).order_by("-created_at")

    def create(self, request, *args, **kwargs):
        user = request.user

        if user.role == "doctor":
            patient_id = _as_int(request.data.get("patient"))
            if not patient_id:
                return Response({"detail": "patient is required."}, status=status.HTTP_400_BAD_REQUEST)
            if patient_id not in set(_allowed_patient_ids(user)):
                return Response({"detail": "You are not allowed to message this patient."}, status=status.HTTP_403_FORBIDDEN)
            doctor_profile, _ = DoctorProfile.objects.get_or_create(user=user)
            conversation, created = Conversation.objects.get_or_create(
                doctor=doctor_profile,
                patient_id=patient_id,
            )
        elif user.role == "patient":
            doctor_id = _as_int(request.data.get("doctor"))
            if not doctor_id:
                return Response({"detail": "doctor is required."}, status=status.HTTP_400_BAD_REQUEST)
            if doctor_id not in set(_allowed_doctor_ids(user)):
                return Response({"detail": "You are not allowed to message this doctor."}, status=status.HTTP_403_FORBIDDEN)
            patient_profile = getattr(user, "patient_profile", None)
            if patient_profile is None:
                return Response({"detail": "Patient profile is missing."}, status=status.HTTP_400_BAD_REQUEST)
            conversation, created = Conversation.objects.get_or_create(
                doctor_id=doctor_id,
                patient=patient_profile,
            )
        else:
            return Response({"detail": "Invalid role for messaging."}, status=status.HTTP_403_FORBIDDEN)

        serializer = self.get_serializer(conversation)
        return Response(serializer.data, status=status.HTTP_201_CREATED if created else status.HTTP_200_OK)


class MessageListCreateView(generics.ListCreateAPIView):
    permission_classes = [IsAuthenticated]
    serializer_class = MessageSerializer

    def get_queryset(self):
        conv_id = self.kwargs["conversation_id"]
        return (
            Message.objects.filter(
                conversation_id=conv_id,
                conversation__patient_id__in=_allowed_patient_ids(self.request.user),
                conversation__doctor_id__in=_allowed_doctor_ids(self.request.user),
            )
            .select_related("sender")
            .order_by("sent_at")
        )

    def perform_create(self, serializer):
        conversation = _conversation_queryset_for_user(self.request.user).filter(id=self.kwargs["conversation_id"]).first()
        if not conversation:
            raise PermissionError("Conversation access denied.")

        now = timezone.now()
        message = serializer.save(
            sender=self.request.user,
            conversation_id=self.kwargs["conversation_id"],
            delivered_at=now,
        )

        payload = {
            "type": "message.created",
            "message": MessageSerializer(message, context={"request": self.request}).data,
        }
        _broadcast_to_conversation(conversation.id, payload)

    def create(self, request, *args, **kwargs):
        try:
            return super().create(request, *args, **kwargs)
        except PermissionError:
            return Response({"detail": "Conversation access denied."}, status=status.HTTP_403_FORBIDDEN)


class MessageMarkReadView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request, conversation_id):
        conversation = _conversation_queryset_for_user(request.user).filter(id=conversation_id).first()
        if not conversation:
            return Response({"detail": "Conversation access denied."}, status=status.HTTP_403_FORBIDDEN)

        now = timezone.now()
        unread_qs = Message.objects.filter(
            conversation_id=conversation_id,
            read_at__isnull=True,
        ).exclude(sender=request.user)
        message_ids = list(unread_qs.values_list("id", flat=True))
        unread_qs.update(read_at=now, delivered_at=now)

        payload = {
            "type": "message.read",
            "conversation_id": conversation_id,
            "message_ids": message_ids,
            "read_at": now.isoformat(),
            "reader_id": request.user.id,
        }
        _broadcast_to_conversation(conversation_id, payload)
        return Response({"updated": len(message_ids)})
