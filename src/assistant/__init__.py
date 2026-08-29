"""Assistantサービスの公開API。"""

from src.assistant.cloud_service import CloudRouteBridge
from src.assistant.contracts import (
    AssistantChannel,
    AssistantError,
    AssistantGenerationError,
    AssistantProfile,
    AssistantRequest,
    AssistantResponse,
)
from src.assistant.factory import build_assistant_service
from src.assistant.service import AssistantService, StreamResult
from src.assistant.stream_queue import QueueStream, stream_to_queue

__all__ = [
    "AssistantChannel",
    "AssistantError",
    "AssistantGenerationError",
    "AssistantProfile",
    "AssistantRequest",
    "AssistantResponse",
    "AssistantService",
    "CloudRouteBridge",
    "QueueStream",
    "StreamResult",
    "build_assistant_service",
    "stream_to_queue",
]
