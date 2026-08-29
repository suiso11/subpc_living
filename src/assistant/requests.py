"""AssistantRequest を検証付きで生成するファクトリ。

呼び出し側 (CLI / Voice / audio-text 等) は channel・profile・privacy を必ず
明示する。requested_provider / requested_model / allow_cloud は既定のまま
発明せず、呼び出し側が現在の意味論で渡した値だけを通す。
"""

from src.assistant.contracts import AssistantChannel, AssistantProfile, AssistantRequest
from src.llm.routing.contracts import PrivacyMode

_VALID_CHANNELS: tuple[AssistantChannel, ...] = (
    "cli",
    "web",
    "discord",
    "voice",
    "internal",
)
_VALID_PROFILES: tuple[AssistantProfile, ...] = (
    "chat_auto",
    "voice_fast",
    "task_local",
    "code_auto",
    "deep_reasoning",
    "private_local",
)
_VALID_PRIVACY: tuple[PrivacyMode, ...] = ("local_only", "local_preferred", "cloud_allowed")


def create_request(
    *,
    text: str,
    conversation_id: str,
    channel: AssistantChannel,
    profile: AssistantProfile,
    privacy: PrivacyMode,
    requested_provider: str | None = None,
    requested_model: str | None = None,
    allow_cloud: bool = False,
    request_id: str | None = None,
) -> AssistantRequest:
    """検証付きで AssistantRequest を生成する。

    channel / profile / privacy は許容値でない場合 ValueError を送出する。
    provider / model / allow_cloud は既定値のまま維持し、このファクトリが
    値を発明することはない。
    """
    if channel not in _VALID_CHANNELS:
        raise ValueError(f"invalid assistant channel: {channel!r}")
    if profile not in _VALID_PROFILES:
        raise ValueError(f"invalid assistant profile: {profile!r}")
    if privacy not in _VALID_PRIVACY:
        raise ValueError(f"invalid privacy mode: {privacy!r}")

    return AssistantRequest(
        text=text,
        conversation_id=conversation_id,
        channel=channel,
        profile=profile,
        privacy=privacy,
        requested_provider=requested_provider,
        requested_model=requested_model,
        allow_cloud=allow_cloud,
        request_id=request_id,
    )