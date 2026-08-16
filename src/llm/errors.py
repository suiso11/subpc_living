"""LLM Provider境界で利用する共通例外。"""


class LLMProviderError(RuntimeError):
    """Provider操作に失敗したときの基底例外。"""

    def __init__(self, provider_id: str, operation: str, message: str) -> None:
        self.provider_id = provider_id
        self.operation = operation
        super().__init__(f"{provider_id}.{operation}: {message}")


class ProviderTimeoutError(LLMProviderError):
    """Providerへの要求がタイムアウトした。"""


class ProviderRequestError(LLMProviderError):
    """Providerとの通信またはHTTP応答処理に失敗した。"""
