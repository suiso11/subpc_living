"""
GPU 検出・デバイス自動設定モジュール
Phase 9: GPU換装に伴い、VRAM容量に応じて最適なデバイス設定を自動決定する。

- P40 (24GB): 全モジュールGPU化、STT medium/float16
- GTX 1060 (6GB): LLMのみGPU、STT/Embedding/VisionはCPU
- GPUなし: 全てCPU
"""
import subprocess
import shutil
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GpuInfo:
    """検出されたGPU情報"""
    available: bool = False
    name: str = ""
    vram_mb: int = 0
    vram_gb: float = 0.0
    cuda_available: bool = False
    driver_version: str = ""


@dataclass
class DeviceConfig:
    """各モジュールのデバイス設定"""
    # GPU情報
    gpu: GpuInfo = field(default_factory=GpuInfo)
    profile: str = "cpu"  # "p40", "gtx1060", "cpu"

    # STT (faster-whisper)
    stt_device: str = "cpu"
    stt_compute_type: str = "int8"
    stt_model_size: str = "small"

    # Embedding (sentence-transformers)
    embedding_device: str = "cpu"

    # Vision ONNX
    onnx_providers: list[str] = field(default_factory=lambda: ["CPUExecutionProvider"])

    # LLM 推奨モデル
    recommended_model: str = "qwen2.5:7b-instruct-q4_K_M"
    recommended_ctx: int = 4096


def detect_gpu() -> GpuInfo:
    """nvidia-smi でGPU情報を検出する"""
    info = GpuInfo()

    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return info

    try:
        result = subprocess.run(
            [nvidia_smi,
             "--query-gpu=gpu_name,memory.total,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return info

        parts = [p.strip() for p in result.stdout.strip().split(",")]
        if len(parts) >= 3:
            info.available = True
            info.name = parts[0]
            info.vram_mb = int(float(parts[1]))
            info.vram_gb = round(info.vram_mb / 1024, 1)
            info.driver_version = parts[2]

            # CUDA利用可能かチェック (PyTorchベース)
            try:
                import torch
                info.cuda_available = torch.cuda.is_available()
            except ImportError:
                # PyTorchなしでもnvidia-smiがあればCUDA自体は利用可能と推定
                info.cuda_available = True

    except (subprocess.TimeoutExpired, Exception):
        pass

    return info


def get_device_config(gpu: Optional[GpuInfo] = None) -> DeviceConfig:
    """GPU情報に基づいて最適なデバイス設定を返す

    Args:
        gpu: GPU情報。Noneの場合は自動検出する。

    Returns:
        DeviceConfig: 各モジュール向けのデバイス設定
    """
    if gpu is None:
        gpu = detect_gpu()

    config = DeviceConfig(gpu=gpu)

    if not gpu.available or not gpu.cuda_available:
        # GPUなし → 全てCPU
        config.profile = "cpu"
        return config

    if gpu.vram_mb >= 20000:
        # === P40クラス (24GB) ===
        config.profile = "p40"

        # STT: GPU + float16 + mediumモデル
        config.stt_device = "cuda"
        config.stt_compute_type = "float16"
        config.stt_model_size = "medium"

        # Embedding: GPU
        config.embedding_device = "cuda"

        # Vision ONNX: GPU優先
        config.onnx_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        # LLM: 13Bモデル推奨
        config.recommended_model = "qwen2.5:14b-instruct-q4_K_M"
        config.recommended_ctx = 8192

    elif gpu.vram_mb >= 4000:
        # === GTX 1060クラス (6GB) ===
        config.profile = "gtx1060"

        # STT: CPU (VRAMはLLM専用)
        config.stt_device = "cpu"
        config.stt_compute_type = "int8"
        config.stt_model_size = "small"

        # Embedding: CPU
        config.embedding_device = "cpu"

        # Vision ONNX: CPU
        config.onnx_providers = ["CPUExecutionProvider"]

        # LLM: 7Bモデル
        config.recommended_model = "qwen2.5:7b-instruct-q4_K_M"
        config.recommended_ctx = 4096

    else:
        # === 低VRAMまたは未対応 ===
        config.profile = "cpu"

    return config


def resolve_device(device: str, module: str = "stt") -> str:
    """
    device="auto" を実際のデバイス名に解決する。

    Args:
        device: "auto", "cpu", "cuda" のいずれか
        module: モジュール名 ("stt", "embedding")

    Returns:
        解決されたデバイス名 ("cpu" or "cuda")
    """
    if device != "auto":
        return device

    config = get_device_config()

    if module == "stt":
        return config.stt_device
    elif module == "embedding":
        return config.embedding_device
    else:
        return "cpu"


def resolve_stt_config(device: str = "auto", compute_type: str = "auto",
                       model_size: str = "auto") -> tuple[str, str, str]:
    """STT用の設定を解決する。

    Returns:
        (device, compute_type, model_size)
    """
    if device != "auto" and compute_type != "auto" and model_size != "auto":
        return device, compute_type, model_size

    config = get_device_config()

    return (
        config.stt_device if device == "auto" else device,
        config.stt_compute_type if compute_type == "auto" else compute_type,
        config.stt_model_size if model_size == "auto" else model_size,
    )


def resolve_onnx_providers(providers: Optional[list[str]] = None) -> list[str]:
    """ONNX Runtime のプロバイダーリストを解決する。

    Args:
        providers: 指定プロバイダー。Noneの場合は自動検出。

    Returns:
        プロバイダーリスト
    """
    if providers is not None:
        return providers

    config = get_device_config()
    return config.onnx_providers


# --- Singleton キャッシュ ---
_cached_config: Optional[DeviceConfig] = None


def get_cached_config() -> DeviceConfig:
    """キャッシュされたDeviceConfigを返す（GPU検出は1回のみ）"""
    global _cached_config
    if _cached_config is None:
        _cached_config = get_device_config()
    return _cached_config


def main():
    """CLI: 現在のGPU設定を表示"""
    import json

    config = get_device_config()

    output = {
        "profile": config.profile,
        "gpu": {
            "available": config.gpu.available,
            "name": config.gpu.name,
            "vram_gb": config.gpu.vram_gb,
            "cuda_available": config.gpu.cuda_available,
            "driver_version": config.gpu.driver_version,
        },
        "stt": {
            "device": config.stt_device,
            "compute_type": config.stt_compute_type,
            "model_size": config.stt_model_size,
        },
        "embedding": {
            "device": config.embedding_device,
        },
        "onnx_providers": config.onnx_providers,
        "llm": {
            "recommended_model": config.recommended_model,
            "recommended_ctx": config.recommended_ctx,
        },
    }

    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
