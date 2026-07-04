"""
GPU 検出・デバイス自動設定モジュール
Phase 9: GPU換装に伴い、VRAM容量に応じて最適なデバイス設定を自動決定する。
Phase 10: デュアルGPU対応

構成:
- P40 (24GB, Compute 6.1): LLM専用 (Ollama)
- RTX 2070 Super 等の Tensor Core GPU: 推論用 (STT/Embedding/Vision)
- P5000 等の Pascal GPU: 推論用にできるが float16 は避ける
- 単GPU: 従来のプロファイルで動作
- GPUなし: 全てCPU
"""
import os
import subprocess
import shutil
from dataclasses import dataclass, field
from typing import Optional

# CUDAデバイス列挙順をPCIバス順に統一 (nvidia-smiと一致させる)
# 必ずtorchインポート前に設定すること
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


@dataclass
class GpuInfo:
    """検出されたGPU情報"""
    available: bool = False
    name: str = ""
    vram_mb: int = 0
    vram_gb: float = 0.0
    cuda_available: bool = False
    driver_version: str = ""
    index: int = 0
    compute_capability: str = ""
    torch_compatible: bool = True


@dataclass
class DeviceConfig:
    """各モジュールのデバイス設定"""
    # GPU情報 (後方互換: 代表GPU)
    gpu: GpuInfo = field(default_factory=GpuInfo)
    gpus: list[GpuInfo] = field(default_factory=list)
    profile: str = "cpu"  # "dual_gpu", "p40", "gtx1060", "cpu"

    # GPU役割分担
    llm_gpu_index: int = 0       # LLM (Ollama) 用
    inference_gpu_index: int = 0  # 推論 (STT/Embedding/Vision) 用

    # STT (faster-whisper)
    stt_device: str = "cpu"
    stt_device_index: int = 0
    stt_compute_type: str = "int8"
    stt_model_size: str = "small"

    # Embedding (sentence-transformers)
    embedding_device: str = "cpu"

    # Vision ONNX
    onnx_providers: list = field(default_factory=lambda: ["CPUExecutionProvider"])

    # LLM 推奨モデル
    recommended_model: str = "qwen2.5:7b-instruct-q4_K_M"
    recommended_ctx: int = 4096


def detect_gpu() -> GpuInfo:
    """nvidia-smi で最初のGPU情報を検出する (後方互換)"""
    gpus = detect_all_gpus()
    return gpus[0] if gpus else GpuInfo()


def detect_all_gpus() -> list[GpuInfo]:
    """nvidia-smi で全GPU情報を検出する"""
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return []

    try:
        result = subprocess.run(
            [nvidia_smi,
             "--query-gpu=index,gpu_name,memory.total,driver_version,compute_cap",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return []

        gpus = []
        cuda_available = False
        torch_supported_caps: list[tuple[int, int]] = []
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            if cuda_available and hasattr(torch.cuda, 'get_arch_list'):
                for arch in torch.cuda.get_arch_list():
                    if arch.startswith("sm_"):
                        try:
                            num = int(arch[3:])
                            torch_supported_caps.append((num // 10, num % 10))
                        except ValueError:
                            pass
        except ImportError:
            cuda_available = True
        except Exception:
            pass

        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                compute_cap = parts[4] if len(parts) >= 5 else ""
                torch_ok = True
                if compute_cap and cuda_available and torch_supported_caps:
                    try:
                        major, minor = map(int, compute_cap.split("."))
                        if (major, minor) not in torch_supported_caps:
                            torch_ok = False
                    except (ValueError, TypeError):
                        pass

                info = GpuInfo(
                    available=True,
                    index=int(parts[0]),
                    name=parts[1],
                    vram_mb=int(float(parts[2])),
                    vram_gb=round(int(float(parts[2])) / 1024, 1),
                    driver_version=parts[3],
                    compute_capability=compute_cap,
                    cuda_available=cuda_available,
                    torch_compatible=torch_ok,
                )
                gpus.append(info)

        return gpus

    except (subprocess.TimeoutExpired, Exception):
        return []


def _classify_gpu(gpu: GpuInfo) -> str:
    """GPUを分類する"""
    name = gpu.name.lower()
    if "p40" in name or "p100" in name or "v100" in name or "a100" in name:
        return "compute"  # 大容量VRAM、LLM向き
    if "2070" in name or "2080" in name or "3060" in name or "3070" in name or "3080" in name or "4060" in name or "4070" in name or "4080" in name or "4090" in name:
        return "inference"  # Tensor Cores、推論向き
    if gpu.vram_mb >= 20000:
        return "compute"
    if gpu.vram_mb >= 6000:
        return "inference"
    return "basic"


def _compute_capability_tuple(gpu: GpuInfo) -> tuple[int, int]:
    try:
        major, minor = gpu.compute_capability.split(".", 1)
        return int(major), int(minor)
    except (AttributeError, ValueError, TypeError):
        return (0, 0)


def _has_fast_fp16(gpu: GpuInfo) -> bool:
    """Tensor Core 世代なら STT の float16 を優先する。"""
    major, _minor = _compute_capability_tuple(gpu)
    return major >= 7


def _configure_stt_for_gpu(config: DeviceConfig, gpu: GpuInfo) -> None:
    """STT設定をGPU世代に合わせる。

    Pascal (sm_61) は CTranslate2 CUDA で float16 非対応になりやすく、
    対応していても低速なので int8 を使う。
    """
    config.stt_device = "cuda"
    config.stt_device_index = gpu.index
    config.stt_model_size = "medium"
    config.stt_compute_type = "float16" if _has_fast_fp16(gpu) else "int8"


def get_device_config(gpu: Optional[GpuInfo] = None) -> DeviceConfig:
    """GPU情報に基づいて最適なデバイス設定を返す

    Args:
        gpu: GPU情報。Noneの場合は自動検出する (後方互換)。

    Returns:
        DeviceConfig: 各モジュール向けのデバイス設定
    """
    gpus = detect_all_gpus()

    # 後方互換: 単一GPUが渡された場合
    if gpu is not None:
        gpus = [gpu]

    if not gpus or not any(g.cuda_available for g in gpus):
        return DeviceConfig(profile="cpu")

    torch_gpus = [g for g in gpus if g.torch_compatible]

    config = DeviceConfig(
        gpu=gpus[0],
        gpus=gpus,
    )

    # === デュアルGPU検出 ===
    if len(gpus) >= 2:
        roles = {i: _classify_gpu(g) for i, g in enumerate(gpus)}

        # 大容量GPU (compute) → LLM、Tensor Core搭載GPU (inference) → 推論
        compute_gpus = [i for i, r in roles.items() if r == "compute"]
        inference_gpus = [i for i, r in roles.items() if r == "inference"]

        if compute_gpus and inference_gpus:
            # === P40 + 2070S のような理想的デュアル構成 ===
            config.profile = "dual_gpu"
            llm_idx = compute_gpus[0]
            inf_idx = inference_gpus[0]
            config.llm_gpu_index = gpus[llm_idx].index
            config.inference_gpu_index = gpus[inf_idx].index

            inf_gpu = gpus[inf_idx]

            # STT: Tensor Core GPU は float16、Pascal GPU は int8
            _configure_stt_for_gpu(config, inf_gpu)

            # Embedding: 推論GPU (PyTorch非対応ならCPU)
            config.embedding_device = f"cuda:{inf_gpu.index}" if inf_gpu.torch_compatible else "cpu"

            # Vision ONNX: 推論GPU (device_id指定)
            config.onnx_providers = [
                ("CUDAExecutionProvider", {"device_id": str(inf_gpu.index)}),
                "CPUExecutionProvider",
            ]

            # LLM: P40の大容量VRAMを活かす
            llm_gpu = gpus[llm_idx]
            if llm_gpu.vram_mb >= 20000:
                config.recommended_model = "qwen3.5:27b"
                config.recommended_ctx = 8192
            else:
                config.recommended_model = "qwen2.5:14b-instruct-q4_K_M"
                config.recommended_ctx = 8192

            return config

        # 同種GPU2枚の場合: 大きい方をLLM、小さい方を推論
        gpus_sorted = sorted(enumerate(gpus), key=lambda x: x[1].vram_mb, reverse=True)
        llm_i, llm_gpu = gpus_sorted[0]
        inf_i, inf_gpu = gpus_sorted[1]

        config.profile = "dual_gpu"
        config.llm_gpu_index = llm_gpu.index
        config.inference_gpu_index = inf_gpu.index

        _configure_stt_for_gpu(config, inf_gpu)

        config.embedding_device = f"cuda:{inf_gpu.index}" if inf_gpu.torch_compatible else "cpu"

        config.onnx_providers = [
            ("CUDAExecutionProvider", {"device_id": str(inf_gpu.index)}),
            "CPUExecutionProvider",
        ]

        config.recommended_model = "qwen2.5:14b-instruct-q4_K_M"
        config.recommended_ctx = 8192

        return config

    # === 単一GPU ===
    gpu = gpus[0]
    config.llm_gpu_index = gpu.index
    config.inference_gpu_index = gpu.index

    if gpu.vram_mb >= 20000:
        # === P40クラス (24GB) ===
        config.profile = "p40"

        _configure_stt_for_gpu(config, gpu)

        config.embedding_device = f"cuda:{gpu.index}" if gpu.torch_compatible else "cpu"

        config.onnx_providers = [
            ("CUDAExecutionProvider", {"device_id": str(gpu.index)}),
            "CPUExecutionProvider",
        ]

        config.recommended_model = "qwen2.5:14b-instruct-q4_K_M"
        config.recommended_ctx = 8192

    elif gpu.vram_mb >= 4000:
        # === GTX 1060クラス (6GB) ===
        config.profile = "gtx1060"

        config.stt_device = "cpu"
        config.stt_compute_type = "int8"
        config.stt_model_size = "small"

        config.embedding_device = "cpu"

        config.onnx_providers = ["CPUExecutionProvider"]

        config.recommended_model = "qwen2.5:7b-instruct-q4_K_M"
        config.recommended_ctx = 4096

    else:
        config.profile = "cpu"

    return config


def resolve_device(device: str, module: str = "stt") -> str:
    """
    device="auto" を実際のデバイス名に解決する。

    Args:
        device: "auto", "cpu", "cuda" のいずれか
        module: モジュール名 ("stt", "embedding")

    Returns:
        解決されたデバイス名 ("cpu" or "cuda" or "cuda:N")
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
                       model_size: str = "auto") -> tuple[str, str, str, int]:
    """STT用の設定を解決する。

    Returns:
        (device, compute_type, model_size, device_index)
    """
    config = get_device_config()

    resolved_device = config.stt_device if device == "auto" else device
    resolved_compute = config.stt_compute_type if compute_type == "auto" else compute_type
    resolved_model = config.stt_model_size if model_size == "auto" else model_size
    resolved_index = config.stt_device_index

    return resolved_device, resolved_compute, resolved_model, resolved_index


def resolve_onnx_providers(providers: Optional[list] = None) -> list:
    """ONNX Runtime のプロバイダーリストを解決する。

    Args:
        providers: 指定プロバイダー。Noneの場合は自動検出。

    Returns:
        プロバイダーリスト (文字列 or (名前, オプション辞書) のタプル)
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

    gpu_list = []
    for g in config.gpus:
        gpu_list.append({
            "index": g.index,
            "name": g.name,
            "vram_gb": g.vram_gb,
            "compute_capability": g.compute_capability,
            "role": _classify_gpu(g),
        })

    # onnx_providers を JSON シリアライズ可能な形にする
    onnx_prov_display = []
    for p in config.onnx_providers:
        if isinstance(p, tuple):
            onnx_prov_display.append({"provider": p[0], "options": p[1]})
        else:
            onnx_prov_display.append(p)

    output = {
        "profile": config.profile,
        "gpus": gpu_list,
        "gpu_assignment": {
            "llm_gpu_index": config.llm_gpu_index,
            "inference_gpu_index": config.inference_gpu_index,
        },
        "stt": {
            "device": config.stt_device,
            "device_index": config.stt_device_index,
            "compute_type": config.stt_compute_type,
            "model_size": config.stt_model_size,
        },
        "embedding": {
            "device": config.embedding_device,
        },
        "onnx_providers": onnx_prov_display,
        "llm": {
            "recommended_model": config.recommended_model,
            "recommended_ctx": config.recommended_ctx,
        },
    }

    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
