import unittest
from unittest.mock import patch

from src.service.gpu_config import GpuInfo, get_device_config


def make_gpu(
    index: int,
    name: str,
    vram_mb: int,
    compute_capability: str,
    *,
    torch_compatible: bool = True,
) -> GpuInfo:
    return GpuInfo(
        available=True,
        name=name,
        vram_mb=vram_mb,
        vram_gb=round(vram_mb / 1024, 1),
        cuda_available=True,
        index=index,
        compute_capability=compute_capability,
        torch_compatible=torch_compatible,
    )


class GpuConfigTest(unittest.TestCase):
    def test_pascal_dual_gpu_uses_int8_stt(self):
        gpus = [
            make_gpu(0, "Tesla P40", 24576, "6.1", torch_compatible=False),
            make_gpu(1, "Quadro P5000", 16384, "6.1", torch_compatible=False),
        ]

        with patch("src.service.gpu_config.detect_all_gpus", return_value=gpus):
            config = get_device_config()

        self.assertEqual(config.profile, "dual_gpu")
        self.assertEqual(config.llm_gpu_index, 0)
        self.assertEqual(config.inference_gpu_index, 1)
        self.assertEqual(config.stt_device, "cuda")
        self.assertEqual(config.stt_device_index, 1)
        self.assertEqual(config.stt_compute_type, "int8")
        self.assertEqual(config.stt_model_size, "medium")
        self.assertEqual(config.embedding_device, "cpu")

    def test_tensor_core_inference_gpu_uses_float16_stt(self):
        gpus = [
            make_gpu(0, "Tesla P40", 24576, "6.1", torch_compatible=False),
            make_gpu(1, "RTX 2070 Super", 8192, "7.5"),
        ]

        with patch("src.service.gpu_config.detect_all_gpus", return_value=gpus):
            config = get_device_config()

        self.assertEqual(config.profile, "dual_gpu")
        self.assertEqual(config.llm_gpu_index, 0)
        self.assertEqual(config.inference_gpu_index, 1)
        self.assertEqual(config.stt_compute_type, "float16")
        self.assertEqual(config.embedding_device, "cuda:1")


if __name__ == "__main__":
    unittest.main()
