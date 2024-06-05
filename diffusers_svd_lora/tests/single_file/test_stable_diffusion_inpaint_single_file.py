import gc
import unittest

import torch

from diffusers import (
    StableDiffusionInpaintPipeline,
)
from diffusers.utils import load_image
from diffusers.utils.testing_utils import (
    enable_full_determinism,
    require_torch_gpu,
    slow,
)

from .single_file_testing_utils import SDSingleFileTesterMixin


enable_full_determinism()


@slow
@require_torch_gpu
class StableDiffusionInpaintPipelineSingleFileSlowTests(unittest.TestCase, SDSingleFileTesterMixin):
    pipeline_class = StableDiffusionInpaintPipeline
    ckpt_path = "https://huggingface.co/runwayml/stable-diffusion-inpainting/blob/main/sd-v1-5-inpainting.ckpt"
    original_config = "https://raw.githubusercontent.com/runwayml/stable-diffusion/main/configs/stable-diffusion/v1-inpainting-inference.yaml"
    repo_id = "runwayml/stable-diffusion-inpainting"

    def setUp(self):
        super().setUp()
        gc.collect()
        torch.cuda.empty_cache()

    def tearDown(self):
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def get_inputs(self, device, generator_device="cpu", dtype=torch.float32, seed=0):
        generator = torch.Generator(device=generator_device).manual_seed(seed)
        init_image = load_image(
            "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main"
            "/stable_diffusion_inpaint/input_bench_image.png"
        )
        mask_image = load_image(
            "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main"
            "/stable_diffusion_inpaint/input_bench_mask.png"
        )
        inputs = {
            "prompt": "Face of a yellow cat, high resolution, sitting on a park bench",
            "image": init_image,
            "mask_image": mask_image,
            "generator": generator,
            "num_inference_steps": 3,
            "guidance_scale": 7.5,
            "output_type": "np",
        }
        return inputs

    def test_single_file_format_inference_is_same_as_pretrained(self):
        super().test_single_file_format_inference_is_same_as_pretrained(expected_max_diff=1e-3)

    def test_single_file_loading_4_channel_unet(self):
        # Test loading single file inpaint with a 4 channel UNet
        ckpt_path = "https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/v1-5-pruned-emaonly.safetensors"
        pipe = self.pipeline_class.from_single_file(ckpt_path)

        assert pipe.unet.config.in_channels == 4


@slow
@require_torch_gpu
class StableDiffusion21InpaintPipelineSingleFileSlowTests(unittest.TestCase, SDSingleFileTesterMixin):
    pipeline_class = StableDiffusionInpaintPipeline
    ckpt_path = (
        "https://huggingface.co/stabilityai/stable-diffusion-2-inpainting/blob/main/512-inpainting-ema.safetensors"
    )
    original_config = "https://raw.githubusercontent.com/Stability-AI/stablediffusion/main/configs/stable-diffusion/v2-inpainting-inference.yaml"
    repo_id = "stabilityai/stable-diffusion-2-inpainting"

    def setUp(self):
        super().setUp()
        gc.collect()
        torch.cuda.empty_cache()

    def tearDown(self):
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def get_inputs(self, device, generator_device="cpu", dtype=torch.float32, seed=0):
        generator = torch.Generator(device=generator_device).manual_seed(seed)
        init_image = load_image(
            "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main"
            "/stable_diffusion_inpaint/input_bench_image.png"
        )
        mask_image = load_image(
            "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main"
            "/stable_diffusion_inpaint/input_bench_mask.png"
        )
        inputs = {
            "prompt": "Face of a yellow cat, high resolution, sitting on a park bench",
            "image": init_image,
            "mask_image": mask_image,
            "generator": generator,
            "num_inference_steps": 3,
            "guidance_scale": 7.5,
            "output_type": "np",
        }
        return inputs

    def test_single_file_format_inference_is_same_as_pretrained(self):
        super().test_single_file_format_inference_is_same_as_pretrained(expected_max_diff=1e-3)
