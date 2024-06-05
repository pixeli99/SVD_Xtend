import gc
import tempfile
import unittest

import torch

from diffusers import EulerDiscreteScheduler, StableDiffusionPipeline
from diffusers.utils.testing_utils import (
    enable_full_determinism,
    require_torch_gpu,
    slow,
)

from .single_file_testing_utils import (
    SDSingleFileTesterMixin,
    download_original_config,
    download_single_file_checkpoint,
)


enable_full_determinism()


@slow
@require_torch_gpu
class StableDiffusionPipelineSingleFileSlowTests(unittest.TestCase, SDSingleFileTesterMixin):
    pipeline_class = StableDiffusionPipeline
    ckpt_path = "https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/v1-5-pruned-emaonly.safetensors"
    original_config = (
        "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/configs/stable-diffusion/v1-inference.yaml"
    )
    repo_id = "runwayml/stable-diffusion-v1-5"

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
        inputs = {
            "prompt": "a fantasy landscape, concept art, high resolution",
            "generator": generator,
            "num_inference_steps": 2,
            "strength": 0.75,
            "guidance_scale": 7.5,
            "output_type": "np",
        }
        return inputs

    def test_single_file_format_inference_is_same_as_pretrained(self):
        super().test_single_file_format_inference_is_same_as_pretrained(expected_max_diff=1e-3)

    def test_single_file_legacy_scheduler_loading(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_filename = self.ckpt_path.split("/")[-1]
            local_ckpt_path = download_single_file_checkpoint(self.repo_id, ckpt_filename, tmpdir)
            local_original_config = download_original_config(self.original_config, tmpdir)

            pipe = self.pipeline_class.from_single_file(
                local_ckpt_path,
                original_config=local_original_config,
                cache_dir=tmpdir,
                local_files_only=True,
                scheduler_type="euler",
            )

        # Default is PNDM for this checkpoint
        assert isinstance(pipe.scheduler, EulerDiscreteScheduler)

    def test_single_file_legacy_scaling_factor(self):
        new_scaling_factor = 10.0
        init_pipe = self.pipeline_class.from_single_file(self.ckpt_path)
        pipe = self.pipeline_class.from_single_file(self.ckpt_path, scaling_factor=new_scaling_factor)

        assert init_pipe.vae.config.scaling_factor != new_scaling_factor
        assert pipe.vae.config.scaling_factor == new_scaling_factor


@slow
class StableDiffusion21PipelineSingleFileSlowTests(unittest.TestCase, SDSingleFileTesterMixin):
    pipeline_class = StableDiffusionPipeline
    ckpt_path = "https://huggingface.co/stabilityai/stable-diffusion-2-1/blob/main/v2-1_768-ema-pruned.safetensors"
    original_config = "https://raw.githubusercontent.com/Stability-AI/stablediffusion/main/configs/stable-diffusion/v2-inference-v.yaml"
    repo_id = "stabilityai/stable-diffusion-2-1"

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
        inputs = {
            "prompt": "a fantasy landscape, concept art, high resolution",
            "generator": generator,
            "num_inference_steps": 2,
            "strength": 0.75,
            "guidance_scale": 7.5,
            "output_type": "np",
        }
        return inputs

    def test_single_file_format_inference_is_same_as_pretrained(self):
        super().test_single_file_format_inference_is_same_as_pretrained(expected_max_diff=1e-3)
