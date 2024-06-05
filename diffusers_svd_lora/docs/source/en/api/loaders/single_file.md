<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Loading Pipelines and Models via `from_single_file`

The `from_single_file` method allows you to load supported pipelines using a single checkpoint file as opposed to Diffusers' multiple folders format. This is useful if you are working with Stable Diffusion Web UI's (such as A1111) that rely on a single file format to distribute all the components of a model.

The `from_single_file` method also supports loading models in their originally distributed format. This means that supported models that have been finetuned with other services can be loaded directly into Diffusers model objects and pipelines.

## Pipelines that currently support `from_single_file` loading

- [`StableDiffusionPipeline`]
- [`StableDiffusionImg2ImgPipeline`]
- [`StableDiffusionInpaintPipeline`]
- [`StableDiffusionControlNetPipeline`]
- [`StableDiffusionControlNetImg2ImgPipeline`]
- [`StableDiffusionControlNetInpaintPipeline`]
- [`StableDiffusionUpscalePipeline`]
- [`StableDiffusionXLPipeline`]
- [`StableDiffusionXLImg2ImgPipeline`]
- [`StableDiffusionXLInpaintPipeline`]
- [`StableDiffusionXLInstructPix2PixPipeline`]
- [`StableDiffusionXLControlNetPipeline`]
- [`StableDiffusionXLKDiffusionPipeline`]
- [`LatentConsistencyModelPipeline`]
- [`LatentConsistencyModelImg2ImgPipeline`]
- [`StableDiffusionControlNetXSPipeline`]
- [`StableDiffusionXLControlNetXSPipeline`]
- [`LEditsPPPipelineStableDiffusion`]
- [`LEditsPPPipelineStableDiffusionXL`]
- [`PIAPipeline`]

## Models that currently support `from_single_file` loading

- [`UNet2DConditionModel`]
- [`StableCascadeUNet`]
- [`AutoencoderKL`]
- [`ControlNetModel`]

## Usage Examples

## Loading a Pipeline using `from_single_file`

```python
from diffusers import StableDiffusionXLPipeline

ckpt_path = "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/sd_xl_base_1.0_0.9vae.safetensors"
pipe = StableDiffusionXLPipeline.from_single_file(ckpt_path)
```

## Setting components in a Pipeline using `from_single_file`

Set components of a pipeline by passing them directly to the `from_single_file` method. For example, here we are swapping out the pipeline's default scheduler with the `DDIMScheduler`.

```python
from diffusers import StableDiffusionXLPipeline, DDIMScheduler

ckpt_path = "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/sd_xl_base_1.0_0.9vae.safetensors"

scheduler = DDIMScheduler()
pipe = StableDiffusionXLPipeline.from_single_file(ckpt_path, scheduler=scheduler)

```

Here we are passing in a ControlNet model to the `StableDiffusionControlNetPipeline`.

```python
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel

ckpt_path = "https://huggingface.co/runwayml/stable-diffusion-v1-5/blob/main/v1-5-pruned-emaonly.safetensors"

controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_canny")
pipe = StableDiffusionControlNetPipeline.from_single_file(ckpt_path, controlnet=controlnet)

```

## Loading a Model using `from_single_file`

```python
from diffusers import StableCascadeUNet

ckpt_path = "https://huggingface.co/stabilityai/stable-cascade/blob/main/stage_b_lite.safetensors"
model = StableCascadeUNet.from_single_file(ckpt_path)

```

## Using a Diffusers model repository to configure single file loading

Under the hood, `from_single_file` will try to automatically determine a model repository to use to configure the components of a pipeline. You can also explicitly set the model repository to configure the pipeline with the `config` argument.

```python
from diffusers import StableDiffusionXLPipeline

ckpt_path = "https://huggingface.co/segmind/SSD-1B/blob/main/SSD-1B.safetensors"
repo_id = "segmind/SSD-1B"

pipe = StableDiffusionXLPipeline.from_single_file(ckpt_path, config=repo_id)

```

In the example above, since we explicitly passed `repo_id="segmind/SSD-1B"` to the `config` argument, it will use this [configuration file](https://huggingface.co/segmind/SSD-1B/blob/main/unet/config.json) from the `unet` subfolder in `"segmind/SSD-1B"` to configure the `unet` component of the pipeline; Similarly, it will use the `config.json` file from `vae` subfolder to configure the `vae` model, `config.json` file from `text_encoder` folder to configure `text_encoder` and so on.

<Tip>

Most of the time you do not need to explicitly set a `config` argument. `from_single_file` will automatically map the checkpoint to the appropriate model repository. However, this option can be useful in cases where model components in the checkpoint might have been changed from what was originally distributed, or in cases where a checkpoint file might not have the necessary metadata to correctly determine the configuration to use for the pipeline.

</Tip>

## Override configuration options when using single file loading

Override the default model or pipeline configuration options by providing the relevant arguments directly to the `from_single_file` method. Any argument supported by the model or pipeline class can be configured in this way:

### Setting a pipeline configuration option

```python
from diffusers import StableDiffusionXLInstructPix2PixPipeline

ckpt_path = "https://huggingface.co/stabilityai/cosxl/blob/main/cosxl_edit.safetensors"
pipe = StableDiffusionXLInstructPix2PixPipeline.from_single_file(ckpt_path, config="diffusers/sdxl-instructpix2pix-768", is_cosxl_edit=True)

```

### Setting a model configuration option

```python
from diffusers import UNet2DConditionModel

ckpt_path = "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/sd_xl_base_1.0_0.9vae.safetensors"
model = UNet2DConditionModel.from_single_file(ckpt_path, upcast_attention=True)

```

<Tip>

To learn more about how to load single file weights, see the [Load different Stable Diffusion formats](../../using-diffusers/other-formats) loading guide.

</Tip>

## Working with local files

As of `diffusers>=0.28.0` the `from_single_file` method will attempt to configure a pipeline or model by first inferring the model type from the keys in the checkpoint file. This inferred model type is then used to determine the appropriate model repository on the Hugging Face Hub to configure the model or pipeline.

For example, any single file checkpoint based on the Stable Diffusion XL base model will use the [`stabilityai/stable-diffusion-xl-base-1.0`](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) model repository to configure the pipeline.

If you are working in an environment with restricted internet access, it is recommended that you download the config files and checkpoints for the model to your preferred directory and pass the local paths to the `pretrained_model_link_or_path` and `config` arguments of the `from_single_file` method.

```python
from huggingface_hub import hf_hub_download, snapshot_download

my_local_checkpoint_path = hf_hub_download(
    repo_id="segmind/SSD-1B",
    filename="SSD-1B.safetensors"
)

my_local_config_path = snapshot_download(
    repo_id="segmind/SSD-1B",
    allowed_patterns=["*.json", "**/*.json", "*.txt", "**/*.txt"]
)

pipe = StableDiffusionXLPipeline.from_single_file(my_local_checkpoint_path, config=my_local_config_path, local_files_only=True)

```

By default this will download the checkpoints and config files to the [Hugging Face Hub cache directory](https://huggingface.co/docs/huggingface_hub/en/guides/manage-cache). You can also specify a local directory to download the files to by passing the `local_dir` argument to the `hf_hub_download` and `snapshot_download` functions.

```python
from huggingface_hub import hf_hub_download, snapshot_download

my_local_checkpoint_path = hf_hub_download(
    repo_id="segmind/SSD-1B",
    filename="SSD-1B.safetensors"
    local_dir="my_local_checkpoints"
)

my_local_config_path = snapshot_download(
    repo_id="segmind/SSD-1B",
    allowed_patterns=["*.json", "**/*.json", "*.txt", "**/*.txt"]
    local_dir="my_local_config"
)

pipe = StableDiffusionXLPipeline.from_single_file(my_local_checkpoint_path, config=my_local_config_path, local_files_only=True)

```

## Working with local files on file systems that do not support symlinking

By default the `from_single_file` method relies on the `huggingface_hub` caching mechanism to fetch and store checkpoints and config files for models and pipelines. If you are working with a file system that does not support symlinking, it is recommended that you first download the checkpoint file to a local directory and disable symlinking by passing the `local_dir_use_symlink=False` argument to the `hf_hub_download` and `snapshot_download` functions.

```python
from huggingface_hub import hf_hub_download, snapshot_download

my_local_checkpoint_path = hf_hub_download(
    repo_id="segmind/SSD-1B",
    filename="SSD-1B.safetensors"
    local_dir="my_local_checkpoints",
    local_dir_use_symlinks=False
)
print("My local checkpoint: ", my_local_checkpoint_path)

my_local_config_path = snapshot_download(
    repo_id="segmind/SSD-1B",
    allowed_patterns=["*.json", "**/*.json", "*.txt", "**/*.txt"]
    local_dir_use_symlinks=False,
)
print("My local config: ", my_local_config_path)

```

Then pass the local paths to the `pretrained_model_link_or_path` and `config` arguments of the `from_single_file` method.

```python
pipe = StableDiffusionXLPipeline.from_single_file(my_local_checkpoint_path, config=my_local_config_path, local_files_only=True)

```

<Tip>

As of `huggingface_hub>=0.23.0` the `local_dir_use_symlinks` argument isn't necessary for the `hf_hub_download` and `snapshot_download` functions.

</Tip>

## Using the original configuration file of a model

If you would like to configure the model components in a pipeline using the orignal YAML configuration file, you can pass a local path or url to the original configuration file via the `original_config` argument.

```python
from diffusers import StableDiffusionXLPipeline

ckpt_path = "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/sd_xl_base_1.0_0.9vae.safetensors"
repo_id = "stabilityai/stable-diffusion-xl-base-1.0"
original_config = "https://raw.githubusercontent.com/Stability-AI/generative-models/main/configs/inference/sd_xl_base.yaml"

pipe = StableDiffusionXLPipeline.from_single_file(ckpt_path, original_config=original_config)
```

<Tip>

When using `original_config` with `local_files_only=True`, Diffusers will attempt to infer the components of the pipeline based on the type signatures of pipeline class, rather than attempting to fetch the configuration files from a model repository on the Hugging Face Hub. This is to prevent backward breaking changes in existing code that might not be able to connect to the internet to fetch the necessary configuration files.

This is not as reliable as providing a path to a local model repository using the `config` argument and might lead to errors when configuring the pipeline. To avoid this, please run the pipeline with `local_files_only=False` once to download the appropriate pipeline configuration files to the local cache.

</Tip>


## FromSingleFileMixin

[[autodoc]] loaders.single_file.FromSingleFileMixin

## FromOriginalModelMixin

[[autodoc]] loaders.single_file_model.FromOriginalModelMixin
