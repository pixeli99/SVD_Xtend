# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import inspect
import os
from collections import defaultdict
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import safetensors
import torch
import torch.nn.functional as F
from huggingface_hub.utils import validate_hf_hub_args
from torch import nn

from ..models.embeddings import (
    ImageProjection,
    IPAdapterFaceIDImageProjection,
    IPAdapterFaceIDPlusImageProjection,
    IPAdapterFullImageProjection,
    IPAdapterPlusImageProjection,
    MultiIPAdapterImageProjection,
)
from ..models.modeling_utils import _LOW_CPU_MEM_USAGE_DEFAULT, load_model_dict_into_meta, load_state_dict
from ..utils import (
    USE_PEFT_BACKEND,
    _get_model_file,
    delete_adapter_layers,
    is_accelerate_available,
    is_torch_version,
    logging,
    set_adapter_layers,
    set_weights_and_activate_adapters,
)
from .unet_loader_utils import _maybe_expand_lora_scales
from .utils import AttnProcsLayers


if is_accelerate_available():
    from accelerate import init_empty_weights
    from accelerate.hooks import AlignDevicesHook, CpuOffload, remove_hook_from_module

logger = logging.get_logger(__name__)


TEXT_ENCODER_NAME = "text_encoder"
UNET_NAME = "unet"

LORA_WEIGHT_NAME = "pytorch_lora_weights.bin"
LORA_WEIGHT_NAME_SAFE = "pytorch_lora_weights.safetensors"

CUSTOM_DIFFUSION_WEIGHT_NAME = "pytorch_custom_diffusion_weights.bin"
CUSTOM_DIFFUSION_WEIGHT_NAME_SAFE = "pytorch_custom_diffusion_weights.safetensors"


class UNet2DConditionLoadersMixin:
    """
    Load LoRA layers into a [`UNet2DCondtionModel`].
    """

    text_encoder_name = TEXT_ENCODER_NAME
    unet_name = UNET_NAME

    @validate_hf_hub_args
    def load_attn_procs(self, pretrained_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]], **kwargs):
        r"""
        Load pretrained attention processor layers into [`UNet2DConditionModel`]. Attention processor layers have to be
        defined in
        [`attention_processor.py`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py)
        and be a `torch.nn.Module` class.

        Parameters:
            pretrained_model_name_or_path_or_dict (`str` or `os.PathLike` or `dict`):
                Can be either:

                    - A string, the model id (for example `google/ddpm-celebahq-256`) of a pretrained model hosted on
                      the Hub.
                    - A path to a directory (for example `./my_model_directory`) containing the model weights saved
                      with [`ModelMixin.save_pretrained`].
                    - A [torch state
                      dict](https://pytorch.org/tutorials/beginner/saving_loading_models.html#what-is-a-state-dict).

            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory where a downloaded pretrained model configuration is cached if the standard cache
                is not used.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            resume_download:
                Deprecated and ignored. All downloads are now resumed by default when possible. Will be removed in v1
                of Diffusers.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, for example, `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            local_files_only (`bool`, *optional*, defaults to `False`):
                Whether to only load local model weights and configuration files or not. If set to `True`, the model
                won't be downloaded from the Hub.
            token (`str` or *bool*, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, the token generated from
                `diffusers-cli login` (stored in `~/.huggingface`) is used.
            low_cpu_mem_usage (`bool`, *optional*, defaults to `True` if torch version >= 1.9.0 else `False`):
                Speed up model loading only loading the pretrained weights and not initializing the weights. This also
                tries to not use more than 1x model size in CPU memory (including peak memory) while loading the model.
                Only supported for PyTorch >= 1.9.0. If you are using an older version of PyTorch, setting this
                argument to `True` will raise an error.
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, a commit id, or any identifier
                allowed by Git.
            subfolder (`str`, *optional*, defaults to `""`):
                The subfolder location of a model file within a larger model repository on the Hub or locally.
            mirror (`str`, *optional*):
                Mirror source to resolve accessibility issues if you’re downloading a model in China. We do not
                guarantee the timeliness or safety of the source, and you should refer to the mirror site for more
                information.

        Example:

        ```py
        from diffusers import AutoPipelineForText2Image
        import torch

        pipeline = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
        ).to("cuda")
        pipeline.unet.load_attn_procs(
            "jbilcke-hf/sdxl-cinematic-1", weight_name="pytorch_lora_weights.safetensors", adapter_name="cinematic"
        )
        ```
        """
        from ..models.attention_processor import CustomDiffusionAttnProcessor
        from ..models.lora import LoRACompatibleConv, LoRACompatibleLinear, LoRAConv2dLayer, LoRALinearLayer

        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", None)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", None)
        token = kwargs.pop("token", None)
        revision = kwargs.pop("revision", None)
        subfolder = kwargs.pop("subfolder", None)
        weight_name = kwargs.pop("weight_name", None)
        use_safetensors = kwargs.pop("use_safetensors", None)
        low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", _LOW_CPU_MEM_USAGE_DEFAULT)
        # This value has the same meaning as the `--network_alpha` option in the kohya-ss trainer script.
        # See https://github.com/darkstorm2150/sd-scripts/blob/main/docs/train_network_README-en.md#execute-learning
        network_alphas = kwargs.pop("network_alphas", None)

        _pipeline = kwargs.pop("_pipeline", None)

        is_network_alphas_none = network_alphas is None

        allow_pickle = False

        if use_safetensors is None:
            use_safetensors = True
            allow_pickle = True

        user_agent = {
            "file_type": "attn_procs_weights",
            "framework": "pytorch",
        }

        model_file = None
        if not isinstance(pretrained_model_name_or_path_or_dict, dict):
            # Let's first try to load .safetensors weights
            if (use_safetensors and weight_name is None) or (
                weight_name is not None and weight_name.endswith(".safetensors")
            ):
                try:
                    model_file = _get_model_file(
                        pretrained_model_name_or_path_or_dict,
                        weights_name=weight_name or LORA_WEIGHT_NAME_SAFE,
                        cache_dir=cache_dir,
                        force_download=force_download,
                        resume_download=resume_download,
                        proxies=proxies,
                        local_files_only=local_files_only,
                        token=token,
                        revision=revision,
                        subfolder=subfolder,
                        user_agent=user_agent,
                    )
                    state_dict = safetensors.torch.load_file(model_file, device="cpu")
                except IOError as e:
                    if not allow_pickle:
                        raise e
                    # try loading non-safetensors weights
                    pass
            if model_file is None:
                model_file = _get_model_file(
                    pretrained_model_name_or_path_or_dict,
                    weights_name=weight_name or LORA_WEIGHT_NAME,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    resume_download=resume_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    token=token,
                    revision=revision,
                    subfolder=subfolder,
                    user_agent=user_agent,
                )
                state_dict = load_state_dict(model_file)
        else:
            state_dict = pretrained_model_name_or_path_or_dict

        # fill attn processors
        lora_layers_list = []

        is_lora = all(("lora" in k or k.endswith(".alpha")) for k in state_dict.keys()) and not USE_PEFT_BACKEND
        is_custom_diffusion = any("custom_diffusion" in k for k in state_dict.keys())

        if is_lora:
            # correct keys
            state_dict, network_alphas = self.convert_state_dict_legacy_attn_format(state_dict, network_alphas)

            if network_alphas is not None:
                network_alphas_keys = list(network_alphas.keys())
                used_network_alphas_keys = set()

            lora_grouped_dict = defaultdict(dict)
            mapped_network_alphas = {}

            all_keys = list(state_dict.keys())
            for key in all_keys:
                value = state_dict.pop(key)
                attn_processor_key, sub_key = ".".join(key.split(".")[:-3]), ".".join(key.split(".")[-3:])
                lora_grouped_dict[attn_processor_key][sub_key] = value

                # Create another `mapped_network_alphas` dictionary so that we can properly map them.
                if network_alphas is not None:
                    for k in network_alphas_keys:
                        if k.replace(".alpha", "") in key:
                            mapped_network_alphas.update({attn_processor_key: network_alphas.get(k)})
                            used_network_alphas_keys.add(k)

            if not is_network_alphas_none:
                if len(set(network_alphas_keys) - used_network_alphas_keys) > 0:
                    raise ValueError(
                        f"The `network_alphas` has to be empty at this point but has the following keys \n\n {', '.join(network_alphas.keys())}"
                    )

            if len(state_dict) > 0:
                raise ValueError(
                    f"The `state_dict` has to be empty at this point but has the following keys \n\n {', '.join(state_dict.keys())}"
                )

            for key, value_dict in lora_grouped_dict.items():
                attn_processor = self
                for sub_key in key.split("."):
                    attn_processor = getattr(attn_processor, sub_key)

                # Process non-attention layers, which don't have to_{k,v,q,out_proj}_lora layers
                # or add_{k,v,q,out_proj}_proj_lora layers.
                rank = value_dict["lora.down.weight"].shape[0]

                if isinstance(attn_processor, LoRACompatibleConv):
                    in_features = attn_processor.in_channels
                    out_features = attn_processor.out_channels
                    kernel_size = attn_processor.kernel_size

                    ctx = init_empty_weights if low_cpu_mem_usage else nullcontext
                    with ctx():
                        lora = LoRAConv2dLayer(
                            in_features=in_features,
                            out_features=out_features,
                            rank=rank,
                            kernel_size=kernel_size,
                            stride=attn_processor.stride,
                            padding=attn_processor.padding,
                            network_alpha=mapped_network_alphas.get(key),
                        )
                elif isinstance(attn_processor, LoRACompatibleLinear):
                    ctx = init_empty_weights if low_cpu_mem_usage else nullcontext
                    with ctx():
                        lora = LoRALinearLayer(
                            attn_processor.in_features,
                            attn_processor.out_features,
                            rank,
                            mapped_network_alphas.get(key),
                        )
                else:
                    raise ValueError(f"Module {key} is not a LoRACompatibleConv or LoRACompatibleLinear module.")

                value_dict = {k.replace("lora.", ""): v for k, v in value_dict.items()}
                lora_layers_list.append((attn_processor, lora))

                if low_cpu_mem_usage:
                    device = next(iter(value_dict.values())).device
                    dtype = next(iter(value_dict.values())).dtype
                    load_model_dict_into_meta(lora, value_dict, device=device, dtype=dtype)
                else:
                    lora.load_state_dict(value_dict)

        elif is_custom_diffusion:
            attn_processors = {}
            custom_diffusion_grouped_dict = defaultdict(dict)
            for key, value in state_dict.items():
                if len(value) == 0:
                    custom_diffusion_grouped_dict[key] = {}
                else:
                    if "to_out" in key:
                        attn_processor_key, sub_key = ".".join(key.split(".")[:-3]), ".".join(key.split(".")[-3:])
                    else:
                        attn_processor_key, sub_key = ".".join(key.split(".")[:-2]), ".".join(key.split(".")[-2:])
                    custom_diffusion_grouped_dict[attn_processor_key][sub_key] = value

            for key, value_dict in custom_diffusion_grouped_dict.items():
                if len(value_dict) == 0:
                    attn_processors[key] = CustomDiffusionAttnProcessor(
                        train_kv=False, train_q_out=False, hidden_size=None, cross_attention_dim=None
                    )
                else:
                    cross_attention_dim = value_dict["to_k_custom_diffusion.weight"].shape[1]
                    hidden_size = value_dict["to_k_custom_diffusion.weight"].shape[0]
                    train_q_out = True if "to_q_custom_diffusion.weight" in value_dict else False
                    attn_processors[key] = CustomDiffusionAttnProcessor(
                        train_kv=True,
                        train_q_out=train_q_out,
                        hidden_size=hidden_size,
                        cross_attention_dim=cross_attention_dim,
                    )
                    attn_processors[key].load_state_dict(value_dict)
        elif USE_PEFT_BACKEND:
            # In that case we have nothing to do as loading the adapter weights is already handled above by `set_peft_model_state_dict`
            # on the Unet
            pass
        else:
            raise ValueError(
                f"{model_file} does not seem to be in the correct format expected by LoRA or Custom Diffusion training."
            )

        # <Unsafe code
        # We can be sure that the following works as it just sets attention processors, lora layers and puts all in the same dtype
        # Now we remove any existing hooks to
        is_model_cpu_offload = False
        is_sequential_cpu_offload = False

        # For PEFT backend the Unet is already offloaded at this stage as it is handled inside `load_lora_weights_into_unet`
        if not USE_PEFT_BACKEND:
            if _pipeline is not None:
                for _, component in _pipeline.components.items():
                    if isinstance(component, nn.Module) and hasattr(component, "_hf_hook"):
                        is_model_cpu_offload = isinstance(getattr(component, "_hf_hook"), CpuOffload)
                        is_sequential_cpu_offload = (
                            isinstance(getattr(component, "_hf_hook"), AlignDevicesHook)
                            or hasattr(component._hf_hook, "hooks")
                            and isinstance(component._hf_hook.hooks[0], AlignDevicesHook)
                        )

                        logger.info(
                            "Accelerate hooks detected. Since you have called `load_lora_weights()`, the previous hooks will be first removed. Then the LoRA parameters will be loaded and the hooks will be applied again."
                        )
                        remove_hook_from_module(component, recurse=is_sequential_cpu_offload)

            # only custom diffusion needs to set attn processors
            if is_custom_diffusion:
                self.set_attn_processor(attn_processors)

            # set lora layers
            for target_module, lora_layer in lora_layers_list:
                target_module.set_lora_layer(lora_layer)

            self.to(dtype=self.dtype, device=self.device)

            # Offload back.
            if is_model_cpu_offload:
                _pipeline.enable_model_cpu_offload()
            elif is_sequential_cpu_offload:
                _pipeline.enable_sequential_cpu_offload()
            # Unsafe code />

    def convert_state_dict_legacy_attn_format(self, state_dict, network_alphas):
        is_new_lora_format = all(
            key.startswith(self.unet_name) or key.startswith(self.text_encoder_name) for key in state_dict.keys()
        )
        if is_new_lora_format:
            # Strip the `"unet"` prefix.
            is_text_encoder_present = any(key.startswith(self.text_encoder_name) for key in state_dict.keys())
            if is_text_encoder_present:
                warn_message = "The state_dict contains LoRA params corresponding to the text encoder which are not being used here. To use both UNet and text encoder related LoRA params, use [`pipe.load_lora_weights()`](https://huggingface.co/docs/diffusers/main/en/api/loaders#diffusers.loaders.LoraLoaderMixin.load_lora_weights)."
                logger.warning(warn_message)
            unet_keys = [k for k in state_dict.keys() if k.startswith(self.unet_name)]
            state_dict = {k.replace(f"{self.unet_name}.", ""): v for k, v in state_dict.items() if k in unet_keys}

        # change processor format to 'pure' LoRACompatibleLinear format
        if any("processor" in k.split(".") for k in state_dict.keys()):

            def format_to_lora_compatible(key):
                if "processor" not in key.split("."):
                    return key
                return key.replace(".processor", "").replace("to_out_lora", "to_out.0.lora").replace("_lora", ".lora")

            state_dict = {format_to_lora_compatible(k): v for k, v in state_dict.items()}

            if network_alphas is not None:
                network_alphas = {format_to_lora_compatible(k): v for k, v in network_alphas.items()}
        return state_dict, network_alphas

    def save_attn_procs(
        self,
        save_directory: Union[str, os.PathLike],
        is_main_process: bool = True,
        weight_name: str = None,
        save_function: Callable = None,
        safe_serialization: bool = True,
        **kwargs,
    ):
        r"""
        Save attention processor layers to a directory so that it can be reloaded with the
        [`~loaders.UNet2DConditionLoadersMixin.load_attn_procs`] method.

        Arguments:
            save_directory (`str` or `os.PathLike`):
                Directory to save an attention processor to (will be created if it doesn't exist).
            is_main_process (`bool`, *optional*, defaults to `True`):
                Whether the process calling this is the main process or not. Useful during distributed training and you
                need to call this function on all processes. In this case, set `is_main_process=True` only on the main
                process to avoid race conditions.
            save_function (`Callable`):
                The function to use to save the state dictionary. Useful during distributed training when you need to
                replace `torch.save` with another method. Can be configured with the environment variable
                `DIFFUSERS_SAVE_MODE`.
            safe_serialization (`bool`, *optional*, defaults to `True`):
                Whether to save the model using `safetensors` or with `pickle`.

        Example:

        ```py
        import torch
        from diffusers import DiffusionPipeline

        pipeline = DiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            torch_dtype=torch.float16,
        ).to("cuda")
        pipeline.unet.load_attn_procs("path-to-save-model", weight_name="pytorch_custom_diffusion_weights.bin")
        pipeline.unet.save_attn_procs("path-to-save-model", weight_name="pytorch_custom_diffusion_weights.bin")
        ```
        """
        from ..models.attention_processor import (
            CustomDiffusionAttnProcessor,
            CustomDiffusionAttnProcessor2_0,
            CustomDiffusionXFormersAttnProcessor,
        )

        if os.path.isfile(save_directory):
            logger.error(f"Provided path ({save_directory}) should be a directory, not a file")
            return

        if save_function is None:
            if safe_serialization:

                def save_function(weights, filename):
                    return safetensors.torch.save_file(weights, filename, metadata={"format": "pt"})

            else:
                save_function = torch.save

        os.makedirs(save_directory, exist_ok=True)

        is_custom_diffusion = any(
            isinstance(
                x,
                (CustomDiffusionAttnProcessor, CustomDiffusionAttnProcessor2_0, CustomDiffusionXFormersAttnProcessor),
            )
            for (_, x) in self.attn_processors.items()
        )
        if is_custom_diffusion:
            model_to_save = AttnProcsLayers(
                {
                    y: x
                    for (y, x) in self.attn_processors.items()
                    if isinstance(
                        x,
                        (
                            CustomDiffusionAttnProcessor,
                            CustomDiffusionAttnProcessor2_0,
                            CustomDiffusionXFormersAttnProcessor,
                        ),
                    )
                }
            )
            state_dict = model_to_save.state_dict()
            for name, attn in self.attn_processors.items():
                if len(attn.state_dict()) == 0:
                    state_dict[name] = {}
        else:
            model_to_save = AttnProcsLayers(self.attn_processors)
            state_dict = model_to_save.state_dict()

        if weight_name is None:
            if safe_serialization:
                weight_name = CUSTOM_DIFFUSION_WEIGHT_NAME_SAFE if is_custom_diffusion else LORA_WEIGHT_NAME_SAFE
            else:
                weight_name = CUSTOM_DIFFUSION_WEIGHT_NAME if is_custom_diffusion else LORA_WEIGHT_NAME

        # Save the model
        save_path = Path(save_directory, weight_name).as_posix()
        save_function(state_dict, save_path)
        logger.info(f"Model weights saved in {save_path}")

    def fuse_lora(self, lora_scale=1.0, safe_fusing=False, adapter_names=None):
        self.lora_scale = lora_scale
        self._safe_fusing = safe_fusing
        self.apply(partial(self._fuse_lora_apply, adapter_names=adapter_names))

    def _fuse_lora_apply(self, module, adapter_names=None):
        if not USE_PEFT_BACKEND:
            if hasattr(module, "_fuse_lora"):
                module._fuse_lora(self.lora_scale, self._safe_fusing)

            if adapter_names is not None:
                raise ValueError(
                    "The `adapter_names` argument is not supported in your environment. Please switch"
                    " to PEFT backend to use this argument by installing latest PEFT and transformers."
                    " `pip install -U peft transformers`"
                )
        else:
            from peft.tuners.tuners_utils import BaseTunerLayer

            merge_kwargs = {"safe_merge": self._safe_fusing}

            if isinstance(module, BaseTunerLayer):
                if self.lora_scale != 1.0:
                    module.scale_layer(self.lora_scale)

                # For BC with prevous PEFT versions, we need to check the signature
                # of the `merge` method to see if it supports the `adapter_names` argument.
                supported_merge_kwargs = list(inspect.signature(module.merge).parameters)
                if "adapter_names" in supported_merge_kwargs:
                    merge_kwargs["adapter_names"] = adapter_names
                elif "adapter_names" not in supported_merge_kwargs and adapter_names is not None:
                    raise ValueError(
                        "The `adapter_names` argument is not supported with your PEFT version. Please upgrade"
                        " to the latest version of PEFT. `pip install -U peft`"
                    )

                module.merge(**merge_kwargs)

    def unfuse_lora(self):
        self.apply(self._unfuse_lora_apply)

    def _unfuse_lora_apply(self, module):
        if not USE_PEFT_BACKEND:
            if hasattr(module, "_unfuse_lora"):
                module._unfuse_lora()
        else:
            from peft.tuners.tuners_utils import BaseTunerLayer

            if isinstance(module, BaseTunerLayer):
                module.unmerge()

    def set_adapters(
        self,
        adapter_names: Union[List[str], str],
        weights: Optional[Union[float, Dict, List[float], List[Dict], List[None]]] = None,
    ):
        """
        Set the currently active adapters for use in the UNet.

        Args:
            adapter_names (`List[str]` or `str`):
                The names of the adapters to use.
            adapter_weights (`Union[List[float], float]`, *optional*):
                The adapter(s) weights to use with the UNet. If `None`, the weights are set to `1.0` for all the
                adapters.

        Example:

        ```py
        from diffusers import AutoPipelineForText2Image
        import torch

        pipeline = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
        ).to("cuda")
        pipeline.load_lora_weights(
            "jbilcke-hf/sdxl-cinematic-1", weight_name="pytorch_lora_weights.safetensors", adapter_name="cinematic"
        )
        pipeline.load_lora_weights("nerijs/pixel-art-xl", weight_name="pixel-art-xl.safetensors", adapter_name="pixel")
        pipeline.set_adapters(["cinematic", "pixel"], adapter_weights=[0.5, 0.5])
        ```
        """
        if not USE_PEFT_BACKEND:
            raise ValueError("PEFT backend is required for `set_adapters()`.")

        adapter_names = [adapter_names] if isinstance(adapter_names, str) else adapter_names

        # Expand weights into a list, one entry per adapter
        # examples for e.g. 2 adapters:  [{...}, 7] -> [7,7] ; None -> [None, None]
        if not isinstance(weights, list):
            weights = [weights] * len(adapter_names)

        if len(adapter_names) != len(weights):
            raise ValueError(
                f"Length of adapter names {len(adapter_names)} is not equal to the length of their weights {len(weights)}."
            )

        # Set None values to default of 1.0
        # e.g. [{...}, 7] -> [{...}, 7] ; [None, None] -> [1.0, 1.0]
        weights = [w if w is not None else 1.0 for w in weights]

        # e.g. [{...}, 7] -> [{expanded dict...}, 7]
        weights = _maybe_expand_lora_scales(self, weights)

        set_weights_and_activate_adapters(self, adapter_names, weights)

    def disable_lora(self):
        """
        Disable the UNet's active LoRA layers.

        Example:

        ```py
        from diffusers import AutoPipelineForText2Image
        import torch

        pipeline = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
        ).to("cuda")
        pipeline.load_lora_weights(
            "jbilcke-hf/sdxl-cinematic-1", weight_name="pytorch_lora_weights.safetensors", adapter_name="cinematic"
        )
        pipeline.disable_lora()
        ```
        """
        if not USE_PEFT_BACKEND:
            raise ValueError("PEFT backend is required for this method.")
        set_adapter_layers(self, enabled=False)

    def enable_lora(self):
        """
        Enable the UNet's active LoRA layers.

        Example:

        ```py
        from diffusers import AutoPipelineForText2Image
        import torch

        pipeline = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
        ).to("cuda")
        pipeline.load_lora_weights(
            "jbilcke-hf/sdxl-cinematic-1", weight_name="pytorch_lora_weights.safetensors", adapter_name="cinematic"
        )
        pipeline.enable_lora()
        ```
        """
        if not USE_PEFT_BACKEND:
            raise ValueError("PEFT backend is required for this method.")
        set_adapter_layers(self, enabled=True)

    def delete_adapters(self, adapter_names: Union[List[str], str]):
        """
        Delete an adapter's LoRA layers from the UNet.

        Args:
            adapter_names (`Union[List[str], str]`):
                The names (single string or list of strings) of the adapter to delete.

        Example:

        ```py
        from diffusers import AutoPipelineForText2Image
        import torch

        pipeline = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
        ).to("cuda")
        pipeline.load_lora_weights(
            "jbilcke-hf/sdxl-cinematic-1", weight_name="pytorch_lora_weights.safetensors", adapter_names="cinematic"
        )
        pipeline.delete_adapters("cinematic")
        ```
        """
        if not USE_PEFT_BACKEND:
            raise ValueError("PEFT backend is required for this method.")

        if isinstance(adapter_names, str):
            adapter_names = [adapter_names]

        for adapter_name in adapter_names:
            delete_adapter_layers(self, adapter_name)

            # Pop also the corresponding adapter from the config
            if hasattr(self, "peft_config"):
                self.peft_config.pop(adapter_name, None)

    def _convert_ip_adapter_image_proj_to_diffusers(self, state_dict, low_cpu_mem_usage=False):
        if low_cpu_mem_usage:
            if is_accelerate_available():
                from accelerate import init_empty_weights

            else:
                low_cpu_mem_usage = False
                logger.warning(
                    "Cannot initialize model with low cpu memory usage because `accelerate` was not found in the"
                    " environment. Defaulting to `low_cpu_mem_usage=False`. It is strongly recommended to install"
                    " `accelerate` for faster and less memory-intense model loading. You can do so with: \n```\npip"
                    " install accelerate\n```\n."
                )

        if low_cpu_mem_usage is True and not is_torch_version(">=", "1.9.0"):
            raise NotImplementedError(
                "Low memory initialization requires torch >= 1.9.0. Please either update your PyTorch version or set"
                " `low_cpu_mem_usage=False`."
            )

        updated_state_dict = {}
        image_projection = None
        init_context = init_empty_weights if low_cpu_mem_usage else nullcontext

        if "proj.weight" in state_dict:
            # IP-Adapter
            num_image_text_embeds = 4
            clip_embeddings_dim = state_dict["proj.weight"].shape[-1]
            cross_attention_dim = state_dict["proj.weight"].shape[0] // 4

            with init_context():
                image_projection = ImageProjection(
                    cross_attention_dim=cross_attention_dim,
                    image_embed_dim=clip_embeddings_dim,
                    num_image_text_embeds=num_image_text_embeds,
                )

            for key, value in state_dict.items():
                diffusers_name = key.replace("proj", "image_embeds")
                updated_state_dict[diffusers_name] = value

        elif "proj.3.weight" in state_dict:
            # IP-Adapter Full
            clip_embeddings_dim = state_dict["proj.0.weight"].shape[0]
            cross_attention_dim = state_dict["proj.3.weight"].shape[0]

            with init_context():
                image_projection = IPAdapterFullImageProjection(
                    cross_attention_dim=cross_attention_dim, image_embed_dim=clip_embeddings_dim
                )

            for key, value in state_dict.items():
                diffusers_name = key.replace("proj.0", "ff.net.0.proj")
                diffusers_name = diffusers_name.replace("proj.2", "ff.net.2")
                diffusers_name = diffusers_name.replace("proj.3", "norm")
                updated_state_dict[diffusers_name] = value

        elif "perceiver_resampler.proj_in.weight" in state_dict:
            # IP-Adapter Face ID Plus
            id_embeddings_dim = state_dict["proj.0.weight"].shape[1]
            embed_dims = state_dict["perceiver_resampler.proj_in.weight"].shape[0]
            hidden_dims = state_dict["perceiver_resampler.proj_in.weight"].shape[1]
            output_dims = state_dict["perceiver_resampler.proj_out.weight"].shape[0]
            heads = state_dict["perceiver_resampler.layers.0.0.to_q.weight"].shape[0] // 64

            with init_context():
                image_projection = IPAdapterFaceIDPlusImageProjection(
                    embed_dims=embed_dims,
                    output_dims=output_dims,
                    hidden_dims=hidden_dims,
                    heads=heads,
                    id_embeddings_dim=id_embeddings_dim,
                )

            for key, value in state_dict.items():
                diffusers_name = key.replace("perceiver_resampler.", "")
                diffusers_name = diffusers_name.replace("0.to", "attn.to")
                diffusers_name = diffusers_name.replace("0.1.0.", "0.ff.0.")
                diffusers_name = diffusers_name.replace("0.1.1.weight", "0.ff.1.net.0.proj.weight")
                diffusers_name = diffusers_name.replace("0.1.3.weight", "0.ff.1.net.2.weight")
                diffusers_name = diffusers_name.replace("1.1.0.", "1.ff.0.")
                diffusers_name = diffusers_name.replace("1.1.1.weight", "1.ff.1.net.0.proj.weight")
                diffusers_name = diffusers_name.replace("1.1.3.weight", "1.ff.1.net.2.weight")
                diffusers_name = diffusers_name.replace("2.1.0.", "2.ff.0.")
                diffusers_name = diffusers_name.replace("2.1.1.weight", "2.ff.1.net.0.proj.weight")
                diffusers_name = diffusers_name.replace("2.1.3.weight", "2.ff.1.net.2.weight")
                diffusers_name = diffusers_name.replace("3.1.0.", "3.ff.0.")
                diffusers_name = diffusers_name.replace("3.1.1.weight", "3.ff.1.net.0.proj.weight")
                diffusers_name = diffusers_name.replace("3.1.3.weight", "3.ff.1.net.2.weight")
                diffusers_name = diffusers_name.replace("layers.0.0", "layers.0.ln0")
                diffusers_name = diffusers_name.replace("layers.0.1", "layers.0.ln1")
                diffusers_name = diffusers_name.replace("layers.1.0", "layers.1.ln0")
                diffusers_name = diffusers_name.replace("layers.1.1", "layers.1.ln1")
                diffusers_name = diffusers_name.replace("layers.2.0", "layers.2.ln0")
                diffusers_name = diffusers_name.replace("layers.2.1", "layers.2.ln1")
                diffusers_name = diffusers_name.replace("layers.3.0", "layers.3.ln0")
                diffusers_name = diffusers_name.replace("layers.3.1", "layers.3.ln1")

                if "norm1" in diffusers_name:
                    updated_state_dict[diffusers_name.replace("0.norm1", "0")] = value
                elif "norm2" in diffusers_name:
                    updated_state_dict[diffusers_name.replace("0.norm2", "1")] = value
                elif "to_kv" in diffusers_name:
                    v_chunk = value.chunk(2, dim=0)
                    updated_state_dict[diffusers_name.replace("to_kv", "to_k")] = v_chunk[0]
                    updated_state_dict[diffusers_name.replace("to_kv", "to_v")] = v_chunk[1]
                elif "to_out" in diffusers_name:
                    updated_state_dict[diffusers_name.replace("to_out", "to_out.0")] = value
                elif "proj.0.weight" == diffusers_name:
                    updated_state_dict["proj.net.0.proj.weight"] = value
                elif "proj.0.bias" == diffusers_name:
                    updated_state_dict["proj.net.0.proj.bias"] = value
                elif "proj.2.weight" == diffusers_name:
                    updated_state_dict["proj.net.2.weight"] = value
                elif "proj.2.bias" == diffusers_name:
                    updated_state_dict["proj.net.2.bias"] = value
                else:
                    updated_state_dict[diffusers_name] = value

        elif "norm.weight" in state_dict:
            # IP-Adapter Face ID
            id_embeddings_dim_in = state_dict["proj.0.weight"].shape[1]
            id_embeddings_dim_out = state_dict["proj.0.weight"].shape[0]
            multiplier = id_embeddings_dim_out // id_embeddings_dim_in
            norm_layer = "norm.weight"
            cross_attention_dim = state_dict[norm_layer].shape[0]
            num_tokens = state_dict["proj.2.weight"].shape[0] // cross_attention_dim

            with init_context():
                image_projection = IPAdapterFaceIDImageProjection(
                    cross_attention_dim=cross_attention_dim,
                    image_embed_dim=id_embeddings_dim_in,
                    mult=multiplier,
                    num_tokens=num_tokens,
                )

            for key, value in state_dict.items():
                diffusers_name = key.replace("proj.0", "ff.net.0.proj")
                diffusers_name = diffusers_name.replace("proj.2", "ff.net.2")
                updated_state_dict[diffusers_name] = value

        else:
            # IP-Adapter Plus
            num_image_text_embeds = state_dict["latents"].shape[1]
            embed_dims = state_dict["proj_in.weight"].shape[1]
            output_dims = state_dict["proj_out.weight"].shape[0]
            hidden_dims = state_dict["latents"].shape[2]
            attn_key_present = any("attn" in k for k in state_dict)
            heads = (
                state_dict["layers.0.attn.to_q.weight"].shape[0] // 64
                if attn_key_present
                else state_dict["layers.0.0.to_q.weight"].shape[0] // 64
            )

            with init_context():
                image_projection = IPAdapterPlusImageProjection(
                    embed_dims=embed_dims,
                    output_dims=output_dims,
                    hidden_dims=hidden_dims,
                    heads=heads,
                    num_queries=num_image_text_embeds,
                )

            for key, value in state_dict.items():
                diffusers_name = key.replace("0.to", "2.to")

                diffusers_name = diffusers_name.replace("0.0.norm1", "0.ln0")
                diffusers_name = diffusers_name.replace("0.0.norm2", "0.ln1")
                diffusers_name = diffusers_name.replace("1.0.norm1", "1.ln0")
                diffusers_name = diffusers_name.replace("1.0.norm2", "1.ln1")
                diffusers_name = diffusers_name.replace("2.0.norm1", "2.ln0")
                diffusers_name = diffusers_name.replace("2.0.norm2", "2.ln1")
                diffusers_name = diffusers_name.replace("3.0.norm1", "3.ln0")
                diffusers_name = diffusers_name.replace("3.0.norm2", "3.ln1")

                if "to_kv" in diffusers_name:
                    parts = diffusers_name.split(".")
                    parts[2] = "attn"
                    diffusers_name = ".".join(parts)
                    v_chunk = value.chunk(2, dim=0)
                    updated_state_dict[diffusers_name.replace("to_kv", "to_k")] = v_chunk[0]
                    updated_state_dict[diffusers_name.replace("to_kv", "to_v")] = v_chunk[1]
                elif "to_q" in diffusers_name:
                    parts = diffusers_name.split(".")
                    parts[2] = "attn"
                    diffusers_name = ".".join(parts)
                    updated_state_dict[diffusers_name] = value
                elif "to_out" in diffusers_name:
                    parts = diffusers_name.split(".")
                    parts[2] = "attn"
                    diffusers_name = ".".join(parts)
                    updated_state_dict[diffusers_name.replace("to_out", "to_out.0")] = value
                else:
                    diffusers_name = diffusers_name.replace("0.1.0", "0.ff.0")
                    diffusers_name = diffusers_name.replace("0.1.1", "0.ff.1.net.0.proj")
                    diffusers_name = diffusers_name.replace("0.1.3", "0.ff.1.net.2")

                    diffusers_name = diffusers_name.replace("1.1.0", "1.ff.0")
                    diffusers_name = diffusers_name.replace("1.1.1", "1.ff.1.net.0.proj")
                    diffusers_name = diffusers_name.replace("1.1.3", "1.ff.1.net.2")

                    diffusers_name = diffusers_name.replace("2.1.0", "2.ff.0")
                    diffusers_name = diffusers_name.replace("2.1.1", "2.ff.1.net.0.proj")
                    diffusers_name = diffusers_name.replace("2.1.3", "2.ff.1.net.2")

                    diffusers_name = diffusers_name.replace("3.1.0", "3.ff.0")
                    diffusers_name = diffusers_name.replace("3.1.1", "3.ff.1.net.0.proj")
                    diffusers_name = diffusers_name.replace("3.1.3", "3.ff.1.net.2")
                    updated_state_dict[diffusers_name] = value

        if not low_cpu_mem_usage:
            image_projection.load_state_dict(updated_state_dict, strict=True)
        else:
            load_model_dict_into_meta(image_projection, updated_state_dict, device=self.device, dtype=self.dtype)

        return image_projection

    def _convert_ip_adapter_attn_to_diffusers(self, state_dicts, low_cpu_mem_usage=False):
        from ..models.attention_processor import (
            AttnProcessor,
            AttnProcessor2_0,
            IPAdapterAttnProcessor,
            IPAdapterAttnProcessor2_0,
        )

        if low_cpu_mem_usage:
            if is_accelerate_available():
                from accelerate import init_empty_weights

            else:
                low_cpu_mem_usage = False
                logger.warning(
                    "Cannot initialize model with low cpu memory usage because `accelerate` was not found in the"
                    " environment. Defaulting to `low_cpu_mem_usage=False`. It is strongly recommended to install"
                    " `accelerate` for faster and less memory-intense model loading. You can do so with: \n```\npip"
                    " install accelerate\n```\n."
                )

        if low_cpu_mem_usage is True and not is_torch_version(">=", "1.9.0"):
            raise NotImplementedError(
                "Low memory initialization requires torch >= 1.9.0. Please either update your PyTorch version or set"
                " `low_cpu_mem_usage=False`."
            )

        # set ip-adapter cross-attention processors & load state_dict
        attn_procs = {}
        key_id = 1
        init_context = init_empty_weights if low_cpu_mem_usage else nullcontext
        for name in self.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else self.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = self.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(self.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = self.config.block_out_channels[block_id]

            if cross_attention_dim is None or "motion_modules" in name:
                attn_processor_class = (
                    AttnProcessor2_0 if hasattr(F, "scaled_dot_product_attention") else AttnProcessor
                )
                attn_procs[name] = attn_processor_class()

            else:
                attn_processor_class = (
                    IPAdapterAttnProcessor2_0 if hasattr(F, "scaled_dot_product_attention") else IPAdapterAttnProcessor
                )
                num_image_text_embeds = []
                for state_dict in state_dicts:
                    if "proj.weight" in state_dict["image_proj"]:
                        # IP-Adapter
                        num_image_text_embeds += [4]
                    elif "proj.3.weight" in state_dict["image_proj"]:
                        # IP-Adapter Full Face
                        num_image_text_embeds += [257]  # 256 CLIP tokens + 1 CLS token
                    elif "perceiver_resampler.proj_in.weight" in state_dict["image_proj"]:
                        # IP-Adapter Face ID Plus
                        num_image_text_embeds += [4]
                    elif "norm.weight" in state_dict["image_proj"]:
                        # IP-Adapter Face ID
                        num_image_text_embeds += [4]
                    else:
                        # IP-Adapter Plus
                        num_image_text_embeds += [state_dict["image_proj"]["latents"].shape[1]]

                with init_context():
                    attn_procs[name] = attn_processor_class(
                        hidden_size=hidden_size,
                        cross_attention_dim=cross_attention_dim,
                        scale=1.0,
                        num_tokens=num_image_text_embeds,
                    )

                value_dict = {}
                for i, state_dict in enumerate(state_dicts):
                    value_dict.update({f"to_k_ip.{i}.weight": state_dict["ip_adapter"][f"{key_id}.to_k_ip.weight"]})
                    value_dict.update({f"to_v_ip.{i}.weight": state_dict["ip_adapter"][f"{key_id}.to_v_ip.weight"]})

                if not low_cpu_mem_usage:
                    attn_procs[name].load_state_dict(value_dict)
                else:
                    device = next(iter(value_dict.values())).device
                    dtype = next(iter(value_dict.values())).dtype
                    load_model_dict_into_meta(attn_procs[name], value_dict, device=device, dtype=dtype)

                key_id += 2

        return attn_procs

    def _load_ip_adapter_weights(self, state_dicts, low_cpu_mem_usage=False):
        if not isinstance(state_dicts, list):
            state_dicts = [state_dicts]
        # Set encoder_hid_proj after loading ip_adapter weights,
        # because `IPAdapterPlusImageProjection` also has `attn_processors`.
        self.encoder_hid_proj = None

        attn_procs = self._convert_ip_adapter_attn_to_diffusers(state_dicts, low_cpu_mem_usage=low_cpu_mem_usage)
        self.set_attn_processor(attn_procs)

        # convert IP-Adapter Image Projection layers to diffusers
        image_projection_layers = []
        for state_dict in state_dicts:
            image_projection_layer = self._convert_ip_adapter_image_proj_to_diffusers(
                state_dict["image_proj"], low_cpu_mem_usage=low_cpu_mem_usage
            )
            image_projection_layers.append(image_projection_layer)

        self.encoder_hid_proj = MultiIPAdapterImageProjection(image_projection_layers)
        self.config.encoder_hid_dim_type = "ip_image_proj"

        self.to(dtype=self.dtype, device=self.device)

    def _load_ip_adapter_loras(self, state_dicts):
        lora_dicts = {}
        for key_id, name in enumerate(self.attn_processors.keys()):
            for i, state_dict in enumerate(state_dicts):
                if f"{key_id}.to_k_lora.down.weight" in state_dict["ip_adapter"]:
                    if i not in lora_dicts:
                        lora_dicts[i] = {}
                    lora_dicts[i].update(
                        {
                            f"unet.{name}.to_k_lora.down.weight": state_dict["ip_adapter"][
                                f"{key_id}.to_k_lora.down.weight"
                            ]
                        }
                    )
                    lora_dicts[i].update(
                        {
                            f"unet.{name}.to_q_lora.down.weight": state_dict["ip_adapter"][
                                f"{key_id}.to_q_lora.down.weight"
                            ]
                        }
                    )
                    lora_dicts[i].update(
                        {
                            f"unet.{name}.to_v_lora.down.weight": state_dict["ip_adapter"][
                                f"{key_id}.to_v_lora.down.weight"
                            ]
                        }
                    )
                    lora_dicts[i].update(
                        {
                            f"unet.{name}.to_out_lora.down.weight": state_dict["ip_adapter"][
                                f"{key_id}.to_out_lora.down.weight"
                            ]
                        }
                    )
                    lora_dicts[i].update(
                        {f"unet.{name}.to_k_lora.up.weight": state_dict["ip_adapter"][f"{key_id}.to_k_lora.up.weight"]}
                    )
                    lora_dicts[i].update(
                        {f"unet.{name}.to_q_lora.up.weight": state_dict["ip_adapter"][f"{key_id}.to_q_lora.up.weight"]}
                    )
                    lora_dicts[i].update(
                        {f"unet.{name}.to_v_lora.up.weight": state_dict["ip_adapter"][f"{key_id}.to_v_lora.up.weight"]}
                    )
                    lora_dicts[i].update(
                        {
                            f"unet.{name}.to_out_lora.up.weight": state_dict["ip_adapter"][
                                f"{key_id}.to_out_lora.up.weight"
                            ]
                        }
                    )
        return lora_dicts
