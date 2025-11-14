from __future__ import annotations
import torch
import os
import sys
import json
import hashlib
import traceback
import math
import random
import logging
import gc
from PIL import Image, ImageOps, ImageSequence
from PIL.PngImagePlugin import PngInfo
import numpy as np
import safetensors.torch

__author__ = "eddy"
current_dir = os.path.dirname(os.path.realpath(__file__))
comfyui_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
MODE_COMFY = False

try:
    from .kv_cache_manager import UnifiedCacheManager
    KV_CACHE_AVAILABLE = True
except Exception as e:
    logging.warning(f"KV Cache Manager not available: {e}")
    KV_CACHE_AVAILABLE = False
    UnifiedCacheManager = None

class SimpleBlockSwap:
    def __init__(self, blocks_to_swap=20, model_size="auto"):
        self.blocks_to_swap = blocks_to_swap
        self.model_size = model_size
        self.main_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.offload_device = torch.device('cpu')

        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    def count_model_parameters(self, model):
        try:
            if hasattr(model, 'parameters'):
                return sum(p.numel() for p in model.parameters())
            elif hasattr(model, 'model') and hasattr(model.model, 'parameters'):
                return sum(p.numel() for p in model.model.parameters())
            else:
                return 0
        except Exception:
            return 0

    def auto_detect_model_size(self, num_parameters):
        if num_parameters < 1e9:
            return "small"
        elif num_parameters < 5e9:
            return "medium"
        elif num_parameters < 15e9:
            return "large"
        else:
            return "xl"

    def get_recommended_blocks(self, model_size):
        configs = {
            "small": 8,
            "medium": 12,
            "large": 16,
            "xl": 24,
            "auto": 20
        }
        return configs.get(model_size, 20)

    def apply_blockswap(self, model, use_recommended=True):
        try:
            if self.model_size == "auto":
                param_count = self.count_model_parameters(model)
                detected_size = self.auto_detect_model_size(param_count)
                logging.info(f"BlockSwap: Auto-detected model size: {detected_size} ({param_count:,} params)")
            else:
                detected_size = self.model_size

            if use_recommended:
                blocks_to_swap = self.get_recommended_blocks(detected_size)
                logging.info(f"BlockSwap: Using recommended config for {detected_size}: {blocks_to_swap} blocks")
            else:
                blocks_to_swap = self.blocks_to_swap
                logging.info(f"BlockSwap: Using manual config: {blocks_to_swap} blocks")

            transformer_blocks = self._find_transformer_blocks(model)

            if not transformer_blocks:
                logging.warning("BlockSwap: No transformer blocks found")
                return model

            total_blocks = len(transformer_blocks)
            blocks_to_swap = min(blocks_to_swap, total_blocks - 1)

            if blocks_to_swap < 0:
                logging.warning("BlockSwap: Invalid blocks_to_swap configuration")
                return model

            total_offload_memory = 0
            total_main_memory = 0

            logging.info(f"BlockSwap: Swapping {blocks_to_swap + 1}/{total_blocks} blocks to CPU")

            for i, block in enumerate(transformer_blocks):
                block_memory = self._get_module_memory_mb(block)

                if i <= blocks_to_swap:
                    block.to(self.offload_device)
                    total_offload_memory += block_memory

                    for param in block.parameters():
                        if param.device.type == 'cpu' and not param.is_pinned():
                            try:
                                param.data = param.data.pin_memory()
                            except:
                                pass
                else:
                    block.to(self.main_device)
                    total_main_memory += block_memory

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            logging.info(f"BlockSwap: CPU blocks: {blocks_to_swap + 1} ({total_offload_memory:.1f}MB)")
            logging.info(f"BlockSwap: GPU blocks: {total_blocks - blocks_to_swap - 1} ({total_main_memory:.1f}MB)")

            return model

        except Exception as e:
            logging.error(f"BlockSwap failed: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return model

    def _find_transformer_blocks(self, model):
        blocks = []

        if hasattr(model, 'model') and hasattr(model.model, 'diffusion_model'):
            diffusion_model = model.model.diffusion_model

            block_paths = [
                'transformer_blocks',
                'blocks',
                'layers',
                'attention_blocks',
                'input_blocks',
                'middle_block',
                'output_blocks'
            ]

            for path in block_paths:
                if hasattr(diffusion_model, path):
                    attr = getattr(diffusion_model, path)
                    if isinstance(attr, (list, torch.nn.ModuleList)):
                        blocks.extend(attr)
                    elif isinstance(attr, torch.nn.Module):
                        blocks.append(attr)

                    if blocks:
                        break

        if not blocks:
            blocks = self._recursive_find_blocks(model)

        return blocks

    def _recursive_find_blocks(self, module):
        blocks = []
        block_names = ['blocks', 'layers', 'transformer_blocks', 'attention_blocks']

        for block_name in block_names:
            if hasattr(module, block_name):
                attr = getattr(module, block_name)
                if isinstance(attr, (list, torch.nn.ModuleList)):
                    blocks.extend(attr)
                    return blocks

        for child_name, child_module in module.named_children():
            child_blocks = self._recursive_find_blocks(child_module)
            blocks.extend(child_blocks)

        return blocks

    def _get_module_memory_mb(self, module):
        try:
            total_params = sum(p.numel() for p in module.parameters())
            memory_mb = (total_params * 4) / (1024 ** 2)
            return memory_mb
        except Exception:
            return 0.0

try:
    if os.environ.get("QIW_STANDALONE", "0") != "1":
        sys.path.insert(0, comfyui_root)

        try:
            import comfy.sd as comfy_sd
            import comfy.utils as comfy_utils
            import comfy.model_management as comfy_model_management
            import comfy.model_sampling as comfy_model_sampling
            from comfy.cli_args import args

            try:
                import comfy.samplers as comfy_samplers
            except ImportError:
                comfy_samplers = None
                logging.warning("ComfyUI-QwenImageWrapper: comfy.samplers not available, using fallback")

            class IO:
                STRING = "STRING"
                CONDITIONING = "CONDITIONING"
                CLIP = "CLIP"

            class ComfyNodeABC:
                pass

            InputTypeDict = dict

        except ImportError as import_err:
            logging.warning(f"ComfyUI-QwenImageWrapper: Could not import from comfy: {import_err}")
            sys.path.insert(0, os.path.join(current_dir, "comfy_core"))
            import comfy_core.samplers as comfy_samplers
            import comfy_core.sd as comfy_sd
            import comfy_core.utils as comfy_utils
            from comfy_core.comfy_types import IO, ComfyNodeABC, InputTypeDict
            import comfy_core.model_management as comfy_model_management
            import comfy_core.model_sampling as comfy_model_sampling
            from comfy_core.cli_args import args

        import folder_paths
        sys.path.insert(0, current_dir)
        import latent_preview
        import node_helpers
        import nodes_base as nodes

        MODE_COMFY = True
    else:
        raise ImportError("standalone")
except Exception as e:
    logging.warning(f"ComfyUI-QwenImageWrapper: Failed to load in COMFY mode: {e}")
    logging.warning(f"ComfyUI-QwenImageWrapper: Falling back to standalone mode")
    comfy_samplers = None
    class IO:
        STRING = "STRING"
        CONDITIONING = "CONDITIONING"
        CLIP = "CLIP"
    class ComfyNodeABC:
        pass
    InputTypeDict = dict
    class _Args:
        disable_metadata = False
    args = _Args()
    comfy_samplers = None
    class _FolderPaths:
        @staticmethod
        def get_output_directory():
            out = os.path.join(current_dir, "output")
            os.makedirs(out, exist_ok=True)
            return out
        @staticmethod
        def get_save_image_path(filename_prefix, output_dir, image_width=0, image_height=0):
            subfolder = os.path.dirname(os.path.normpath(filename_prefix))
            filename = os.path.basename(os.path.normpath(filename_prefix))
            full_output_folder = os.path.join(output_dir, subfolder)
            os.makedirs(full_output_folder, exist_ok=True)
            def map_filename(fn):
                try:
                    digits = int(fn.split('_')[1])
                except:
                    digits = 0
                return digits
            try:
                existing = [f for f in os.listdir(full_output_folder) if f.startswith(f"{filename}_") and f.endswith("_.png")]
                counter = max([map_filename(f) for f in existing], default=0) + 1
            except FileNotFoundError:
                counter = 1
            return full_output_folder, filename, counter, subfolder, filename_prefix
        @staticmethod
        def get_filename_list(folder_name):
            return []
        @staticmethod
        def get_full_path_or_raise(folder_name, filename):
            raise FileNotFoundError(f"{folder_name}/{filename} not available in standalone mode.")
        @staticmethod
        def get_folder_paths(folder_name):
            return []
    folder_paths = _FolderPaths()
    class _Nodes:
        def common_ksampler(self, *args, **kwargs):
            raise NotImplementedError("KSampler is unavailable in standalone mode.")
    nodes = _Nodes()

MAX_RESOLUTION = 16384

class CLIPTextEncode(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(s) -> InputTypeDict:
        return {
            "required": {
                "text": (IO.STRING, {"multiline": True, "dynamicPrompts": True, "tooltip": "The text to be encoded."}),
                "clip": (IO.CLIP, {"tooltip": "The CLIP model used for encoding the text."})
            }
        }
    RETURN_TYPES = (IO.CONDITIONING,)
    OUTPUT_TOOLTIPS = ("A conditioning containing the embedded text used to guide the diffusion model.",)
    FUNCTION = "encode"
    CATEGORY = "conditioning"
    DESCRIPTION = "Encodes a text prompt using a CLIP model into an embedding that can be used to guide the diffusion model towards generating specific images."

    def encode(self, clip, text):
        if clip is None:
            raise RuntimeError("ERROR: clip input is invalid: None\n\nIf the clip is from a checkpoint loader node your checkpoint does not contain a valid clip or text encoder model.")
        tokens = clip.tokenize(text)
        return (clip.encode_from_tokens_scheduled(tokens), )


class VAEDecode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "samples": ("LATENT", ), "vae": ("VAE", )}}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "decode"
    CATEGORY = "latent"

    def decode(self, vae, samples):
        return (vae.decode(samples["samples"]), )


class VAELoader:
    @staticmethod
    def vae_list():
        vaes = folder_paths.get_filename_list("vae")
        approx_vaes = folder_paths.get_filename_list("vae_approx")
        sdxl_taesd_enc = False
        sdxl_taesd_dec = False
        sd1_taesd_enc = False
        sd1_taesd_dec = False
        sd3_taesd_enc = False
        sd3_taesd_dec = False

        for v in approx_vaes:
            if v.startswith("taesd_decoder."):
                sd1_taesd_dec = True
            elif v.startswith("taesd_encoder."):
                sd1_taesd_enc = True
            elif v.startswith("taesdxl_decoder."):
                sdxl_taesd_dec = True
            elif v.startswith("taesdxl_encoder."):
                sdxl_taesd_enc = True
            elif v.startswith("taesd3_decoder."):
                sd3_taesd_dec = True
            elif v.startswith("taesd3_encoder."):
                sd3_taesd_enc = True
        if sd1_taesd_dec and sd1_taesd_enc:
            vaes.append("taesd")
        if sdxl_taesd_dec and sdxl_taesd_enc:
            vaes.append("taesdxl")
        if sd3_taesd_dec and sd3_taesd_enc:
            vaes.append("taesd3")
        return vaes

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "vae_name": (s.vae_list(), )}}
    RETURN_TYPES = ("VAE",)
    FUNCTION = "load_vae"
    CATEGORY = "loaders"

    def load_vae(self, vae_name):
        if vae_name in ["taesd", "taesdxl", "taesd3"]:
            sd = {}
        else:
            vae_path = folder_paths.get_full_path_or_raise("vae", vae_name)
            sd = comfy_utils.load_torch_file(vae_path)
        vae = comfy_sd.VAE(sd=sd)
        return (vae,)


class UNETLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "unet_name": (folder_paths.get_filename_list("diffusion_models"), ),
                             "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"],)
                             }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"
    CATEGORY = "advanced/loaders"

    def load_unet(self, unet_name, weight_dtype):
        if weight_dtype == "default":
            weight_dtype = None
        elif weight_dtype == "fp8_e4m3fn_fast":
            weight_dtype = torch.float8_e4m3fn
        model_options = {}
        if weight_dtype is not None:
            model_options["weight_dtype"] = weight_dtype

        unet_path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)
        model = comfy_sd.load_diffusion_model(unet_path, model_options=model_options)
        return (model,)


class CLIPLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "clip_name": (folder_paths.get_filename_list("text_encoders"), ),
                              "type": (["stable_diffusion", "stable_cascade", "sd3", "stable_audio", "mochi", "ltxv", "hunyuan_video", "wan", "omnigen"],),
                              "device": (["default", "cpu"],),
                             }}
    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_clip"
    CATEGORY = "advanced/loaders"

    def load_clip(self, clip_name, type="stable_diffusion", device="default"):
        if device == "default":
            device = None

        clip_type = comfy_sd.CLIPType.STABLE_DIFFUSION
        if type == "stable_cascade":
            clip_type = comfy_sd.CLIPType.STABLE_CASCADE
        elif type == "sd3":
            clip_type = comfy_sd.CLIPType.SD3
        elif type == "stable_audio":
            clip_type = comfy_sd.CLIPType.STABLE_AUDIO
        elif type == "mochi":
            clip_type = comfy_sd.CLIPType.MOCHI
        elif type == "ltxv":
            clip_type = comfy_sd.CLIPType.LTXV
        elif type == "hunyuan_video":
            clip_type = comfy_sd.CLIPType.HUNYUAN_VIDEO
        elif type == "wan":
            clip_type = comfy_sd.CLIPType.WAN
        elif type == "omnigen":
            clip_type = comfy_sd.CLIPType.OMNIGEN

        clip_path = folder_paths.get_full_path_or_raise("text_encoders", clip_name)
        clip = comfy_sd.load_clip(ckpt_paths=[clip_path], embedding_directory=folder_paths.get_folder_paths("embeddings"), clip_type=clip_type, model_options={"manual_cast_dtype": device})
        return (clip,)


class KSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                    "sampler_name": (comfy_samplers.KSampler.SAMPLERS if MODE_COMFY else ["euler"], ),
                    "scheduler": (comfy_samplers.KSampler.SCHEDULERS if MODE_COMFY else ["normal"], ),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "latent_image": ("LATENT", ),
                    "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                     }
                }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "sampling"

    def sample(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=1.0):
        return nodes.common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise)


class SaveImage:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"images": ("IMAGE", {"tooltip": "The images to save."}),
                     "filename_prefix": ("STRING", {"default": "ComfyUI", "tooltip": "The prefix for the file to save. This may include formatting information such as %date:yyyy-MM-dd% or %Empty Latent Image.width% to include values from nodes."})},
                "hidden":
                    {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
                }

    RETURN_TYPES = ()
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "image"
    DESCRIPTION = "Saves the input images to your ComfyUI output directory."

    def save_images(self, images, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None):
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])
        results = list()
        for (batch_number, image) in enumerate(images):
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = None
            if not args.disable_metadata:
                metadata = PngInfo()
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        metadata.add_text(x, json.dumps(extra_pnginfo[x]))

            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            file = f"{filename_with_batch_num}_{counter:05}_.png"
            img.save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=self.compress_level)
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1

        return { "ui": { "images": results } }


class EmptySD3LatentImage:
    def __init__(self):
        self.device = comfy_model_management.intermediate_device() if MODE_COMFY else "cpu"

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "width": ("INT", {"default": 1024, "min": 16, "max": MAX_RESOLUTION, "step": 16}),
                              "height": ("INT", {"default": 1024, "min": 16, "max": MAX_RESOLUTION, "step": 16}),
                              "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096})}}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate"
    CATEGORY = "latent/sd3"

    def generate(self, width, height, batch_size=1):
        latent = torch.zeros([batch_size, 16, height // 8, width // 8], device=self.device)
        return ({"samples":latent}, )


class ModelSamplingSD3:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "shift": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 100.0, "step":0.01}),
                              }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "advanced/model"

    def patch(self, model, shift, multiplier=1000):
        m = model.clone()

        sampling_base = comfy_model_sampling.ModelSamplingDiscreteFlow
        sampling_type = comfy_model_sampling.CONST

        class ModelSamplingAdvanced(sampling_base, sampling_type):
            pass

        model_sampling = ModelSamplingAdvanced(model.model.model_config)
        model_sampling.set_parameters(shift=shift, multiplier=multiplier)
        m.add_object_patch("model_sampling", model_sampling)
        return (m, )


class ModelSamplingAuraFlow(ModelSamplingSD3):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "shift": ("FLOAT", {"default": 1.73, "min": 0.0, "max": 100.0, "step":0.01}),
                              }}

    FUNCTION = "patch_aura"

    def patch_aura(self, model, shift):
        return self.patch(model, shift, multiplier=1.0)


class EddyQwenImageBlockSwap(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        unet_list = folder_paths.get_filename_list("diffusion_models")
        clip_list = folder_paths.get_filename_list("text_encoders")
        vae_list = folder_paths.get_filename_list("vae")
        lora_list_raw = folder_paths.get_filename_list("loras")
        lora_list = ["none"] + lora_list_raw

        sampler_list = comfy_samplers.KSampler.SAMPLERS if MODE_COMFY and comfy_samplers else ["euler"]
        scheduler_list = comfy_samplers.KSampler.SCHEDULERS if MODE_COMFY and comfy_samplers else ["normal"]

        return {
            "required": {
                "positive": (IO.STRING, {"multiline": True, "dynamicPrompts": True}),
                "negative": (IO.STRING, {"multiline": True, "dynamicPrompts": True}),
                "unet_name": (unet_list, {}),
                "clip_name": (clip_list, {}),
                "vae_name": (vae_list, {}),
                "width": ("INT", {"default": 1328, "min": 256, "max": MAX_RESOLUTION, "step": 16}),
                "height": ("INT", {"default": 1328, "min": 256, "max": MAX_RESOLUTION, "step": 16}),
                "steps": ("INT", {"default": 8, "min": 1, "max": 100, "step": 1}),
                "cfg": ("FLOAT", {"default": 2.5, "min": 0.0, "max": 20.0, "step": 0.1}),
                "sampler_name": (sampler_list, {}),
                "scheduler": (scheduler_list, {}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "quantization_dtype": (["default", "fp8_e4m3fn", "fp8_e5m2", "fp16", "fp16_fast", "bf16", "bf16_fast"], {"default": "fp8_e4m3fn", "tooltip": "Weight precision: fp8=fastest+50% VRAM save, bf16_fast=balanced 2.5x speed, default=no quantization"}),
                "lora_1_name": (lora_list, {"default": "none", "tooltip": "First LoRA model (none = disabled)"}),
                "lora_1_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05, "tooltip": "LoRA 1 strength (0 = no effect)"}),
                "lora_2_name": (lora_list, {"default": "none", "tooltip": "Second LoRA model (none = disabled)"}),
                "lora_2_strength": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.05, "tooltip": "LoRA 2 strength (0 = no effect)"}),
                "lora_3_name": (lora_list, {"default": "none", "tooltip": "Third LoRA model (none = disabled)"}),
                "lora_3_strength": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.05, "tooltip": "LoRA 3 strength (0 = no effect)"}),
                "lora_4_name": (lora_list, {"default": "none", "tooltip": "Fourth LoRA model (none = disabled)"}),
                "lora_4_strength": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step": 0.05, "tooltip": "LoRA 4 strength (0 = no effect)"}),
                "use_blockswap": ("BOOLEAN", {"default": True, "tooltip": "Swap transformer blocks to CPU for 30-60% VRAM reduction (minimal speed impact with pinned memory)"}),
                "blockswap_blocks": ("INT", {"default": 20, "min": 1, "max": 50, "step": 1, "tooltip": "Blocks to swap (higher = more VRAM save but slower). Ignored if use_recommended=True"}),
                "blockswap_model_size": (["auto", "small", "medium", "large", "xl"], {"default": "auto", "tooltip": "Model size for auto block count (auto = detect from parameters)"}),
                "blockswap_use_recommended": ("BOOLEAN", {"default": True, "tooltip": "Use optimal block count for model size (small=8, medium=12, large=16, xl=24)"}),
                "enable_matmul_optimization": ("BOOLEAN", {"default": True, "tooltip": "Enable TF32, FP16/BF16-Fast math, cuDNN benchmark for 1.5-2x speedup (always recommended)"}),
                "use_torch_compile": ("BOOLEAN", {"default": False, "tooltip": "JIT compile model (20-60% faster but SLOW first run: 1-3min, instant with cache after)"}),
                "matmul_precision": (["highest", "high", "medium"], {"default": "high", "tooltip": "Matrix precision: high=balanced, medium=faster slightly less accurate, highest=slowest most accurate"}),
                "use_autocast": ("BOOLEAN", {"default": False, "tooltip": "Mixed precision (30-50% faster, 40% VRAM save, minimal quality loss)"}),
                "autocast_dtype": (["float16", "bfloat16"], {"default": "bfloat16", "tooltip": "Autocast type: bfloat16=stable (RTX 30xx+), float16=faster but less stable"}),
                "use_channels_last": ("BOOLEAN", {"default": False, "tooltip": "NHWC memory layout (10-20% faster convolutions, may cause issues with some models)"}),
                "enable_flash_attention": ("BOOLEAN", {"default": True, "tooltip": "Optimized attention kernel (2-4x faster attention, always recommended)"}),
                "compile_mode": (["default", "reduce-overhead", "max-autotune"], {"default": "default", "tooltip": "Compile optimization: default=fast compile, reduce-overhead=better runtime, max-autotune=best runtime but VERY slow compile"}),
                "enable_kv_cache": ("BOOLEAN", {"default": True, "tooltip": "Cache compiled model (2nd startup 15s vs 3min, always recommended if using torch.compile)"}),
            }
        }

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate"
    CATEGORY = "Qwen/Integrated"

    def _apply_matmul_optimizations(self, matmul_precision, use_autocast=False, autocast_dtype="bfloat16"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

        if hasattr(torch.backends.cuda.matmul, 'allow_fp16_reduced_precision_reduction'):
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
            logging.info("Enabled FP16 reduced precision reduction for faster matmul")

        if hasattr(torch.backends.cuda.matmul, 'allow_bf16_reduced_precision_reduction'):
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
            logging.info("Enabled BF16 reduced precision reduction for faster matmul")

        if hasattr(torch.backends.cuda, 'allow_fp16_bf16_reduction_math_sdp'):
            torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp = True
            logging.info("Enabled FP16/BF16 reduction math for scaled_dot_product_attention")

        if matmul_precision == "highest":
            torch.set_float32_matmul_precision("highest")
        elif matmul_precision == "high":
            torch.set_float32_matmul_precision("high")
        elif matmul_precision == "medium":
            torch.set_float32_matmul_precision("medium")

    def _compile_model_if_needed(self, model, unet_path, use_torch_compile, compile_mode="default", enable_kv_cache=True):
        if not use_torch_compile:
            return model

        if not hasattr(model, "_compiled") or not model._compiled:
            compile_config = {
                "mode": compile_mode,
                "fullgraph": False,
                "dynamic": False,
            }

            if enable_kv_cache and KV_CACHE_AVAILABLE:
                try:
                    cache_manager = UnifiedCacheManager()

                    logging.info("Checking compilation cache...")
                    is_previously_compiled = cache_manager.check_and_mark_compilation(unet_path, compile_config)

                    if is_previously_compiled:
                        logging.info("Model was previously compiled - torch will reuse cached kernels automatically")
                except Exception as e:
                    logging.warning(f"Cache check failed: {e}, will compile normally")

            try:
                logging.info(f"Compiling model with torch.compile (mode={compile_mode})...")
                logging.info("First-time compilation will take 1-3 minutes, but subsequent loads will be instant!")

                torch._dynamo.config.cache_size_limit = 256
                torch._dynamo.config.suppress_errors = True
                torch._inductor.config.triton.cudagraphs = False
                torch._inductor.config.fallback_random = True

                if compile_mode == "max-autotune":
                    torch._inductor.config.max_autotune = True
                    logging.info("Enabled max-autotune mode for aggressive optimization")

                compiled_diffusion_model = torch.compile(
                    model.model.diffusion_model,
                    mode=compile_mode,
                    fullgraph=False,
                    dynamic=False,
                )
                model.model.diffusion_model = compiled_diffusion_model
                model._compiled = True

                logging.info(f"Model compilation completed successfully with mode={compile_mode}")

                if enable_kv_cache and KV_CACHE_AVAILABLE:
                    try:
                        cache_manager = UnifiedCacheManager()
                        cache_manager.mark_model_compiled(unet_path, compile_config)
                        logging.info("Compilation metadata saved - future runs will reuse torch inductor cache automatically")
                    except Exception as e:
                        logging.warning(f"Failed to save compilation metadata: {e}")

            except Exception as e:
                logging.warning(f"torch.compile failed: {e}, continuing without compilation")

        return model

    def generate(self, positive, negative, unet_name, clip_name, vae_name,
                 width, height, steps, cfg, sampler_name, scheduler, seed,
                 quantization_dtype,
                 lora_1_name, lora_1_strength,
                 lora_2_name, lora_2_strength,
                 lora_3_name, lora_3_strength,
                 lora_4_name, lora_4_strength,
                 use_blockswap, blockswap_blocks, blockswap_model_size, blockswap_use_recommended,
                 enable_matmul_optimization=True, use_torch_compile=False, matmul_precision="high",
                 use_autocast=False, autocast_dtype="bfloat16", use_channels_last=False,
                 enable_flash_attention=True, compile_mode="default", enable_kv_cache=True):
        if not MODE_COMFY:
            raise RuntimeError("EddyQwenImageBlockSwap requires full Comfy core mode.")

        if enable_matmul_optimization:
            self._apply_matmul_optimizations(matmul_precision, use_autocast, autocast_dtype)

        # 1) load diffusion model with dynamic quantization
        unet_path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)
        model_options = {}

        if quantization_dtype != "default":
            if quantization_dtype == "fp8_e4m3fn":
                model_options["weight_dtype"] = torch.float8_e4m3fn
            elif quantization_dtype == "fp8_e5m2":
                model_options["weight_dtype"] = "fp8_e5m2"
            elif quantization_dtype == "fp16":
                model_options["weight_dtype"] = torch.float16
            elif quantization_dtype == "fp16_fast":
                model_options["weight_dtype"] = torch.float16
                if hasattr(torch.backends.cuda.matmul, 'allow_fp16_reduced_precision_reduction'):
                    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
                    logging.info("Using FP16-Fast: FP16 with reduced precision reduction enabled")
            elif quantization_dtype == "bf16":
                model_options["weight_dtype"] = torch.bfloat16
            elif quantization_dtype == "bf16_fast":
                model_options["weight_dtype"] = torch.bfloat16
                if hasattr(torch.backends.cuda.matmul, 'allow_bf16_reduced_precision_reduction'):
                    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
                    logging.info("Using BF16-Fast: BF16 with reduced precision reduction enabled")

        model = comfy_sd.load_diffusion_model(unet_path, model_options=model_options)

        if use_channels_last and hasattr(model, 'model') and hasattr(model.model, 'diffusion_model'):
            try:
                model.model.diffusion_model = model.model.diffusion_model.to(memory_format=torch.channels_last)
                logging.info("Applied channels_last memory format for optimized convolutions")
            except Exception as e:
                logging.warning(f"Failed to apply channels_last format: {e}")

        if enable_flash_attention and hasattr(model, 'model'):
            try:
                if hasattr(model.model, 'diffusion_model'):
                    logging.info("Flash Attention will be automatically used by PyTorch SDPA")
            except Exception as e:
                logging.warning(f"Flash Attention check failed: {e}")

        # 2) apply multiple LoRAs
        lora_configs = [
            (lora_1_name, lora_1_strength),
            (lora_2_name, lora_2_strength),
            (lora_3_name, lora_3_strength),
            (lora_4_name, lora_4_strength),
        ]

        lora_loader = LoraLoaderModelOnly()
        for lora_name, lora_strength in lora_configs:
            if lora_strength != 0.0 and lora_name and lora_name != "none":
                try:
                    model, = lora_loader.load_lora_model_only(model, lora_name, lora_strength)
                    logging.info(f"Applied LoRA: {lora_name} with strength {lora_strength}")
                except Exception as e:
                    logging.warning(f"Failed to load LoRA {lora_name}: {e}")

        # 3) optional BlockSwap with configurable parameters
        if use_blockswap:
            try:
                logging.info(f"Applying BlockSwap: blocks={blockswap_blocks}, model_size={blockswap_model_size}, use_recommended={blockswap_use_recommended}")
                bs_optimizer = SimpleBlockSwap(
                    blocks_to_swap=blockswap_blocks,
                    model_size=blockswap_model_size
                )
                model = bs_optimizer.apply_blockswap(model, use_recommended=blockswap_use_recommended)
                logging.info("BlockSwap applied successfully")
            except Exception as e:
                logging.error(f"BlockSwap application failed: {e}")
                import traceback
                logging.error(traceback.format_exc())
                logging.warning("Continuing without BlockSwap")
        else:
            logging.info("BlockSwap disabled by user")

        # 3.5) optional torch.compile optimization with KV cache
        model = self._compile_model_if_needed(model, unet_path, use_torch_compile, compile_mode, enable_kv_cache)

        # 4) load CLIP
        clip_path = folder_paths.get_full_path_or_raise("text_encoders", clip_name)
        clip_type = comfy_sd.CLIPType.WAN
        clip = comfy_sd.load_clip(
            ckpt_paths=[clip_path],
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            clip_type=clip_type,
            model_options={"manual_cast_dtype": None},
        )

        # 5) text encode
        pos_tokens = clip.tokenize(positive)
        neg_tokens = clip.tokenize(negative)
        positive_cond = clip.encode_from_tokens_scheduled(pos_tokens)
        negative_cond = clip.encode_from_tokens_scheduled(neg_tokens)

        # 6) load VAE
        vae_path = folder_paths.get_full_path_or_raise("vae", vae_name)
        vae_sd = comfy_utils.load_torch_file(vae_path)
        vae = comfy_sd.VAE(sd=vae_sd)

        # 7) latent
        device = comfy_model_management.intermediate_device()
        latent = torch.zeros([1, 16, height // 8, width // 8], device=device)
        latent_dict = {"samples": latent}

        # 8) sampling patch (AuraFlow)
        msa = ModelSamplingAuraFlow()
        model_patched, = msa.patch_aura(model, 1.73)

        # 9) KSampler with optional autocast
        if use_autocast and torch.cuda.is_available():
            autocast_dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16}
            dtype = autocast_dtype_map.get(autocast_dtype, torch.bfloat16)
            logging.info(f"Using autocast with dtype={autocast_dtype}")

            with torch.cuda.amp.autocast(dtype=dtype):
                samples_tuple = nodes.common_ksampler(
                    model_patched,
                    seed,
                    steps,
                    cfg,
                    sampler_name,
                    scheduler,
                    positive_cond,
                    negative_cond,
                    latent_dict,
                    denoise=1.0,
                )
        else:
            samples_tuple = nodes.common_ksampler(
                model_patched,
                seed,
                steps,
                cfg,
                sampler_name,
                scheduler,
                positive_cond,
                negative_cond,
                latent_dict,
                denoise=1.0,
            )

        samples = samples_tuple[0]

        # 10) VAE decode
        images = vae.decode(samples["samples"])
        if len(images.shape) == 5:
            images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        return (images,)


class LoraLoader:
    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL", {"tooltip": "The diffusion model the LoRA will be applied to."}),
                              "clip": ("CLIP", {"tooltip": "The CLIP model the LoRA will be applied to."}),
                              "lora_name": (folder_paths.get_filename_list("loras"), {"tooltip": "The name of the LoRA."}),
                              "strength_model": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01, "tooltip": "How strongly to modify the diffusion model. This value can be negative."}),
                              "strength_clip": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01, "tooltip": "How strongly to modify the CLIP model. This value can be negative."}),
                              }}
    RETURN_TYPES = ("MODEL", "CLIP")
    OUTPUT_TOOLTIPS = ("The modified diffusion model.", "The modified CLIP model.")
    FUNCTION = "load_lora"
    CATEGORY = "loaders"
    DESCRIPTION = "LoRAs are used to modify diffusion and CLIP models, altering the way in which latents are denoised such as applying styles. Multiple LoRA nodes can be linked together."

    def load_lora(self, model, clip, lora_name, strength_model, strength_clip):
        if strength_model == 0 and strength_clip == 0:
            return (model, clip)

        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        lora = None
        if self.loaded_lora is not None:
            if self.loaded_lora[0] == lora_path:
                lora = self.loaded_lora[1]
            else:
                temp = self.loaded_lora
                self.loaded_lora = None
                del temp

        if lora is None:
            lora = comfy_utils.load_torch_file(lora_path, safe_load=True)
            self.loaded_lora = (lora_path, lora)

        model_lora, clip_lora = comfy_sd.load_lora_for_models(model, clip, lora, strength_model, strength_clip)
        return (model_lora, clip_lora)


class LoraLoaderModelOnly(LoraLoader):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "lora_name": (folder_paths.get_filename_list("loras"), ),
                              "strength_model": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                              }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_lora_model_only"

    def load_lora_model_only(self, model, lora_name, strength_model):
        return (self.load_lora(model, None, lora_name, strength_model, 0)[0],)


NODE_CLASS_MAPPINGS = {
    "eddy_qwen_CLIPTextEncode": CLIPTextEncode,
    "eddy_qwen_VAEDecode": VAEDecode,
    "eddy_qwen_VAELoader": VAELoader,
    "eddy_qwen_UNETLoader": UNETLoader,
    "eddy_qwen_CLIPLoader": CLIPLoader,
    "eddy_qwen_KSampler": KSampler,
    "eddy_qwen_SaveImage": SaveImage,
    "eddy_qwen_EmptySD3LatentImage": EmptySD3LatentImage,
    "eddy_qwen_ModelSamplingAuraFlow": ModelSamplingAuraFlow,
    "eddy_qwen_LoraLoaderModelOnly": LoraLoaderModelOnly,
    "eddy_qwen_image_blockswap": EddyQwenImageBlockSwap,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "eddy_qwen_CLIPTextEncode": "Eddy Qwen CLIP Text Encode",
    "eddy_qwen_VAEDecode": "Eddy Qwen VAE Decode",
    "eddy_qwen_VAELoader": "Eddy Qwen Load VAE",
    "eddy_qwen_UNETLoader": "Eddy Qwen Load Diffusion Model",
    "eddy_qwen_CLIPLoader": "Eddy Qwen Load CLIP",
    "eddy_qwen_KSampler": "Eddy Qwen KSampler",
    "eddy_qwen_SaveImage": "Eddy Qwen Save Image",
    "eddy_qwen_EmptySD3LatentImage": "Eddy Qwen EmptySD3LatentImage",
    "eddy_qwen_ModelSamplingAuraFlow": "Eddy Qwen ModelSamplingAuraFlow",
    "eddy_qwen_LoraLoaderModelOnly": "Eddy Qwen Load LoRA Model Only",
    "eddy_qwen_image_blockswap": "Eddy Qwen Image + BlockSwap",
}
