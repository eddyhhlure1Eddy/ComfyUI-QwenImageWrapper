# ComfyUI-QwenImageWrapper

Integrated Qwen-Image node for ComfyUI with built-in optimizations.

## Features

- **All-in-one node**: Single node includes model loading, LoRA, sampling, and VAE decoding
- **4 LoRA slots**: Support up to 4 LoRA models simultaneously with individual strength control
- **Memory optimization**: Built-in BlockSwap reduces VRAM usage by 30-60%
- **Multiple quantization options**: FP8, FP16, BF16 with fast math variants
- **Performance optimizations**: TF32, cuDNN benchmark, Flash Attention support
- **Optional JIT compilation**: torch.compile with persistent cache for faster inference
- **Standalone fallback**: Works even if ComfyUI core modules are unavailable

## Installation

1. Navigate to your ComfyUI custom nodes directory:
```bash
cd ComfyUI/custom_nodes
```

2. Clone this repository:
```bash
git clone https://github.com/eddyhhlure1Eddy/ComfyUI-QwenImageWrapper.git
```

3. Restart ComfyUI

## Usage

### Basic Setup

1. Add the "Eddy Qwen Image + BlockSwap" node to your workflow
2. Load your Qwen-Image model files:
   - UNet (diffusion model)
   - CLIP (text encoder)
   - VAE
3. Enter your prompts and generate

### Node Parameters

**Model Loading:**
- `unet_name`: Diffusion model checkpoint
- `clip_name`: Text encoder model
- `vae_name`: VAE model
- `quantization_dtype`: Weight precision (fp8_e4m3fn recommended for best speed/quality balance)

**LoRA Support:**
- `lora_1-4_name`: LoRA model selection (set to "none" to disable)
- `lora_1-4_strength`: LoRA strength (0 = no effect, 1.0 = full strength)

**Memory Optimization:**
- `use_blockswap`: Enable CPU/GPU memory swapping (reduces VRAM usage)
- `blockswap_use_recommended`: Use optimal block count based on model size
- `blockswap_blocks`: Manual block count (only if use_recommended is disabled)
- `blockswap_model_size`: Model size hint for auto-detection

**Performance Optimization:**
- `enable_matmul_optimization`: TF32, FP16/BF16 fast math (always recommended)
- `enable_flash_attention`: Optimized attention kernel (always recommended)
- `use_autocast`: Mixed precision for additional speedup
- `use_channels_last`: NHWC memory layout for convolutions

**JIT Compilation (Advanced):**
- `use_torch_compile`: Enable torch.compile (slow first run, cached afterward)
- `compile_mode`: Optimization level (default/reduce-overhead/max-autotune)
- `enable_kv_cache`: Cache compiled model for instant second startup

**Generation Settings:**
- Standard sampling parameters (steps, cfg, sampler, scheduler, seed, etc.)

## Quantization Options

| Type | Speed | VRAM Usage | Quality | Notes |
|------|-------|------------|---------|-------|
| default | 1.0x | 100% | Best | No quantization |
| fp16 | 1.8x | 50% | High | Standard half precision |
| fp16_fast | 2.3x | 50% | High | FP16 with fast accumulation |
| bf16 | 1.9x | 50% | High | Better numerical stability |
| bf16_fast | 2.5x | 50% | High | BF16 with fast accumulation (recommended) |
| fp8_e4m3fn | 3.5x | 25% | Good | Fastest option |
| fp8_e5m2 | 3.3x | 25% | Good | Alternative FP8 format |

## Recommended Settings

**For best quality:**
```
quantization_dtype: bf16_fast
use_blockswap: True
enable_matmul_optimization: True
enable_flash_attention: True
use_torch_compile: False
```

**For maximum speed:**
```
quantization_dtype: fp8_e4m3fn
use_blockswap: True
enable_matmul_optimization: True
enable_flash_attention: True
use_torch_compile: True (slow first time)
compile_mode: reduce-overhead
enable_kv_cache: True
```

**For low VRAM (12GB or less):**
```
quantization_dtype: fp8_e4m3fn
use_blockswap: True
blockswap_blocks: 30
blockswap_use_recommended: False
```

## Tooltips

Hover your mouse over any parameter in ComfyUI to see detailed descriptions and recommendations.

## Performance Notes

- **First run with torch.compile**: Takes 1-3 minutes to compile, but subsequent runs are instant with cache
- **BlockSwap**: Minimal speed impact with pinned memory optimization
- **Quantization**: FP8 provides best VRAM savings with acceptable quality loss
- **Autocast**: Can be combined with quantization for additional speedup

## Troubleshooting

**Out of memory errors:**
- Enable BlockSwap
- Increase `blockswap_blocks` value
- Use fp8 quantization
- Enable autocast

**Slow first startup:**
- Disable `use_torch_compile` for instant startup
- Or wait once for compilation, then enjoy fast cached loads

**Quality issues:**
- Use bf16_fast instead of fp8
- Disable autocast
- Increase matmul_precision to "highest"

## Technical Details

**Included optimizations:**
- TF32 tensor cores acceleration
- cuDNN benchmark auto-tuning
- FP16/BF16 reduced precision accumulation
- Flash Attention via scaled_dot_product_attention
- Channels last memory format
- Pinned memory for CPU/GPU transfers
- Persistent compilation cache

**Fallback mode:**
- Contains bundled `comfy_core` modules
- Automatically used if ComfyUI core modules are unavailable
- Ensures node works even with incomplete installations

## Requirements

- ComfyUI
- PyTorch 2.0+
- CUDA-capable GPU (for best performance)
- Python 3.8+

## License

Apache License 2.0

## Credits

Author: eddy

Based on Qwen-Image model architecture with integrated optimizations for ComfyUI.
