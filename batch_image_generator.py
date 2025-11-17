"""
Batch Image Generator Node - Generate multiple images with preset prompts
Author: eddy
Date: 2025-11-16
"""

import torch
import comfy.samplers
import comfy.sample
import nodes


class BatchImageGenerator:
    """
    Batch Image Generator - Generate 1-16 images with preset prompts
    All-in-one node: prompts → images
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "base_prompt": ("STRING", {
                    "multiline": True,
                    "default": "Asian woman, professional photo, high quality, detailed",
                    "tooltip": "Base prompt for all variants"
                }),
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": "low quality, blurry, bad anatomy",
                    "tooltip": "Negative prompt"
                }),
                "width": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 8}),
                "height": ("INT", {"default": 512, "min": 64, "max": 8192, "step": 8}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 200}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 30.0, "step": 0.5}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "denoise": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Denoise strength (1.0=txt2img, 0.5-0.8=img2img)"}),
                
                # Preset prompts with enables
                "prompt_1": ("STRING", {
                    "multiline": True,
                    "default": "face close-up portrait, front view, eye contact, white background"
                }),
                "enable_1": ("BOOLEAN", {"default": True}),
                
                "prompt_2": ("STRING", {
                    "multiline": True,
                    "default": "side profile portrait, 90 degree angle, clean background"
                }),
                "enable_2": ("BOOLEAN", {"default": True}),
                
                "prompt_3": ("STRING", {
                    "multiline": True,
                    "default": "3/4 view portrait, slightly turned, soft lighting"
                }),
                "enable_3": ("BOOLEAN", {"default": True}),
                
                "prompt_4": ("STRING", {
                    "multiline": True,
                    "default": "turning head to the left, side glance, dynamic pose"
                }),
                "enable_4": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "image": ("IMAGE", {
                    "tooltip": "Input image for img2img mode (optional, if not provided uses txt2img)"
                }),
                "prompt_5": ("STRING", {
                    "multiline": True,
                    "default": "turning head to the right, looking over shoulder"
                }),
                "enable_5": ("BOOLEAN", {"default": False}),
                "prompt_6": ("STRING", {
                    "multiline": True,
                    "default": "back view, looking back over shoulder, rear angle"
                }),
                "enable_6": ("BOOLEAN", {"default": False}),
                "prompt_7": ("STRING", {
                    "multiline": True,
                    "default": "full back view, rear portrait, hair visible"
                }),
                "enable_7": ("BOOLEAN", {"default": False}),
                "prompt_8": ("STRING", {
                    "multiline": True,
                    "default": "upper body portrait, waist up, standing pose"
                }),
                "enable_8": ("BOOLEAN", {"default": False}),
                "prompt_9": ("STRING", {
                    "multiline": True,
                    "default": "full body standing, hands at sides, neutral pose"
                }),
                "enable_9": ("BOOLEAN", {"default": False}),
                "prompt_10": ("STRING", {
                    "multiline": True,
                    "default": "head tilted, looking up, chin raised"
                }),
                "enable_10": ("BOOLEAN", {"default": False}),
                "prompt_11": ("STRING", {
                    "multiline": True,
                    "default": "head tilted down, looking down, contemplative"
                }),
                "enable_11": ("BOOLEAN", {"default": False}),
                "prompt_12": ("STRING", {
                    "multiline": True,
                    "default": "extreme close-up, facial features detail, macro shot"
                }),
                "enable_12": ("BOOLEAN", {"default": False}),
                "prompt_13": ("STRING", {
                    "multiline": True,
                    "default": "45 degree angle from above, top-down view"
                }),
                "enable_13": ("BOOLEAN", {"default": False}),
                "prompt_14": ("STRING", {
                    "multiline": True,
                    "default": "low angle shot, looking up at subject, dramatic perspective"
                }),
                "enable_14": ("BOOLEAN", {"default": False}),
                "prompt_15": ("STRING", {
                    "multiline": True,
                    "default": "walking pose, motion, dynamic movement"
                }),
                "enable_15": ("BOOLEAN", {"default": False}),
                "prompt_16": ("STRING", {
                    "multiline": True,
                    "default": "sitting pose, seated portrait, relaxed posture"
                }),
                "enable_16": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "generate_images"
    CATEGORY = "Qwen/Batch Generation"
    
    def generate_images(self, model, clip, vae, base_prompt, negative_prompt,
                       width, height, steps, cfg, sampler_name, scheduler, seed, denoise,
                       prompt_1, enable_1, prompt_2, enable_2,
                       prompt_3, enable_3, prompt_4, enable_4,
                       image=None,
                       prompt_5="", enable_5=False,
                       prompt_6="", enable_6=False,
                       prompt_7="", enable_7=False,
                       prompt_8="", enable_8=False,
                       prompt_9="", enable_9=False,
                       prompt_10="", enable_10=False,
                       prompt_11="", enable_11=False,
                       prompt_12="", enable_12=False,
                       prompt_13="", enable_13=False,
                       prompt_14="", enable_14=False,
                       prompt_15="", enable_15=False,
                       prompt_16="", enable_16=False):
        """
        Generate multiple images with different prompts
        """
        # Collect enabled prompts
        all_prompts = [
            (prompt_1, enable_1), (prompt_2, enable_2),
            (prompt_3, enable_3), (prompt_4, enable_4),
            (prompt_5, enable_5), (prompt_6, enable_6),
            (prompt_7, enable_7), (prompt_8, enable_8),
            (prompt_9, enable_9), (prompt_10, enable_10),
            (prompt_11, enable_11), (prompt_12, enable_12),
            (prompt_13, enable_13), (prompt_14, enable_14),
            (prompt_15, enable_15), (prompt_16, enable_16),
        ]
        
        # Build prompt list
        prompts = []
        for prompt, enabled in all_prompts:
            if enabled and prompt.strip():
                combined = f"{base_prompt}, {prompt}" if base_prompt.strip() else prompt
                prompts.append(combined)
        
        if not prompts:
            print("[BatchImageGen] Warning: No prompts enabled, using base only")
            prompts = [base_prompt]
        
        num_images = len(prompts)
        
        # Determine mode
        use_img2img = image is not None
        mode = "img2img" if use_img2img else "txt2img"
        print(f"\n[BatchImageGen] Mode: {mode}, generating {num_images} images...")
        
        # Encode all prompts
        clip_encoder = nodes.CLIPTextEncode()
        positive_conds = []
        negative_cond = clip_encoder.encode(clip, negative_prompt)[0]
        
        for i, prompt in enumerate(prompts, 1):
            print(f"  [{i}/{num_images}] {prompt[:60]}...")
            positive_cond = clip_encoder.encode(clip, prompt)[0]
            positive_conds.append(positive_cond)
        
        # Prepare latent source
        if use_img2img:
            # Img2img mode: encode input image
            vae_encoder = nodes.VAEEncode()
            print(f"  Using img2img mode with denoise={denoise}")
            # Use first image if batch
            input_image = image[0:1] if image.shape[0] > 1 else image
            base_latent = vae_encoder.encode(vae, input_image)[0]
        else:
            # Txt2img mode: create empty latent
            empty_latent = nodes.EmptyLatentImage()
            print(f"  Using txt2img mode")
        
        # Generate images
        ksampler = nodes.KSampler()
        vae_decoder = nodes.VAEDecode()
        
        generated_images = []
        
        for i, positive_cond in enumerate(positive_conds, 1):
            print(f"  Generating image {i}/{num_images}...")
            
            # Get latent
            if use_img2img:
                latent = base_latent
            else:
                latent = empty_latent.generate(width, height, 1)[0]
            
            # Sample
            current_seed = seed + i - 1  # Different seed for each image
            sampled_latent = ksampler.sample(
                model, current_seed, steps, cfg, sampler_name, scheduler,
                positive_cond, negative_cond, latent, denoise
            )[0]
            
            # Decode
            decoded_image = vae_decoder.decode(vae, sampled_latent)[0]
            generated_images.append(decoded_image)
        
        # Concatenate all images along batch dimension
        result = torch.cat(generated_images, dim=0)
        
        print(f"[BatchImageGen] ✓ Generated {num_images} images, shape: {result.shape}")
        
        return (result,)


# Node registration
NODE_CLASS_MAPPINGS = {
    "eddy_batch_image_generator": BatchImageGenerator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "eddy_batch_image_generator": "Eddy Batch Image Generator",
}
