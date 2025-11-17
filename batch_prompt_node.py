"""
Batch Prompt Node - Generate multiple prompts for batch generation
Author: eddy
Date: 2025-11-16
"""


class BatchPromptNode:
    """
    Batch Prompt Node - Output multiple text prompts at once for batch generation
    Enables efficient batch generation with different prompts
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Base prompt appended to all variants (e.g., 'high quality, detailed')"
                }),
                "separator": (["comma", "comma_space", "newline", "none"], {
                    "default": "comma_space",
                    "tooltip": "Separator between base and variant"
                }),
                # First 4 prompts always visible
                "prompt_1": ("STRING", {
                    "multiline": True,
                    "default": "face close-up portrait, front view, eye contact, white background",
                    "tooltip": "Preset 1: Face close-up, front view"
                }),
                "enable_1": ("BOOLEAN", {"default": True, "tooltip": "Enable variant 1"}),
                
                "prompt_2": ("STRING", {
                    "multiline": True,
                    "default": "side profile portrait, 90 degree angle, clean background",
                    "tooltip": "Preset 2: Side profile"
                }),
                "enable_2": ("BOOLEAN", {"default": True, "tooltip": "Enable variant 2"}),
                
                "prompt_3": ("STRING", {
                    "multiline": True,
                    "default": "3/4 view portrait, slightly turned, soft lighting",
                    "tooltip": "Preset 3: 3/4 view"
                }),
                "enable_3": ("BOOLEAN", {"default": True, "tooltip": "Enable variant 3"}),
                
                "prompt_4": ("STRING", {
                    "multiline": True,
                    "default": "turning head to the left, side glance, dynamic pose",
                    "tooltip": "Preset 4: Turning head left"
                }),
                "enable_4": ("BOOLEAN", {"default": True, "tooltip": "Enable variant 4"}),
            },
            "optional": {
                # Optional prompts 5-16
                "prompt_5": ("STRING", {
                    "multiline": True, 
                    "default": "turning head to the right, looking over shoulder",
                    "tooltip": "Preset 5: Turning head right"
                }),
                "enable_5": ("BOOLEAN", {"default": False}),
                "prompt_6": ("STRING", {
                    "multiline": True, 
                    "default": "back view, looking back over shoulder, rear angle",
                    "tooltip": "Preset 6: Back view with look back"
                }),
                "enable_6": ("BOOLEAN", {"default": False}),
                "prompt_7": ("STRING", {
                    "multiline": True, 
                    "default": "full back view, rear portrait, hair visible",
                    "tooltip": "Preset 7: Full back view"
                }),
                "enable_7": ("BOOLEAN", {"default": False}),
                "prompt_8": ("STRING", {
                    "multiline": True, 
                    "default": "upper body portrait, waist up, standing pose",
                    "tooltip": "Preset 8: Upper body"
                }),
                "enable_8": ("BOOLEAN", {"default": False}),
                "prompt_9": ("STRING", {
                    "multiline": True, 
                    "default": "full body standing, hands at sides, neutral pose",
                    "tooltip": "Preset 9: Full body standing"
                }),
                "enable_9": ("BOOLEAN", {"default": False}),
                "prompt_10": ("STRING", {
                    "multiline": True, 
                    "default": "head tilted, looking up, chin raised",
                    "tooltip": "Preset 10: Looking up"
                }),
                "enable_10": ("BOOLEAN", {"default": False}),
                "prompt_11": ("STRING", {
                    "multiline": True, 
                    "default": "head tilted down, looking down, contemplative",
                    "tooltip": "Preset 11: Looking down"
                }),
                "enable_11": ("BOOLEAN", {"default": False}),
                "prompt_12": ("STRING", {
                    "multiline": True, 
                    "default": "extreme close-up, facial features detail, macro shot",
                    "tooltip": "Preset 12: Extreme close-up"
                }),
                "enable_12": ("BOOLEAN", {"default": False}),
                "prompt_13": ("STRING", {
                    "multiline": True, 
                    "default": "45 degree angle from above, top-down view",
                    "tooltip": "Preset 13: High angle"
                }),
                "enable_13": ("BOOLEAN", {"default": False}),
                "prompt_14": ("STRING", {
                    "multiline": True, 
                    "default": "low angle shot, looking up at subject, dramatic perspective",
                    "tooltip": "Preset 14: Low angle"
                }),
                "enable_14": ("BOOLEAN", {"default": False}),
                "prompt_15": ("STRING", {
                    "multiline": True, 
                    "default": "walking pose, motion, dynamic movement",
                    "tooltip": "Preset 15: Walking pose"
                }),
                "enable_15": ("BOOLEAN", {"default": False}),
                "prompt_16": ("STRING", {
                    "multiline": True, 
                    "default": "sitting pose, seated portrait, relaxed posture",
                    "tooltip": "Preset 16: Sitting pose"
                }),
                "enable_16": ("BOOLEAN", {"default": False}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("batch_prompts",)
    FUNCTION = "generate_batch_prompts"
    CATEGORY = "Qwen/Batch Generation"
    OUTPUT_IS_LIST = (True,)  # Output as list for batch processing
    
    def generate_batch_prompts(self, base_prompt, separator,
                               prompt_1, enable_1, prompt_2, enable_2,
                               prompt_3, enable_3, prompt_4, enable_4,
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
        Generate batch prompts from enabled variants
        
        Returns:
            List of prompts for batch generation
        """
        # Determine separator
        sep_map = {
            "comma": ",",
            "comma_space": ", ",
            "newline": "\n",
            "none": " "
        }
        sep = sep_map.get(separator, ", ")
        
        # Collect all prompts with their enable flags
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
        
        # Build final prompt list
        final_prompts = []
        for prompt, enabled in all_prompts:
            if enabled and prompt.strip():
                # Combine base + variant
                if base_prompt.strip():
                    combined = f"{base_prompt.strip()}{sep}{prompt.strip()}"
                else:
                    combined = prompt.strip()
                final_prompts.append(combined)
        
        if not final_prompts:
            print("[BatchPrompt] Warning: No prompts enabled, using base prompt only")
            final_prompts = [base_prompt.strip() if base_prompt.strip() else ""]
        
        print(f"[BatchPrompt] Generated {len(final_prompts)} prompts for batch generation:")
        for i, p in enumerate(final_prompts, 1):
            preview = p[:60] + "..." if len(p) > 60 else p
            print(f"  [{i}] {preview}")
        
        return (final_prompts,)


# Node registration
NODE_CLASS_MAPPINGS = {
    "eddy_batch_prompt": BatchPromptNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "eddy_batch_prompt": "Eddy Batch Prompt",
}
