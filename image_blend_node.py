"""
Image Blend Node - Dynamic Multi-Image Blending
Supports dynamic input of multiple images with various blend modes
Author: eddy
Date: 2025-11-16
"""

import torch
import numpy as np


class ImageBlendNode:
    """
    Image Blend Node - Supports dynamic blending of 2-8 images
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "blend_mode": ([
                    "grid",             # Grid layout (concat images)
                    "average",           # Average blend
                    "weighted",          # Weighted blend
                    "add",              # Add
                    "multiply",         # Multiply
                    "screen",           # Screen
                    "overlay",          # Overlay
                    "lighten",          # Lighten
                    "darken",           # Darken
                    "difference",       # Difference
                    "max",              # Maximum
                    "min",              # Minimum
                ], {
                    "default": "grid",
                    "tooltip": "Blend mode: grid=concat images in grid layout, others=pixel blending"
                }),
                "grid_layout": (["auto", "horizontal", "vertical", "2x2", "3x3", "2x4"], {
                    "default": "auto",
                    "tooltip": "Grid layout: auto=detect from image count, horizontal=1xN, vertical=Nx1, 2x2/3x3/2x4=fixed grid"
                }),
                "resize_mode": ([
                    "stretch",          # Stretch to uniform size
                    "crop",             # Crop to minimum size
                    "pad",              # Pad to maximum size
                ], {
                    "default": "stretch",
                    "tooltip": "Resize mode"
                }),
                "clamp_output": ("BOOLEAN", {"default": True, "tooltip": "Clamp output values to [0,1] range"}),
                "image_1": ("IMAGE", {"tooltip": "First input image"}),
                "blend_factor_1": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1, "tooltip": "Blend factor for image 1"}),
                "image_2": ("IMAGE", {"tooltip": "Second input image"}),
                "blend_factor_2": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1, "tooltip": "Blend factor for image 2"}),
            },
            "optional": {
                "image_3": ("IMAGE", {"tooltip": "Third image (optional)"}),
                "blend_factor_3": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1, "tooltip": "Blend factor for image 3"}),
                "image_4": ("IMAGE", {"tooltip": "Fourth image (optional)"}),
                "blend_factor_4": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1, "tooltip": "Blend factor for image 4"}),
                "image_5": ("IMAGE", {"tooltip": "Fifth image (optional)"}),
                "blend_factor_5": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1, "tooltip": "Blend factor for image 5"}),
                "image_6": ("IMAGE", {"tooltip": "Sixth image (optional)"}),
                "blend_factor_6": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1, "tooltip": "Blend factor for image 6"}),
                "image_7": ("IMAGE", {"tooltip": "Seventh image (optional)"}),
                "blend_factor_7": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1, "tooltip": "Blend factor for image 7"}),
                "image_8": ("IMAGE", {"tooltip": "Eighth image (optional)"}),
                "blend_factor_8": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1, "tooltip": "Blend factor for image 8"}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("blended_image",)
    FUNCTION = "blend_images"
    CATEGORY = "Qwen/Image Processing"
    
    def normalize_sizes(self, images, mode="stretch"):
        """
        Normalize all image sizes
        
        Args:
            images: List of images [B, H, W, C]
            mode: Size processing mode
        
        Returns:
            List of images with normalized sizes
        """
        if len(images) == 0:
            return images
        
        # Get all image sizes
        sizes = [(img.shape[1], img.shape[2]) for img in images]  # (H, W)
        
        if mode == "stretch":
            # Stretch to first image size
            target_h, target_w = sizes[0]
        elif mode == "crop":
            # Crop to minimum size
            target_h = min(h for h, w in sizes)
            target_w = min(w for h, w in sizes)
        elif mode == "pad":
            # Pad to maximum size
            target_h = max(h for h, w in sizes)
            target_w = max(w for h, w in sizes)
        else:
            target_h, target_w = sizes[0]
        
        normalized_images = []
        for img in images:
            b, h, w, c = img.shape
            
            if (h, w) == (target_h, target_w):
                normalized_images.append(img)
                continue
            
            if mode == "stretch":
                # Use bilinear interpolation to stretch
                img_nchw = img.permute(0, 3, 1, 2)  # [B, C, H, W]
                img_resized = torch.nn.functional.interpolate(
                    img_nchw,
                    size=(target_h, target_w),
                    mode='bilinear',
                    align_corners=False
                )
                img_nhwc = img_resized.permute(0, 2, 3, 1)  # [B, H, W, C]
                normalized_images.append(img_nhwc)
            
            elif mode == "crop":
                # Center crop
                start_h = (h - target_h) // 2
                start_w = (w - target_w) // 2
                img_cropped = img[:, start_h:start_h+target_h, start_w:start_w+target_w, :]
                normalized_images.append(img_cropped)
            
            elif mode == "pad":
                # Center pad (pad with black)
                pad_h = target_h - h
                pad_w = target_w - w
                pad_top = pad_h // 2
                pad_bottom = pad_h - pad_top
                pad_left = pad_w // 2
                pad_right = pad_w - pad_left
                
                img_padded = torch.nn.functional.pad(
                    img.permute(0, 3, 1, 2),  # [B, C, H, W]
                    (pad_left, pad_right, pad_top, pad_bottom),
                    mode='constant',
                    value=0
                )
                img_nhwc = img_padded.permute(0, 2, 3, 1)  # [B, H, W, C]
                normalized_images.append(img_nhwc)
        
        return normalized_images
    
    def blend_average(self, images):
        """Average blend"""
        return torch.mean(torch.stack(images), dim=0)
    
    def blend_weighted(self, images, weights):
        """Weighted blend"""
        total_weight = sum(weights)
        if total_weight == 0:
            return self.blend_average(images)
        
        weighted_sum = torch.zeros_like(images[0])
        for img, weight in zip(images, weights):
            weighted_sum += img * (weight / total_weight)
        
        return weighted_sum
    
    def blend_add(self, images):
        """Add blend"""
        result = images[0].clone()
        for img in images[1:]:
            result = result + img
        return result
    
    def blend_multiply(self, images):
        """Multiply blend"""
        result = images[0].clone()
        for img in images[1:]:
            result = result * img
        return result
    
    def blend_screen(self, images):
        """Screen blend: 1 - (1-a) * (1-b)"""
        result = images[0].clone()
        for img in images[1:]:
            result = 1.0 - (1.0 - result) * (1.0 - img)
        return result
    
    def blend_overlay(self, images):
        """Overlay blend"""
        result = images[0].clone()
        for img in images[1:]:
            # overlay: if base < 0.5: 2*a*b, else: 1 - 2*(1-a)*(1-b)
            mask = result < 0.5
            result = torch.where(
                mask,
                2.0 * result * img,
                1.0 - 2.0 * (1.0 - result) * (1.0 - img)
            )
        return result
    
    def blend_lighten(self, images):
        """Lighten blend (take maximum)"""
        result = images[0].clone()
        for img in images[1:]:
            result = torch.maximum(result, img)
        return result
    
    def blend_darken(self, images):
        """Darken blend (take minimum)"""
        result = images[0].clone()
        for img in images[1:]:
            result = torch.minimum(result, img)
        return result
    
    def blend_difference(self, images):
        """Difference blend"""
        result = images[0].clone()
        for img in images[1:]:
            result = torch.abs(result - img)
        return result
    
    def blend_max(self, images):
        """Max blend"""
        return torch.max(torch.stack(images), dim=0)[0]
    
    def blend_min(self, images):
        """Min blend"""
        return torch.min(torch.stack(images), dim=0)[0]
    
    def blend_grid(self, images, grid_layout="auto"):
        """
        Concat images in grid layout
        
        Args:
            images: List of images [B, H, W, C]
            grid_layout: Grid layout mode
        
        Returns:
            Concatenated image in grid layout
        """
        num_images = len(images)
        
        # Determine grid layout
        if grid_layout == "auto":
            if num_images == 2:
                rows, cols = 1, 2
            elif num_images == 3:
                rows, cols = 1, 3
            elif num_images == 4:
                rows, cols = 2, 2
            elif num_images <= 6:
                rows, cols = 2, 3
            elif num_images <= 9:
                rows, cols = 3, 3
            else:
                rows, cols = 3, 3
                print(f"[ImageBlend] Warning: {num_images} images exceed 3x3 grid, using first 9")
                images = images[:9]
                num_images = 9
        elif grid_layout == "horizontal":
            rows, cols = 1, num_images
        elif grid_layout == "vertical":
            rows, cols = num_images, 1
        elif grid_layout == "2x2":
            rows, cols = 2, 2
            if num_images > 4:
                images = images[:4]
                num_images = 4
        elif grid_layout == "3x3":
            rows, cols = 3, 3
            if num_images > 9:
                images = images[:9]
                num_images = 9
        elif grid_layout == "2x4":
            rows, cols = 2, 4
            if num_images > 8:
                images = images[:8]
                num_images = 8
        else:
            rows, cols = 2, 2
        
        # Pad with black images if needed
        cells_needed = rows * cols
        if num_images < cells_needed:
            print(f"[ImageBlend] Padding {cells_needed - num_images} empty cells for {rows}x{cols} grid")
            # Create black image with same size as first image
            black_img = torch.zeros_like(images[0])
            while len(images) < cells_needed:
                images.append(black_img)
        
        # Get cell size (all images already normalized to same size)
        cell_h, cell_w = images[0].shape[1], images[0].shape[2]
        
        # Create grid
        grid_rows = []
        for r in range(rows):
            row_images = []
            for c in range(cols):
                idx = r * cols + c
                if idx < len(images):
                    row_images.append(images[idx])
            # Concatenate along width (dim=2)
            row_concat = torch.cat(row_images, dim=2)
            grid_rows.append(row_concat)
        
        # Concatenate rows along height (dim=1)
        result = torch.cat(grid_rows, dim=1)
        
        print(f"[ImageBlend] Grid layout: {rows}x{cols}, cell size: {cell_h}x{cell_w}, output: {result.shape}")
        return result
    
    def blend_images(self, blend_mode, grid_layout, resize_mode, clamp_output,
                    image_1, blend_factor_1, image_2, blend_factor_2,
                    image_3=None, blend_factor_3=1.0,
                    image_4=None, blend_factor_4=1.0,
                    image_5=None, blend_factor_5=1.0,
                    image_6=None, blend_factor_6=1.0,
                    image_7=None, blend_factor_7=1.0,
                    image_8=None, blend_factor_8=1.0):
        """
        Blend multiple images
        
        Args:
            blend_mode: Blend mode
            resize_mode: Resize mode
            clamp_output: Whether to clamp output range
            image_1, image_2: Required input images
            blend_factor_1~8: Blend factors (weights for weighted mode)
            image_3~8: Optional input images
        
        Returns:
            Blended image
        """
        # Collect all input images
        images = [image_1, image_2]
        blend_factors = [blend_factor_1, blend_factor_2]
        
        optional_images = [image_3, image_4, image_5, image_6, image_7, image_8]
        optional_factors = [blend_factor_3, blend_factor_4, blend_factor_5, blend_factor_6, blend_factor_7, blend_factor_8]
        
        for img, factor in zip(optional_images, optional_factors):
            if img is not None:
                images.append(img)
                blend_factors.append(factor)
        
        print(f"[ImageBlend] Processing {len(images)} images, mode: {blend_mode}, resize: {resize_mode}")
        
        # Normalize image sizes
        normalized_images = self.normalize_sizes(images, resize_mode)
        
        # Execute blending based on mode
        if blend_mode == "grid":
            result = self.blend_grid(normalized_images, grid_layout)
        elif blend_mode == "average":
            result = self.blend_average(normalized_images)
        elif blend_mode == "weighted":
            result = self.blend_weighted(normalized_images, blend_factors[:len(images)])
        elif blend_mode == "add":
            result = self.blend_add(normalized_images)
        elif blend_mode == "multiply":
            result = self.blend_multiply(normalized_images)
        elif blend_mode == "screen":
            result = self.blend_screen(normalized_images)
        elif blend_mode == "overlay":
            result = self.blend_overlay(normalized_images)
        elif blend_mode == "lighten":
            result = self.blend_lighten(normalized_images)
        elif blend_mode == "darken":
            result = self.blend_darken(normalized_images)
        elif blend_mode == "difference":
            result = self.blend_difference(normalized_images)
        elif blend_mode == "max":
            result = self.blend_max(normalized_images)
        elif blend_mode == "min":
            result = self.blend_min(normalized_images)
        else:
            print(f"[ImageBlend] Unknown blend mode: {blend_mode}, using average blend")
            result = self.blend_average(normalized_images)
        
        # Clamp output range
        if clamp_output:
            result = torch.clamp(result, 0.0, 1.0)
        
        print(f"[ImageBlend] Output shape: {result.shape}, dtype: {result.dtype}, range: [{result.min():.3f}, {result.max():.3f}]")
        
        return (result,)


# Node registration
NODE_CLASS_MAPPINGS = {
    "eddy_image_blend": ImageBlendNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "eddy_image_blend": "Eddy Image Blend",
}
