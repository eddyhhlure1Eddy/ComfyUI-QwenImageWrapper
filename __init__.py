from __future__ import annotations

__author__ = "eddy"

from . import standalone_official_nodes as _standalone
from . import image_blend_node as _blend
from . import batch_prompt_node as _batch
from . import batch_image_generator as _batch_gen

# 合并所有节点映射
NODE_CLASS_MAPPINGS = {
    **_standalone.NODE_CLASS_MAPPINGS,
    **_blend.NODE_CLASS_MAPPINGS,
    **_batch.NODE_CLASS_MAPPINGS,
    **_batch_gen.NODE_CLASS_MAPPINGS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **_standalone.NODE_DISPLAY_NAME_MAPPINGS,
    **_blend.NODE_DISPLAY_NAME_MAPPINGS,
    **_batch.NODE_DISPLAY_NAME_MAPPINGS,
    **_batch_gen.NODE_DISPLAY_NAME_MAPPINGS,
}
