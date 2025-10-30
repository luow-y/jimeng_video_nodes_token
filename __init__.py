"""
即梦AI视频生成节点包（Token版）
支持文生视频和图生视频，直接使用Token连接国内端点
"""

from .jimeng_video_token_node import JimengVideoTokenNode

# 节点类映射
NODE_CLASS_MAPPINGS = {
    "Jimeng_Video_Token": JimengVideoTokenNode,
}

# 节点显示名称映射
NODE_DISPLAY_NAME_MAPPINGS = {
    "Jimeng_Video_Token": "即梦AI视频生成（Token版）",
}

__version__ = "1.0.0"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]



