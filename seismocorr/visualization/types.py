# seismocorr/visualization/types.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Literal

# -----------------------------
# 中间表示：Layer / PlotSpec
# -----------------------------

LayerType = Literal[
    "heatmap",
    "lines",
    "wiggle",
    "vlines",
    "annotations",
    "polar_heatmap",
]

@dataclass
class Layer:
    """
    一层绘图元素（后端无关）
    - type: 图层类型，如 wiggle/heatmap/vlines 等
    - data: 绘图数据（后端读取）
    - style: 样式参数（后端读取）
    - name: 用于图例/识别（可选）
    """
    type: LayerType
    data: Dict[str, Any] = field(default_factory=dict)
    style: Dict[str, Any] = field(default_factory=dict)
    name: Optional[str] = None


@dataclass
class PlotSpec:
    """
    一张图的“说明书”（后端无关）
    - plot_id: 插件 ID（便于调试）
    - layers: 多个图层叠加
    - layout: 标题、坐标轴标签、画布大小等（后端会尽量理解）
    """
    plot_id: str
    layers: List[Layer]
    layout: Dict[str, Any] = field(default_factory=dict)


# -----------------------------
# FigureHandle：统一返回
# -----------------------------

BackendName = Literal["mpl", "plotly"]

@dataclass
class FigureHandle:
    """
    plot() 的统一返回对象
    - backend: 'mpl' 或 'plotly'
    - handle: matplotlib Figure 或 plotly Figure
    - spec: 可选，保留 PlotSpec 便于调试/再渲染
    """
    backend: BackendName
    handle: Any
    spec: Optional[PlotSpec] = None
    extra: Optional[dict] = field(default_factory=dict)  # 新增

# -----------------------------
# 方案B：参数元数据（ParamSpec）
# -----------------------------

@dataclass
class Param:
    """
    参数元信息（机器可读）

    字段说明：
    - ptype: 参数类型（字符串形式，便于打印和轻量检查）
      常用：'bool'/'int'/'float'/'str'/'dict'/'list'/'list[dict]'
    - default: 默认值（required=False 且用户未提供则用它）
    - doc: 中文说明
    - required: 是否必填
    - choices: 枚举值（可选）
    - item_schema: list[dict] 时用于描述 dict 的子字段（例如 highlights）
    """
    ptype: str
    default: Any = None
    doc: str = ""
    required: bool = False
    choices: Optional[List[Any]] = None
    item_schema: Optional[Dict[str, "Param"]] = None


ParamSpec = Dict[str, Param]


# -----------------------------
# Plugin：插件定义
# -----------------------------

@dataclass
class Plugin:
    """
    插件：把业务数据 data + kwargs -> PlotSpec（后端无关）

    重要字段：
    - id: 图的唯一标识（用户 plot() 用）
    - title: 展示名称（help/list 用）
    - build: Callable(data, **kwargs) -> PlotSpec
    - params: ParamSpec（方案B：用于 help/默认值/校验）
    - data_spec: dict（新增：用于说明 data 输入格式，不参与 kwargs 校验）
    """
    id: str
    title: str
    build: Callable[..., PlotSpec]

    default_layout: Dict[str, Any] = field(default_factory=dict)
    params: ParamSpec = field(default_factory=dict)

    # ✅新增：data 输入规范（只用于文档/帮助，不参与参数校验）
    data_spec: Dict[str, Any] = field(default_factory=dict)
