# seismocorr/visualization/plugins/ccf.py
from __future__ import annotations

from typing import Any, Optional
import numpy as np

from ..types import Plugin, PlotSpec, Layer, Param


def _build_ccf_wiggle(
    data: Any,
    *,
    title: Optional[str] = None,
    normalize: bool = True,
    normalize_method: Optional[str] = None,
    norm_method: str = "trace",
    clip: Optional[float] = None,
    scale: float = 1.0,
    dy: Optional[float] = None,
    excursion: float = 2.0,
    bias: float = 0.0,
    fill_mode: Optional[str] = "pos",
    fill_alpha: float = 0.10,
    trace_step: int = 1,
    ytick_step: int = 5,
    x_label: str = "Lag (s)",
    y_label: str = "Trace index",
    highlights=None,
    sort=None,
    show_zero_line: bool = False,
) -> PlotSpec:
    """
    CCF wiggle 插件：生成 PlotSpec（后端无关）

    data 约定：
      dict:
        - "cc": (n_tr, n_lags)
        - "lags": (n_lags,)
        - "labels": optional list[str], length n_tr
    """
    if not isinstance(data, dict):
        raise TypeError("ccf.wiggle 当前仅支持 dict 输入：{'cc','lags','labels?'}")

    cc = np.asarray(data["cc"], dtype=float)
    lags = np.asarray(data["lags"], dtype=float)
    labels = data.get("labels")

    if cc.ndim != 2:
        raise ValueError("cc 必须是二维矩阵 (n_traces, n_lags)")
    if lags.ndim != 1 or lags.shape[0] != cc.shape[1]:
        raise ValueError("lags 必须是一维，且长度等于 cc.shape[1]")

    if normalize_method is not None:
        normalize_style = normalize_method
    else:
        normalize_style = "max" if normalize else None

    wiggle = Layer(
        type="wiggle",
        data={
            "x": lags,
            "traces": cc,
            "labels": labels,
            "highlights": highlights,
            "sort": sort,
        },
        style={
            "dy": float(scale if dy is None else dy),
            "scale": float(scale),
            "excursion": float(excursion),
            "bias": float(bias),
            "norm_method": str(norm_method),
            "normalize": normalize_style,
            "clip": clip,
            "linewidth": 0.6,
            "alpha": 0.85,
            "color": "k",
            "fill_mode": fill_mode,
            "fill_alpha": float(fill_alpha),
            "ytick_step": int(ytick_step),
            "trace_step": int(trace_step),
            "zero_line": bool(show_zero_line),
        },
        name="CCF",
    )

    layers = [wiggle]

    if show_zero_line:
        v0 = Layer(
            type="vlines",
            data={"xs": [0.0]},
            style={"linewidth": 0.8, "alpha": 0.8},
            name="lag=0",
        )
        layers.append(v0)

    layout = {
        "title": title or "CCF Wiggle",
        "x_label": x_label,
        "y_label": y_label,
    }

    return PlotSpec(plot_id="ccf.wiggle", layers=layers, layout=layout)


PLUGINS = [
    Plugin(
        id="ccf.wiggle",
        title="互相关多道 Wiggle",
        build=_build_ccf_wiggle,
        default_layout={"figsize": (10, 6)},
        data_spec={
            "type": "dict",
            "required_keys": {
                "cc": "2D array, shape (n_tr, n_lags) 互相关矩阵",
                "lags": "1D array, shape (n_lags,) lag/时间轴",
            },
            "optional_keys": {
                "labels": "list[str], length n_tr 道名/台站名（<=60 时显示在 y 轴）",
            },
            "notes": [
                "normalize 控制是否启用归一化；normalize_method 可指定 'max'/'p95'/'rms'",
                "norm_method 可选 'trace'(逐道) 或 'stream'(全局)",
                "excursion 控制单道最大跨越的道间距倍数",
                "fill_mode 支持 variable-area 填充：pos/neg/both",
                "trace_step 可抽稀绘制减少拥挤",
            ],
        },
        params={
            "title": Param("str", None, "图标题"),
            "normalize": Param("bool", True, "是否归一化（默认 True）"),
            "normalize_method": Param("str", None, "归一化幅度统计：'max'/'p95'/'rms'（优先于 normalize）"),
            "norm_method": Param("str", "trace", "归一化范围：'trace' 或 'stream'"),
            "clip": Param("float", None, "归一化后限幅阈值"),
            "scale": Param("float", 1.0, "垂直缩放系数（兼容旧参数）"),
            "dy": Param("float", None, "垂直缩放系数（优先于 scale）"),
            "excursion": Param("float", 2.0, "单道最大跨越道间距倍数"),
            "bias": Param("float", 0.0, "填充阈值偏置（归一化幅度单位）"),
            "fill_mode": Param("str", "pos", "填充方式：None/'pos'/'neg'/'both'"),
            "fill_alpha": Param("float", 0.10, "填充透明度"),
            "trace_step": Param("int", 1, "抽稀绘制步长（每隔 N 道画一条）"),
            "ytick_step": Param("int", 5, "y 轴刻度抽样步长"),
            "x_label": Param("str", "x", "x 轴标签"),
            "y_label": Param("str", "y", "y 轴标签"),
            "show_zero_line": Param("bool", False, "是否显示 x=0 的参考线"),
            "highlights": Param(
                "list[dict]",
                None,
                "高亮配置：对指定道、指定时间段叠加彩色线段。",
                item_schema={
                    "trace": Param("int", None, "道号（0-based，必填）", required=True),
                    "t0": Param("float", None, "起始时间（可选，缺省=全时段）"),
                    "t1": Param("float", None, "终止时间（可选，缺省=全时段）"),
                    "color": Param("str", None, "高亮颜色"),
                    "linewidth": Param("float", None, "高亮线宽"),
                    "alpha": Param("float", 1.0, "高亮透明度"),
                },
            ),
            "sort": Param(
                "dict",
                None,
                "排序/纵轴控制：{'by': 距离数组(n_tr,), 'ascending': True/False, "
                "'y_mode': 'index'|'distance', 'label': 'Distance (km)'}",
            ),
        },
    )
]
