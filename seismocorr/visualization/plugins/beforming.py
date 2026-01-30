from __future__ import annotations

from typing import Any, Optional
import numpy as np
import matplotlib.pyplot as plt

from ..types import Plugin, PlotSpec, Layer, Param


def _build_beamforming_polar_heatmap(
    data: Any,
    *,
    title: str = "Beamforming Polar Heatmap",
    slowness_unit: str = "s/km",        # "s/m" 或 "s/km"
    azimuth_convention: str = "math",   # 方位角约定
    ax: Optional[plt.Axes] = None,
) -> PlotSpec:
    """
    Beamforming 极坐标热力图插件：生成 PlotSpec（后端无关）

    data 约定：
      dict:
        - "power": (n_azimuth, n_radius) 波束形成功率矩阵
        - "azimuth_deg": (n_azimuth,) 方位角（度）
        - "slowness_s_per_m": (n_radius,) 慢度（s/m）
    """
    if not isinstance(data, dict):
        raise TypeError("beamforming.polar_heatmap 当前仅支持 dict 输入：{'power', 'azimuth_deg', 'slowness_s_per_m'}")

    power = np.asarray(data["power"], dtype=float)
    azimuth_deg = np.asarray(data["azimuth_deg"], dtype=float)
    slowness_s_per_m = np.asarray(data["slowness_s_per_m"], dtype=float)

    if power.ndim != 2:
        raise ValueError("power 必须是二维矩阵 (n_azimuth, n_radius)")
    if azimuth_deg.ndim != 1 or slowness_s_per_m.ndim != 1:
        raise ValueError("azimuth_deg 和 slowness_s_per_m 必须是一维")

    # 根据 slowness 单位进行处理
    if slowness_unit == "s/km":
        r = slowness_s_per_m * 1000.0  # 转换为 s/km
        radial_title = "Slowness (s/km)"
    elif slowness_unit == "s/m":
        r = slowness_s_per_m
        radial_title = "Slowness (s/m)"
    else:
        raise ValueError('slowness_unit 必须是 "s/m" 或 "s/km"')

    # ============ 传递 theta（方位角）数据 ============  
    # 在极坐标热力图中，theta 是方位角，直接使用 azimuth_deg 作为 theta
    theta = azimuth_deg  # 方位角即为 theta

    # 生成热力图层
    polar_heatmap_layer = Layer(
        type="polar_heatmap",
        data={
            "z": power,
            "theta": theta,  # 将 azimuth_deg 映射为 theta
            "r": r,  # slowness_s_per_m 对应半径 r
        },
        style={
            "title": title,
            "azimuth_convention": azimuth_convention,
        },
        name="Beamforming Polar Heatmap",
    )

    # 返回 PlotSpec 对象
    return PlotSpec(
        plot_id="beamforming.polar_heatmap",
        layers=[polar_heatmap_layer],
        layout={
            "title": title,
            "radial_title": radial_title,
            "x_label": "Azimuth (°)",
            "y_label": radial_title,
        },
    )


PLUGINS = [
    Plugin(
        id="beamforming.polar_heatmap",
        title="Beamforming 极坐标热力图",
        build=_build_beamforming_polar_heatmap,
        default_layout={"figsize": (10, 6)},

        # 插件参数说明
        data_spec={
            "type": "dict",
            "required_keys": {
                "power": "2D array, shape (slowness, azimuth) Beamforming Power Matrix",
                "azimuth_deg": "1D array, shape (n_azimuth,) Azimuth angles (degrees)",
                "slowness_s_per_m": "1D array, shape (n_radius,) Slowness in s/m",
            },
            "notes": [
                "slowness_unit 可以是 s/m 或 s/km",
            ],
        },

        # 参数列表，支持插件配置
        params={
            "title": Param("str", "Beamforming Polar Heatmap", "图标题"),
            "slowness_unit": Param("str", "s/km", '慢度单位，"s/m" 或 "s/km"'),
            "azimuth_convention": Param("str", "math", "方位角约定，可以是数学、地理等"),
            "ax": Param("plt.Axes", None, "matplotlib 的 Axes 对象，可选"),
        },
    )
]
