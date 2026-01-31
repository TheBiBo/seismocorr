# seismocorr/visualization/plugins/vel1d.py
from __future__ import annotations

from typing import Any, Optional
import numpy as np

from ..types import Plugin, PlotSpec, Layer, Param


def _v_structure_1d(thickness: np.ndarray, velocity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    将分层模型(thickness, velocity)转换为阶梯状(step)折线点集：
    x = velocity, y = depth（深度向下增加）
    """
    thickness = np.asarray(thickness, dtype=float).reshape(-1)
    velocity = np.asarray(velocity, dtype=float).reshape(-1)

    if thickness.ndim != 1 or velocity.ndim != 1:
        raise ValueError("thickness 和 velocity 必须是一维数组")
    if thickness.shape[0] != velocity.shape[0]:
        raise ValueError("thickness 和 velocity 的长度必须一致")
    if np.any(thickness < 0):
        raise ValueError("thickness 不能为负数")

    depth_edges = np.concatenate([[0.0], np.cumsum(thickness)])  # (n+1,)
    n = thickness.shape[0]

    # 构造阶梯折线点（水平段 + 层间垂直跳变）
    xs = [float(velocity[0])]
    ys = [0.0]
    for i in range(n):
        # 水平到本层底界
        xs.append(float(velocity[i]))
        ys.append(float(depth_edges[i + 1]))

        # 层间速度跳变（垂直线）
        if i < n - 1:
            xs.append(float(velocity[i + 1]))
            ys.append(float(depth_edges[i + 1]))

    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def _build_vel1d(
    data: Any,
    *,
    title: Optional[str] = None,
    x_label: str = "Velocity",
    y_label: str = "Depth",
    x_lim: Optional[list[float]] = None,
    y_lim: Optional[list[float]] = None,
    colors: Optional[list[str]] = None,
    labels: Optional[list[str]] = None,
    invert_y: bool = True,  # 深度向下：True 表示 y 轴反向
) -> PlotSpec:
    """
    1D 速度结构绘制插件（支持多条曲线）
    输入 data 约定为 2D array:
        - shape (n, 2):  第一列为层厚 thickness，第二列为 velocity（单条）
        - shape (n, 1+m): 第一列为层厚 thickness，后 m 列为不同曲线的 velocity（多条）
    """

    arr = np.asarray(data, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise TypeError("data 必须是二维数组，且至少两列：[thickness, velocity]")

    thickness = arr[:, 0]
    vel_mat = arr[:, 1:]  # (n, m) 或 (n, 1)
    if vel_mat.ndim == 1:
        vel_mat = vel_mat.reshape(-1, 1)

    n_layers, n_lines = vel_mat.shape

    if colors is None or len(colors) == 0:
        colors = ["black"]
    if labels is None or len(labels) == 0:
        labels = ["vel"]

    line_layers: list[Layer] = []
    all_x, all_y = [], []

    for i in range(n_lines):
        v = vel_mat[:, i]
        xs, ys = _v_structure_1d(thickness, v)

        all_x.append(xs)
        all_y.append(ys)

        color = colors[min(i, len(colors) - 1)]
        label = labels[min(i, len(labels) - 1)]

        line_layers.append(
            Layer(
                type="lines",
                data={"x": xs, "y": ys},
                style={"linewidth": 2, "color": color},
                name=label,
            )
        )

    all_xc = np.concatenate(all_x) if all_x else np.asarray([], dtype=float)
    all_yc = np.concatenate(all_y) if all_y else np.asarray([], dtype=float)

    # 轴范围
    if x_lim is None:
        if all_xc.size:
            x_lim = [float(np.nanmin(all_xc)), float(np.nanmax(all_xc))]
        else:
            x_lim = [0.0, 1.0]

    if y_lim is None:
        if all_yc.size:
            y_min, y_max = float(np.nanmin(all_yc)), float(np.nanmax(all_yc))
            y_lim = [y_max, y_min] if invert_y else [y_min, y_max]
        else:
            y_lim = [1.0, 0.0] if invert_y else [0.0, 1.0]
    else:
        y_lim = [float(y_lim[0]), float(y_lim[1])]

    layout = {
        "title": title or "1D Velocity Structure",
        "x_label": x_label,
        "y_label": y_label,
        "x_lim": [float(x_lim[0]), float(x_lim[1])],
        "y_lim": [float(y_lim[0]), float(y_lim[1])],
        "invert_y": bool(invert_y),
    }

    return PlotSpec(plot_id="vel1d_plot", layers=line_layers, layout=layout)


PLUGINS = [
    Plugin(
        id="vel1d",
        title="绘制 1D 速度结构",
        build=_build_vel1d,
        default_layout={"figsize": (7, 9)},
        data_spec={
            "type": "2d-array",
            "required_keys": None,
            "description": "2D array: 第一列为层厚 thickness，后续列为 velocity（可多条）",
        },
        params={
            "title": Param("str", "1D Velocity Structure", "图标题"),
            "x_label": Param("str", "Velocity", "x轴标签"),
            "y_label": Param("str", "Depth", "y轴标签"),
            "x_lim": Param("list[float]", None, "x轴范围：[xmin, xmax]"),
            "y_lim": Param("list[float]", None, "y轴范围：[ymin, ymax]（深度向下时可传 [max, min]）"),
            "colors": Param("list[str]", ["black"], "线条颜色(不够时使用最后一个)"),
            "labels": Param("list[str]", ["vel"], "线条label(不够时使用最后一个)"),
            "invert_y": Param("bool", True, "深度向下：y轴反向"),
        },
    )
]
