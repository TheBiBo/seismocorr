from __future__ import annotations

from typing import Any, Literal, Optional
import numpy as np

from ..types import Plugin, PlotSpec, Layer, Param


InterpMethod = Literal["none", "linear", "nearest", "cubic"]


def _interp1d_nearest(x_new: np.ndarray, x_old: np.ndarray, y_old: np.ndarray) -> np.ndarray:
    """
    最近邻 1D 插值（x_old 单调递增）。
    y_old 可以是 (n,) 或 (n, k)；返回匹配 x_new 的 shape。
    """
    x_old = np.asarray(x_old, float)
    x_new = np.asarray(x_new, float)
    y_old = np.asarray(y_old, float)

    idx = np.searchsorted(x_old, x_new, side="left")
    idx = np.clip(idx, 0, x_old.size - 1)

    left = np.clip(idx - 1, 0, x_old.size - 1)
    right = idx

    # 选更近的点
    choose_right = (np.abs(x_old[right] - x_new) <= np.abs(x_new - x_old[left]))
    nn = np.where(choose_right, right, left)

    return y_old[nn]  # 自动广播到后维


def _centers_to_edges(c: np.ndarray) -> np.ndarray:
    c = np.asarray(c, float)
    if c.size < 2:
        dc = 0.5
        return np.array([c[0] - dc, c[0] + dc], dtype=float)
    if np.any(np.diff(c) <= 0):
        raise ValueError("centers 必须严格递增才能转换为 edges。")
    mid = 0.5 * (c[:-1] + c[1:])
    first = c[0] - (mid[0] - c[0])
    last = c[-1] + (c[-1] - mid[-1])
    return np.concatenate([[first], mid, [last]])


def _edges_cover(start: float, stop: float, step: float) -> np.ndarray:
    if step <= 0:
        raise ValueError("step must be > 0")
    e = np.arange(start, stop, step, dtype=float)   # 注意：不把 stop 放进去
    if e.size == 0:
        return np.array([start, stop], dtype=float)
    if e[0] != start:
        e = np.insert(e, 0, start)
    # 强制把 stop 加进去（确保覆盖到边界）
    if e[-1] < stop - 1e-9:
        e = np.append(e, stop)
    else:
        e[-1] = stop  # 如果非常接近，就直接贴边
    return e


def _build_vel2d(
    data: Any,
    *,
    title: str = "2D Velocity Structure",
    x_label: str = "Distance (km)",
    y_label: str = "Depth (m)",
    cmap: str = "jet",
    colorbar_label: str = "Velocity (m/s)",
    invert_y: bool = True,
    interp: bool = True,
    interp_method: InterpMethod = "linear",
    # 插值强度：>1 表示加密（分辨率提高）；=1 表示按原始网格；<1 表示变稀（不推荐）
    interp_strength: float = 3.0,
    # 可选：手动指定目标网格分辨率（优先级高于 interp_strength）
    dx_km: Optional[float] = None,
    dy_m: Optional[float] = None,
) -> PlotSpec:
    """
    data 约定（dict）：
      required:
        - "Velocities": 2D array, shape (n_t, n_d)
        - "thickness":  1D array, shape (n_t,)
        - "distance":   1D array, shape (n_d,)
    """

    # -------- 数据读取与校验 --------
    if not isinstance(data, dict):
        raise TypeError("vel2d 当前仅支持 dict 输入：{'Velocities','thickness','distance'}")
    for k in ("Velocities", "thickness", "distance"):
        if k not in data:
            raise KeyError(f"data 必须包含键：{k}")

    Velocities = np.asarray(data["Velocities"], dtype=float)
    thickness = np.asarray(data["thickness"], dtype=float)
    distance = np.asarray(data["distance"], dtype=float)

    if Velocities.ndim != 2:
        raise ValueError("Velocities 必须是二维矩阵 (n_t, n_d)")
    n_t, n_d = Velocities.shape

    if thickness.ndim != 1 or thickness.shape[0] != n_t:
        raise ValueError("thickness 必须是一维，且长度等于 Velocities.shape[0] (n_t)")
    if distance.ndim != 1 or distance.shape[0] != n_d:
        raise ValueError("distance 必须是一维，且长度等于 Velocities.shape[1] (n_d)")

    if not np.isfinite(Velocities).all():
        raise ValueError("Velocities 包含 NaN/Inf，请先清理。")
    if not np.isfinite(thickness).all() or (thickness <= 0).any():
        raise ValueError("thickness 需为有限且全为正数。")
    if not np.isfinite(distance).all():
        raise ValueError("distance 包含 NaN/Inf，请先清理。")

    # distance 必须单调递增（否则插值会出错）
    if np.any(np.diff(distance) <= 0):
        idx = np.argsort(distance)
        distance = distance[idx]
        Velocities = Velocities[:, idx]

    # -------- 深度坐标（由 thickness 累积得到）--------
    depth_edges = np.concatenate([[0.0], np.cumsum(thickness)])          # (n_t+1,)
    depth_centers = 0.5 * (depth_edges[:-1] + depth_edges[1:])           # (n_t,)
    max_depth = float(depth_edges[-1])

    # -------- 深度头尾补齐：把 y_old 扩展到 [0, ..., max_depth] --------
    # 这样 y_new 可以从 0 到 max_depth，不会在顶部/底部空一截
    # z_old 同步补两行：首层/末层复制
    y_support = np.concatenate([[0.0], depth_centers, [max_depth]])      # (n_t+2,)
    z_support = np.vstack([Velocities[0:1, :], Velocities, Velocities[-1:, :]])  # (n_t+2, n_d)

    # -------- 插值参数解析 --------
    if interp_method not in ("none", "linear", "nearest", "cubic"):
        raise ValueError("interp_method 必须是 'none'|'linear'|'nearest'|'cubic'")

    if not interp or interp_method == "none":
        distance_plot = distance
        depth_plot = depth_centers  # 不插值时保留中心；你也可以改成 depth_edges 看渲染端需求
        vel_plot = Velocities
    else:
        if interp_strength <= 0:
            raise ValueError("interp_strength 必须为正数（建议 > 1 用于加密）。")

        # 基于原始网格的“典型步长”
        # distance 可能不等距，用 median(diff) 比较稳
        base_dx = float(np.median(np.diff(distance))) if distance.size > 1 else 1.0
        base_dy = float(np.median(np.diff(depth_centers))) if depth_centers.size > 1 else max_depth / max(n_t, 1)

        # 若用户未手动指定 dx/dy，则用强度推导
        dx = float(dx_km) if dx_km is not None else max(base_dx / interp_strength, 1e-9)
        dy = float(dy_m) if dy_m is not None else max(base_dy / interp_strength, 1e-9)

        x_new = np.arange(float(distance.min()), float(distance.max()) + 1e-12, dx)
        y_new = np.arange(0.0, max_depth + 1e-12, dy)  # 注意：从 0 到 max_depth，补齐头尾

        x_old = distance
        y_old = y_support
        z_old = z_support

        # 极端退化：无法插值
        if x_old.size < 2 or y_old.size < 2:
            distance_plot, depth_plot, vel_plot = x_old, y_old, z_old
        else:
            method = interp_method

            # ===== 方法 1/2：纯 numpy 的 linear / nearest（两步 1D）=====
            if method in ("linear", "nearest"):
                # (1) 沿 x 插值：对每一行（固定深度）插到 x_new
                z_x = np.empty((z_old.shape[0], x_new.size), dtype=float)
                for i in range(z_old.shape[0]):
                    if method == "linear":
                        z_x[i, :] = np.interp(x_new, x_old, z_old[i, :])
                    else:
                        z_x[i, :] = _interp1d_nearest(x_new, x_old, z_old[i, :])

                # (2) 沿 y 插值：对每一列（固定距离）插到 y_new
                z_xy = np.empty((y_new.size, x_new.size), dtype=float)
                for j in range(x_new.size):
                    if method == "linear":
                        z_xy[:, j] = np.interp(y_new, y_old, z_x[:, j])
                    else:
                        z_xy[:, j] = _interp1d_nearest(y_new, y_old, z_x[:, j])
                
                x_edges = _centers_to_edges(x_new)     # (nx+1,)
                y_edges = _centers_to_edges(y_new)     # (ny+1,)

                distance_plot, depth_plot, vel_plot = x_edges, y_edges, z_xy

            # ===== 方法 3：cubic（优先用 SciPy，缺失则回退 linear）=====
            else:  # method == "cubic"
                try:
                    from scipy.interpolate import RegularGridInterpolator  # type: ignore
                except Exception:
                    # SciPy 不可用：回退线性
                    z_x = np.empty((z_old.shape[0], x_new.size), dtype=float)
                    for i in range(z_old.shape[0]):
                        z_x[i, :] = np.interp(x_new, x_old, z_old[i, :])
                    z_xy = np.empty((y_new.size, x_new.size), dtype=float)
                    for j in range(x_new.size):
                        z_xy[:, j] = np.interp(y_new, y_old, z_x[:, j])
                        
                        x_edges = _centers_to_edges(x_new)     # (nx+1,)
                        y_edges = _centers_to_edges(y_new)     # (ny+1,)
                        distance_plot, depth_plot, vel_plot = x_edges, y_edges, z_xy
                else:
                    from scipy.interpolate import RectBivariateSpline  
                    x_edges = _centers_to_edges(x_new)
                    y_edges = _edges_cover(0.0, max_depth, dy)     
                    y_new = 0.5 * (y_edges[:-1] + y_edges[1:])          
                    spline = RectBivariateSpline(y_old, x_old, z_old, kx=3, ky=3)
                    z_xy = spline(y_new, x_new)                       
                    distance_plot, depth_plot, vel_plot = x_edges, y_edges, np.asarray(z_xy, float)
    # -------- 绘图 --------
    layer = Layer(
        type="heatmap",
        data={"x": distance_plot, "y": depth_plot, "z": vel_plot},
        style={"cmap": cmap, "colorbar_label": colorbar_label},
        name="Velocity",
    )

    layout = {
        "title": title,
        "x_label": x_label,
        "y_label": y_label,
        "x_lim": [float(distance_plot.min()), float(distance_plot.max())],
        "y_lim": [0.0, max_depth],
        "invert_y": invert_y,
    }

    return PlotSpec(plot_id="vel2d_plot", layers=[layer], layout=layout)


PLUGINS = [
    Plugin(
        id="vel2d",
        title="2D Velocity Structure Heatmap",
        build=_build_vel2d,
        default_layout={"figsize": (8, 5)},
        data_spec={"type": "2d-array"},
        params={
            "title": Param("str", "2D Velocity Structure", "标题"),
            "x_label": Param("str", "Distance (km)", "x轴标签"),
            "y_label": Param("str", "Depth (m)", "y轴标签"),
            "cmap": Param("str", "jet_r", "色带"),
            "colorbar_label": Param("str", "Velocity (m/s)", "颜色条标签"),
            "invert_y": Param("bool", True, "深度向下"),
            "interp": Param("bool", True, "是否插值"),
            "interp_method": Param("str", "linear", "插值方法: none/linear/nearest/cubic"),
            "interp_strength": Param("float", 3.0, "插值强度(>1加密)"),
            "dx_km": Param("float", None, "手动指定x步长(km), 优先于强度"),
            "dy_m": Param("float", None, "手动指定y步长(m), 优先于强度"),
        },
    )
]
