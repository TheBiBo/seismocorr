# visualization/backends/plotly/render.py
from __future__ import annotations

from typing import Any, Dict
from ...types import FigureHandle, PlotSpec
from . import primitives


def is_available() -> bool:
    """判断 plotly 是否安装"""
    try:
        import plotly  # noqa: F401
        return True
    except Exception:
        return False


def render(spec: PlotSpec) -> FigureHandle:
    """
    将 PlotSpec 渲染为 plotly Figure
    支持的 layer.type：
      - heatmap / lines / wiggle / vlines / annotations(可后续加)
    """
    import plotly.graph_objects as go

    layout = spec.layout or {}
    fig = go.Figure()

    # 竖线 shapes 汇总（plotly 用 layout.shapes）
    shapes = []
    yaxis_overrides: Dict[str, Any] = {}

    for layer in spec.layers:
        if layer.type == "heatmap":
            primitives.add_heatmap(
                fig,
                layer.data["z"],
                x=layer.data.get("x"),
                y=layer.data.get("y"),
                cmap=layer.style.get("cmap", "viridis"),
                colorbar_label=layer.style.get("colorbar_label", ""),
                name=layer.name,
            )

        elif layer.type == "lines":
            primitives.add_lines(
                fig,
                layer.data["x"],
                layer.data["y"],
                linewidth=layer.style.get("linewidth", 1.0),
                alpha=layer.style.get("alpha", 1.0),
                name=layer.name,
            )

        elif layer.type == "vlines":
            xs = layer.data["xs"]
            shapes.extend(
                primitives.add_vlines(
                    fig,
                    xs,
                    linewidth=layer.style.get("linewidth", 1.0),
                    alpha=layer.style.get("alpha", 1.0),
                    name=layer.name,
                )
            )

        elif layer.type == "wiggle":
            # 获取 layer 的数据和样式
            yaxis_part = primitives.add_wiggle(
                fig,
                x=layer.data["x"],
                traces=layer.data["traces"],
                scale=layer.style.get("scale", 1.0),
                linewidth=layer.style.get("linewidth", 0.8),
                alpha=layer.style.get("alpha", 1.0),
                labels=layer.data.get("labels"),
                name=layer.name,
                highlights=layer.data.get("highlights"),  # 传入高亮数据
                sort=layer.data.get("sort")  # 传入排序信息
            )
            
            # 更新 y 轴设置（可能包括标签和刻度）
            yaxis_overrides.update(yaxis_part)

        elif layer.type == "annotations":
            # 预留：文本标注
            # layer.data 约定：{"texts":[{"x":..,"y":..,"text":".."}, ...]}
            ann = []
            for item in layer.data.get("texts", []):
                ann.append(dict(x=float(item["x"]), y=float(item["y"]), text=str(item["text"]), showarrow=False))
            fig.update_layout(annotations=ann)
        elif layer.type == "polar_heatmap":
            import numpy as np
            import plotly.graph_objects as go

            theta = np.asarray(layer.data["theta"], dtype=float)
            r = np.asarray(layer.data["r"], dtype=float)
            z = np.asarray(layer.data["z"], dtype=float)
            theta_unit = layer.data.get("theta_unit", "deg")

            if theta_unit == "deg":
                theta = np.deg2rad(theta)
            elif theta_unit != "rad":
                raise ValueError("theta_unit 必须是 'deg' 或 'rad'")

            if z.shape != (r.shape[0], theta.shape[0]):
                raise ValueError("z 形状必须是 (n_r, n_theta)")

            # 生成规则网格（分辨率可调，越大越细但更重）
            nr = int(layer.style.get("nr", 300))
            nxy = int(layer.style.get("nxy", 400))

            r_min, r_max = float(np.min(r)), float(np.max(r))
            x = np.linspace(-r_max, r_max, nxy)
            y = np.linspace(-r_max, r_max, nxy)
            X, Y = np.meshgrid(x, y)
            RR = np.sqrt(X**2 + Y**2)
            TT = (np.arctan2(X, Y) + 2*np.pi) % (2*np.pi)  # 0在N，顺时针

            # 把 RR,TT 映射到 z（最近邻/线性插值）
            # 简化实现：用最近邻（足够好且代码短）
            r_idx = np.searchsorted(r, RR, side="left")
            r_idx = np.clip(r_idx, 0, len(r)-1)

            theta_sorted = theta
            t_idx = np.searchsorted(theta_sorted, TT, side="left")
            t_idx = np.clip(t_idx, 0, len(theta_sorted)-1)

            Z = z[r_idx, t_idx]

            # 圆外区域设为 NaN（透明）
            Z = np.where(RR <= r_max, Z, np.nan)

            colorscale = primitives.mpl_cmap_to_plotly(layer.style.get("cmap", "viridis"))
            cbl = layer.style.get("colorbar_label", "")

            fig.add_trace(
                go.Heatmap(
                    z=Z,
                    x=x,
                    y=y,
                    colorscale=colorscale,
                    colorbar=dict(title=cbl) if cbl else None,
                    zsmooth="best",
                )
            )

            fig.update_yaxes(scaleanchor="x", scaleratio=1)  # 保持圆形不变形
        else:
            raise ValueError(f"plotly 后端不支持的 layer.type: {layer.type}")

    # 通用布局：标题/坐标轴
    fig.update_layout(
        title=layout.get("title", "") or "",
        xaxis_title=layout.get("x_label", "") or "",
        yaxis_title=layout.get("y_label", "") or "",
        shapes=shapes if shapes else None,
        margin=dict(l=60, r=20, t=60, b=50),
    )

    # figsize（plotly 用像素，做一个粗略换算）
    figsize = layout.get("figsize")
    if figsize:
        w = int(float(figsize[0]) * 110)
        h = int(float(figsize[1]) * 110)
        fig.update_layout(width=w, height=h)

    # y轴标签（wiggle 的 ticktext）
    if yaxis_overrides:
        fig.update_yaxes(**yaxis_overrides)

    return FigureHandle(backend="plotly", handle=fig, extra={}, spec=spec)
