# visualization/backends/mpl/render.py
from __future__ import annotations

from typing import Any, Dict
from ...types import FigureHandle, PlotSpec, Layer
from . import primitives


def is_available() -> bool:
    """判断 matplotlib 是否安装"""
    try:
        import matplotlib  # noqa: F401
        return True
    except Exception:
        return False


def render(spec: PlotSpec) -> FigureHandle:
    """
    将 PlotSpec 渲染为 matplotlib Figure
    支持的 layer.type：
      - heatmap / lines / wiggle / vlines / annotations / polar_heatmap
    """
    import matplotlib.pyplot as plt

    layout = spec.layout or {}
    figsize = layout.get("figsize", (9, 4))

    fig = plt.figure(figsize=figsize)

    has_polar = any(layer.type == "polar_heatmap" for layer in spec.layers)
    if has_polar:
        ax = fig.add_subplot(111, projection="polar")
    else:
        ax = fig.add_subplot(111)

    extra: Dict[str, Any] = {"ax": ax}

    # 逐层渲染
    for layer in spec.layers:
        if layer.type == "heatmap":
            z = layer.data["z"]
            x = layer.data.get("x")
            y = layer.data.get("y")
            cmap = layer.style.get("cmap", "viridis")
            cbl = layer.style.get("colorbar_label", "")
            r = primitives.plot_heatmap(ax, z, x=x, y=y, cmap=cmap, colorbar_label=cbl)
            extra.setdefault("artists", []).append(r)

        elif layer.type == "lines":
            x = layer.data["x"]
            y = layer.data["y"]
            r = primitives.plot_lines(
                ax, x, y,
                linewidth=layer.style.get("linewidth", 1.0),
                alpha=layer.style.get("alpha", 1.0),
                label=layer.name,
            )
            extra.setdefault("artists", []).append(r)

        elif layer.type == "vlines":
            xs = layer.data["xs"]
            r = primitives.plot_vlines(
                ax, xs,
                linewidth=layer.style.get("linewidth", 1.0),
                alpha=layer.style.get("alpha", 1.0),
                label=layer.name,
            )
            extra.setdefault("artists", []).append(r)

        elif layer.type == "wiggle":
            r = primitives.plot_wiggle(
                ax,
                layer.data["x"],
                layer.data["traces"],
                scale=layer.style.get("scale", 1.0),
                linewidth=layer.style.get("linewidth", 0.8),
                alpha=layer.style.get("alpha", 1.0),
                labels=layer.data.get("labels"),
                color="k",
                highlights=layer.data.get("highlights"),
                sort = layer.data.get("sort")
            )
            extra.setdefault("artists", []).append(r)

        elif layer.type == "annotations":
            # 预留：可后续实现文本标注
            for item in layer.data.get("texts", []):
                ax.text(float(item["x"]), float(item["y"]), str(item["text"]))
        
        elif layer.type == "polar_heatmap":
            theta = layer.data["theta"]
            r_data = layer.data["r"]  # 避免命名冲突
            z = layer.data["z"]
            theta_unit = layer.data.get("theta_unit", "deg")
            cmap = layer.style.get("cmap", "viridis")
            cbl = layer.style.get("colorbar_label", "")
            
            r = primitives.plot_polar_heatmap(
                ax, theta, r_data, z,
                theta_unit=theta_unit,
                cmap=cmap,
                colorbar_label=cbl,
            )
            extra.setdefault("artists", []).append(r)
        
        else:
            raise ValueError(f"mpl 后端不支持的 layer.type: {layer.type}")

    # 通用布局
    ax.set_title(layout.get("title", "") or "")
    ax.set_xlabel(layout.get("x_label", "") or "")
    ax.set_ylabel(layout.get("y_label", "") or "")

    # 是否显示图例
    if any(l.name for l in spec.layers):
        # 只有在存在可用 legend 项时才显示图例，避免无意义 warning
        handles, labels = ax.get_legend_handles_labels()
        if labels:  # labels 非空表示至少有一个可显示的图例项
            ax.legend(loc="best")

    fig.tight_layout()
    return FigureHandle(backend="mpl", handle=fig, extra=extra, spec=spec)