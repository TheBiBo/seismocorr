# visualization/backends/plotly/primitives.py
from __future__ import annotations

from typing import Any, Dict, Optional
import numpy as np
import plotly.graph_objects as go
import numpy as np

def mpl_cmap_to_plotly(cmap: str) -> str:
    """
    做一个最小映射：让常见 mpl cmap 名称在 plotly 里能用
    plotly 的 colorscale 名称是首字母大写形式居多
    """
    mapping = {
        "viridis": "Viridis",
        "plasma": "Plasma",
        "inferno": "Inferno",
        "magma": "Magma",
        "cividis": "Cividis",
        "turbo": "Turbo",
        "gray": "Greys",
        "grey": "Greys",
        "Greys": "Greys",
    }
    return mapping.get(cmap, "Viridis")


def add_heatmap(fig, z, x=None, y=None, *, cmap="viridis", colorbar_label="", name=None):
    import plotly.graph_objects as go

    z = np.asarray(z)
    if x is None:
        x = np.arange(z.shape[1])
    if y is None:
        y = np.arange(z.shape[0])

    fig.add_trace(
        go.Heatmap(
            z=z,
            x=x,
            y=y,
            colorscale=mpl_cmap_to_plotly(str(cmap)),
            colorbar=dict(title=str(colorbar_label)) if colorbar_label else None,
            name=name,
            showscale=True,
        )
    )


def add_lines(fig, x, y, *, linewidth=1.0, alpha=1.0, name=None):
    import plotly.graph_objects as go
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            line=dict(width=float(linewidth)),
            opacity=float(alpha),
            name=name,
        )
    )


def add_wiggle(fig, x, traces, *, scale=1.0, linewidth=0.8, alpha=1.0, labels=None, name=None, highlights=None, sort=None):
    """
    Plotly 版 wiggle：每道作为一个 trace（Scatter lines）
    注意：道数过多会比较重（>200 道可能明显卡）
    """
    x = np.asarray(x)
    tr = np.asarray(traces)
    n_tr, _ = tr.shape

    # 处理排序
    order = np.arange(n_tr)
    inv_order = None

    sort_by = None
    sort_ascending = True
    y_mode = "index"
    y_label = None

    if sort is not None:
        if not isinstance(sort, dict) or "by" not in sort:
            raise ValueError("sort 必须是 dict 且包含 'by'，例如 {'by': dist, 'ascending': True, 'y_mode': 'distance'}")

        sort_by = np.asarray(sort["by"], dtype=float)
        if sort_by.ndim != 1 or sort_by.shape[0] != n_tr:
            raise ValueError("sort['by'] 必须是一维且长度等于 traces.shape[0]（道数）")

        sort_ascending = bool(sort.get("ascending", True))
        y_mode = str(sort.get("y_mode", "index"))
        y_label = sort.get("label", None)

        order = np.argsort(sort_by)
        if not sort_ascending:
            order = order[::-1]

        # 重排 traces
        tr = tr[order, :]

        # 重排 labels
        if labels is not None:
            if len(labels) != n_tr:
                raise ValueError("labels 长度必须等于道数 n_tr")
            labels = [labels[i] for i in order]

        # old_index -> new_index 映射（用于 highlights）
        inv_order = np.empty(n_tr, dtype=int)
        inv_order[order] = np.arange(n_tr)

        sort_by_sorted = sort_by[order]
    else:
        sort_by_sorted = None  # 未排序

    # 计算每道的“基线纵坐标”
    if y_mode == "distance":
        if sort_by_sorted is None:
            raise ValueError("y_mode='distance' 时必须提供 sort={'by': distance,...}")
        y0 = sort_by_sorted.astype(float)  # 每道基线就是距离值
    elif y_mode == "index":
        y0 = np.arange(n_tr, dtype=float)
    else:
        raise ValueError("sort['y_mode'] 只能是 'index' 或 'distance'")

    # 预处理 highlights（输入的 trace 默认按原始索引）
    hl_by_trace = {}
    if highlights:
        if not isinstance(highlights, (list, tuple)):
            raise ValueError("highlights 必须是 list[dict] 或 tuple[dict]")
        for item in highlights:
            if not isinstance(item, dict) or "trace" not in item:
                raise ValueError("highlights 每项必须是 dict 且包含 'trace' 字段")
            old_i = int(item["trace"])
            if old_i < 0 or old_i >= n_tr:
                raise ValueError(f"highlights.trace 越界: {old_i} (0~{n_tr-1})")

            # 如果发生排序，把 old_i 映射成新序号 new_i
            new_i = int(inv_order[old_i]) if inv_order is not None else old_i
            hl_by_trace.setdefault(new_i, []).append(item)

    # 添加 wiggle 曲线
    for i in range(n_tr):
        y = y0[i] + tr[i] * float(scale)

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                line=dict(width=float(linewidth), color="black"),
                opacity=float(alpha),
                name=(name if (name and i == 0) else None),
                showlegend=bool(name and i == 0),
                hoverinfo="skip",
            )
        )

        # 高亮处理
        if i in hl_by_trace:
            for cfg in hl_by_trace[i]:
                t0 = float(cfg.get("t0", min(x)))
                t1 = float(cfg.get("t1", max(x)))
                if t1 < t0:
                    t0, t1 = t1, t0

                mask = (x >= t0) & (x <= t1)
                if np.any(mask):
                    hl_color = cfg.get("color", "red")
                    hl_lw = float(cfg.get("linewidth", linewidth))
                    hl_alpha = float(cfg.get("alpha", 1.0))

                    fig.add_trace(
                        go.Scatter(
                            x=x[mask],
                            y=y[mask],
                            mode="lines",
                            line=dict(width=hl_lw, color=hl_color),
                            opacity=hl_alpha,
                            hoverinfo="skip",
                        )
                    )

    # 设置 y 轴刻度/标签
    yaxis = {}
    if labels is not None and len(labels) == n_tr and n_tr <= 60:
        yaxis.update(
            tickmode="array",
            tickvals=list(range(n_tr)),
            ticktext=[str(s) for s in labels],
        )

    return yaxis
