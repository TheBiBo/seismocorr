# visualization/backends/mpl/primitives.py
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Sequence
import numpy as np


def plot_heatmap(ax, z, x=None, y=None, *, cmap="", colorbar_label=""):
    """
    绘制二维热力图（rect heatmap）

    支持两种坐标输入：
      1) centers: len(x)=ncols, len(y)=nrows
      2) edges:   len(x)=ncols+1, len(y)=nrows+1  (推荐，最严谨、不会头尾空)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    def _centers_to_edges(c: np.ndarray) -> np.ndarray:
        """把中心点坐标转换为边界坐标（长度 +1）。要求 c 单调递增。"""
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
    z = np.asarray(z, float)
    nrows, ncols = z.shape

    if x is None:
        x = np.arange(ncols, dtype=float)
    else:
        x = np.asarray(x, float)

    if y is None:
        y = np.arange(nrows, dtype=float)
    else:
        y = np.asarray(y, float)

    # 判定 edges / centers，并统一转换为 edges
    if x.size == ncols + 1:
        x_edges = x
    elif x.size == ncols:
        x_edges = _centers_to_edges(x)
    else:
        raise ValueError(f"x 长度应为 {ncols} (centers) 或 {ncols+1} (edges)，但得到 {x.size}")

    if y.size == nrows + 1:
        y_edges = y
    elif y.size == nrows:
        y_edges = _centers_to_edges(y)
    else:
        raise ValueError(f"y 长度应为 {nrows} (centers) 或 {nrows+1} (edges)，但得到 {y.size}")

    # 用 pcolormesh：每个 z[i,j] 对应一个矩形单元格 [x_edges[j],x_edges[j+1]]×[y_edges[i],y_edges[i+1]]
    im = ax.pcolormesh(
        x_edges,
        y_edges,
        z,
        shading="auto",
        cmap=cmap,
    )

    cbar = plt.colorbar(im, ax=ax, shrink=0.9)
    if colorbar_label:
        cbar.set_label(colorbar_label, fontsize=14)

    return {"im": im, "colorbar": cbar}


def plot_lines(ax, x, y, *, linewidth=1.0, alpha=1.0, label=None, color:str):
    """绘制折线/曲线"""
    (ln,) = ax.plot(x, y, linewidth=linewidth, alpha=alpha, label=label, color = color)
    return {"line": ln}


def plot_vlines(ax, xs, *, linewidth=1.0, alpha=1.0, label=None):
    """绘制多条竖线"""
    arts = []
    for i, xv in enumerate(xs):
        arts.append(ax.axvline(x=float(xv), linewidth=linewidth, alpha=alpha, label=(label if i == 0 else None)))
    return {"vlines": arts}


def plot_wiggle(
    ax,
    x,
    traces,
    *,
    # 显示控制
    dy: Optional[float] = None,
    scale: float = 1.0,  
    excursion: float = 2.0,
    bias: float = 0.0,
    # 归一化与限幅
    norm_method: str = "trace",
    normalize: Optional[str] = "p95",
    clip: Optional[float] = 2.5,
    # 线与填充
    linewidth: float = 0.6,
    alpha: float = 0.85,
    color: str = "k",
    fill_mode: Optional[str] = "pos",
    fill_alpha: float = 0.10,
    # 标签与高亮
    labels: Optional[Sequence[str]] = None,
    highlights=None,
    # 排序与纵轴
    sort=None,
    ytick_step: int = 5,
    trace_step: int = 1,
    # 参考线
    zero_line: bool = True,
    zero_line_kwargs: Optional[Dict] = None,
):
    """
    绘制多道 wiggle（支持 variable-area 填充、排序、抽稀与局部高亮）。

    参数说明（核心）
    ----------------
    dy :
        单道显示的“单位高度”。若为 None，则使用 scale。
    excursion :
        单道最大跨越的道间距倍数（控制拥挤程度）。建议 1.5~2.5。
    bias :
        填充阈值偏置（以“归一化后的幅度单位”计）。bias>0 会抬高填充阈值。
    norm_method :
        "trace"：每道独立归一化；"stream"：全局统一归一化。
    normalize :
        None / "max" / "p95" / "rms"。决定归一化的幅度统计量。
    clip :
        归一化后限幅阈值；None 表示不限幅。
    fill_mode :
        None / "pos" / "neg" / "both"。variable-area 填充方式。
    trace_step :
        抽稀绘制步长（每隔 trace_step 道画一条），用于大规模道数时减负担。

    返回
    ----
    dict:
      - lines: 背景曲线 Line2D 列表
      - highlight_lines: 高亮曲线 Line2D 列表
      - order: 新索引 -> 原索引 映射（排序后）
    """
    x = np.asarray(x, dtype=float)
    tr = np.asarray(traces, dtype=float)

    if tr.ndim != 2:
        raise ValueError("traces 必须是二维数组 (n_traces, n_samples)")
    n_tr, n_samp = tr.shape
    if x.ndim != 1 or x.shape[0] != n_samp:
        raise ValueError("x 必须是一维，且长度等于 traces.shape[1]")

    if trace_step < 1:
        raise ValueError("trace_step 必须 >= 1")

    if norm_method not in ("trace", "stream"):
        raise ValueError("norm_method 只能是 'trace' 或 'stream'")

    if fill_mode not in (None, "pos", "neg", "both"):
        raise ValueError("fill_mode 只能是 None/'pos'/'neg'/'both'")

    # -------------------- 1) 排序 --------------------
    order = np.arange(n_tr)
    inv_order = None

    sort_by_sorted = None
    y_mode = "index"
    y_label = None

    if sort is not None:
        if not isinstance(sort, dict) or "by" not in sort:
            raise ValueError("sort 必须是 dict 且包含 'by'")

        sort_by = np.asarray(sort["by"], dtype=float)
        if sort_by.ndim != 1 or sort_by.shape[0] != n_tr:
            raise ValueError("sort['by'] 必须是一维且长度等于道数")

        ascending = bool(sort.get("ascending", True))
        y_mode = str(sort.get("y_mode", "index"))
        y_label = sort.get("label", None)

        order = np.argsort(sort_by)
        if not ascending:
            order = order[::-1]

        tr = tr[order, :]

        if labels is not None:
            if len(labels) != n_tr:
                raise ValueError("labels 长度必须等于道数")
            labels = [labels[i] for i in order]

        inv_order = np.empty(n_tr, dtype=int)
        inv_order[order] = np.arange(n_tr)
        sort_by_sorted = sort_by[order]

    # -------------------- 2) 纵轴基线 y0 --------------------
    if y_mode == "distance":
        if sort_by_sorted is None:
            raise ValueError("y_mode='distance' 时必须提供 sort={'by': distance,...}")
        y0 = sort_by_sorted.astype(float)
        spacing = float(np.median(np.diff(y0))) if n_tr > 1 else 1.0
        if not np.isfinite(spacing) or spacing == 0.0:
            spacing = 1.0
    elif y_mode == "index":
        y0 = np.arange(n_tr, dtype=float)
        spacing = 1.0
    else:
        raise ValueError("sort['y_mode'] 只能是 'index' 或 'distance'")

    # dy 优先，其次 scale（兼容旧参数）
    dy_eff = float(scale if dy is None else dy)

    # -------------------- 3) 归一化 + 限幅（仅用于显示） --------------------
    trn = tr.copy()

    def _amp_stat(a: np.ndarray) -> np.ndarray:
        if normalize is None:
            return np.ones((a.shape[0], 1), dtype=float)
        if normalize == "max":
            s = np.max(np.abs(a), axis=1, keepdims=True)
        elif normalize == "p95":
            s = np.percentile(np.abs(a), 95, axis=1, keepdims=True)
        elif normalize == "rms":
            s = np.sqrt(np.mean(a**2, axis=1, keepdims=True))
        else:
            raise ValueError("normalize 只能是 None/'max'/'p95'/'rms'")
        s = np.where(s == 0, 1.0, s)
        return s

    if norm_method == "trace":
        denom = _amp_stat(trn)
        trn = trn / denom
    else:
        if normalize is None:
            denom_g = 1.0
        else:
            if normalize == "max":
                denom_g = float(np.max(np.abs(trn)))
            elif normalize == "p95":
                denom_g = float(np.percentile(np.abs(trn), 95))
            elif normalize == "rms":
                denom_g = float(np.sqrt(np.mean(trn**2)))
            else:
                raise ValueError("normalize 只能是 None/'max'/'p95'/'rms'")
            if denom_g == 0.0:
                denom_g = 1.0
        trn = trn / denom_g

    if clip is not None:
        c = float(abs(clip))
        trn = np.clip(trn, -c, c)
        amp_cap = c
    else:
        amp_cap = float(np.max(np.abs(trn))) if trn.size else 1.0
        if amp_cap == 0.0:
            amp_cap = 1.0

    # 将“幅度单位”映射到“道间距单位”
    # 目标：|trn| 最大不超过 excursion * spacing
    amp_to_y = (float(excursion) * spacing) / amp_cap

    # -------------------- 4) 高亮映射 --------------------
    hl_by_trace = {}
    if highlights:
        if not isinstance(highlights, (list, tuple)):
            raise ValueError("highlights 必须是 list[dict] 或 tuple[dict]")
        for item in highlights:
            if not isinstance(item, dict) or "trace" not in item:
                raise ValueError("highlights 每项必须是 dict 且包含 'trace'")
            old_i = int(item["trace"])
            if old_i < 0 or old_i >= n_tr:
                raise ValueError(f"highlights.trace 越界: {old_i} (0~{n_tr-1})")
            new_i = int(inv_order[old_i]) if inv_order is not None else old_i
            hl_by_trace.setdefault(new_i, []).append(item)

    arts = []
    hl_arts = []

    xmin = float(np.min(x))
    xmax = float(np.max(x))

    # bias 以“归一化幅度单位”计
    bias_u = float(bias)

    # -------------------- 5) 绘制 --------------------
    idxs = np.arange(0, n_tr, int(trace_step), dtype=int)

    for i in idxs:
        y = y0[i] + dy_eff * amp_to_y * trn[i]

        (ln,) = ax.plot(x, y, linewidth=linewidth, alpha=alpha, color=color)
        arts.append(ln)

        if fill_mode is not None:
            ybase = y0[i]
            yfill = y0[i] + dy_eff * amp_to_y * (trn[i] - bias_u)

            if fill_mode in ("pos", "both"):
                ax.fill_between(
                    x,
                    ybase,
                    y,
                    where=(trn[i] - bias_u) >= 0,
                    color=color,
                    alpha=float(fill_alpha),
                    linewidth=0.0,
                )
            if fill_mode in ("neg", "both"):
                ax.fill_between(
                    x,
                    ybase,
                    y,
                    where=(trn[i] - bias_u) <= 0,
                    color=color,
                    alpha=float(fill_alpha),
                    linewidth=0.0,
                )

        if i in hl_by_trace:
            for cfg in hl_by_trace[i]:
                t0 = float(cfg.get("t0", xmin))
                t1 = float(cfg.get("t1", xmax))
                if t1 < t0:
                    t0, t1 = t1, t0

                mask = (x >= t0) & (x <= t1)
                if not np.any(mask):
                    continue

                hl_color = cfg.get("color", "crimson")
                hl_lw = float(cfg.get("linewidth", max(1.6, linewidth * 2.5)))
                hl_alpha = float(cfg.get("alpha", 1.0))

                (hln,) = ax.plot(
                    x[mask],
                    y[mask],
                    linewidth=hl_lw,
                    alpha=hl_alpha,
                    color=hl_color,
                )
                hl_arts.append(hln)

    # -------------------- 6) 参考线与刻度 --------------------
    if zero_line:
        kw = dict(color="0.2", linewidth=0.8, alpha=0.8, zorder=0)
        if zero_line_kwargs:
            kw.update(zero_line_kwargs)
        ax.axvline(0.0, **kw)

    if ytick_step and ytick_step > 0 and n_tr > 0:
        if y_mode == "index":
            ticks = np.arange(0, n_tr, int(ytick_step))
            ax.set_yticks(ticks)
            if labels is not None and len(labels) == n_tr:
                ax.set_yticklabels([labels[i] for i in ticks], fontsize=8)
            else:
                ax.set_yticklabels([str(i) for i in ticks], fontsize=8)
        else:
            # 距离模式：刻度与标签一律显示距离值，避免被台站名覆盖
            ticks_idx = np.arange(0, n_tr, int(ytick_step))
            yticks = y0[ticks_idx]
            ax.set_yticks(yticks)
            ax.set_yticklabels([f"{v:.1f}" for v in yticks], fontsize=8)

    # 距离模式默认给一个 y 轴标签（若 sort 里提供 label 则用之）
    if y_mode == "distance":
        ax.set_ylabel(str(y_label) if y_label else "Distance")
    else:
        if y_label:
            ax.set_ylabel(str(y_label))

    ax.tick_params(direction="out", length=3, width=0.8)

    return {"lines": arts, "highlight_lines": hl_arts, "order": order}


def plot_polar_heatmap(
    ax,
    theta,
    r,
    z,
    *,
    theta_unit="deg",
    cmap="viridis",
    colorbar_label="",
    title="Beamforming Polar Heatmap",
    radial_label="Slowness (s/km)",
):
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    import matplotlib as mpl

    theta = np.asarray(theta, dtype=float)
    r = np.asarray(r, dtype=float)
    z = np.asarray(z, dtype=float)

    if theta_unit == "deg":
        theta = np.deg2rad(theta)
    elif theta_unit != "rad":
        raise ValueError("theta_unit 必须是 'deg' 或 'rad'")

    if z.shape != (r.shape[0], theta.shape[0]):
        raise ValueError(f"z 形状必须是 (n_r={r.shape[0]}, n_theta={theta.shape[0]})，但得到 {z.shape}")

    def _edges(v):
        v = np.asarray(v, float)
        if v.size == 1:
            return np.array([v[0] - 0.5, v[0] + 0.5])
        dv = np.diff(v)
        mid = (v[:-1] + v[1:]) / 2.0
        left = v[0] - dv[0] / 2.0
        right = v[-1] + dv[-1] / 2.0
        return np.concatenate([[left], mid, [right]])

    order = np.argsort(theta)
    theta = theta[order]
    z = z[:, order]

    th_e = _edges(theta)          # (n_theta+1,)
    r_e = _edges(r)               # (n_r+1,)
    th_c = 0.5 * (th_e[:-1] + th_e[1:])      # (n_theta,)
    th_w = (th_e[1:] - th_e[:-1])            # (n_theta,)

    r_b = r_e[:-1]                            # (n_r,)
    r_h = (r_e[1:] - r_e[:-1])                # (n_r,)

    norm = mpl.colors.Normalize(vmin=np.nanmin(z), vmax=np.nanmax(z))
    mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

    for i in range(len(r)):
        colors = mappable.to_rgba(z[i, :])
        ax.bar(
            th_c,
            height=r_h[i],
            width=th_w,
            bottom=r_b[i],
            align="center",
            color=colors,
            edgecolor="none",
            linewidth=0,
        )

    cax = inset_axes(ax, width="3%", height="70%", loc="center right", borderpad=-10)
    cbar = ax.figure.colorbar(mappable, cax=cax)
    if colorbar_label:
        cbar.set_label(colorbar_label, color="bl")
    cbar.ax.tick_params(colors="black")

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    ax.set_xticks(np.linspace(0, 2 * np.pi, 8, endpoint=False))
    ax.set_xticklabels(
        ["0°", "45°", "90°", "135°", "180°", "225°", "270°", "315°"],
        fontsize=12, rotation=45, color="black"
    )

    yt = np.linspace(0, np.max(r), 6)
    ax.set_yticks(yt[1:-1])
    ax.set_yticklabels([f"{int(x)}" for x in yt[1:-1]], fontsize=18, rotation=0, color="white")
    ax.set_rlabel_position(0)
    ax.tick_params(axis="y", colors="white")

    ax.set_title(title, fontsize=32, pad=20, color="white")

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.yaxis.label.set_visible(False)

    ax.text(-0.10, 0.5, radial_label, transform=ax.transAxes,
            rotation=90, ha="center", va="center", fontsize=14, color="black")

    ax.set_thetagrids([0, 90, 180, 270])  # 只保留十字方向
    ax.xaxis.grid(True, color="white", lw=1.5, alpha=0.8)

    ax.yaxis.grid(True, linestyle="-", alpha=0.8, linewidth=1.0, color="white")

    ax.spines["polar"].set_visible(False)

    return {"im": mappable, "colorbar": cbar}


