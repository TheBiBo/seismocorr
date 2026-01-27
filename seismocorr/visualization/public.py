# seismocorr/visualization/public.py
from __future__ import annotations

from typing import Any, Dict, Optional

from .core import Visualizer, PluginRegistry
from .plugins import make_plugins
from .types import BackendName, FigureHandle
import matplotlib.pyplot as plt
# -----------------------------
# 初始化：注册插件 -> 生成一个全局可用的 Visualizer
# -----------------------------

_REGISTRY = PluginRegistry(plugins=make_plugins())
_VIS = Visualizer(_REGISTRY)

_DEFAULT_BACKEND: BackendName = "mpl"


def set_default_backend(name: BackendName) -> None:
    global _DEFAULT_BACKEND
    _DEFAULT_BACKEND = name


def get_default_backend() -> BackendName:
    return _DEFAULT_BACKEND


def list_plots():
    """列出所有可用 plot_id（简单版）"""
    return _REGISTRY.list_ids()


def plot(
    plot_id: str,
    data: Any,
    *,
    backend: Optional[BackendName] = None,
    fallback: bool = True,
    **kwargs,
) -> FigureHandle:
    """
    统一绘图入口：
    - plot_id：例如 'ccf.wiggle'
    - data：业务数据（结构由插件定义）
    - backend：'mpl' 或 'plotly'（默认使用 set_default_backend 设置）
    - fallback：后端不可用时是否自动回退
    - kwargs：传递给插件 build 的参数（会经过方案B校验/默认值填充）
    """
    if backend is None:
        backend = _DEFAULT_BACKEND
    return _VIS.plot(plot_id, data, backend=backend, fallback=fallback, **kwargs)


def get_plot_schema(plot_id: str) -> Dict[str, Any]:
    """
    返回机器可读的 schema，包含：
    - plot_id/title
    - data 输入规范（data_spec）
    - kwargs 参数规范（params）
    """
    plugin = _REGISTRY.get(plot_id)

    schema: Dict[str, Any] = {
        "id": plugin.id,
        "title": plugin.title,
        "data_spec": plugin.data_spec or {},
        "params": {},
    }

    for k, p in (plugin.params or {}).items():
        schema["params"][k] = {
            "type": p.ptype,
            "default": p.default,
            "required": p.required,
            "doc": p.doc,
            "choices": p.choices,
        }
        if p.item_schema:
            schema["params"][k]["item_schema"] = {
                kk: {
                    "type": pp.ptype,
                    "default": pp.default,
                    "required": pp.required,
                    "doc": pp.doc,
                    "choices": pp.choices,
                }
                for kk, pp in p.item_schema.items()
            }

    return schema


def help_plot(plot_id: str) -> str:
    """
    返回“人类可读”的帮助说明：
    - plot_id / title
    - data 输入结构说明
    - 可用 kwargs（类型/默认值/说明）
    """
    plugin = _REGISTRY.get(plot_id)
    lines = []
    lines.append(f"Plot ID: {plugin.id}")
    lines.append(f"Title: {plugin.title}")

    # 1) data 输入规范
    if plugin.data_spec:
        ds = plugin.data_spec
        lines.append("")
        lines.append("data 输入格式：")
        if "type" in ds:
            lines.append(f"  type: {ds.get('type')}")
        rk = ds.get("required_keys", {})
        if rk:
            lines.append("  必填字段：")
            for k, v in rk.items():
                lines.append(f"    - {k}: {v}")
        ok = ds.get("optional_keys", {})
        if ok:
            lines.append("  可选字段：")
            for k, v in ok.items():
                lines.append(f"    - {k}: {v}")
        notes = ds.get("notes", [])
        if notes:
            lines.append("  备注：")
            for s in notes:
                lines.append(f"    - {s}")

    # 2) kwargs 参数规范（方案B）
    lines.append("")
    lines.append("可用参数（kwargs）：")
    if not plugin.params:
        lines.append("  （该插件未提供 ParamSpec，无法列出参数；请查看 build() 定义/文档）")
        return "\n".join(lines)

    for k in sorted(plugin.params.keys()):
        p = plugin.params[k]
        default_str = f"default={p.default!r}" if not p.required else "required=True"
        choice_str = f" choices={p.choices!r}" if p.choices else ""
        lines.append(f"  - {k} ({p.ptype}) {default_str}{choice_str}")
        if p.doc:
            lines.append(f"      {p.doc}")
        if p.item_schema:
            lines.append("      子字段（list[dict] 的 dict keys）：")
            for kk, pp in p.item_schema.items():
                d2 = f"default={pp.default!r}" if not pp.required else "required=True"
                lines.append(f"        * {kk} ({pp.ptype}) {d2} - {pp.doc}")

    return "\n".join(lines)


def show(fig_handle: FigureHandle) -> None:
    """
    显示绘制的图形。
    
    参数:
    - fig_handle: FigureHandle，包含后端图形和相应的图形对象
    """
    if fig_handle.backend == "mpl":
        plt.show()  # 如果是 matplotlib 后端，调用 plt.show()
    elif fig_handle.backend == "plotly":
        fig_handle.handle.show()  # 如果是 plotly 后端，直接调用 plotly 图形的 show 方法
    else:
        raise ValueError(f"Unsupported backend: {fig_handle.backend}")