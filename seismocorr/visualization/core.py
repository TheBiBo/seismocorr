# visualization/core.py
from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Dict, Optional, List

from .types import BackendName, FigureHandle, PlotSpec, Plugin, Param, ParamSpec

def _type_check(name: str, value, param: Param) -> None:
    """轻量类型检查：不追求覆盖所有情况，只要能抓住常见错误"""
    if value is None:
        return

    t = param.ptype
    ok = True

    if t == "bool":
        ok = isinstance(value, bool)
    elif t == "int":
        ok = isinstance(value, int) and not isinstance(value, bool)
    elif t == "float":
        ok = isinstance(value, (int, float)) and not isinstance(value, bool)
    elif t == "str":
        ok = isinstance(value, str)
    elif t == "dict":
        ok = isinstance(value, dict)
    elif t == "list":
        ok = isinstance(value, list)
    elif t == "list[dict]":
        ok = isinstance(value, list) and all(isinstance(x, dict) for x in value)
    # 其他类型先不强检，保持宽松
    else:
        ok = True

    if not ok:
        raise TypeError(f"参数 {name!r} 类型不符合要求：期望 {t}，实际 {type(value).__name__}")

    # choices 检查
    if param.choices is not None and value is not None:
        if value not in param.choices:
            raise ValueError(f"参数 {name!r} 必须是 {param.choices} 之一，实际为 {value!r}")


def _prepare_kwargs(params: ParamSpec, user_kwargs: dict) -> dict:
    """
    根据 ParamSpec：
    - 检查未知参数
    - 填默认值
    - 检查 required
    - 基础类型检查
    """
    # 1) 未知参数检查
    for k in user_kwargs.keys():
        if k not in params:
            raise TypeError(f"未知参数: {k!r}。可用参数: {sorted(params.keys())}")

    # 2) 合并默认值
    merged = {}
    for k, p in params.items():
        if k in user_kwargs:
            merged[k] = user_kwargs[k]
        else:
            if p.required:
                raise TypeError(f"缺少必填参数: {k!r}")
            merged[k] = p.default

        # 3) 基础类型检查
        _type_check(k, merged[k], p)

    return merged
# =========================
# Registry：管理“插件”（PlotSpec 生成器）
# =========================
# seismocorr/visualization/registry.py (或类似位置)

class PluginRegistry:
    def __init__(self, plugins: Optional[List] = None):
        """
        初始化 PluginRegistry，可以传入一个插件列表
        :param plugins: 一个插件列表，默认为 None 时为空列表
        """
        self.plugins = plugins if plugins else []

    def add_plugin(self, plugin: 'Plugin') -> None:
        """添加插件到注册表"""
        self.plugins.append(plugin)

    def get(self, plot_id: str) -> 'Plugin':
        """根据 plot_id 获取插件"""
        for plugin in self.plugins:
            if plugin.id == plot_id:
                return plugin
        raise ValueError(f"Plugin with id {plot_id} not found.")

    def list_ids(self) -> List[str]:
        """列出所有插件的 id"""
        return [plugin.id for plugin in self.plugins]


# =========================
# Backend：管理“后端渲染器”
# =========================
@dataclass
class Backend:
    """
    后端对象：用于把 PlotSpec 渲染成 FigureHandle
    - name: mpl/plotly/...
    - available: 是否可用（库是否安装）
    - render: render(spec) -> FigureHandle
    """
    name: BackendName
    available: bool
    render: Any  # Callable[[PlotSpec], FigureHandle]


def _load_backend(name: BackendName) -> Backend:
    """
    动态加载后端模块（相对本包，不写死顶层名字）
    实际导入路径：
      seismocorr.visualization.backends.mpl.render
      seismocorr.visualization.backends.plotly.render
    """
    # __package__ 在 seismocorr.visualization.core 中等于 "seismocorr.visualization"
    base = __package__  # => "seismocorr.visualization"

    if name == "mpl":
        mod = importlib.import_module(f"{base}.backends.mpl.render")
        return Backend(name="mpl", available=mod.is_available(), render=mod.render)

    if name == "plotly":
        mod = importlib.import_module(f"{base}.backends.plotly.render")
        return Backend(name="plotly", available=mod.is_available(), render=mod.render)

    raise ValueError(f"未知后端: {name}")



# =========================
# Visualizer：统一入口
# =========================
class Visualizer:
    def __init__(self, registry: PluginRegistry, default_backend: BackendName = "mpl") -> None:
        self.registry = registry
        self._default_backend: BackendName = default_backend

    def set_default_backend(self, backend: BackendName) -> None:
        self._default_backend = backend

    def get_default_backend(self) -> BackendName:
        return self._default_backend

    def plot(
        self,
        plugin_id: str,
        data: Any,
        backend: Optional[BackendName] = None,
        *,
        fallback: bool = True,
        **kwargs: Any,
    ) -> FigureHandle:
        """
        统一绘图函数：
        1) plugin.build(data, **kwargs) -> PlotSpec（后端无关）
        2) backend.render(spec) -> FigureHandle（后端相关）

        参数：
        - plugin_id: 如 "ccf.wiggle"
        - data: 业务数据对象（可以是 ndarray / dict / 自定义类）
        - backend: 指定后端（mpl/plotly），不传则用默认后端
        - fallback: 如果指定后端不可用，是否自动回退到另一个可用后端
        - kwargs: 传给插件 build 的参数（如 title/cmap/normalize...）
        """
        plugin = self.registry.get(plugin_id)

        # ✅ 先做参数校验/默认值合并
        prepared = _prepare_kwargs(plugin.params, kwargs) if plugin.params else kwargs

        # ✅ 再调用 build（插件可以放心用 prepared 里的字段）
        spec: PlotSpec = plugin.build(data, **prepared)

        # 合并插件默认 layout（用户可在插件 build 内覆盖，也可在 kwargs 里传）
        merged_layout = dict(plugin.default_layout)
        merged_layout.update(spec.layout or {})
        spec.layout = merged_layout

        be = backend or self._default_backend
        be_obj = _load_backend(be)

        if not be_obj.available:
            if not fallback:
                raise RuntimeError(f"后端 {be} 不可用（可能未安装依赖），且 fallback=False")
            # 回退策略：优先选择另一个后端
            alt = "plotly" if be == "mpl" else "mpl"
            alt_obj = _load_backend(alt)  # type: ignore[arg-type]
            if not alt_obj.available:
                raise RuntimeError(f"后端 {be} 不可用，且回退后端 {alt} 也不可用。请安装对应依赖。")
            fig = alt_obj.render(spec)
            fig.spec = spec
            return fig

        fig = be_obj.render(spec)
        fig.spec = spec
        return fig
