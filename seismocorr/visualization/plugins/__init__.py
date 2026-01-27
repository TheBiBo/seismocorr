# seismocorr/visualization/plugins/__init__.py

from __future__ import annotations
from typing import List, Any
import importlib
import pkgutil
import warnings
from ..types import Plugin


def make_plugins() -> List[Plugin]:
    """
    自动发现并返回当前 plugins 包下的所有 Plugin。
    返回：list[Plugin]：来自各模块的 PLUGINS 合并后的结果
    """
    plugins: List[Plugin] = []

    # __path__ 是包的搜索路径列表，pkgutil.iter_modules 会遍历其中的模块
    for modinfo in pkgutil.iter_modules(__path__):  # type: ignore[name-defined]
        name = modinfo.name

        # 跳过私有/测试模块（可按你习惯调整规则）
        if name.startswith("_"):
            continue

        full_name = f"{__name__}.{name}"  # e.g. visualization.plugins.ccf

        try:
            module = importlib.import_module(full_name)
        except Exception as e:
            warnings.warn(
                f"[visualization.plugins] 插件模块导入失败，已跳过: {full_name}\n"
                f"原因: {type(e).__name__}: {e}"
            )
            continue

        # 读取模块的 PLUGINS
        mod_plugins: Any = getattr(module, "PLUGINS", None)
        if mod_plugins is None:
            # 模块没定义 PLUGINS，说明它不是插件模块，直接跳过
            continue

        if not isinstance(mod_plugins, list):
            warnings.warn(
                f"[visualization.plugins] {full_name}.PLUGINS 不是 list，已跳过该模块。"
            )
            continue

        # 简单校验：list 里应当都是 Plugin（允许子类/兼容对象）
        for p in mod_plugins:
            if not isinstance(p, Plugin):
                warnings.warn(
                    f"[visualization.plugins] {full_name}.PLUGINS 中包含非 Plugin 对象: {p!r}，已跳过该对象。"
                )
                continue
            plugins.append(p)

    return plugins


__all__ = ["make_plugins"]
