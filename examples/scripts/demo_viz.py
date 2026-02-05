
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import sys
sys.path.append(r"C:\Users\Admin\Desktop\成像python包\seismocorr")
from seismocorr.visualization import help_plot, plot, set_default_backend, show

# =========================
# 数据生成
# =========================
def generate_ccf_data(n_tr: int = 40, n_lags: int = 401, *, seed: int = 0) -> Dict[str, Any]:
    """生成模拟 CCF 数据。

    Args:
        n_tr: 道数（traces 数量）。
        n_lags: lag 采样点数。
        seed: 随机种子，用于可复现。

    Returns:
        dict，包含：
            - cc: (n_tr, n_lags) 的 CCF 矩阵
            - lags: (n_lags,) 的 lag 轴
            - labels: (n_tr,) 的标签列表
            - dist_km: (n_tr,) 的距离数组（用于排序示例）
    """
    rng = np.random.default_rng(seed)

    lags = np.linspace(-20.0, 20.0, int(n_lags))
    cc = 0.08 * rng.standard_normal((int(n_tr), int(n_lags)))

    center = int(n_lags) // 2
    for i in range(int(n_tr)):
        shift = int(round((i - n_tr / 2.0) * 0.5))
        idx = int(np.clip(center + shift, 0, int(n_lags) - 1))
        cc[i, idx] += 1.0

    labels = [f"STA{i:02d}" for i in range(int(n_tr))]
    dist_km = rng.uniform(10.0, 300.0, size=int(n_tr))

    return {"cc": cc, "lags": lags, "labels": labels, "dist_km": dist_km}


def generate_beamforming_data(
    n_azimuth: int = 12,
    n_radius: int = 10,
    *,
    seed: int = 1,
) -> Dict[str, Any]:
    """生成模拟波束形成功率矩阵数据。

    Args:
        n_azimuth: 方位角采样数。
        n_radius: 慢度采样数（极坐标半径方向）。
        seed: 随机种子，用于可复现。

    Returns:
        dict，包含：
            - power: (n_radius, n_azimuth) 波束功率矩阵
            - azimuth_deg: (n_azimuth,) 方位角数组（0..360）
            - slowness_s_per_m: (n_radius,) 慢度数组
    """
    rng = np.random.default_rng(seed)

    azimuth_deg = np.linspace(0.0, 360.0, int(n_azimuth))
    slowness_s_per_m = np.linspace(0.0, 1.0, int(n_radius))

    # 原始生成是 (n_azimuth, n_radius)，插件约定这里用 (n_radius, n_azimuth)
    power = rng.random((int(n_azimuth), int(n_radius))).T

    return {
        "power": power,
        "azimuth_deg": azimuth_deg,
        "slowness_s_per_m": slowness_s_per_m,
    }


def generate_dispersion_energy_data(
    n_v: int = 50, n_f: int = 100, *, seed: int = 42
) -> Dict[str, Any]:
    """生成模拟的频散能量数据（能量与频率/速度的关系）。

    Args:
        n_v: 速度采样点数（或波数轴）。
        n_f: 频率采样点数。
        seed: 随机种子，用于可复现。

    Returns:
        dict，包含：
            - E: (n_v, n_f) 的能量矩阵。
            - f: (n_f,) 的频率轴。
            - v: (n_v,) 的速度轴（或群速度/相速度轴）。
    """
    rng = np.random.default_rng(seed)

    # 频率轴，假设频率从 0.01 Hz 到 10 Hz
    f = np.logspace(-2, 1, n_f)
    
    # 速度轴，假设速度从 1000 m/s 到 5000 m/s
    v = np.linspace(1000, 5000, n_v)

    # 模拟能量矩阵 E：假设能量与频率和速度相关，使用正弦波与指数衰减相结合
    E = np.zeros((n_v, n_f))
    for i in range(n_v):
        for j in range(n_f):
            E[i, j] = np.exp(-((f[j] - 5.0)**2) / 2.0) * np.sin(v[i] * f[j] / 1000)

    return {"E": E, "f": f, "v": v}


def generate_seismic_data(
    n_tr: int = 3,  # 道数（traces）
    n_samples: int = 1000,  # 每道采样点数
    duration: float = 20.0,  # 持续时间（秒）s
    *, 
    seed: int = 42
) -> Dict[str, Any]:
    """生成模拟地震波形数据。

    Args:
        n_tr: 道数（traces 数量）。
        n_samples: 每道的采样点数。
        duration: 每个波形的持续时间。
        seed: 随机种子，用于可复现。

    Returns:
        dict，包含：
            - traces: (n_tr, n_samples) 的地震波形矩阵
            - time: (n_samples,) 时间轴
            - labels: (n_tr,) 的标签列表
    """
    rng = np.random.default_rng(seed)

    time = np.linspace(0, duration, n_samples)  # 时间轴
    traces = np.zeros((n_tr, n_samples))  # 初始化地震波形矩阵

    for i in range(n_tr):
        # 使用正弦波模拟地震波形
        frequency = rng.uniform(0.1, 1.0)  # 随机频率
        traces[i, :] = np.sin(2 * np.pi * frequency * time) + 0.1 * rng.standard_normal(n_samples)

    # 标签为道的编号
    labels = [f"STA{i:02d}" for i in range(n_tr)]

    return {"traces": traces, "time": time, "labels": labels}


def generate_velocity_data() -> np.ndarray:
    """生成1D速度结构数据。"""
    # 层厚度 (单位：米)
    thickness = np.array([100, 200, 300, 400, 500])  # 示例层厚度，单位米

    # 不同速度模型的速度值 (单位：m/s)
    velocity_model_1 = np.array([1500, 1700, 1900, 2100, 2300])  # 模型1的速度值
    velocity_model_2 = np.array([1600, 1800, 2000, 2200, 2400])  # 模型2的速度值

    # 将层厚度和速度值组合成2D数组
    # 第一列为层厚度，后续列为每个模型的速度值
    data = np.vstack([thickness, velocity_model_1, velocity_model_2]).T  # 转置以符合 (n, m) 格式

    return data


def generate_vel2d_data(
    n_d: int = 60,          # distance 采样点数
    n_t: int = 25,          # 层数（深度方向）
    max_depth_m: float = 2000.0,
    *,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    生成模拟 2D 速度结构数据（dict 格式）：
      - Velocities: (n_t, n_d)
      - thickness:  (n_t,)
      - distance:   (n_d,)
    """
    rng = np.random.default_rng(seed)

    # distance 轴（km）
    distance = np.linspace(0.0, 30.0, int(n_d))

    # thickness（m）：这里用随机扰动后再归一化到 max_depth_m
    thickness_raw = rng.uniform(0.8, 1.2, size=int(n_t))
    thickness = thickness_raw / thickness_raw.sum() * float(max_depth_m)

    # 深度中心（m），用于合成一个随深度增大、随距离有起伏的速度场
    depth_edges = np.concatenate([[0.0], np.cumsum(thickness)])
    depth_centers = 0.5 * (depth_edges[:-1] + depth_edges[1:])  # (n_t,)

    # 合成速度场 Velocities (n_t, n_d)
    # 基础：随深度线性增加
    v0 = 1500.0 + 0.6 * depth_centers[:, None]  # (n_t,1)

    # 距离方向变化：小幅线性+正弦扰动
    lateral = 40.0 * np.sin(distance[None, :] / 4.0) + 5.0 * distance[None, :]

    # 加一点随机扰动（可选）
    noise = rng.normal(0.0, 15.0, size=(int(n_t), int(n_d)))

    Velocities = v0 + lateral + noise

    return {"Velocities": Velocities, "thickness": thickness, "distance": distance}


# =========================
# 绘图测试
# =========================
def test_ccf_wiggle_plot() -> None:
    """测试并绘制 CCF wiggle 图。"""
    set_default_backend("mpl")

    ccf = generate_ccf_data()

    fig = plot(
        "ccf.wiggle",
        data={"cc": ccf["cc"], "lags": ccf["lags"], "labels": ccf["labels"]},
        sort={
            "by": ccf["dist_km"],
            "ascending": True,
            "y_mode": "distance",
            "label": "Distance (km)",
        },
        highlights=[
            {"trace": 5, "t0": -2.0, "t1": 2.0},
            {"trace": 10, "color": "royalblue"},
            {"trace": 12, "t0": 1.0, "t1": 3.5, "color": "#ff00ff", "linewidth": 2.5},
        ],
        normalize=True,
        normalize_method="p95",
        clip=2.5,
        norm_method="trace",
        dy=1.0,
        excursion=2.0,
        bias=0.0,
        fill_mode="pos",
        fill_alpha=0.5,
        ytick_step=5,
        trace_step=1,
        show_zero_line=False,
    )

    show(fig)
    print(help_plot("ccf.wiggle"))


def test_beamforming_polar_heatmap() -> None:
    """测试并绘制波束形成极坐标热力图。"""
    data = generate_beamforming_data()

    fig = plot("beamforming.polar_heatmap", data)
    # 如需指定 plotly 后端：
    # fig = plot("beamforming.polar_heatmap", data, backend="plotly")

    show(fig)
    print(help_plot("beamforming.polar_heatmap"))


def test_dispersion_energy_plot() -> None:
    """测试并绘制频散能量图（f-v）。"""
    # 生成频散能量数据
    dispersion_data = generate_dispersion_energy_data()

    # 绘制频散能量图
    fig = plot(
        "heat_map",
        data={"E": dispersion_data["E"], "f": dispersion_data["f"], "v": dispersion_data["v"]},
        title="Dispersion Energy Map",
        x_label="Frequency (Hz)",
        y_label="Velocity (m/s)",
        x_lim=[8, 10], 
        y_lim=[3000, 5000],
        cmap = "jet",
        colorbar_label = "Energy",

    )

    # 显示图形
    show(fig)

    # 打印插件的帮助信息
    print(help_plot("heat_map"))


def test_seismic_waveform_plot() -> None:
    """测试并绘制地震波形图。"""
    set_default_backend("mpl")  # 设置默认后端为 matplotlib

    # 生成地震波形数据
    seismic_data = generate_seismic_data()

    fig = plot(
        "lines",  # 使用 line 插件绘制波形图
        data={"x": seismic_data["time"], "y": seismic_data["traces"]},
        title="Seismic Waveforms",  # 设置图标题
        x_label="Time (s)",  # 设置 x 轴标签
        y_label="Amplitude (m/s)",  # 设置 y 轴标签
        x_lim=[0, 20],  # 设置 x 轴范围
        y_lim=[-2, 2],  # 设置 y 轴范围
        colors = ["blue","black"],
        labels = ["s1", "s2"] )

    # 显示图形
    show(fig)

    # 打印插件的帮助信息
    print(help_plot("lines"))  # 根据插件 id 调用 help_plot 获取帮助信息


def test_velocity_structure_1d_plot() -> None:
    """测试并绘制1D速度结构图。"""
    set_default_backend("mpl")  # 设置默认后端为 matplotlib

    # 生成1D速度结构数据
    velocity_data = generate_velocity_data()

    # 绘制1D速度结构图
    fig = plot(
        "vel1d",  # 使用 vel1d 插件绘制1D速度结构图
        data=velocity_data,  # 数据格式：二维数组，第一列为层厚，后续列为每条曲线的速度值
        title="1D Velocity Structure",  # 设置图标题
        x_label="Velocity (m/s)",  # 设置 x 轴标签
        y_label="Depth (m)",  # 设置 y 轴标签
        x_lim=[1500, 2000],  # 设置 x 轴范围
        y_lim=[350, 0],  # 设置 y 轴范围（深度向下）
        colors=["blue", "green"],  # 设置两条曲线的颜色
        labels=["Model 1", "Model 2"],  # 设置两条曲线的标签
        invert_y=True  # 深度向下，y轴反向
    )

    # 显示图形
    show(fig)

    # 打印插件的帮助信息
    print(help_plot("vel1d"))  # 根据插件 id 调用 help_plot 获取帮助信息


def test_velocity_structure_2d_plot() -> None:
    """测试并绘制 2D 速度结构图（vel2d）：对比不同插值方法/强度，并检查深度头尾是否补齐。"""
    set_default_backend("mpl")

    data = generate_vel2d_data(n_d=80, n_t=30, max_depth_m=2500.0, seed=7)

    fig0 = plot(
        "vel2d",
        data=data,
        title="vel2d | no interp (baseline)",
        x_label="Distance (km)",
        y_label="Depth (m)",
        cmap="jet",
        colorbar_label="Velocity (m/s)",
        invert_y=False,
        interp=False,
        interp_method="none",
    )
    show(fig0)

    fig1 = plot(
        "vel2d",
        data=data,
        title="vel2d | linear interp (strength=3)",
        x_label="Distance (km)",
        y_label="Depth (m)",
        cmap="jet",
        colorbar_label="Velocity (m/s)",
        invert_y=False,
        interp=True,
        interp_method="linear",
        interp_strength=3.0,
    )
    show(fig1)

    fig2 = plot(
        "vel2d",
        data=data,
        title="vel2d | nearest interp (strength=6)",
        x_label="Distance (km)",
        y_label="Depth (m)",
        cmap="jet",
        colorbar_label="Velocity (m/s)",
        invert_y=False,
        interp=True,
        interp_method="nearest",
        interp_strength=6.0,
    )
    show(fig2)

    fig3 = plot(
        "vel2d",
        data=data,
        title="vel2d | linear interp (dx=0.2 km, dy=5 m)",
        x_label="Distance (km)",
        y_label="Depth (m)",
        cmap="jet",
        colorbar_label="Velocity (m/s)",
        invert_y=False,
        interp=True,
        interp_method="linear",
        dx_km=0.2,
        dy_m=5.0,
    )
    show(fig3)

    fig4 = plot(
        "vel2d",
        data=data,
        title="vel2d | cubic interp (strength=4) [requires scipy]",
        x_label="Distance (km)",
        y_label="Depth (m)",
        cmap="jet_r",
        colorbar_label="Velocity (m/s)",
        invert_y=True,
        interp=True,
        interp_method="cubic",
        interp_strength=4.0,
    )
    show(fig4)

    print(help_plot("vel2d"))


def main() -> None:
    """运行示例测试。"""
    # test_ccf_wiggle_plot()
    # test_beamforming_polar_heatmap()
    # test_dispersion_energy_plot()
    # test_seismic_waveform_plot()
    # test_velocity_structure_1d_plot()
    test_velocity_structure_2d_plot()
if __name__ == "__main__":
    main()
