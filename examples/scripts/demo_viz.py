import numpy as np
from seismocorr.visualization import plot, set_default_backend, show, help_plot

# -------------------------
# 1) 生成 CCF 数据
# -------------------------
def generate_ccf_data(n_tr=40, n_lags=401):
    """
    生成模拟的 CCF 数据
    """
    np.random.seed(0)
    lags = np.linspace(-20, 20, n_lags)
    
    # 模拟 CCF：噪声 + 中心脉冲 + 不同道的轻微延迟
    cc = 0.08 * np.random.randn(n_tr, n_lags)
    center = n_lags // 2
    for i in range(n_tr):
        shift = int((i - n_tr / 2) * 0.5)  # 随道变化一点点延迟
        idx = np.clip(center + shift, 0, n_lags - 1)
        cc[i, idx] += 1.0

    labels = [f"STA{i:02d}" for i in range(n_tr)]
    dist_km = np.random.uniform(10, 300, size=n_tr)  # 距离数组（示例：随机生成并排序前混乱）
    
    return {"cc": cc, "lags": lags, "labels": labels, "dist_km": dist_km}

# -------------------------
# 2) 生成波束形成数据
# -------------------------
def generate_beamforming_data(n_azimuth=100, n_radius=50):
    """
    生成模拟的波束形成功率矩阵数据
    """
    # 生成一个随机的波束形成功率矩阵
    power = np.random.rand(n_azimuth, n_radius)  # 形状 (n_azimuth, n_radius)
    
    # 生成方位角（从0到360度）
    azimuth_deg = np.linspace(0, 360, n_azimuth)  # 100个方位角
    
    # 生成慢度值（假设为从 0.1 到 1.0 的线性变化）
    slowness_s_per_m = np.linspace(0.1, 1.0, n_radius)  # 50个慢度值
    
    # 打包数据
    data = {
        "power": power.T,  # 波束形成功率矩阵
        "azimuth_deg": azimuth_deg,  # 方位角数组
        "slowness_s_per_m": slowness_s_per_m,  # 慢度值数组
    }
    
    return data

# -------------------------
# 3) 测试 CCF Wiggle 图绘制
# -------------------------
def test_ccf_wiggle_plot():
    """
    测试并绘制 CCF Wiggle 图
    """
    # 设置默认后端为 matplotlib
    set_default_backend("mpl")

    # 生成 CCF 数据
    ccf_data = generate_ccf_data()

    # 绘制 CCF Wiggle 图
    fig = plot(
        "ccf.wiggle",
        data={"cc": ccf_data["cc"], "lags": ccf_data["lags"], "labels": ccf_data["labels"]},
        normalize=True,
        clip=0.9,
        sort={"by": ccf_data["dist_km"], "ascending": True, "y_mode": "index", "label": "Distance (km)"},
        highlights=[
            {"trace": 5, "t0": -2.0, "t1": 2.0},  # 默认红色
            {"trace": 10, "color": "blue"},  # 全时段蓝色
            {"trace": 12, "t0": 1.0, "t1": 3.5, "color": "#ff00ff", "linewidth": 3},
        ],
        scale=5,
        backend='plotly'
    )

    # 显示图表
    show(fig)

    # 打印帮助信息
    print(help_plot('ccf.wiggle'))

# -------------------------
# 4) 测试波束形成极坐标热力图
# -------------------------
def test_beamforming_polar_heatmap():
    """
    测试并绘制波束形成极坐标热力图
    """
    # 生成波束形成数据
    beamforming_data = generate_beamforming_data()

    # 绘制极坐标热力图
    # fig = plot("beamforming.polar_heatmap", beamforming_data)
    fig = plot("beamforming.polar_heatmap", data=beamforming_data,backend='plotly')
    # 显示图表
    show(fig)

    # 打印帮助信息
    print(help_plot('beamforming.polar_heatmap'))

# -------------------------
# 5) 主程序：运行测试
# -------------------------
if __name__ == "__main__":
    # 测试 CCF Wiggle 图绘制
    test_ccf_wiggle_plot()
    
    # # 测试波束形成极坐标热力图
    test_beamforming_polar_heatmap()
