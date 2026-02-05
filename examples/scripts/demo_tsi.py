from __future__ import annotations

from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from seismocorr.core.correlation.correlation import CorrelationConfig, CorrelationEngine
from seismocorr.core.correlation.stacking import stack_ccfs
from seismocorr.plugins.processing.three_stations_interferometry import (
    ThreeStationConfig,
    ThreeStationInterferometry,
)


def snr_peak_over_rms(
    lags: np.ndarray,
    ccf: np.ndarray,
    *,
    signal_win: Tuple[float, float],
    noise_win: Tuple[float, float],
    eps: float = 1e-12,
) -> Dict[str, float]:
    """计算 peak/RMS 定义的 SNR。

    SNR = max(|ccf| in signal window) / RMS(ccf in noise window)
    """
    t1, t2 = signal_win
    n1, n2 = noise_win
    if t1 > t2 or n1 > n2:
        raise ValueError("window order must be (start, end) with start <= end")

    lags = np.asarray(lags, dtype=float).reshape(-1)
    ccf = np.asarray(ccf, dtype=float).reshape(-1)
    if lags.shape != ccf.shape:
        raise ValueError("lags and ccf must have the same shape")

    sig_mask = (lags >= t1) & (lags <= t2)
    noi_mask = (lags >= n1) & (lags <= n2)

    if not np.any(sig_mask):
        raise ValueError("signal window has no samples; adjust signal_win")
    if not np.any(noi_mask):
        raise ValueError("noise window has no samples; adjust noise_win")

    sig = ccf[sig_mask]
    noi = ccf[noi_mask]

    peak_idx = int(np.argmax(np.abs(sig)))
    peak_val = float(np.abs(sig[peak_idx]))
    peak_time = float(lags[sig_mask][peak_idx])

    noise_rms = float(np.sqrt(np.mean(noi**2)))
    snr = float(peak_val / (noise_rms + eps))

    return {
        "snr": snr,
        "peak": peak_val,
        "noise_rms": noise_rms,
        "peak_time": peak_time,
    }


def align_by_interpolate(
    lags_ref: np.ndarray,
    y_ref: np.ndarray,
    lags_other: np.ndarray,
    y_other: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """把 other 插值到 ref 的 lag 网格上，并裁剪到共同 lag 范围。

    这能避免 “只裁剪再硬截断长度” 导致的 lag 不一一对应（绘图像显示不全/错位）。
    """
    lags_ref = np.asarray(lags_ref, dtype=float).reshape(-1)
    y_ref = np.asarray(y_ref, dtype=float).reshape(-1)
    lags_other = np.asarray(lags_other, dtype=float).reshape(-1)
    y_other = np.asarray(y_other, dtype=float).reshape(-1)

    if lags_ref.shape != y_ref.shape or lags_other.shape != y_other.shape:
        raise ValueError("lags and y must have matching shapes for both series")

    # 共同 lag 范围
    tmin = float(max(lags_ref.min(), lags_other.min()))
    tmax = float(min(lags_ref.max(), lags_other.max()))
    if tmin >= tmax:
        raise ValueError("no overlapping lag range to align")

    mask = (lags_ref >= tmin) & (lags_ref <= tmax)
    if not np.any(mask):
        raise ValueError("no samples in overlapping lag range after masking")

    lags = lags_ref[mask]
    y1 = y_ref[mask]
    y2 = np.interp(lags, lags_other, y_other)  # 插值对齐
    return lags, y1, y2


def make_demo_traces(
    n_stations: int,
    n_samples: int,
    *,
    seed: int = 0,
) -> np.ndarray:
    """生成演示用 traces（带公共成分）。"""
    rng = np.random.default_rng(seed)
    common = rng.standard_normal(int(n_samples))
    traces = np.stack(
        [0.5 * common + rng.standard_normal(int(n_samples)) for _ in range(int(n_stations))],
        axis=0,
    )
    return traces


def _print_lag_info(name: str, lags: np.ndarray) -> None:
    lags = np.asarray(lags, float).ravel()
    dt = float(np.median(np.diff(lags))) if lags.size > 1 else float("nan")
    print(f"{name}: n={lags.size}, range=[{lags.min():.6f}, {lags.max():.6f}], dt~{dt:.6g}, "
          f"closest_to_0={lags[np.argmin(np.abs(lags))]:.6e}")


def main() -> None:
    """运行：直接互相关 vs 三站干涉（叠加）并比较 SNR。"""
    sr = 200.0
    n_stations = 60
    n_samples = int(sr * 60.0)

    traces = make_demo_traces(n_stations=n_stations, n_samples=n_samples, seed=0)

    engine = CorrelationEngine()
    max_lag_1 = 2.0
    cc_cfg = CorrelationConfig(method="freq-domain", max_lag=max_lag_1, nfft=None)

    def xcorr_func(x: np.ndarray, y: np.ndarray, *, config: CorrelationConfig):
        return engine.compute_cross_correlation(x, y, sampling_rate=sr, config=config)

    tsi = ThreeStationInterferometry(
        sampling_rate=sr,
        xcorr_func=xcorr_func,
        cfg=ThreeStationConfig(mode="auto", max_lag2=2.0),
    )

    i, j = 10, 40
    k_list: Optional[np.ndarray] = None

    # A) 直接互相关
    lags_ij, ccf_ij = xcorr_func(traces[i], traces[j], config=cc_cfg)

    # B) 三站干涉：得到多条 ncf_ijk
    res = tsi.compute_pair(traces, i=i, j=j, k_list=k_list, config=cc_cfg)
    lags2 = res["lags2"]
    ccfs_ijk = res["ccfs"]

    stacked_3s_linear = stack_ccfs(ccfs_ijk, method="linear")
    stacked_3s_pws = stack_ccfs(ccfs_ijk, method="pws", power=2)

    # ===== 自检：看看两者 lag 网格是否一致（你之前“显示不全/错位”的关键）=====
    print("\n=== Lag diagnostics ===")
    _print_lag_info("Direct lags", lags_ij)
    _print_lag_info("3S lags2  ", lags2)

    # ===== 真正对齐：把 3S 结果插值到 direct 的 lag 网格 =====
    lags, direct_aligned, lin_aligned = align_by_interpolate(lags_ij, ccf_ij, lags2, stacked_3s_linear)
    _, _, pws_aligned = align_by_interpolate(lags_ij, ccf_ij, lags2, stacked_3s_pws)

    signal_win = (-0.3, 0.3)
    noise_win = (1.0, 2.0)

    snr_direct = snr_peak_over_rms(lags, direct_aligned, signal_win=signal_win, noise_win=noise_win)
    snr_3s_lin = snr_peak_over_rms(lags, lin_aligned, signal_win=signal_win, noise_win=noise_win)
    snr_3s_pws = snr_peak_over_rms(lags, pws_aligned, signal_win=signal_win, noise_win=noise_win)

    print("\n=== SNR comparison (peak/RMS) ===")
    print(
        "Direct CCF:       "
        f"SNR={snr_direct['snr']:.3f}, peak={snr_direct['peak']:.4g} "
        f"at {snr_direct['peak_time']:.3f}s, noise_rms={snr_direct['noise_rms']:.4g}"
    )
    print(
        "3-station linear: "
        f"SNR={snr_3s_lin['snr']:.3f}, peak={snr_3s_lin['peak']:.4g} "
        f"at {snr_3s_lin['peak_time']:.3f}s, noise_rms={snr_3s_lin['noise_rms']:.4g}"
    )
    print(
        "3-station PWS:    "
        f"SNR={snr_3s_pws['snr']:.3f}, peak={snr_3s_pws['peak']:.4g} "
        f"at {snr_3s_pws['peak_time']:.3f}s, noise_rms={snr_3s_pws['noise_rms']:.4g}"
    )

    # 可视化对比（所有曲线共用同一条 lag 轴 -> 绝对不会“显示不全/错位”）
    plt.figure(figsize=(10, 4))
    plt.plot(lags, direct_aligned, label=f"Direct CCF (SNR={snr_direct['snr']:.2f})")
    plt.plot(lags, lin_aligned, label=f"3-station linear stack (SNR={snr_3s_lin['snr']:.2f})")
    plt.plot(lags, pws_aligned, label=f"3-station PWS stack (SNR={snr_3s_pws['snr']:.2f})")
    plt.axvspan(signal_win[0], signal_win[1], alpha=0.15, label="signal window")
    plt.axvspan(noise_win[0], noise_win[1], alpha=0.10, label="noise window")
    plt.title(f"Direct CCF vs Three-station Interferometry (pair {i}-{j})")
    plt.xlabel("Lag (s)")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
