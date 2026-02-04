import numpy as np
import matplotlib.pyplot as plt

from seismocorr.plugins.processing.beamforming import Beamformer


def band_limited_noise(rng, n, fs, fmin, fmax):
    """用频域置带生成带限噪声"""
    freqs = np.fft.rfftfreq(n, d=1 / fs)
    spec = rng.normal(size=freqs.size) + 1j * rng.normal(size=freqs.size)
    band = (freqs >= fmin) & (freqs <= fmax)
    spec[~band] = 0.0
    x = np.fft.irfft(spec, n=n)
    x = x / (np.std(x) + 1e-12)
    return x


def simulate_ambient_plane_wave(
    fs: float,
    duration_s: float,
    xy_m: np.ndarray,
    azimuth_deg: float,
    slowness_s_per_m: float,
    fmin_hz: float,
    fmax_hz: float,
    snr_db: float = 0.0,
    seed: int = 0,
):
    """
    合成“背景噪声里有一个占优方向”的平面波模型：
      x_i(t) = a * n_band(t - τ_i) + noise
      τ_i = s * (r_i · u)
    """
    rng = np.random.default_rng(seed)
    n_chan = xy_m.shape[0]
    n_samples = int(round(duration_s * fs))
    t = np.arange(n_samples) / fs

    az = np.deg2rad(azimuth_deg)
    u = np.array([np.sin(az), np.cos(az)])  # East, North
    tau = slowness_s_per_m * (xy_m @ u)     # (n_chan,)

    src = band_limited_noise(rng, n_samples, fs, fmin_hz, fmax_hz)

    sig = np.zeros((n_chan, n_samples), float)
    for c in range(n_chan):
        tt = t - tau[c]
        sig[c] = np.interp(tt, t, src, left=0.0, right=0.0)

    signal_power = np.mean(sig**2)
    snr_lin = 10 ** (snr_db / 10.0)
    noise_power = signal_power / snr_lin if snr_lin > 0 else signal_power * 10
    noise = rng.normal(scale=np.sqrt(noise_power), size=sig.shape)

    return sig + noise


def simulate_point_source(
    fs: float,
    duration_s: float,
    xy_m: np.ndarray,
    src_xy_m: np.ndarray,
    velocity_m_per_s: float,
    fmin_hz: float,
    fmax_hz: float,
    snr_db: float = 0.0,
    seed: int = 0,
):
    """
    合成“点源辐射”的带限噪声模型：
      x_c(t) = a * s(t - τ_c) + noise
      τ_c = ||r_c - r_src|| / v
    """
    rng = np.random.default_rng(seed)
    n_chan = xy_m.shape[0]
    n_samples = int(round(duration_s * fs))
    t = np.arange(n_samples) / fs

    src = band_limited_noise(rng, n_samples, fs, fmin_hz, fmax_hz)

    d = np.linalg.norm(xy_m - src_xy_m[None, :], axis=1)  # (n_chan,)
    tau = d / velocity_m_per_s

    sig = np.zeros((n_chan, n_samples), float)
    for c in range(n_chan):
        tt = t - tau[c]
        sig[c] = np.interp(tt, t, src, left=0.0, right=0.0)

    signal_power = np.mean(sig**2)
    snr_lin = 10 ** (snr_db / 10.0)
    noise_power = signal_power / snr_lin if snr_lin > 0 else signal_power * 10
    noise = rng.normal(scale=np.sqrt(noise_power), size=sig.shape)

    return sig + noise


def compute_travel_times_point_source_grid(
    grid_xy_m: np.ndarray,   # (n_grid, 2)
    xy_m: np.ndarray,        # (n_chan, 2)
    velocity_m_per_s: float,
    ref_chan: int = 0,
):
    """
    对点源 BP：计算走时表 tau[g,c] = ||r_c - r_g||/v，并转成相对走时减少偏差：
      tau_rel[g,c] = tau[g,c] - tau[g,ref]
    """
    # (n_grid, n_chan)
    d = np.linalg.norm(grid_xy_m[:, None, :] - xy_m[None, :, :], axis=-1)
    tau = d / velocity_m_per_s
    tau_rel = tau - tau[:, [ref_chan]]
    return tau_rel


def test_delay_and_sum(bf: Beamformer, xy_m, fs, fmin, fmax):
    # 平面波真值
    true_az = 60.0
    true_v = 1200.0
    true_s = 1.0 / true_v
    snr_db = -3.0
    duration_s = 10 * 60.0

    data = simulate_ambient_plane_wave(
        fs=fs,
        duration_s=duration_s,
        xy_m=xy_m,
        azimuth_deg=true_az,
        slowness_s_per_m=true_s,
        fmin_hz=fmin,
        fmax_hz=fmax,
        snr_db=snr_db,
        seed=1,
    )

    az_scan = np.arange(0, 360, 2.0)
    s_scan = np.linspace(0.0003, 0.0030, 136)

    res = bf.delay_and_sum(data=data, xy_m=xy_m, azimuth_deg=az_scan, slowness_s_per_m=s_scan)

    idx = np.unravel_index(np.argmax(res.power), res.power.shape)
    est_s = res.slowness_s_per_m[idx[0]]
    est_az = res.azimuth_deg[idx[1]]

    print("=== Beamforming (plane wave) ===")
    print(f"True az={true_az:.1f} deg, True s={true_s:.6f} s/m (v={true_v:.1f} m/s)")
    print(f"Est  az={est_az:.1f} deg, Est  s={est_s:.6f} s/m (v={1/est_s:.1f} m/s)")

    plt.figure()
    plt.imshow(
        res.power,
        aspect="auto",
        origin="lower",
        extent=[
            res.azimuth_deg[0], res.azimuth_deg[-1],
            res.slowness_s_per_m[0], res.slowness_s_per_m[-1],
        ],
    )
    plt.xlabel("Azimuth (deg, from North clockwise)")
    plt.ylabel("Slowness (s/m)")
    plt.title("Beamforming Power (Bartlett) - Plane wave")
    plt.colorbar(label="Power")
    plt.scatter([true_az], [true_s], marker="o", label="True")
    plt.scatter([est_az], [est_s], marker="x", label="Estimated")
    plt.legend()


def test_backprojection(bf: Beamformer, xy_m, fs, fmin, fmax):
    true_src = np.array([25.0, -35.0])  # (x=East, y=North) in meters
    true_v = 1000.0
    snr_db = -6.0
    duration_s = 6 * 60.0

    data = simulate_point_source(
        fs=fs,
        duration_s=duration_s,
        xy_m=xy_m,
        src_xy_m=true_src,
        velocity_m_per_s=true_v,
        fmin_hz=fmin,
        fmax_hz=fmax,
        snr_db=snr_db,
        seed=2,
    )

    # 扫描网格（覆盖阵列附近）
    gx = np.linspace(-80, 80, 81)
    gy = np.linspace(-80, 80, 81)
    grid_x, grid_y = np.meshgrid(gx, gy, indexing="xy")
    grid_xy = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)  # (n_grid,2)

    tau_rel = compute_travel_times_point_source_grid(
        grid_xy_m=grid_xy,
        xy_m=xy_m,
        velocity_m_per_s=true_v,
        ref_chan=0,
    )

    res = bf.backproject(
        data=data,
        travel_times_s=tau_rel,
        return_frame_power=False,   # 你也可以 True 看时变能量
        chunk_size=512,
    )

    est_idx = int(np.argmax(res.power))
    est_src = grid_xy[est_idx]

    print("\n=== Backprojection (point source) ===")
    print(f"True src (x,y)=({true_src[0]:.1f}, {true_src[1]:.1f}) m, v={true_v:.1f} m/s")
    print(f"Est  src (x,y)=({est_src[0]:.1f}, {est_src[1]:.1f}) m")

    power_map = res.power.reshape(grid_x.shape)

    plt.figure()
    plt.imshow(
        power_map,
        origin="lower",
        extent=[gx[0], gx[-1], gy[0], gy[-1]],
        aspect="equal",
    )
    plt.colorbar(label="Power")
    plt.scatter([true_src[0]], [true_src[1]], marker="o", label="True src")
    plt.scatter([est_src[0]], [est_src[1]], marker="x", label="Estimated")
    plt.scatter(xy_m[:, 0], xy_m[:, 1], s=20, marker="^", label="Stations")
    plt.xlabel("x (East, m)")
    plt.ylabel("y (North, m)")
    plt.title("Backprojection Power - Point source")
    plt.legend()


def make_array_square(spacing=50.0, size = 4):
    grid = np.array([(i, j) for j in range(size) for i in range(size)], dtype=float)
    xy_m = (grid - grid.mean(axis=0)) * spacing
    return xy_m


def main():

    xy_m = make_array_square()

    fs = 50.0
    fmin, fmax = 5, 15

    bf = Beamformer(
        fs=fs,
        fmin=fmin,
        fmax=fmax,
        frame_len_s=20.0,
        hop_s=10.0,
        window="hann",
        whiten=True,
    )

    test_delay_and_sum(bf, xy_m, fs, fmin, fmax)
    test_backprojection(bf, xy_m, fs, fmin, fmax)

    plt.show()


if __name__ == "__main__":
    main()
