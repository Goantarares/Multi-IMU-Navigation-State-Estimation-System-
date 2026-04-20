import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# ============================================================
# INCARCARE
# ============================================================
df = pd.read_csv('imu_stationary_data_brut.csv')

# Corecteaza header-ul (acelasi bug ca inainte)
correct_columns = ['packet_id'] + [f'timestamp_s{i}' for i in range(1, 6)]
for axis in ['gx', 'gy', 'gz']:
    for s in range(1, 6):
        correct_columns.append(f'{axis}_s{s}')
for axis in ['ax', 'ay', 'az']:
    for s in range(1, 6):
        correct_columns.append(f'{axis}_s{s}')
df.columns = correct_columns

# Frecventa reala din timestamp
dt_arr = df['timestamp_s5'].diff().dropna().values / 1_000_000
fs = 1.0 / np.mean(dt_arr)
print(f"Frecventa: {fs:.2f} Hz")
print(f"Durata: {len(df)/fs:.1f} s")
print(f"Total cadre: {len(df)}")

# ============================================================
# ANALIZA BIAS SI ZGOMOT - SENZORUL x
# ============================================================
s = 4
gyro_raw  = df[[f'gx_s{s}', f'gy_s{s}', f'gz_s{s}']].values  # dps
accel_raw = df[[f'ax_s{s}', f'ay_s{s}', f'az_s{s}']].values  # g

# Bias = media pe intervalul static
gyro_bias_dps  = np.mean(gyro_raw,  axis=0)
accel_bias_g   = np.mean(accel_raw, axis=0)
accel_bias_g[2] -= 1.0  # scade 1g de pe axa Z (gravitatie reala)

print(f"\n=== BIAS GIROSCOP S{s} ===")
print(f"  X: {gyro_bias_dps[0]:+.6f} dps")
print(f"  Y: {gyro_bias_dps[1]:+.6f} dps")
print(f"  Z: {gyro_bias_dps[2]:+.6f} dps")
print(f"  Norma: {np.linalg.norm(gyro_bias_dps):.6f} dps")

print(f"\n=== BIAS ACCELEROMETRU S{s} (dupa scaderea gravitatiei) ===")
print(f"  X: {accel_bias_g[0]:+.6f} g")
print(f"  Y: {accel_bias_g[1]:+.6f} g")
print(f"  Z: {accel_bias_g[2]:+.6f} g")

# Zgomot = deviatia standard
gyro_std  = np.std(gyro_raw,  axis=0)
accel_std = np.std(accel_raw, axis=0)

print(f"\n=== ZGOMOT (std dev) ===")
print(f"  Giroscop  (dps): X={gyro_std[0]:.6f}  Y={gyro_std[1]:.6f}  Z={gyro_std[2]:.6f}")
print(f"  Accel      (g):  X={accel_std[0]:.6f}  Y={accel_std[1]:.6f}  Z={accel_std[2]:.6f}")

# R pentru EKF (varianta accelerometrului in m/s²)
accel_std_ms2 = accel_std * 9.81
print(f"\n=== VALORI PENTRU config.py ===")
print(f"GYRO_BIAS_INIT = np.array([{gyro_bias_dps[0]:.8f}, "
      f"{gyro_bias_dps[1]:.8f}, {gyro_bias_dps[2]:.8f}]) * DEG2RAD")
print(f"R_ACCEL_STD = {np.mean(accel_std_ms2):.6f}   # m/s²")

# ============================================================
# ALLAN VARIANCE
# ============================================================
def allan_deviation(data, fs):
    N = len(data)
    max_m = int(N / 2)
    m_values = np.unique(
        np.logspace(0, np.log10(max_m), 300).astype(int)
    )
    tau_list  = []
    adev_list = []
    for m in m_values:
        tau = m / fs
        n_clusters = N // m
        if n_clusters < 2:
            break
        clusters = data[:n_clusters * m].reshape(n_clusters, m)
        means    = clusters.mean(axis=1)
        avar     = 0.5 * np.mean(np.diff(means)**2)
        tau_list.append(tau)
        adev_list.append(np.sqrt(avar))
    return np.array(tau_list), np.array(adev_list)

print(f"\n=== ALLAN VARIANCE ===")
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle(f'Allan Deviation - Senzor S{s} static', fontsize=13)

axes_labels = ['X', 'Y', 'Z']

for i in range(3):
    # Giroscop
    gyro_rads = gyro_raw[:, i] * np.pi / 180.0
    tau, adev = allan_deviation(gyro_rads, fs)

    ax = axes[0, i]
    ax.loglog(tau, adev, 'b-', linewidth=1.5)
    ax.set_title(f'Giroscop {axes_labels[i]}')
    ax.set_xlabel('τ (s)')
    ax.set_ylabel('Allan Dev (rad/s)')
    ax.grid(True, which='both', alpha=0.3)

    # Citeste ARW la tau=1s
    if tau[-1] >= 1.0:
        idx_1s = np.argmin(np.abs(tau - 1.0))
        arw = adev[idx_1s]
        ax.axvline(x=1.0, color='r', linestyle='--', alpha=0.5)
        ax.axhline(y=arw, color='r', linestyle='--', alpha=0.5)
        ax.text(0.05, 0.95, f'ARW={arw:.2e} rad/s',
                transform=ax.transAxes, fontsize=9,
                verticalalignment='top', color='red')

    # Minimul = Bias Instability
    min_idx = np.argmin(adev)
    bi = adev[min_idx]
    ax.scatter([tau[min_idx]], [bi], color='green', s=50, zorder=5)
    ax.text(0.05, 0.80, f'BI={bi:.2e} rad/s',
            transform=ax.transAxes, fontsize=9,
            verticalalignment='top', color='green')

    # Accelerometru
    accel_ms2_col = accel_raw[:, i] * 9.81
    tau_a, adev_a = allan_deviation(accel_ms2_col, fs)

    ax2 = axes[1, i]
    ax2.loglog(tau_a, adev_a, 'r-', linewidth=1.5)
    ax2.set_title(f'Accelerometru {axes_labels[i]}')
    ax2.set_xlabel('τ (s)')
    ax2.set_ylabel('Allan Dev (m/s²)')
    ax2.grid(True, which='both', alpha=0.3)

    if tau_a[-1] >= 1.0:
        idx_1s_a = np.argmin(np.abs(tau_a - 1.0))
        vrw = adev_a[idx_1s_a]
        ax2.text(0.05, 0.95, f'VRW={vrw:.2e} m/s²',
                transform=ax2.transAxes, fontsize=9,
                verticalalignment='top', color='darkred')

plt.tight_layout()
plt.savefig('allan_variance_static_s4.png', dpi=150, bbox_inches='tight')
print("Grafic salvat: allan_variance_static_s4.png")