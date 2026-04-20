import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ekf_mare import EKF_MARE
from quaternion_utils import quat_to_euler, quat_from_gyro, quat_multiply, quat_normalize
from quaternion_utils import quat_to_rotation_matrix
from config import G, DEG2RAD

# ============================================================
# CONFIGURARE
# ============================================================
DATA_FILE  = 'dataset_filtru_mare_combined.csv'
SENSOR_IDS = [1, 2, 3, 4, 5]

# Zgomot de proces Q (10x10)
Q_DIAG = np.array([
    2.05e-9, 2.05e-9, 2.05e-9, 2.05e-9,  # q - din ARW
    1e-3,    1e-3,    1e-3,               # v
    1e-4,    1e-4,    1e-4,               # p
])

# R de baza accelerometru (va fi scalat dinamic cu sigma_q)
R_ACCEL_STD = 0.17   # m/s² - punct de start, ajusteaza dupa NIS

# Covarianta initiala
SIGMA_INIT_DIAG = np.array([
    1e-6, 1e-6, 1e-6, 1e-6,   # q
    1.0,  1.0,  1.0,           # v
    1e-6, 1e-6, 1e-6,          # p
])

# ============================================================
# INCARCARE DATE
# ============================================================
print("=" * 60)
print("EKF MARE — Fuziune 5 senzori")
print("=" * 60)

print(f"\n[1] Incarcare date din {DATA_FILE}...")
df = pd.read_csv(DATA_FILE)
print(f"    Total cadre: {len(df)}")
print(f"    Coloane: {len(df.columns)}")
print(f"    dt mediu: {df['dt'].mean()*1000:.2f} ms (~{1/df['dt'].mean():.1f} Hz)")

dt_data = df['dt'].values

# Extrage datele per senzor
gyros         = {}
accels        = {}
sigma_traces  = {}

for s in SENSOR_IDS:
    gyros[s] = df[[f'gx_corr_s{s}', f'gy_corr_s{s}', f'gz_corr_s{s}']].values  # rad/s
    accels[s] = df[[f'ax_corr_s{s}', f'ay_corr_s{s}', f'az_corr_s{s}']].values  # m/s²
    sigma_traces[s] = df[f'sigma_q_trace_s{s}'].values

print(f"\n    Sigma_q_trace medie per senzor:")
for s in SENSOR_IDS:
    print(f"      S{s}: {np.mean(sigma_traces[s]):.2e}  "
          f"(min={np.min(sigma_traces[s]):.2e}, max={np.max(sigma_traces[s]):.2e})")

# ============================================================
# INITIALIZARE EKF MARE
# ============================================================
print(f"\n[2] Initializare EKF Mare...")

# Estimeaza orientarea initiala din primul cadru al senzorului cel mai sigur
# (cel cu sigma_q_trace minima la primul cadru)
sigma_first = {s: sigma_traces[s][0] for s in SENSOR_IDS}
s_best      = min(sigma_first, key=sigma_first.get)
print(f"    Senzorul cel mai sigur la start: S{s_best} "
      f"(sigma_q={sigma_first[s_best]:.2e})")

# Quaternionul initial din senzorul cel mai sigur
q0_init = df[[f'qw_s{s_best}', f'qx_s{s_best}',
              f'qy_s{s_best}', f'qz_s{s_best}']].iloc[0].values

ekf = EKF_MARE(
    n_sensors=len(SENSOR_IDS),
    q0_init=q0_init,
    sigma_init_diag=SIGMA_INIT_DIAG,
    Q_diag=Q_DIAG,
    R_accel_std=R_ACCEL_STD,
)
print("    EKF Mare initializat cu succes")

# ============================================================
# INTEGRARE BRUTA (senzor de referinta S5) — pentru comparatie
# ============================================================
print(f"\n[3] Integrare bruta bruta senzor S5 (pentru comparatie)...")

raw_euler = []
q_raw     = np.array([1.0, 0.0, 0.0, 0.0])
vel_raw   = np.zeros(3)
pos_raw   = np.zeros(3)
raw_pos   = [np.zeros(3)]

for i in range(len(df)):
    dt = dt_data[i]
    if dt <= 0:
        continue
    gyro_raw  = gyros[5][i]        # rad/s deja
    accel_raw = accels[5][i]       # m/s² deja

    dq    = quat_from_gyro(gyro_raw, dt)
    q_raw = quat_normalize(quat_multiply(q_raw, dq))

    R_raw       = quat_to_rotation_matrix(q_raw)
    a_glob      = R_raw @ accel_raw - np.array([0, 0, G])
    vel_raw    += a_glob * dt
    pos_raw     = pos_raw + vel_raw * dt + 0.5 * a_glob * dt**2

    raw_euler.append(quat_to_euler(q_raw))
    raw_pos.append(pos_raw.copy())

raw_euler = np.array(raw_euler)
raw_pos   = np.array(raw_pos[1:])
print("    Integrare bruta completa")

# ============================================================
# RULARE EKF MARE
# ============================================================
print(f"\n[4] Rulare EKF Mare...")

ekf_euler = []
ekf_pos   = []
ekf_vel   = []

for i in range(len(df)):
    dt = dt_data[i]

    # Extrage datele tuturor senzorilor la pasul i
    gyros_i        = [gyros[s][i]        for s in SENSOR_IDS]
    accels_i       = [accels[s][i]       for s in SENSOR_IDS]
    sigma_traces_i = [sigma_traces[s][i] for s in SENSOR_IDS]

    # Prediction cu giroscop si accelerometru fuzionat ponderat
    ekf.predict(gyros_i, accels_i, sigma_traces_i, dt)

    # Update secvential cu accelerometrul fiecarui senzor
    ekf.update_all_sensors(accels_i, sigma_traces_i)

    ekf_euler.append(ekf.get_orientation_euler())
    ekf_pos.append(ekf.get_position())
    ekf_vel.append(ekf.get_velocity())

    if i % 500 == 0:
        state = ekf.get_state_summary()
        print(f"    Cadru {i:5d}/{len(df)} | "
              f"Roll={state['roll_deg']:+6.1f}° "
              f"Pitch={state['pitch_deg']:+6.1f}° "
              f"Yaw={state['yaw_deg']:+6.1f}° | "
              f"Pos=[{state['pos_m'][0]:+.2f}, "
              f"{state['pos_m'][1]:+.2f}, "
              f"{state['pos_m'][2]:+.2f}] m")

ekf_euler = np.array(ekf_euler)
ekf_pos   = np.array(ekf_pos)
ekf_vel   = np.array(ekf_vel)

# ============================================================
# NIS
# ============================================================
nis_stats = ekf.get_nis_stats()
if nis_stats:
    print(f"\n    === NIS ===")
    print(f"    Mediana:      {nis_stats['mediana']:.3f}  (asteptat: 3.0)")
    print(f"    Medie (trim): {nis_stats['medie']:.3f}")
    print(f"    Medie bruta:  {nis_stats['medie_bruta']:.3f}")
    print(f"    Outlieri:     {nis_stats['n_outlieri']}")
    if nis_stats['consistent']:
        print(f"    Filtrul e CONSISTENT ✅")
    else:
        print(f"    Filtrul e INCONSISTENT ⚠️  — ajusteaza R_ACCEL_STD")

# ============================================================
# VIZUALIZARE
# ============================================================
print(f"\n[5] Generare grafice...")

time         = np.cumsum(dt_data)
labels_euler = ['Roll (X)', 'Pitch (Y)', 'Yaw (Z) ⚠️ deriva']
labels_xyz   = ['X', 'Y', 'Z']
colors_raw   = ['#ff7f7f', '#7fbf7f', '#7f7fff']
colors_ekf   = ['#cc0000', '#008000', '#0000cc']

fig, axes = plt.subplots(3, 3, figsize=(16, 12))
fig.suptitle('EKF Mare — Fuziune 5 Senzori', fontsize=14, fontweight='bold')

# Rand 1: Orientare
for i in range(3):
    ax = axes[0, i]
    ax.plot(time, raw_euler[:, i], color=colors_raw[i],
            alpha=0.6, linewidth=0.8, label='Brut S5')
    ax.plot(time, ekf_euler[:, i], color=colors_ekf[i],
            linewidth=1.2, label='EKF Mare')
    ax.set_title(labels_euler[i])
    ax.set_xlabel('Timp (s)')
    ax.set_ylabel('Grade (°)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

# Rand 2: Pozitie
for i in range(3):
    ax = axes[1, i]
    ax.plot(time, raw_pos[:, i], color=colors_raw[i],
            alpha=0.6, linewidth=0.8, label='Brut S5')
    ax.plot(time, ekf_pos[:, i], color=colors_ekf[i],
            linewidth=1.2, label='EKF Mare')
    ax.set_title(f'Pozitie {labels_xyz[i]}')
    ax.set_xlabel('Timp (s)')
    ax.set_ylabel('m')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

# Rand 3: NIS + Traiectorie + Sigma per senzor
if nis_stats:
    nis_arr = nis_stats['valori']
    ax_nis = axes[2, 0]
    ax_nis.plot(nis_arr, color='purple', linewidth=0.8, alpha=0.7)
    ax_nis.axhline(y=3.0,   color='green', linestyle='--', linewidth=1.5, label='Asteptat (3)')
    ax_nis.axhline(y=7.815, color='red',   linestyle='--', linewidth=1.0, label='95% bound')
    ax_nis.axhline(y=nis_stats['mediana'], color='orange', linestyle='-',
                   linewidth=1.5, label=f"Mediana ({nis_stats['mediana']:.2f})")
    ax_nis.set_title('NIS in timp')
    ax_nis.set_xlabel('Index update')
    ax_nis.set_ylabel('NIS')
    ax_nis.legend(fontsize=8)
    ax_nis.grid(True, alpha=0.3)
    ax_nis.set_ylim(0, min(np.percentile(nis_arr, 95) * 1.5, 50))
else:
    axes[2, 0].set_visible(False)

# Traiectorie XY
ax_traj = axes[2, 1]
ax_traj.plot(raw_pos[:, 0], raw_pos[:, 1],
             color='#ff7f7f', alpha=0.6, linewidth=0.8, label='Brut S5')
ax_traj.plot(ekf_pos[:, 0], ekf_pos[:, 1],
             color='#cc0000', linewidth=1.2, label='EKF Mare')
ax_traj.scatter([0], [0], color='green', s=50, zorder=5, label='Start')
ax_traj.scatter([ekf_pos[-1, 0]], [ekf_pos[-1, 1]],
                color='red', s=50, zorder=5, label='Final')
ax_traj.set_title('Traiectorie XY')
ax_traj.set_xlabel('X (m)')
ax_traj.set_ylabel('Y (m)')
ax_traj.legend(fontsize=8)
ax_traj.grid(True, alpha=0.3)
ax_traj.set_aspect('equal', adjustable='datalim')

# Sigma_q_trace per senzor in timp
ax_sig = axes[2, 2]
colors_s = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
for idx, s in enumerate(SENSOR_IDS):
    ax_sig.semilogy(time, sigma_traces[s],
                    color=colors_s[idx], linewidth=0.8,
                    alpha=0.8, label=f'S{s}')
ax_sig.set_title('Sigma_q_trace per senzor (incredere)')
ax_sig.set_xlabel('Timp (s)')
ax_sig.set_ylabel('sigma_q_trace (log)')
ax_sig.legend(fontsize=8)
ax_sig.grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.savefig('ekf_mare_rezultate.png', dpi=150, bbox_inches='tight')
print("    Grafic salvat: ekf_mare_rezultate.png")

# ============================================================
# SUMAR FINAL
# ============================================================
print("\n" + "=" * 60)
print("SUMAR FINAL EKF MARE")
print("=" * 60)
state = ekf.get_state_summary()
print(f"  Orientare:  Roll={state['roll_deg']:+.2f}°  "
      f"Pitch={state['pitch_deg']:+.2f}°  "
      f"Yaw={state['yaw_deg']:+.2f}°")
print(f"  Pozitie:    X={state['pos_m'][0]:+.3f} m  "
      f"Y={state['pos_m'][1]:+.3f} m  "
      f"Z={state['pos_m'][2]:+.3f} m")
print(f"  Viteza:     X={state['vel_mps'][0]:+.3f} m/s  "
      f"Y={state['vel_mps'][1]:+.3f} m/s  "
      f"Z={state['vel_mps'][2]:+.3f} m/s")
if nis_stats:
    print(f"\n  NIS Mediana: {nis_stats['mediana']:.3f}  "
          f"({'consistent ✅' if nis_stats['consistent'] else 'inconsistent ⚠️'})")
print(f"\n  De tunat:")
print(f"    R_ACCEL_STD — ajusteaza pana NIS mediana ≈ 3")
print(f"    Q_DIAG[4:7] — mareste daca viteza reactioneaza prea lent")
print("=" * 60)