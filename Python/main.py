import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ekf import EKF_IMU
from quaternion_utils import quat_to_euler
from config import G, DEG2RAD

# ============================================================
# CONFIGURARE
# ============================================================
DATA_FILE = 'imu_data_corrected_2.csv'
SENSOR_ID = 5
N_STATIC  = 50

# ============================================================
# INCARCARE DATE
# ============================================================
print("=" * 60)
print(f"EKF IMU - Senzorul S{SENSOR_ID}")
print("=" * 60)

print(f"\n[1] Incarcare date din {DATA_FILE}...")
df = pd.read_csv(DATA_FILE)
print(f"    Total cadre: {len(df)}")
print(f"    dt mediu: {df['dt'].mean()*1000:.2f} ms  "
      f"(~{1/df['dt'].mean():.1f} Hz)")

s = SENSOR_ID
gyro_data  = df[[f'gx_s{s}', f'gy_s{s}', f'gz_s{s}']].values  # dps
accel_data = df[[f'ax_s{s}', f'ay_s{s}', f'az_s{s}']].values  # g
dt_data    = df['dt'].values

# ============================================================
# ESTIMARE BIAS INITIAL
# ============================================================
print(f"\n[2] Estimare bias din primele {N_STATIC} cadre statice...")

if N_STATIC > 0:
    gyro_static  = gyro_data[:N_STATIC]
    accel_static = accel_data[:N_STATIC]

    gyro_bias_dps = np.mean(gyro_static, axis=0)
    gyro_bias_rad = gyro_bias_dps * DEG2RAD
    accel_var     = np.var(accel_static * G, axis=0)

    print(f"    Bias giroscop (dps):          {gyro_bias_dps}")
    print(f"    Bias giroscop (rad/s):        {gyro_bias_rad}")
    print(f"    Varianta accelerometru (m/s²)²: {accel_var}")
else:
    gyro_bias_rad = np.zeros(3)
    print("    Bias setat la zero")

# ============================================================
# INITIALIZARE EKF
# ============================================================
print(f"\n[3] Initializare EKF...")
from config import GYRO_BIAS_Z_FIXED
gyro_bias_xy_rad = gyro_bias_rad[:2]
ekf = EKF_IMU(gyro_bias_xy_init=gyro_bias_xy_rad)
print(f"    Bias gyro XY init (dps): {gyro_bias_xy_rad * 180/np.pi}")
print(f"    Bias gyro Z fix   (dps): {GYRO_BIAS_Z_FIXED * 180/np.pi:.6f}")
print("    EKF initializat cu succes")

# Verificare Jacobieni analitici vs numerici (ruleaza o singura data)
# Dezactiveaza dupa prima rulare de succes pentru viteza maxima
VERIFY_JACOBIANS = True
if VERIFY_JACOBIANS:
    ekf.verify_jacobians(gyro_data[0], accel_data[0], dt_data[0])

# ============================================================
# INTEGRARE BRUTA
# ============================================================
print(f"\n[4] Rulare integrare bruta (fara filtru)...")

from quaternion_utils import quat_multiply, quat_from_gyro, quat_normalize
from quaternion_utils import quat_to_rotation_matrix

raw_pos   = [np.zeros(3)]
raw_vel   = [np.zeros(3)]
raw_euler = []
vel_raw   = np.zeros(3)
pos_raw   = np.zeros(3)
q_raw     = np.array([1.0, 0.0, 0.0, 0.0])

for i in range(len(df)):
    gyro  = gyro_data[i]
    accel = accel_data[i]
    dt    = dt_data[i]

    if dt <= 0:
        continue

    dq    = quat_from_gyro(gyro * DEG2RAD, dt)
    q_raw = quat_normalize(quat_multiply(q_raw, dq))

    R_raw        = quat_to_rotation_matrix(q_raw)
    a_global_raw = R_raw @ (accel * G) - np.array([0, 0, G])
    vel_raw      = vel_raw + a_global_raw * dt
    pos_raw      = pos_raw + vel_raw * dt + 0.5 * a_global_raw * dt**2

    raw_vel.append(vel_raw.copy())
    raw_pos.append(pos_raw.copy())
    raw_euler.append(quat_to_euler(q_raw))

raw_pos   = np.array(raw_pos[1:])
raw_vel   = np.array(raw_vel[1:])
raw_euler = np.array(raw_euler)
print("    Integrare bruta completa")

# ============================================================
# RULARE EKF
# ============================================================
print(f"\n[5] Rulare EKF...")

ekf_euler  = []
ekf_bias_g = []

for i in range(len(df)):
    gyro  = gyro_data[i]
    accel = accel_data[i]
    dt    = dt_data[i]

    ekf.predict(gyro, accel, dt)
    ekf.update(accel, dt, gyro)

    ekf_euler.append(ekf.get_orientation_euler())
    ekf_bias_g.append(ekf.get_gyro_bias_xy())

    if i % 500 == 0:
        state = ekf.get_state_summary()
        print(f"    Cadru {i:5d}/{len(df)} | "
              f"Roll={state['roll_deg']:+6.1f}° "
              f"Pitch={state['pitch_deg']:+6.1f}° "
              f"Yaw={state['yaw_deg']:+6.1f}°")

ekf_euler  = np.array(ekf_euler)
ekf_bias_g = np.array(ekf_bias_g)

# ============================================================
# SALVARE MASURATORI FILTRATE PENTRU FILTRUL MARE
# ============================================================
print(f"\n[5b] Salvare masuratori filtrate pentru filtrul mare...")

corrected_gyro  = []
corrected_accel = []
quaternions     = []
sigma_q_norms   = []

for i in range(len(df)):
    meas = ekf.get_corrected_measurements(gyro_data[i], accel_data[i])
    corrected_gyro.append(meas['gyro_corrected_rads'])
    corrected_accel.append(meas['accel_corrected_ms2'])
    quaternions.append(meas['quaternion'])
    sigma_q_norms.append(np.trace(meas['sigma_q']))

corrected_gyro  = np.array(corrected_gyro)
corrected_accel = np.array(corrected_accel)
quaternions     = np.array(quaternions)
sigma_q_norms   = np.array(sigma_q_norms)

df_out = pd.DataFrame({
    'timestamp_us':  df['timestamp_us'].values,
    'dt':            dt_data,
    'gx_corr':       corrected_gyro[:, 0],
    'gy_corr':       corrected_gyro[:, 1],
    'gz_corr':       corrected_gyro[:, 2],
    'ax_corr':       corrected_accel[:, 0],
    'ay_corr':       corrected_accel[:, 1],
    'az_corr':       corrected_accel[:, 2],
    'qw':            quaternions[:, 0],
    'qx':            quaternions[:, 1],
    'qy':            quaternions[:, 2],
    'qz':            quaternions[:, 3],
    'sigma_q_trace': sigma_q_norms,
})

output_csv = f'dataset_filtru_mare_s{SENSOR_ID}.csv'
df_out.to_csv(output_csv, index=False)
print(f"    Salvat: {output_csv}  ({len(df_out)} randuri, {df_out.shape[1]} coloane)")

stats = ekf.get_update_stats()
print(f"\n    Update-uri normale   (cvasi-static):     {stats['normale']}")
print(f"    Update-uri ponderate (R scalat):          {stats['ponderate']}")
print(f"    Update-uri sarite    (> 3g):              {stats['sarite']} "
      f"({stats['procent_sarite']:.1f}%)")
print(f"    Sarite din cauza rotatiei (Directia 3):  {stats['sarite_gyro']}")

# ============================================================
# NIS - Directia 4
# ============================================================
nis_stats = ekf.get_nis_stats()
if nis_stats:
    print(f"\n    === NIS (Normalized Innovation Squared) ===")
    print(f"    Medie:    {nis_stats['medie']:.3f}  (asteptat: {nis_stats['asteptat']:.1f})")
    print(f"    Mediana:  {nis_stats['mediana']:.3f}")
    print(f"    Std:      {nis_stats['std']:.3f}")
    if nis_stats['consistent']:
        print(f"    Filtrul e CONSISTENT — Q si R sunt bine calibrate ✅")
    else:
        medie = nis_stats['medie']
        if medie > 3.0:
            print(f"    NIS > 3 — R prea mic sau Q prea mic ⚠️")
            print(f"    Sugestie: mareste R_ACCEL_STD sau Q_DIAG")
        else:
            print(f"    NIS < 3 — R prea mare ⚠️")
            print(f"    Sugestie: micsoreaza R_ACCEL_STD")

# ============================================================
# VIZUALIZARE
# ============================================================
print(f"\n[6] Generare grafice...")

time         = np.cumsum(dt_data)
labels_euler = ['Roll (X)', 'Pitch (Y)', 'Yaw (Z) ⚠️ deriva']
labels_xyz   = ['X', 'Y', 'Z']
colors_raw   = ['#ff7f7f', '#7fbf7f', '#7f7fff']
colors_ekf   = ['#cc0000', '#008000', '#0000cc']

fig, axes = plt.subplots(4, 3, figsize=(16, 16))
fig.suptitle(f'EKF IMU - Senzor S{SENSOR_ID}', fontsize=14, fontweight='bold')

# --- RAND 1: Orientare ---
for i in range(3):
    ax = axes[0, i]
    ax.plot(time, raw_euler[:, i], color=colors_raw[i],
            alpha=0.6, linewidth=0.8, label='Brut')
    ax.plot(time, ekf_euler[:, i], color=colors_ekf[i],
            linewidth=1.2, label='EKF')
    ax.set_title(labels_euler[i])
    ax.set_xlabel('Timp (s)')
    ax.set_ylabel('Grade (°)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    if i == 2:
        ax.text(0.02, 0.95, 'Yaw deriva fara magnetometru!',
                transform=ax.transAxes, fontsize=8,
                color='red', verticalalignment='top')

# --- RAND 2: Orientare bruta integrare (fara EKF) ---
for i in range(3):
    ax = axes[1, i]
    ax.plot(time, raw_euler[:, i], color=colors_raw[i],
            alpha=0.8, linewidth=1.0, label='Brut')
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.set_title(f'Brut - {labels_euler[i]}')
    ax.set_xlabel('Timp (s)')
    ax.set_ylabel('Grade (°)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

# --- RAND 3: Bias giroscop estimat ---
for i in range(2):
    ax = axes[2, i]
    ax.plot(time, ekf_bias_g[:, i] * 180/np.pi,
            color=colors_ekf[i], linewidth=1.2)
    ax.set_title(f'Bias giroscop {labels_xyz[i]} estimat')
    ax.set_xlabel('Timp (s)')
    ax.set_ylabel('°/s')
    ax.grid(True, alpha=0.3)

# --- RAND 3, col 3: Diferenta orientare EKF vs Brut ---
ax_diff = axes[2, 2]
for i in range(3):
    diff = ekf_euler[:, i] - raw_euler[:, i]
    ax_diff.plot(time, diff, linewidth=1.0,
                 label=labels_euler[i].split(' ')[0])
ax_diff.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
ax_diff.set_title('Diferenta EKF - Brut (orientare)')
ax_diff.set_xlabel('Timp (s)')
ax_diff.set_ylabel('Grade (°)')
ax_diff.legend(fontsize=8)
ax_diff.grid(True, alpha=0.3)

# --- RAND 4: NIS (Directia 4) ---
if nis_stats:
    nis_arr = nis_stats['valori']
    nis_time = time[np.where(np.array([1 if ekf.nis_values else 0
                                        for _ in range(len(time))]))]

    # Grafic NIS in timp
    ax_nis1 = axes[3, 0]
    ax_nis1.plot(nis_arr, color='purple', linewidth=0.8, alpha=0.7)
    ax_nis1.axhline(y=3.0,   color='green', linestyle='--',
                    linewidth=1.5, label='Asteptat (3)')
    ax_nis1.axhline(y=7.815, color='red',   linestyle='--',
                    linewidth=1.0, label='95% bound (7.815)')
    ax_nis1.axhline(y=nis_stats['medie'], color='orange', linestyle='-',
                    linewidth=1.5, label=f"Medie ({nis_stats['medie']:.2f})")
    ax_nis1.set_title('NIS in timp')
    ax_nis1.set_xlabel('Index update')
    ax_nis1.set_ylabel('NIS')
    ax_nis1.legend(fontsize=8)
    ax_nis1.grid(True, alpha=0.3)
    ax_nis1.set_ylim(0, min(np.percentile(nis_arr, 95) * 1.5, 50))

    # Histograma NIS
    ax_nis2 = axes[3, 1]
    ax_nis2.hist(nis_arr, bins=40, color='purple', alpha=0.7,
                 density=True, label='NIS masurat')
    ax_nis2.axvline(x=3.0, color='green', linestyle='--',
                    linewidth=2, label='Asteptat (3)')
    ax_nis2.axvline(x=nis_stats['medie'], color='orange', linestyle='-',
                    linewidth=2, label=f"Medie ({nis_stats['medie']:.2f})")
    ax_nis2.set_title('Distributia NIS')
    ax_nis2.set_xlabel('NIS')
    ax_nis2.set_ylabel('Densitate')
    ax_nis2.legend(fontsize=8)
    ax_nis2.grid(True, alpha=0.3)
    ax_nis2.set_xlim(0, 20)

    # NIS mediu pe fereastra glisanta
    window = 20
    if len(nis_arr) > window:
        nis_rolling = np.convolve(nis_arr,
                                  np.ones(window)/window,
                                  mode='valid')
        ax_nis3 = axes[3, 2]
        ax_nis3.plot(nis_rolling, color='purple', linewidth=1.2)
        ax_nis3.axhline(y=3.0, color='green', linestyle='--',
                        linewidth=1.5, label='Asteptat (3)')
        ax_nis3.axhline(y=7.815, color='red', linestyle='--',
                        linewidth=1.0, label='95% bound')
        ax_nis3.set_title(f'NIS medie glisanta (fereastra {window})')
        ax_nis3.set_xlabel('Index update')
        ax_nis3.set_ylabel('NIS mediu')
        ax_nis3.legend(fontsize=8)
        ax_nis3.grid(True, alpha=0.3)
    else:
        axes[3, 2].set_visible(False)
else:
    for i in range(3):
        axes[3, i].set_visible(False)

plt.tight_layout()
plt.savefig('ekf_rezultate_senzor_5.png', dpi=150, bbox_inches='tight')
print("    Grafic salvat: ekf_rezultate_senzor_5.png")

# ============================================================
# SUMAR FINAL
# ============================================================
print("\n" + "=" * 60)
print("SUMAR FINAL EKF")
print("=" * 60)
state = ekf.get_state_summary()
print(f"Stare finala EKF:")
print(f"  Orientare:  Roll={state['roll_deg']:+.2f}°  "
      f"Pitch={state['pitch_deg']:+.2f}°  "
      f"Yaw={state['yaw_deg']:+.2f}°")
print(f"  Bias gyro:  {state['gyro_bias'] * 180/np.pi} °/s")
print(f"  Bias accel: {state['accel_bias']} m/s²")
if nis_stats:
    print(f"\nConsistenta filtru (NIS):")
    print(f"  Medie NIS: {nis_stats['medie']:.3f}  "
          f"({'consistent ✅' if nis_stats['consistent'] else 'inconsistent ⚠️'})")
print(f"\nDe tunat in config.py:")
print(f"  Q_DIAG            - mareste daca EKF reactioneaza prea lent")
print(f"  R_ACCEL_STD       - ajusteaza dupa NIS (>3 mareste, <3 micsoreaza)")
print(f"  STATIC_THRESHOLD  - mareste daca prea putine update-uri normale")
print(f"  GYRO_STATIC_THRESHOLD - mareste daca rotatiile lente sunt sarite")
print("=" * 60)