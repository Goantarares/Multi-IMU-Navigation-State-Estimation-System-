import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from quaternion_utils import (
    quat_multiply, quat_from_gyro, quat_normalize,
    quat_to_rotation_matrix, quat_to_euler
)
from config import G, DEG2RAD

# ============================================================
# CONFIGURARE
# ============================================================
DATA_FILE = 'imu_stationary_data_corrected.csv'
SENSOR_ID = 5

# Bias giroscop masurat din date statice (din sesiunea de calibrare)
# Folosit pentru varianta "brut cu bias corectat"
GYRO_BIAS_DPS = np.array([0.02873156, 0.80534179, -0.72649666])

# ============================================================
# INCARCARE DATE
# ============================================================
print("=" * 60)
print(f"INTEGRARE BRUTA - Senzor S{SENSOR_ID}")
print("=" * 60)

df = pd.read_csv(DATA_FILE)
s  = SENSOR_ID

gyro_data  = df[[f'gx_s{s}', f'gy_s{s}', f'gz_s{s}']].values  # dps
accel_data = df[[f'ax_s{s}', f'ay_s{s}', f'az_s{s}']].values  # g
dt_data    = df['dt'].values

print(f"Cadre: {len(df)}  |  Durata: {np.sum(dt_data):.1f}s  |  "
      f"Frecventa: {1/np.mean(dt_data):.1f} Hz")

# ============================================================
# VARIANTA 1 — Integrare bruta pura (fara bias, fara filtru)
# Cel mai simplu caz — exact ce ai face fara niciun post-procesing
# ============================================================
print("\n[1] Integrare bruta pura (fara bias, fara filtru)...")

q1     = np.array([1.0, 0.0, 0.0, 0.0])
vel1   = np.zeros(3)
pos1   = np.zeros(3)
euler1 = []
pos1_arr = []
vel1_arr = []

for i in range(len(df)):
    gyro  = gyro_data[i] * DEG2RAD   # rad/s, fara corectie bias
    accel = accel_data[i] * G         # m/s²
    dt    = dt_data[i]

    if dt <= 0:
        continue

    dq = quat_from_gyro(gyro, dt)
    q1 = quat_normalize(quat_multiply(q1, dq))

    R       = quat_to_rotation_matrix(q1)
    a_glob  = R @ accel - np.array([0, 0, G])
    vel1   += a_glob * dt
    pos1   += vel1 * dt + 0.5 * a_glob * dt**2

    euler1.append(quat_to_euler(q1))
    pos1_arr.append(pos1.copy())
    vel1_arr.append(vel1.copy())

euler1   = np.array(euler1)
pos1_arr = np.array(pos1_arr)
vel1_arr = np.array(vel1_arr)

print(f"   Orientare finala:  Roll={euler1[-1,0]:+.2f}°  "
      f"Pitch={euler1[-1,1]:+.2f}°  Yaw={euler1[-1,2]:+.2f}°")
print(f"   Pozitie finala:    X={pos1_arr[-1,0]:+.3f}m  "
      f"Y={pos1_arr[-1,1]:+.3f}m  Z={pos1_arr[-1,2]:+.3f}m")
print(f"   Viteza finala:     X={vel1_arr[-1,0]:+.3f}m/s  "
      f"Y={vel1_arr[-1,1]:+.3f}m/s  Z={vel1_arr[-1,2]:+.3f}m/s")

# ============================================================
# VARIANTA 2 — Integrare bruta cu bias corectat (fara EKF)
# Bias masurat static aplicat fix, fara estimare dinamica
# ============================================================
print("\n[2] Integrare bruta cu bias corectat (fara EKF)...")

gyro_bias_rad = GYRO_BIAS_DPS * DEG2RAD
q2     = np.array([1.0, 0.0, 0.0, 0.0])
vel2   = np.zeros(3)
pos2   = np.zeros(3)
euler2 = []
pos2_arr = []
vel2_arr = []

for i in range(len(df)):
    gyro  = gyro_data[i] * DEG2RAD - gyro_bias_rad  # cu bias corectat
    accel = accel_data[i] * G
    dt    = dt_data[i]

    if dt <= 0:
        continue

    dq = quat_from_gyro(gyro, dt)
    q2 = quat_normalize(quat_multiply(q2, dq))

    R       = quat_to_rotation_matrix(q2)
    a_glob  = R @ accel - np.array([0, 0, G])
    vel2   += a_glob * dt
    pos2   += vel2 * dt + 0.5 * a_glob * dt**2

    euler2.append(quat_to_euler(q2))
    pos2_arr.append(pos2.copy())
    vel2_arr.append(vel2.copy())

euler2   = np.array(euler2)
pos2_arr = np.array(pos2_arr)
vel2_arr = np.array(vel2_arr)

print(f"   Orientare finala:  Roll={euler2[-1,0]:+.2f}°  "
      f"Pitch={euler2[-1,1]:+.2f}°  Yaw={euler2[-1,2]:+.2f}°")
print(f"   Pozitie finala:    X={pos2_arr[-1,0]:+.3f}m  "
      f"Y={pos2_arr[-1,1]:+.3f}m  Z={pos2_arr[-1,2]:+.3f}m")
print(f"   Viteza finala:     X={vel2_arr[-1,0]:+.3f}m/s  "
      f"Y={vel2_arr[-1,1]:+.3f}m/s  Z={vel2_arr[-1,2]:+.3f}m/s")

# ============================================================
# SUMAR COMPARATIV
# ============================================================
print("\n" + "=" * 60)
print("SUMAR COMPARATIV")
print("=" * 60)
print(f"{'Metrica':<30} {'Brut pur':>15} {'Brut+bias':>15}")
print("-" * 60)
print(f"{'Roll final (°)':<30} {euler1[-1,0]:>+15.2f} {euler2[-1,0]:>+15.2f}")
print(f"{'Pitch final (°)':<30} {euler1[-1,1]:>+15.2f} {euler2[-1,1]:>+15.2f}")
print(f"{'Yaw final (°)':<30} {euler1[-1,2]:>+15.2f} {euler2[-1,2]:>+15.2f}")
print(f"{'|Pozitie| finala (m)':<30} "
      f"{np.linalg.norm(pos1_arr[-1]):>15.2f} "
      f"{np.linalg.norm(pos2_arr[-1]):>15.2f}")
print(f"{'|Viteza| finala (m/s)':<30} "
      f"{np.linalg.norm(vel1_arr[-1]):>15.2f} "
      f"{np.linalg.norm(vel2_arr[-1]):>15.2f}")
print(f"{'Deriva yaw (°/s)':<30} "
      f"{euler1[-1,2]/np.sum(dt_data):>+15.4f} "
      f"{euler2[-1,2]/np.sum(dt_data):>+15.4f}")
print("=" * 60)
print("\nNote:")
print("  Brut pur      = integrare fara nicio corectie")
print("  Brut+bias     = bias masurat static aplicat fix, fara EKF")
print("  Compara cu rezultatele EKF pentru a vedea imbunatatirile")

# ============================================================
# GRAFICE
# ============================================================
time = np.cumsum(dt_data)

fig, axes = plt.subplots(2, 3, figsize=(16, 8))
fig.suptitle(f'Integrare bruta - Senzor S{SENSOR_ID}', fontsize=13)

labels_euler = ['Roll (X)', 'Pitch (Y)', 'Yaw (Z)']
labels_xyz   = ['X', 'Y', 'Z']

# Rand 1: Orientare
for i in range(3):
    ax = axes[0, i]
    ax.plot(time, euler1[:, i], color='#aaaaaa',
            linewidth=0.8, label='Brut pur')
    ax.plot(time, euler2[:, i], color='#cc6600',
            linewidth=1.2, label='Brut + bias corectat')
    ax.set_title(labels_euler[i])
    ax.set_xlabel('Timp (s)')
    ax.set_ylabel('Grade (°)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

# Rand 2: Pozitie
for i in range(3):
    ax = axes[1, i]
    ax.plot(time, pos1_arr[:, i], color='#aaaaaa',
            linewidth=0.8, label='Brut pur')
    ax.plot(time, pos2_arr[:, i], color='#cc6600',
            linewidth=1.2, label='Brut + bias corectat')
    ax.set_title(f'Pozitie {labels_xyz[i]}')
    ax.set_xlabel('Timp (s)')
    ax.set_ylabel('m')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('brut_rezultate_stationar_15_min.png', dpi=150, bbox_inches='tight')
print("\nGrafic salvat: brut_rezultate.png")