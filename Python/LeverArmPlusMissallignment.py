import numpy as np
import pandas as pd

# ============================================================
# CONFIGURARE - pune valorile reale aici
# ============================================================

# Unghiurile de rotatie ale senzorilor fata de senzorul central (S5)
# in grade, in jurul axei Z
# Senzorul 5 e central cu unghi 0
rotations_deg = {
    1: 0.0,   # S1 - pune unghiul real
    2: 270.0,   # S2 - pune unghiul real
    3: 180.0,   # S3 - pune unghiul real
    4: 90.0,   # S4 - pune unghiul real
    5: 0.0    # S5 - central, referinta
}

# Pozitiile relative fata de senzorul central (S5) in metri
# format: [x, y, z]
positions_m = {
    1: np.array([0.003, -0.0025, 0.0]),  # S1 - pune valorile reale
    2: np.array([-0.003, -0.0025, 0.0]),  # S2 - pune valorile reale
    3: np.array([0.003, 0.0025, 0.0]),  # S3 - pune valorile reale
    4: np.array([-0.003, 0.0025, 0.0]),  # S4 - pune valorile reale
    5: np.array([0.0, 0.0, 0.0])   # S5 - central
}

# ============================================================
# FUNCTII AJUTATOARE
# ============================================================

def rotation_matrix_z(theta_deg):
    """Matrice de rotatie in jurul axei Z"""
    theta = np.radians(theta_deg)
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0,              0,             1]
    ])
    return R


def correct_misalignment(gyro, accel, theta_deg):
    """Roteste masuratorile in sistemul de coordonate al senzorului central"""
    R = rotation_matrix_z(theta_deg)
    gyro_corrected  = R @ gyro
    accel_corrected = R @ accel
    return gyro_corrected, accel_corrected


def correct_lever_arm(accel_corrected, gyro_corrected, gyro_prev, position, dt):
    """Corecteaza acceleratia pentru efectul lever arm"""
    if dt <= 0:
        return accel_corrected

    omega = gyro_corrected  # rad/s
    r = position            # metri

    # Acceleratie centripeta: omega x (omega x r)
    a_centripetal = np.cross(omega, np.cross(omega, r))

    # Acceleratie tangentiala: (d_omega/dt) x r
    d_omega = (gyro_corrected - gyro_prev) / dt
    a_tangential = np.cross(d_omega, r)

    accel_final = accel_corrected - a_centripetal - a_tangential
    return accel_final


# ============================================================
# INCARCARE DATE
# ============================================================

print("Incarcare date...")
df = pd.read_csv('imu_data_20260411_202444.csv')

# corectare

# Corecteaza ordinea coloanelor - datele sunt axis-major, header-ul era sensor-major
correct_columns = ['packet_id'] + [f'timestamp_s{i}' for i in range(1, 6)]
for axis in ['gx', 'gy', 'gz']:
    for s in range(1, 6):
        correct_columns.append(f'{axis}_s{s}')
for axis in ['ax', 'ay', 'az']:
    for s in range(1, 6):
        correct_columns.append(f'{axis}_s{s}')

df.columns = correct_columns
print("Coloane corectate. Verificare S5 static:")
print(f"  ax_s5={df['ax_s5'].iloc[0]:.4f}g  (ar trebui ~0)")
print(f"  ay_s5={df['ay_s5'].iloc[0]:.4f}g  (ar trebui ~0)")
print(f"  az_s5={df['az_s5'].iloc[0]:.4f}g  (ar trebui ~1)")

# gata corectarea

print(f"Randuri incarcate: {len(df)}")

# ============================================================
# PASUL 1 - SINCRONIZARE TIMESTAMPURI
# Folosim senzorul 5 (central) ca referinta
# ============================================================

print("Sincronizare timestampuri...")

t_ref = df['timestamp_s5'].values.astype(float)

for s in range(1, 6):
    if s == 5:
        continue

    t_s = df[f'timestamp_s{s}'].values.astype(float)

    for axis in ['gx', 'gy', 'gz', 'ax', 'ay', 'az']:
        col = f'{axis}_s{s}'
        df[col] = np.interp(t_ref, t_s, df[col].values)

    df[f'timestamp_s{s}'] = t_ref

print("Sincronizare completa")

# ============================================================
# PASUL 2 - CORECTIE MISALIGNMENT SI LEVER ARM
# ============================================================

print("Corectie misalignment si lever arm...")

# dt in secunde bazat pe senzorul de referinta
df['dt'] = df['timestamp_s5'].diff() / 1_000_000
df['dt'] = df['dt'].fillna(1.0 / 104.0)  # prima valoare = rata nominala

# Gyro anterior pentru fiecare senzor (pentru calculul d_omega)
gyro_prev = {s: np.zeros(3) for s in range(1, 6)}

corrected_rows = []

for idx, row in df.iterrows():
    dt = row['dt']
    corrected_row = {
        'packet_id':    row['packet_id'],
        'timestamp_us': row['timestamp_s5'],
        'dt':           dt
    }

    for s in range(1, 6):
        # Extrage masuratorile brute
        gyro = np.array([
            row[f'gx_s{s}'],
            row[f'gy_s{s}'],
            row[f'gz_s{s}']
        ]) * np.pi / 180.0  # dps -> rad/s

        accel = np.array([
            row[f'ax_s{s}'],
            row[f'ay_s{s}'],
            row[f'az_s{s}']
        ])  # in g

        # Pas 1: corectie misalignment
        gyro_rot, accel_rot = correct_misalignment(
            gyro, accel, rotations_deg[s]
        )

        # Pas 2: corectie lever arm
        accel_final = correct_lever_arm(
            accel_rot,
            gyro_rot,
            gyro_prev[s],
            positions_m[s],
            dt
        )

        gyro_prev[s] = gyro_rot.copy()

        # Salveaza valorile corectate (gyro inapoi in dps)
        corrected_row[f'gx_s{s}'] = gyro_rot[0]  * 180.0 / np.pi
        corrected_row[f'gy_s{s}'] = gyro_rot[1]  * 180.0 / np.pi
        corrected_row[f'gz_s{s}'] = gyro_rot[2]  * 180.0 / np.pi
        corrected_row[f'ax_s{s}'] = accel_final[0]
        corrected_row[f'ay_s{s}'] = accel_final[1]
        corrected_row[f'az_s{s}'] = accel_final[2]

    corrected_rows.append(corrected_row)

    if idx % 1000 == 0:
        print(f"Procesat {idx}/{len(df)} randuri...")

# ============================================================
# SALVARE
# ============================================================

df_corrected = pd.DataFrame(corrected_rows)
output_filename = 'imu_data_corrected_stationar_pentru_ekf.csv'
df_corrected.to_csv(output_filename, index=False)

print(f"\nSalvat in {output_filename}")
print(f"Total randuri: {len(df_corrected)}")
print(f"\nPrimele valori corectate:")
print(df_corrected[['packet_id', 'timestamp_us', 'dt',
                     'gx_s5', 'gy_s5', 'gz_s5',
                     'ax_s5', 'ay_s5', 'az_s5']].head())