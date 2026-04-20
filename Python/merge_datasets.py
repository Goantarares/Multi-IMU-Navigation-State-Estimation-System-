import numpy as np
import pandas as pd

# ============================================================
# CONFIGURARE
# ============================================================
SENSOR_IDS  = [1, 2, 3, 4, 5]
INPUT_PATTERN  = 'dataset_filtru_mare_s{}.csv'
OUTPUT_FILE    = 'dataset_filtru_mare_combined.csv'

# Coloane comune — identice pentru toti senzorii, nu se dubleaza
COMMON_COLS = ['timestamp_us', 'dt']

# Coloane specifice per senzor — vor primi sufixul _sN
SENSOR_COLS = ['gx_corr', 'gy_corr', 'gz_corr',
               'ax_corr', 'ay_corr', 'az_corr',
               'qw', 'qx', 'qy', 'qz',
               'sigma_q_trace']

# ============================================================
# INCARCARE SI MERGE
# ============================================================
print("Incarcare fisiere filtrate per senzor...")

dfs = []
for s in SENSOR_IDS:
    filename = INPUT_PATTERN.format(s)
    try:
        df_s = pd.read_csv(filename)
        print(f"  S{s}: {filename} — {len(df_s)} randuri, coloane: {list(df_s.columns)}")

        # Redenumeste coloanele specifice cu sufixul _sN
        rename_map = {col: f'{col}_s{s}' for col in SENSOR_COLS if col in df_s.columns}
        df_s = df_s.rename(columns=rename_map)

        dfs.append(df_s)
    except FileNotFoundError:
        print(f"  S{s}: {filename} — NEGASIT, sarit")

if len(dfs) == 0:
    print("Niciun fisier gasit. Verifica ca ai rulat main.py pentru fiecare senzor.")
    exit(1)

# Merge pe timestamp_us — inner join pastreaza doar randurile comune
print(f"\nMerge pe timestamp_us...")
df_combined = dfs[0]
for df_s in dfs[1:]:
    # Pastreaza doar coloanele specifice + timestamp pentru merge
    cols_to_merge = COMMON_COLS + [c for c in df_s.columns if c not in COMMON_COLS]
    # Evita duplicarea dt daca e identic
    merge_on = ['timestamp_us']
    df_combined = pd.merge(
        df_combined,
        df_s.drop(columns=['dt'], errors='ignore'),
        on=merge_on,
        how='inner'
    )

print(f"Randuri dupa merge: {len(df_combined)}")
print(f"Coloane: {len(df_combined.columns)}")

# ============================================================
# VERIFICARE CONSISTENTA
# ============================================================
print(f"\nVerificare consistenta quaternioni intre senzori...")
if len(dfs) > 1:
    for s in SENSOR_IDS:
        col_qw = f'qw_s{s}'
        col_qx = f'qx_s{s}'
        if col_qw in df_combined.columns and f'qw_s{SENSOR_IDS[0]}' in df_combined.columns:
            # Unghiul dintre quaternionii S1 si Sx
            q1 = df_combined[[f'qw_s{SENSOR_IDS[0]}',
                               f'qx_s{SENSOR_IDS[0]}',
                               f'qy_s{SENSOR_IDS[0]}',
                               f'qz_s{SENSOR_IDS[0]}']].values
            qs = df_combined[[f'qw_s{s}',
                               f'qx_s{s}',
                               f'qy_s{s}',
                               f'qz_s{s}']].values
            dot = np.clip(np.sum(q1 * qs, axis=1), -1, 1)
            angle_deg = 2 * np.degrees(np.arccos(np.abs(dot)))
            print(f"  S{SENSOR_IDS[0]} vs S{s}: unghi mediu = {np.mean(angle_deg):.2f}°  "
                  f"max = {np.max(angle_deg):.2f}°")

# ============================================================
# SALVARE
# ============================================================
df_combined.to_csv(OUTPUT_FILE, index=False)
print(f"\nSalvat: {OUTPUT_FILE}")
print(f"  Randuri:  {len(df_combined)}")
print(f"  Coloane:  {len(df_combined.columns)}")
print(f"\nStructura coloane:")
print(f"  Comune:   {COMMON_COLS}")
for s in SENSOR_IDS:
    cols_s = [c for c in df_combined.columns if c.endswith(f'_s{s}')]
    if cols_s:
        print(f"  S{s}:      {cols_s}")