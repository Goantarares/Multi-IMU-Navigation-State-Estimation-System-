import numpy as np

# ============================================================
# CONSTANTE FIZICE
# ============================================================
G       = 9.81
DEG2RAD = np.pi / 180.0
RAD2DEG = 180.0 / np.pi

# ============================================================
# STAREA INITIALA
# ============================================================
Q0_INIT = np.array([1.0, 0.0, 0.0, 0.0])

# Biasuri masurate din 14.000 cadre statice
# b_g_z e scos din stare - tratat ca parametru fix

# Pentru s5
GYRO_BIAS_XY_INIT = np.array([0.02873156, 0.80534179]) * DEG2RAD
GYRO_BIAS_Z_FIXED = -0.72649666 * DEG2RAD
ACCEL_BIAS_INIT   = np.array([-0.025295, -0.008218, 0.020665]) * G

# Pentru s1
# GYRO_BIAS_XY_INIT = np.array([-0.07631086, -0.63531217]) * DEG2RAD
# GYRO_BIAS_Z_FIXED = -0.054145 * DEG2RAD
# ACCEL_BIAS_INIT   = np.array([-0.032403, 0.029412, 0.010184]) * G

# Pentru s2
# GYRO_BIAS_XY_INIT = np.array([-0.10231907, -0.43315500]) * DEG2RAD
# GYRO_BIAS_Z_FIXED = -0.12976349 * DEG2RAD
# ACCEL_BIAS_INIT   = np.array([-0.007933, -0.036265, 0.009405]) * G

# Pentru s3
# GYRO_BIAS_XY_INIT = np.array([0.03753549, -0.84950084]) * DEG2RAD
# GYRO_BIAS_Z_FIXED = -0.58816799 * DEG2RAD
# ACCEL_BIAS_INIT   = np.array([0.013097, 0.027969, 0.010117]) * G

# Pentru s4
# GYRO_BIAS_XY_INIT = np.array([0.44734626, 0.08069586]) * DEG2RAD
# GYRO_BIAS_Z_FIXED = -0.05682561 * DEG2RAD
# ACCEL_BIAS_INIT   = np.array([0.038954, -0.020466, 0.011260]) * G


# ============================================================
# COVARIANTA INITIALA Sigma_0 (9x9)
# Pozitia si viteza eliminate din stare — estimate in filtrul mare
# ============================================================
SIGMA_INIT_DIAG = np.array([
    1e-6, 1e-6, 1e-6, 1e-6,   # q   - orientare initiala cunoscuta
    1e-8, 1e-8,                # b_g XY - bias cunoscut precis
    1e-8, 1e-8, 1e-8,          # b_a - bias cunoscut precis
])

# ============================================================
# ZGOMOT DE PROCES Q (9x9)
# ============================================================
dt_nominal = 1.0 / 15.62

# Pentru s5
ARW_X  = 1.79e-4
ARW_Y  = 9.39e-5
Q_q    = (ARW_X ** 2) * dt_nominal   # ≈ 2.05e-9

BI_X   = 3.84e-5
BI_Y   = 2.08e-5
Q_bg_X = BI_X ** 2   # ≈ 1.47e-9
Q_bg_Y = BI_Y ** 2   # ≈ 4.33e-10

# Pentru s1
# ARW_X  = 1.17e-4
# ARW_Y  = 9.21e-5
# Q_q    = (ARW_X ** 2) * dt_nominal
#
# BI_X   = 8.08e-7
# BI_Y   = 2.16e-5
# Q_bg_X = BI_X ** 2
# Q_bg_Y = BI_Y ** 2

# Pentru s2
# ARW_X  = 1.78e-4
# ARW_Y  = 8.60e-5
# Q_q    = (ARW_X ** 2) * dt_nominal   # ≈ 2.05e-9
#
# BI_X   = 9.07e-5
# BI_Y   = 6.86e-6
# Q_bg_X = BI_X ** 2
# Q_bg_Y = BI_Y ** 2

# Pentru s3
# ARW_X  = 2.52e-4
# ARW_Y  = 9.57e-5
# Q_q    = (ARW_X ** 2) * dt_nominal   # ≈ 2.05e-9
#
# BI_X   = 5.25e-5
# BI_Y   = 2.12e-5
# Q_bg_X = BI_X ** 2
# Q_bg_Y = BI_Y ** 2

# Pentru s4
# ARW_X  = 1.02e-4
# ARW_Y  = 1.05e-5
# Q_q    = (ARW_X ** 2) * dt_nominal
#
# BI_X   = 6.16e-6
# BI_Y   = 1.17e-5
# Q_bg_X = BI_X ** 2   # ≈ 1.47e-9
# Q_bg_Y = BI_Y ** 2   # ≈ 4.33e-10

Q_DIAG = np.array([
    Q_q,    Q_q,    Q_q,    Q_q,   # q   - din ARW masurat
    Q_bg_X, Q_bg_Y,                # b_g XY - din Bias Instability
    1e-12,  1e-12,  1e-12,         # b_a - random walk foarte lent
])

# ============================================================
# ZGOMOT DE MASURARE R (3x3)
# ============================================================

# Pentru s5
R_ACCEL_STD_STATIC  = 0.008165  # m/s² - masurat static, update agresiv
R_ACCEL_STD_DYNAMIC = 0.44      # m/s² - calibrat din NIS

# Pentru s1
# R_ACCEL_STD_STATIC  = 0.009114  # m/s² - masurat static, update agresiv
# R_ACCEL_STD_DYNAMIC = 0.44      # m/s² - calibrat din NIS

# Pentru s2
# R_ACCEL_STD_STATIC  = 0.008398  # m/s² - masurat static, update agresiv
# R_ACCEL_STD_DYNAMIC = 0.44      # m/s² - calibrat din NIS

# Pentru s3
# R_ACCEL_STD_STATIC  = 0.009036  # m/s² - masurat static, update agresiv
# R_ACCEL_STD_DYNAMIC = 0.44      # m/s² - calibrat din NIS

# Pentru s4
# R_ACCEL_STD_STATIC  = 0.009756  # m/s² - masurat static, update agresiv
# R_ACCEL_STD_DYNAMIC = 0.44      # m/s² - calibrat din NIS

R_DIAG = np.array([R_ACCEL_STD_STATIC**2] * 3)

# ============================================================
# DETECTIE STATICA
# ============================================================
STATIC_THRESHOLD      = 0.5    # m/s²
GYRO_STATIC_THRESHOLD = 0.05   # rad/s

# ============================================================
# ALTELE
# ============================================================
EPS_JACOBIAN = 1e-7
EPS_QUAT     = 1e-10