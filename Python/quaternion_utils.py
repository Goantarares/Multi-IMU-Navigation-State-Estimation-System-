import numpy as np
from config import EPS_QUAT

# ============================================================
# Conventie quaternion folosita in TOT codul:
# q = [w, x, y, z]  unde w e partea scalara
# ============================================================

def quat_normalize(q):
    """Normalizeaza quaternionul la norma unitara."""
    norm = np.linalg.norm(q)
    if norm < EPS_QUAT:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / norm


def quat_multiply(q1, q2):
    """
    Inmultire quaternioni: q1 ⊗ q2
    q = [w, x, y, z]
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])


def quat_from_gyro(omega, dt):
    """
    Calculeaza quaternionul de rotatie din viteza unghiulara.
    omega: [wx, wy, wz] in rad/s
    dt: pas de timp in secunde
    Returneaza Δq = [w, x, y, z]
    """
    angle = np.linalg.norm(omega) * dt
    if angle < EPS_QUAT:
        # Rotatie neglijabila - quaternion identitate
        return np.array([1.0, 0.0, 0.0, 0.0])

    axis = omega / np.linalg.norm(omega)
    half_angle = angle / 2.0
    return np.array([
        np.cos(half_angle),
        axis[0] * np.sin(half_angle),
        axis[1] * np.sin(half_angle),
        axis[2] * np.sin(half_angle)
    ])


def quat_to_rotation_matrix(q):
    """
    Matrice de rotatie din quaternion [w, x, y, z].
    R transforma un vector din cadrul CORPULUI in cadrul GLOBAL:
        v_global = R @ v_body
    """
    w, x, y, z = q
    R = np.array([
        [1 - 2*(y**2 + z**2),   2*(x*y - w*z),       2*(x*z + w*y)     ],
        [2*(x*y + w*z),          1 - 2*(x**2 + z**2), 2*(y*z - w*x)     ],
        [2*(x*z - w*y),          2*(y*z + w*x),       1 - 2*(x**2 + y**2)]
    ])
    return R


def quat_to_euler(q):
    """
    Converteste quaternion [w, x, y, z] in unghiuri Euler [roll, pitch, yaw]
    in grade. Conventie ZYX (yaw-pitch-roll).
    ATENTIE: yaw va deriva - folosit doar pentru vizualizare.
    """
    w, x, y, z = q

    # Roll (rotatie in jurul axei X)
    sinr_cosp = 2 * (w*x + y*z)
    cosr_cosp = 1 - 2 * (x**2 + y**2)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (rotatie in jurul axei Y)
    sinp = 2 * (w*y - z*x)
    sinp = np.clip(sinp, -1.0, 1.0)  # evita erori numerice la arcsin
    pitch = np.arcsin(sinp)

    # Yaw (rotatie in jurul axei Z) - VA DERIVA fara magnetometru
    siny_cosp = 2 * (w*z + x*y)
    cosy_cosp = 1 - 2 * (y**2 + z**2)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.degrees(np.array([roll, pitch, yaw]))