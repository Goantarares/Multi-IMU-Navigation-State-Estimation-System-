import numpy as np
from quaternion_utils import (
    quat_normalize, quat_multiply, quat_from_gyro,
    quat_to_rotation_matrix, quat_to_euler
)
from config import G, DEG2RAD, EPS_JACOBIAN

HARD_SKIP_THRESHOLD = 3.0 * G


class EKF_MARE:
    """
    Extended Kalman Filter pentru fuziunea celor 5 senzori IMU.

    Preia ca input datele deja filtrate de filtrele mici:
        - giroscop corectat de bias (rad/s)
        - accelerometru corectat de bias (m/s²)
        - quaternion estimat per senzor
        - sigma_q_trace per senzor (masura de incredere)

    Vector de stare (10 componente):
        mu = [q0, q1, q2, q3,    <- quaternion orientare [w,x,y,z]  (4)
              vx, vy, vz,        <- viteza in cadru global (m/s)     (3)
              px, py, pz]        <- pozitie in cadru global (m)      (3)

    Biasurile NU sunt in stare — deja eliminate de filtrele mici.

    Prediction: giroscop fuzionat ponderat (1/sigma_q_trace per senzor)
    Update:     update secvential cu accelerometrul fiecarui senzor,
                R scalat cu sigma_q_trace al senzorului respectiv
    """

    IDX_Q   = slice(0, 4)
    IDX_V   = slice(4, 7)
    IDX_P   = slice(7, 10)
    N_STATE = 10

    def __init__(self, n_sensors=5,
                 q0_init=None,
                 sigma_init_diag=None,
                 Q_diag=None,
                 R_accel_std=0.01):
        """
        n_sensors:        numarul de senzori (default 5)
        q0_init:          quaternion initial [w,x,y,z]
        sigma_init_diag:  diagonala covariantei initiale (10,)
        Q_diag:           diagonala zgomotului de proces (10,)
        R_accel_std:      deviatia standard a accelerometrului (m/s²)
                          — punct de start, scalat dinamic cu sigma_q
        """
        self.n_sensors = n_sensors

        # Starea initiala
        self.mu = np.zeros(self.N_STATE)
        self.mu[self.IDX_Q] = q0_init if q0_init is not None \
                              else np.array([1.0, 0.0, 0.0, 0.0])

        # Covarianta initiala
        if sigma_init_diag is not None:
            self.Sigma = np.diag(sigma_init_diag)
        else:
            self.Sigma = np.diag([
                1e-6, 1e-6, 1e-6, 1e-6,   # q - orientare initiala cunoscuta
                1.0,  1.0,  1.0,           # v - viteza necunoscuta
                1e-6, 1e-6, 1e-6,          # p - pozitia e originea
            ])

        # Zgomot de proces
        if Q_diag is not None:
            self.Q = np.diag(Q_diag)
        else:
            self.Q = np.diag([
                2.05e-9, 2.05e-9, 2.05e-9, 2.05e-9,  # q
                1e-3,    1e-3,    1e-3,                # v
                1e-4,    1e-4,    1e-4,                # p
            ])

        # R de baza per senzor — va fi scalat cu sigma_q_trace
        self.R_base = np.diag([R_accel_std**2] * 3)

        self._v_prev = np.zeros(3)

        # Statistici
        self.n_updates         = 0
        self.n_skipped_updates = 0
        self.nis_values        = []

    # ================================================================
    # FUNCTIA DE TRANZITIE A STARII: g(mu, u)
    # ================================================================

    def _state_transition(self, mu, omega_fuz, accel_fuz, dt):
        """
        Propaga starea cu giroscopul si accelerometrul fuzionat.

        omega_fuz: viteza unghiulara fuzionata (rad/s)
        accel_fuz: acceleratie fuzionata corectata (m/s²)
        """
        q = mu[self.IDX_Q]
        v = mu[self.IDX_V]
        p = mu[self.IDX_P]

        # Propaga quaternionul
        delta_q = quat_from_gyro(omega_fuz, dt)
        q_new   = quat_normalize(quat_multiply(q, delta_q))

        # Acceleratie in cadru global
        R_mat    = quat_to_rotation_matrix(q_new)
        a_global = R_mat @ accel_fuz - np.array([0.0, 0.0, G])

        v_new = v + a_global * dt
        p_new = p + v * dt + 0.5 * a_global * dt**2

        mu_new = np.zeros(self.N_STATE)
        mu_new[self.IDX_Q] = q_new
        mu_new[self.IDX_V] = v_new
        mu_new[self.IDX_P] = p_new
        return mu_new

    # ================================================================
    # FUNCTIA DE MASURARE: h(mu)
    # ================================================================

    def _measurement_model(self, mu):
        q     = mu[self.IDX_Q]
        R_mat = quat_to_rotation_matrix(q)
        return R_mat.T @ np.array([0.0, 0.0, G])

    # ================================================================
    # JACOBIAN G — NUMERIC (10x10)
    # ================================================================

    def _compute_jacobian_G(self, omega_fuz, accel_fuz, dt):
        n     = self.N_STATE
        G_jac = np.zeros((n, n))
        f0    = self._state_transition(self.mu, omega_fuz, accel_fuz, dt)

        for i in range(n):
            mu_plus = self.mu.copy()
            mu_plus[i] += EPS_JACOBIAN
            mu_plus[self.IDX_Q] = quat_normalize(mu_plus[self.IDX_Q])
            f_plus = self._state_transition(mu_plus, omega_fuz, accel_fuz, dt)
            G_jac[:, i] = (f_plus - f0) / EPS_JACOBIAN

        return G_jac

    # ================================================================
    # JACOBIAN H — ANALITIC (3x10)
    # ================================================================

    def _compute_jacobian_H(self):
        """
        h = g * [2(xz-wy), 2(yz+wx), 1-2(x²+y²)]
        Coloanele 4-9 (v, p) sunt zero.
        """
        w, x, y, z = self.mu[self.IDX_Q]

        H = np.zeros((3, self.N_STATE))
        H[0:3, 0:4] = G * np.array([
            [-2*y,  2*z, -2*w,  2*x],
            [ 2*x,  2*w,  2*z,  2*y],
            [  0,  -4*x, -4*y,   0 ]
        ])
        return H

    # ================================================================
    # FUZIUNEA GIROSCOPULUI SI ACCELEROMETRULUI
    # ================================================================

    @staticmethod
    def fuse_weighted(measurements, weights):
        """
        Media ponderata a unui set de masuratori.
        measurements: lista de vectori (n_sensors, 3)
        weights:      lista de scalari (n_sensors,) — mai mare = mai de incredere
        """
        weights = np.array(weights)
        weights = weights / weights.sum()   # normalizeaza
        return sum(w * m for w, m in zip(weights, measurements))

    # ================================================================
    # PASUL DE PREDICTIE
    # ================================================================

    def predict(self, gyros, accels, sigma_q_traces, dt):
        """
        Prediction step cu giroscop si accelerometru fuzionat.

        gyros:          lista de 5 vectori [gx,gy,gz] in rad/s
        accels:         lista de 5 vectori [ax,ay,az] in m/s²
        sigma_q_traces: lista de 5 scalari — incertitudinea quaternionului
                        (mai mica = filtrul mic era mai sigur → contribuie mai mult)
        dt:             pas de timp in secunde
        """
        if dt <= 0:
            return

        # Ponderile sunt inversul sigma — senzorul mai sigur contribuie mai mult
        # Adaugam epsilon pentru a evita impartirea la zero
        eps      = 1e-12
        weights  = [1.0 / (s + eps) for s in sigma_q_traces]

        omega_fuz = self.fuse_weighted(gyros,  weights)
        accel_fuz = self.fuse_weighted(accels, weights)

        G_jac = self._compute_jacobian_G(omega_fuz, accel_fuz, dt)

        self._v_prev = self.mu[self.IDX_V].copy()

        self.mu = self._state_transition(self.mu, omega_fuz, accel_fuz, dt)
        self.mu[self.IDX_Q] = quat_normalize(self.mu[self.IDX_Q])

        self.Sigma = G_jac @ self.Sigma @ G_jac.T + self.Q
        self.Sigma = (self.Sigma + self.Sigma.T) / 2.0

    # ================================================================
    # PASUL DE UPDATE — SECVENTIAL PENTRU FIECARE SENZOR
    # ================================================================

    def update_sensor(self, accel_ms2, sigma_q_trace):
        """
        Update cu accelerometrul unui singur senzor.
        R e scalat cu sigma_q_trace — senzorul mai incert are R mai mare.

        accel_ms2:     [ax, ay, az] in m/s²
        sigma_q_trace: scalarul de incertitudine al senzorului
        """
        a_norm    = np.linalg.norm(accel_ms2)
        deviation = abs(a_norm - G)

        if deviation > HARD_SKIP_THRESHOLD:
            self.n_skipped_updates += 1
            return False

        # R scalat cu sigma_q_trace al senzorului
        # Senzor mai incert → R mai mare → contribuie mai putin
        sigma_scale = 1.0 + sigma_q_trace / 1e-6   # normalizeaza fata de valoare tipica
        dev_scale   = 1.0 + (deviation / G) ** 2   # scalare patratica cu deviatia de la 1g
        R_adaptiv   = self.R_base * sigma_scale * dev_scale

        H_jac = self._compute_jacobian_H()

        z     = accel_ms2
        z_hat = self._measurement_model(self.mu)
        innov = z - z_hat

        S = H_jac @ self.Sigma @ H_jac.T + R_adaptiv
        K = self.Sigma @ H_jac.T @ np.linalg.inv(S)

        self.mu = self.mu + K @ innov

        # Joseph form
        I   = np.eye(self.N_STATE)
        IKH = I - K @ H_jac
        self.Sigma = IKH @ self.Sigma @ IKH.T + K @ R_adaptiv @ K.T

        self.mu[self.IDX_Q] = quat_normalize(self.mu[self.IDX_Q])
        self.Sigma          = (self.Sigma + self.Sigma.T) / 2.0

        # NIS
        try:
            nis = float(innov.T @ np.linalg.inv(S) @ innov)
            self.nis_values.append(nis)
        except np.linalg.LinAlgError:
            pass

        self.n_updates += 1
        return True

    def update_all_sensors(self, accels, sigma_q_traces):
        """
        Update secvential cu toti senzorii.

        accels:         lista de 5 vectori [ax,ay,az] in m/s²
        sigma_q_traces: lista de 5 scalari
        """
        for accel, sigma in zip(accels, sigma_q_traces):
            self.update_sensor(accel, sigma)

    # ================================================================
    # ACCES LA STARE
    # ================================================================

    def get_orientation_euler(self):
        return quat_to_euler(self.mu[self.IDX_Q])

    def get_quaternion(self):
        return self.mu[self.IDX_Q].copy()

    def get_velocity(self):
        return self.mu[self.IDX_V].copy()

    def get_position(self):
        return self.mu[self.IDX_P].copy()

    def get_nis_stats(self):
        if not self.nis_values:
            return None
        arr         = np.array(self.nis_values)
        threshold   = np.percentile(arr, 95)
        arr_trimmed = arr[arr <= threshold]
        return {
            'medie':      float(np.mean(arr_trimmed)),
            'medie_bruta': float(np.mean(arr)),
            'mediana':    float(np.median(arr)),
            'valori':     arr,
            'asteptat':   3.0,
            'consistent': abs(np.median(arr) - 3.0) < 1.0,
            'n_outlieri': int(np.sum(arr > threshold)),
        }

    def get_state_summary(self):
        euler = self.get_orientation_euler()
        return {
            'roll_deg':  euler[0],
            'pitch_deg': euler[1],
            'yaw_deg':   euler[2],
            'vel_mps':   self.get_velocity(),
            'pos_m':     self.get_position(),
        }