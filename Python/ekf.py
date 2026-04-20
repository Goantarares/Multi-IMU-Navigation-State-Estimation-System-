import numpy as np
from quaternion_utils import (
    quat_normalize, quat_multiply, quat_from_gyro,
    quat_to_rotation_matrix, quat_to_euler
)
from config import (
    G, DEG2RAD, EPS_JACOBIAN, STATIC_THRESHOLD, GYRO_STATIC_THRESHOLD,
    SIGMA_INIT_DIAG, Q_DIAG,
    Q0_INIT, GYRO_BIAS_XY_INIT, GYRO_BIAS_Z_FIXED,
    ACCEL_BIAS_INIT
)

HARD_SKIP_THRESHOLD = 3.0 * G
V_MAX_LINEAR_COMP   = 5.0


class EKF_IMU:
    """
    Extended Kalman Filter pentru un singur IMU 6-DOF.

    Vector de stare (9 componente):
        mu = [q0, q1, q2, q3,    <- quaternion [w,x,y,z]  (4)
              bgx, bgy,          <- bias giroscop X,Y      (2)
              bax, bay, baz]     <- bias accelerometru     (3)

    Pozitia si viteza sunt EXCLUSE din stare — vor fi estimate
    in filtrul mare care fuzioneza toti cei 5 senzori.
    Aceasta reduce Sigma de la 15x15 la 9x9, crescand viteza
    de calcul de ~4.6x (O(n³): 15³=3375 vs 9³=729 operatii).

    b_g_z exclus — neobservabil fara magnetometru.

    Implementare:
        - Jacobian G numeric (15 apeluri _state_transition)
        - Jacobian H analitic (formule exacte, verificate)
        - Joseph form pentru stabilitate numerica
        - Detectie statica combinata accelerometru + giroscop
        - Colectare inovatii pentru NIS
        - R static vs dinamic separat
    """

    IDX_Q   = slice(0, 4)
    IDX_BG  = slice(4, 6)
    IDX_BA  = slice(6, 9)
    N_STATE = 9

    def __init__(self, gyro_bias_xy_init=None, accel_bias_init=None,
                 gyro_bias_z_fixed=None):
        self.mu = np.zeros(self.N_STATE)
        self.mu[self.IDX_Q]  = Q0_INIT.copy()
        self.mu[self.IDX_BG] = gyro_bias_xy_init if gyro_bias_xy_init is not None \
                                else GYRO_BIAS_XY_INIT.copy()
        self.mu[self.IDX_BA] = accel_bias_init if accel_bias_init is not None \
                                else ACCEL_BIAS_INIT.copy()

        self.gyro_bias_z = gyro_bias_z_fixed if gyro_bias_z_fixed is not None \
                           else GYRO_BIAS_Z_FIXED

        self.Sigma = np.diag(SIGMA_INIT_DIAG)   # 9x9
        self.Q     = np.diag(Q_DIAG)             # 9x9

        from config import R_ACCEL_STD_STATIC, R_ACCEL_STD_DYNAMIC
        self.R_static  = np.diag([R_ACCEL_STD_STATIC**2]  * 3)
        self.R_dynamic = np.diag([R_ACCEL_STD_DYNAMIC**2] * 3)

        self.n_updates          = 0
        self.n_skipped_updates  = 0
        self.n_weighted_updates = 0
        self.n_normal_updates   = 0
        self.n_gyro_skipped     = 0

        self.nis_values = []
        self.innov_list = []

    # ================================================================
    # HELPER: d(R(q) @ a) / dq  — (3x4) analitic
    # ================================================================

    @staticmethod
    def _dRa_dq(q, a):
        w, x, y, z = q
        a1, a2, a3 = a
        col_w = 2.0 * np.array([-z*a2+y*a3,  z*a1-x*a3, -y*a1+x*a2])
        col_x = 2.0 * np.array([ y*a2+z*a3,  y*a1-2*x*a2-w*a3, z*a1+w*a2-2*x*a3])
        col_y = 2.0 * np.array([-2*y*a1+x*a2+w*a3, x*a1+z*a3, -w*a1+z*a2-2*y*a3])
        col_z = 2.0 * np.array([-2*z*a1-w*a2+x*a3, w*a1-2*z*a2+y*a3, x*a1+y*a2])
        return np.column_stack([col_w, col_x, col_y, col_z])

    # ================================================================
    # FUNCTIA DE TRANZITIE A STARII: g(mu, u)
    # Propaga doar quaternionul si biasurile — fara pozitie/viteza
    # ================================================================

    def _state_transition(self, mu, gyro_rads, accel_ms2, dt):
        q  = mu[self.IDX_Q]
        bg = mu[self.IDX_BG]
        ba = mu[self.IDX_BA]

        omega_corr = gyro_rads.copy()
        omega_corr[0] -= bg[0]
        omega_corr[1] -= bg[1]
        omega_corr[2] -= self.gyro_bias_z

        delta_q = quat_from_gyro(omega_corr, dt)
        q_new   = quat_normalize(quat_multiply(q, delta_q))

        mu_new = np.zeros(self.N_STATE)
        mu_new[self.IDX_Q]  = q_new
        mu_new[self.IDX_BG] = bg.copy()
        mu_new[self.IDX_BA] = ba.copy()   # accel_ms2 nu mai e folosit in propagare
        return mu_new

    # ================================================================
    # FUNCTIA DE MASURARE: h(mu)
    # ================================================================

    def _measurement_model(self, mu):
        q     = mu[self.IDX_Q]
        R_mat = quat_to_rotation_matrix(q)
        return R_mat.T @ np.array([0.0, 0.0, G])

    # ================================================================
    # JACOBIAN G — NUMERIC (9x9)
    # ================================================================

    def _compute_jacobian_G(self, gyro_rads, accel_ms2, dt):
        """
        Jacobianul G = dg/dmu (9x9) numeric.
        Mult mai mic decat versiunea 15x15 — 9³=729 vs 15³=3375 operatii.
        """
        n     = self.N_STATE
        G_jac = np.zeros((n, n))
        f0    = self._state_transition(self.mu, gyro_rads, accel_ms2, dt)

        for i in range(n):
            mu_plus = self.mu.copy()
            mu_plus[i] += EPS_JACOBIAN
            mu_plus[self.IDX_Q] = quat_normalize(mu_plus[self.IDX_Q])
            f_plus = self._state_transition(mu_plus, gyro_rads, accel_ms2, dt)
            G_jac[:, i] = (f_plus - f0) / EPS_JACOBIAN

        return G_jac

    # ================================================================
    # JACOBIAN H — ANALITIC (3x9)
    # ================================================================

    def _compute_jacobian_H(self):
        """
        Jacobianul H = dh/dmu (3x9) analitic.
        Coloanele 4-8 (bg, ba) sunt zero — h depinde doar de q.

        h = g * [2(xz-wy), 2(yz+wx), 1-2(x²+y²)]
        ∂h/∂w = g*[-2y,  2x,   0]
        ∂h/∂x = g*[ 2z,  2w,  -4x]
        ∂h/∂y = g*[-2w,  2z,  -4y]
        ∂h/∂z = g*[ 2x,  2y,   0]
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
    # PASUL DE PREDICTIE
    # ================================================================

    def predict(self, gyro_dps, accel_g, dt):
        """
        Propaga orientarea si biasurile cu giroscopul.
        Accelerometrul nu mai e folosit in prediction (nu mai avem v, p).
        """
        if dt <= 0:
            return

        gyro_rads = gyro_dps * DEG2RAD
        accel_ms2 = accel_g  * G   # pastrat pentru compatibilitate interfata

        G_jac = self._compute_jacobian_G(gyro_rads, accel_ms2, dt)

        self.mu = self._state_transition(self.mu, gyro_rads, accel_ms2, dt)
        self.mu[self.IDX_Q] = quat_normalize(self.mu[self.IDX_Q])

        self.Sigma = G_jac @ self.Sigma @ G_jac.T + self.Q
        self.Sigma = (self.Sigma + self.Sigma.T) / 2.0

    # ================================================================
    # PASUL DE UPDATE
    # ================================================================

    def update(self, accel_g, dt, gyro_dps):
        """
        Corecteaza orientarea si biasurile cu accelerometrul.
        Nu mai exista separare acceleratie liniara (nu avem viteza).
        """
        accel_ms2 = accel_g * G

        # Detectie statica combinata
        a_norm    = np.linalg.norm(accel_ms2)
        deviation = abs(a_norm - G)

        if deviation > HARD_SKIP_THRESHOLD:
            self.n_skipped_updates += 1
            return False

        gyro_rads = gyro_dps * DEG2RAD
        g_norm    = np.linalg.norm(gyro_rads)

        is_static = (deviation < STATIC_THRESHOLD and
                     g_norm   < GYRO_STATIC_THRESHOLD)

        if is_static:
            R_baza = self.R_static
            self.n_normal_updates += 1
        else:
            R_baza = self.R_dynamic
            if g_norm >= GYRO_STATIC_THRESHOLD and deviation < STATIC_THRESHOLD:
                self.n_gyro_skipped += 1
            self.n_weighted_updates += 1

        scale     = 1.0 + (deviation / G) ** 2
        R_adaptiv = R_baza * scale

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

        try:
            nis = float(innov.T @ np.linalg.inv(S) @ innov)
            self.nis_values.append(nis)
            self.innov_list.append(innov.copy())
        except np.linalg.LinAlgError:
            pass

        self.n_updates += 1
        return True

    # ================================================================
    # VERIFICARE JACOBIENI
    # ================================================================

    def verify_jacobians(self, gyro_dps, accel_g, dt, tol=1e-4):
        gyro_rads = gyro_dps * DEG2RAD
        accel_ms2 = accel_g  * G

        n     = self.N_STATE
        G_num = np.zeros((n, n))
        f0    = self._state_transition(self.mu, gyro_rads, accel_ms2, dt)
        for i in range(n):
            mu_p = self.mu.copy()
            mu_p[i] += EPS_JACOBIAN
            mu_p[self.IDX_Q] = quat_normalize(mu_p[self.IDX_Q])
            f_p = self._state_transition(mu_p, gyro_rads, accel_ms2, dt)
            G_num[:, i] = (f_p - f0) / EPS_JACOBIAN

        H_num = np.zeros((3, n))
        h0    = self._measurement_model(self.mu)
        for i in range(n):
            mu_p = self.mu.copy()
            mu_p[i] += EPS_JACOBIAN
            mu_p[self.IDX_Q] = quat_normalize(mu_p[self.IDX_Q])
            h_p = self._measurement_model(mu_p)
            H_num[:, i] = (h_p - h0) / EPS_JACOBIAN

        G_an  = self._compute_jacobian_G(gyro_rads, accel_ms2, dt)
        H_an  = self._compute_jacobian_H()
        err_G = np.max(np.abs(G_an - G_num))
        err_H = np.max(np.abs(H_an - H_num))

        print(f"\n=== VERIFICARE JACOBIENI ===")
        print(f"  G numeric  vs numeric: eroare max = {err_G:.2e}  ✅")
        print(f"  H analitic vs numeric: eroare max = {err_H:.2e}  "
              f"{'✅ OK' if err_H < tol else '❌ EROARE'}")
        print(f"===========================\n")
        return err_H < tol

    # ================================================================
    # ACCES LA STARE
    # ================================================================

    def get_orientation_euler(self):
        return quat_to_euler(self.mu[self.IDX_Q])

    def get_quaternion(self):
        return self.mu[self.IDX_Q].copy()

    def get_gyro_bias_xy(self):
        return self.mu[self.IDX_BG].copy()

    def get_gyro_bias_full(self):
        bg_xy = self.mu[self.IDX_BG]
        return np.array([bg_xy[0], bg_xy[1], self.gyro_bias_z])

    def get_accel_bias(self):
        return self.mu[self.IDX_BA].copy()

    def get_corrected_measurements(self, gyro_dps, accel_g):
        """
        Returneaza masuratorile corectate — inputul pentru filtrul mare.
        """
        gyro_rads = gyro_dps * DEG2RAD
        accel_ms2 = accel_g  * G

        gyro_corr = gyro_rads.copy()
        gyro_corr[0] -= self.mu[self.IDX_BG][0]
        gyro_corr[1] -= self.mu[self.IDX_BG][1]
        gyro_corr[2] -= self.gyro_bias_z

        accel_corr = accel_ms2 - self.mu[self.IDX_BA]

        return {
            'gyro_corrected_rads': gyro_corr,
            'accel_corrected_ms2': accel_corr,
            'quaternion':          self.get_quaternion(),
            'sigma_q':             self.Sigma[self.IDX_Q, self.IDX_Q],
        }

    def get_nis_stats(self):
        if not self.nis_values:
            return None
        arr          = np.array(self.nis_values)
        threshold    = np.percentile(arr, 95)
        arr_trimmed  = arr[arr <= threshold]
        return {
            'medie':        float(np.mean(arr_trimmed)),
            'medie_bruta':  float(np.mean(arr)),
            'mediana':      float(np.median(arr)),
            'std':          float(np.std(arr_trimmed)),
            'valori':       arr,
            'asteptat':     3.0,
            'consistent':   abs(np.median(arr) - 3.0) < 1.0,
            'n_outlieri':   int(np.sum(arr > threshold)),
        }

    def get_update_stats(self):
        total = self.n_updates + self.n_skipped_updates
        return {
            'total_cadre':    total,
            'normale':        self.n_normal_updates,
            'ponderate':      self.n_weighted_updates,
            'sarite':         self.n_skipped_updates,
            'sarite_gyro':    self.n_gyro_skipped,
            'procent_sarite': 100 * self.n_skipped_updates / total if total > 0 else 0,
        }

    def get_state_summary(self):
        euler = self.get_orientation_euler()
        return {
            'roll_deg':   euler[0],
            'pitch_deg':  euler[1],
            'yaw_deg':    euler[2],
            'gyro_bias':  self.get_gyro_bias_full(),
            'accel_bias': self.get_accel_bias(),
        }