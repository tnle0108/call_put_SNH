import json
import numpy as np
from scipy.optimize import minimize
from quantmr.config.loader import _SPECS_DIR
from quantmr.config import load_spec
from quantmr.utils import DayCount, KalmanFilter


class HullWhite:
    HULLWHITE_CACHE: dict[str, "HullWhite"] = {}
    # HULLWHITE_CACHE.clear()

    def __init__(
        self,
        a: float | np.float64 = 0.0,
        sigma: float | np.float64 = 0.0,
        sigma_eps: float | np.float64 = 0.0,
    ):
        self._a = a
        self._sigma = sigma
        self._sigma_eps = sigma_eps

    @property
    def a(self) -> np.float64:
        return np.float64(self._a)

    @a.setter
    def a(self, value: float | np.float64) -> None:
        self._a = float(value)

    @property
    def sigma(self) -> np.float64:
        sigma = np.float64(self._sigma)
        if sigma < 0.0:
            raise ValueError("Sigma must be non-negative")
        return sigma

    @sigma.setter
    def sigma(self, value: float | np.float64) -> None:
        self._sigma = float(value)

    @property
    def sigma_eps(self) -> np.float64:
        sigma_eps = np.float64(self._sigma_eps)
        if sigma_eps < 0.0:
            raise ValueError("sigma_eps must be non-negative")
        return sigma_eps

    @sigma_eps.setter
    def sigma_eps(self, value: float | np.float64) -> None:
        self._sigma_eps = float(value)

    @staticmethod
    def _B(a: np.float64, delta_t: np.ndarray) -> np.ndarray:
        if np.isclose(a, 0.0):
            return delta_t
        return (1.0 - np.exp(-a * delta_t)) / a

    @staticmethod
    def _V(a: np.float64, sigma: np.float64, t: np.float64) -> np.ndarray:
        if np.isclose(a, 0.0):
            return np.asarray(sigma**2 * t, dtype=np.float64)
        return np.asarray(
            sigma**2 * (1.0 - np.exp(-2.0 * a * t)) / (2.0 * a),
            dtype=np.float64,
        )

    @staticmethod
    def _mu(a: np.float64, sigma: np.float64, t: np.float64 | np.ndarray) -> np.ndarray:
        if np.isclose(a, 0.0):
            return np.zeros_like(np.asarray(t, dtype=np.float64))
        return np.asarray(
            sigma**2 / (2.0 * a**2) * (1.0 - np.exp(-a * t)) ** 2,
            dtype=np.float64,
        )

    def sim_x(
        self,
        t: float | np.float64,
        epsilon: np.ndarray,
    ) -> np.ndarray:
        a, sigma, t = self.a, self.sigma, np.float64(t)
        return self._mu(a, sigma, t) + np.sqrt(self._V(a, sigma, t)) * epsilon

    @staticmethod
    def zero_rate(
        t: float | np.float64,
        T: float | np.float64 | np.ndarray,
        P0_t: float | np.ndarray,
        P0_T: float | np.ndarray,
        x_t: float | np.ndarray,
        a: float | np.float64,
        sigma: float | np.float64,
    ) -> np.ndarray:
        t = np.float64(t)
        a = np.float64(a)
        sigma = np.float64(sigma)
        T = np.atleast_1d(np.asarray(T, dtype=np.float64))
        x_t = np.atleast_1d(np.asarray(x_t, dtype=np.float64))
        P0_T = np.atleast_1d(np.asarray(P0_T, dtype=np.float64))
        P0_t = np.float64(P0_t)

        delta_t = T - t
        B = HullWhite._B(a, delta_t)
        V = HullWhite._V(a, sigma, t)

        log_ratio = np.where(delta_t > 0.0, np.log(P0_T / P0_t), 0.0)

        bracket = (
            log_ratio[..., None]
            - B[..., None] * x_t
            - 0.5 * (B**2)[..., None] * V
        )

        safe_dt = np.where(delta_t > 0.0, delta_t, 1.0)[..., None]
        return np.where(delta_t[..., None] > 0.0, -bracket / safe_dt, 0.0)

    def rate_adjustment(
        self,
        t: float | np.float64,
        delta_t: float | np.float64 | list | np.ndarray,
        epsilon: np.ndarray,
    ) -> np.ndarray:
        t = np.float64(t)
        delta_t = np.atleast_1d(np.asarray(delta_t, dtype=np.float64))
        x = self.sim_x(t, epsilon)
        return self.zero_rate(t, t + delta_t, 1.0, np.ones_like(delta_t), x, self.a, self.sigma)

    def filter_state(
        self,
        curve_name: str,
        tau: list | np.ndarray | None = None,
        as_of_range: (
            tuple[str | np.datetime64, str | np.datetime64] | None
        ) = None,
        selected_col: list[str] | slice | None = None,
        exclude_tenors: list[str] | None = None,
        convention: str = "act365",
    ) -> dict:
        from quantmr.curve.zerocurve import ZeroCurve
        from quantmr.curve.curvenode import CurveNode

        a = float(self.a)
        sigma = float(self.sigma)
        sigma_eps = float(self.sigma_eps)
        if a <= 0 or sigma <= 0 or sigma_eps <= 0:
            raise ValueError(
                f"filter_state() requires calibrated a>0, sigma>0, sigma_eps>0; "
                f"got a={a}, sigma={sigma}, sigma_eps={sigma_eps}. "
                f"Run calibrate(method='kfmh', save=True) first."
            )

        if exclude_tenors is None:
            exclude_tenors = ["ON", "1W", "2W"]
        if selected_col is None and exclude_tenors:
            all_cols = list(CurveNode.get(curve_name).terms)
            selected_col = [c for c in all_cols if c not in exclude_tenors]

        if tau is None:
            tau = [1/12, 2/12, 3/12, 6/12, 9/12, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
        tau = np.asarray(tau, dtype=np.float64)
        m = len(tau)

        daycount = DayCount.get(convention)
        curvenode = CurveNode.get(
            curve_name, as_of_range=as_of_range, selected_col=selected_col,
        )
        dates = curvenode.as_of
        N = len(dates)
        if N < 2:
            raise ValueError(f"Need at least 2 dates, got {N}")

        t_abs = daycount.yearfrac(dates[0], dates)
        dt = daycount.yearfrac(dates[:-1], dates[1:])

        y = np.empty((N, m), dtype=np.float64)
        for i, d in enumerate(dates):
            zc = ZeroCurve.get(curve_name, d, selected_col=selected_col)
            y[i] = zc.rate(tau, compounding="continuous")

        zc0 = ZeroCurve.get(curve_name, dates[0], selected_col=selected_col)

        kf_inputs = _build_kf_inputs(
            a, sigma, sigma_eps, tau, t_abs, dt, zc0, m, N,
        )
        kf = KalmanFilter(**kf_inputs)
        filt = kf.filter(y)

        x_filt = filt["x_filt"][:, 0]
        P_filt = filt["P_filt"][:, 0, 0]
        mu_t = HullWhite._mu(a, sigma, t_abs)
        tilde_x = x_filt - mu_t

        f0_t = zc0.forward_rate(
            np.zeros_like(t_abs), np.maximum(t_abs, 1e-8), dt=1e-6,
        )
        r_filt = x_filt + f0_t

        return {
            "a": a, "sigma": sigma, "sigma_eps": sigma_eps,
            "dates": dates,
            "t_abs": t_abs,
            "y": y,
            "x_filt": x_filt,
            "P_filt": P_filt,
            "mu_t": mu_t,
            "tilde_x": tilde_x,
            "f0_t": f0_t,
            "r_filt": r_filt,
            "tau": tau,
            "filter_result": filt,
        }

    def calibrate(
        self,
        curve_name: str,
        method: str = "kf",
        save: bool = False,
        **kwargs,
    ) -> dict:
        match method.lower():
            case "kf":
                result = self._calibrate_kf(curve_name, **kwargs)
            case "kfmh":
                result = self._calibrate_kfmh(curve_name, **kwargs)
            case _:
                raise ValueError(f"Unknown calibration method: {method}")

        if save:
            self.save(curve_name)
        return result

    def _calibrate_kf(
        self,
        curve_name: str,
        tau: list | np.ndarray | None = None,
        as_of_range: (
            tuple[str | np.datetime64, str | np.datetime64] | None
        ) = None,
        selected_col: list[str] | slice | None = None,
        exclude_tenors: list[str] | None = None,
        convention: str = "act365",
        a0: float | None = None,
        sigma0: float | None = None,
        sigma_eps0: float = 0.001,
        bounds: dict | None = None,
    ) -> dict:
        from quantmr.curve.zerocurve import ZeroCurve
        from quantmr.curve.curvenode import CurveNode

        if exclude_tenors is None:
            exclude_tenors = ["ON", "1W", "2W"]
        if selected_col is None and exclude_tenors:
            all_cols = list(CurveNode.get(curve_name).terms)
            selected_col = [c for c in all_cols if c not in exclude_tenors]

        if tau is None:
            tau = [
                1 / 12,
                2 / 12,
                3 / 12,
                6 / 12,
                9 / 12,
                1.0,
                2.0,
                3.0,
                5.0,
                7.0,
                10.0,
            ]
        tau = np.asarray(tau, dtype=np.float64)
        m = len(tau)

        daycount = DayCount.get(convention)

        curvenode = CurveNode.get(
            curve_name,
            as_of_range=as_of_range,
            selected_col=selected_col,
        )
        dates = curvenode.as_of
        N = len(dates)
        if N < 3:
            raise ValueError(f"Need at least 3 dates for calibration, got {N}")

        t_abs = daycount.yearfrac(dates[0], dates)
        dt = daycount.yearfrac(dates[:-1], dates[1:])

        y = np.empty((N, m), dtype=np.float64)
        for i, d in enumerate(dates):
            zc = ZeroCurve.get(curve_name, d, selected_col=selected_col)
            y[i] = zc.rate(tau, compounding="continuous")

        zc0 = ZeroCurve.get(curve_name, dates[0], selected_col=selected_col)

        def neg_loglik(params: np.ndarray) -> float:
            try:
                kf_inputs = _build_kf_inputs(
                    params[0],
                    params[1],
                    params[2],
                    tau,
                    t_abs,
                    dt,
                    zc0,
                    m,
                    N,
                )
            except (ValueError, FloatingPointError):
                return 1e15
            kf = KalmanFilter(**kf_inputs)
            ll = kf.loglikelihood(y)
            return -ll if np.isfinite(ll) else 1e15

        a_init = a0 if a0 is not None else (float(self._a) or 0.1)
        sigma_init = (
            sigma0 if sigma0 is not None else (float(self._sigma) or 0.01)
        )
        x0 = np.array([a_init, sigma_init, sigma_eps0])

        default_bounds = {
            "a": (1e-6, 5.0),
            "sigma": (1e-6, 1.0),
            "sigma_eps": (1e-8, 0.1),
        }
        if bounds is not None:
            default_bounds.update(bounds)
        opt_bounds = [default_bounds[k] for k in ("a", "sigma", "sigma_eps")]

        opt = minimize(neg_loglik, x0, method="L-BFGS-B", bounds=opt_bounds)
        a_cal, sigma_cal, sigma_eps_cal = opt.x

        self.a = a_cal
        self.sigma = sigma_cal
        self.sigma_eps = sigma_eps_cal

        kf_inputs = _build_kf_inputs(
            a_cal,
            sigma_cal,
            sigma_eps_cal,
            tau,
            t_abs,
            dt,
            zc0,
            m,
            N,
        )
        kf = KalmanFilter(**kf_inputs)
        filt = kf.filter(y)

        return {
            "a": a_cal,
            "sigma": sigma_cal,
            "sigma_eps": sigma_eps_cal,
            "loglik": -opt.fun,
            "result": opt,
            "filter_result": filt,
            "dates": dates,
            "t_abs": t_abs,
            "y": y,
            "tau": tau,
        }

    def _calibrate_kfmh(
        self,
        curve_name: str,
        tau: list | np.ndarray | None = None,
        as_of_range: (
            tuple[str | np.datetime64, str | np.datetime64] | None
        ) = None,
        selected_col: list[str] | slice | None = None,
        exclude_tenors: list[str] | None = None,
        convention: str = "act365",
        horizons: list[int] | None = None,
        max_tau_plus_h: float = 11.0,
        a0: float | None = None,
        sigma0: float | None = None,
    ) -> dict:
        from quantmr.curve.zerocurve import ZeroCurve
        from quantmr.curve.curvenode import CurveNode
        # CurveNode.CURVENODE_CACHE.clear()

        if exclude_tenors is None:
            exclude_tenors = ["ON", "1W", "2W"]
        if selected_col is None and exclude_tenors:
            all_cols = list(CurveNode.get(curve_name).terms)
            selected_col = [c for c in all_cols if c not in exclude_tenors]

        if tau is None:
            tau = [1/12, 2/12, 3/12, 6/12, 9/12, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
        tau = np.asarray(tau, dtype=np.float64)
        m = len(tau)

        if horizons is None:
            horizons = [21, 63, 126, 252, 504, 756, 1260, 1764, 2520]

        daycount = DayCount.get(convention)

        curvenode = CurveNode.get(
            curve_name, as_of_range=as_of_range, selected_col=selected_col,
        )
        dates = curvenode.as_of
        N = len(dates)
        if N < 10:
            raise ValueError(f"Need at least 10 dates for calibration, got {N}")

        horizons = [h for h in horizons if h < N - 1]
        if not horizons:
            raise ValueError("No valid horizons for the given data length")

        t_abs = daycount.yearfrac(dates[0], dates)
        dt = daycount.yearfrac(dates[:-1], dates[1:])

        y = np.empty((N, m), dtype=np.float64)
        for i, d in enumerate(dates):
            zc = ZeroCurve.get(curve_name, d, selected_col=selected_col)
            y[i] = zc.rate(tau, compounding="continuous")

        zc0 = ZeroCurve.get(curve_name, dates[0], selected_col=selected_col)

        def s1_neg_loglik(log_params: np.ndarray) -> float:
            a_v = np.exp(log_params[0])
            sigma_v = np.exp(log_params[1])
            sigma_eps_v = np.exp(log_params[2])
            try:
                kf_inputs = _build_kf_inputs(
                    a_v, sigma_v, sigma_eps_v, tau, t_abs, dt, zc0, m, N,
                )
                kf = KalmanFilter(**kf_inputs)
                ll = kf.loglikelihood(y)
            except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                return 1e15
            return -ll if np.isfinite(ll) else 1e15

        if a0 is not None and sigma0 is not None:
            log_x0 = np.log([a0, sigma0, 0.003])
        else:
            a_grid = [0.05, 0.15, 0.3, 0.5, 1.0, 1.5]
            s_grid = [0.003, 0.01, 0.02, 0.04]
            se_grid = [0.001, 0.003, 0.01, 0.03, 0.05]
            best_val = np.inf
            log_x0 = np.log([0.15, 0.01, 0.003])
            for ag in a_grid:
                for sg in s_grid:
                    for seg in se_grid:
                        val = s1_neg_loglik(np.log([ag, sg, seg]))
                        if val < best_val:
                            best_val = val
                            log_x0 = np.log([ag, sg, seg])

        opt1 = minimize(
            s1_neg_loglik, log_x0, method="Nelder-Mead",
            options={"maxiter": 10000, "xatol": 1e-7, "fatol": 1e-9},
        )
        a_kf = float(np.exp(opt1.x[0]))
        sigma_kf = float(np.exp(opt1.x[1]))
        sigma_eps_kf = float(np.exp(opt1.x[2]))

        kf_inputs = _build_kf_inputs(
            a_kf, sigma_kf, sigma_eps_kf, tau, t_abs, dt, zc0, m, N,
        )
        kf1 = KalmanFilter(**kf_inputs)
        filt1 = kf1.filter(y)
        x_hat = filt1["x_filt"][:, 0]
        P_hat = filt1["P_filt"][:, 0, 0]

        f0_t = zc0.forward_rate(
            np.zeros_like(t_abs), np.maximum(t_abs, 1e-8), dt=1e-6,
        )
        r_hat = x_hat + f0_t

        stage1 = {
            "a": a_kf, "sigma": sigma_kf, "sigma_eps": sigma_eps_kf,
            "loglik": -opt1.fun, "result": opt1,
            "filter_result": filt1,
        }

        R0_times = np.where(
            t_abs > 0,
            zc0.rate(np.maximum(t_abs, 1e-10), compounding="continuous") * t_abs,
            0.0,
        )
        R0T_times = np.empty((N, m), dtype=np.float64)
        for i in range(N):
            T_ij = t_abs[i] + tau
            R0T_times[i] = zc0.rate(T_ij, compounding="continuous") * T_ij

        horizon_data = []
        for h in horizons:
            if h >= N:
                continue
            h_yf = daycount.yearfrac(dates[:N - h], dates[h:N])
            h_yf_max = float(h_yf.max())
            valid_tenors = np.where(tau + h_yf_max <= max_tau_plus_h)[0]
            if len(valid_tenors) == 0:
                continue
            horizon_data.append({
                "h": h,
                "h_yf": h_yf,
                "valid_tenors": valid_tenors,
                "y_future": y[h:N],
                "x_past": x_hat[:N - h],
                "P_past": P_hat[:N - h],
                "t_past": t_abs[:N - h],
                "t_future": t_abs[h:N],
                "R0t_future": R0_times[h:N],
                "R0T_future": R0T_times[h:N],
            })

        if not horizon_data:
            raise ValueError("No valid horizon/tenor pairs for Stage 2")

        def s2_neg_avg_loglik(log_params: np.ndarray) -> float:
            a_v = np.exp(log_params[0])
            sigma_v = np.exp(log_params[1])
            try:
                total_ll, n_obs_v = _stage2_eval(
                    a_v, sigma_v, sigma_eps_kf, horizon_data, tau
                )
            except (ValueError, FloatingPointError, np.linalg.LinAlgError,
                    ZeroDivisionError):
                return 1e15
            if n_obs_v == 0 or not np.isfinite(total_ll):
                return 1e15
            return -total_ll / n_obs_v

        opt2 = minimize(
            s2_neg_avg_loglik, np.log([a_kf, sigma_kf]), method="Nelder-Mead",
            options={"maxiter": 5000, "xatol": 1e-7, "fatol": 1e-9},
        )
        a_cal = float(np.exp(opt2.x[0]))
        sigma_cal = float(np.exp(opt2.x[1]))
        total_ll, n_obs = _stage2_eval(
            a_cal, sigma_cal, sigma_eps_kf, horizon_data, tau
        )

        self.a = a_cal
        self.sigma = sigma_cal
        self.sigma_eps = sigma_eps_kf

        return {
            "a": a_cal,
            "sigma": sigma_cal,
            "sigma_eps": sigma_eps_kf,
            "loglik": total_ll,
            "avg_loglik": total_ll / n_obs if n_obs else float("nan"),
            "n_obs": n_obs,
            "stage1": stage1,
            "filter_result": filt1,
            "dates": dates,
            "t_abs": t_abs,
            "t0": dates[0],
            "y": y,
            "x_hat": x_hat,
            "P_hat": P_hat,
            "f0_t": f0_t,
            "r_hat": r_hat,
            "tau": tau,
            "horizons": horizons,
        }

    @classmethod
    def get(cls, curve_name: str) -> "HullWhite":
        curve_name = curve_name.lower()
        if curve_name not in cls.HULLWHITE_CACHE:
            spec = load_spec("hullwhite").get(curve_name, {})
            hw = cls(
                a=spec.get("a", 0.0),
                sigma=spec.get("sigma", 0.0),
                sigma_eps=spec.get("sigma_eps", 0.0),
            )
            cls.HULLWHITE_CACHE[curve_name] = hw
        return cls.HULLWHITE_CACHE[curve_name]

    def save(self, curve_name: str) -> None:
        spec_path = _SPECS_DIR / "hullwhite.json"
        if spec_path.exists():
            with open(spec_path, "r") as f:
                all_specs = json.load(f)
        else:
            all_specs = {}
        all_specs[curve_name.lower()] = {
            "a": float(self.a),
            "sigma": float(self.sigma),
            "sigma_eps": float(self.sigma_eps),
        }
        with open(spec_path, "w") as f:
            json.dump(all_specs, f, indent=4)
        self.HULLWHITE_CACHE[curve_name.lower()] = self


def _stage2_eval(
    a: float,
    sigma: float,
    sigma_eps: float,
    horizon_data: list,
    tau: np.ndarray,
) -> tuple[float, int]:
    a = max(a, 1e-10)
    B = (1.0 - np.exp(-a * tau)) / a
    H = B / tau
    conv_coeff = 0.5 * B**2 / tau

    total_ll = 0.0
    n_obs = 0

    for hd in horizon_data:
        h_yf = hd["h_yf"]
        vt = hd["valid_tenors"]
        y_fut = hd["y_future"]
        x_past = hd["x_past"]
        P_past = hd["P_past"]
        t_past = hd["t_past"]
        t_future = hd["t_future"]
        R0t_future = hd["R0t_future"]
        R0T_future = hd["R0T_future"]

        exp_ah = np.exp(-a * h_yf)
        Q_h = sigma**2 / (2.0 * a) * (1.0 - np.exp(-2.0 * a * h_yf))
        mu_past = HullWhite._mu(a, sigma, t_past)
        mu_future = HullWhite._mu(a, sigma, t_future)
        drift_shift = mu_future - exp_ah * mu_past
        V_future = HullWhite._V(a, sigma, t_future)

        for j in vt:
            H_j = H[j]
            tau_j = tau[j]
            F_mj = (R0T_future[:, j] - R0t_future) / tau_j
            d_fh_j = F_mj + conv_coeff[j] * V_future

            mean = d_fh_j + H_j * (exp_ah * x_past + drift_shift)
            V_y = H_j**2 * (Q_h + exp_ah**2 * P_past) + sigma_eps**2

            resid = y_fut[:, j] - mean
            contrib = -0.5 * float(
                np.sum(np.log(2.0 * np.pi * V_y) + resid**2 / V_y)
            )
            if not np.isfinite(contrib):
                return -1e15, 0
            total_ll += contrib
            n_obs += len(resid)

    return total_ll, n_obs


def _build_kf_inputs(
    a: float,
    sigma: float,
    sigma_eps: float,
    tau: np.ndarray,
    t_abs: np.ndarray,
    dt: np.ndarray,
    zc0,
    m: int,
    N: int,
) -> dict:
    a = max(a, 1e-10)

    B = (1.0 - np.exp(-a * tau)) / a
    H = (B / tau).reshape(m, 1)
    R = np.eye(m) * sigma_eps**2

    F_i = np.exp(-a * dt).reshape(-1, 1, 1)
    Q_i = (sigma**2 / (2.0 * a) * (1.0 - np.exp(-2.0 * a * dt))).reshape(
        -1, 1, 1
    )

    mu = HullWhite._mu(a, sigma, t_abs)
    c_i = (mu[1:] - np.exp(-a * dt) * mu[:-1]).reshape(-1, 1)

    V = HullWhite._V(a, sigma, t_abs)

    d = np.empty((N, m), dtype=np.float64)
    R0_t = np.where(
        t_abs > 0,
        zc0.rate(np.maximum(t_abs, 1e-10), compounding="continuous") * t_abs,
        0.0,
    )
    conv_coeff = 0.5 * B**2 / tau

    for i in range(N):
        T_ij = t_abs[i] + tau
        R0_T = zc0.rate(T_ij, compounding="continuous") * T_ij
        fwd = (R0_T - R0_t[i]) / tau
        d[i] = fwd + conv_coeff * V[i]

    x0 = np.zeros(1)
    P0 = np.array([[1e-10]])

    F = np.concatenate([F_i, np.ones((1, 1, 1))], axis=0)
    Q = np.concatenate([Q_i, np.ones((1, 1, 1))], axis=0)
    c = np.concatenate([c_i, np.zeros((1, 1))], axis=0)

    return {
        "F": F,
        "H": H,
        "Q": Q,
        "R": R,
        "c": c,
        "d": d,
        "x0": x0,
        "P0": P0,
    }


if __name__ == "__main__":
    print("Calibrating Hull-White for USD short rate on KF")
    hw = HullWhite.get("zc usd sr")
    result = hw.calibrate(
        "zc usd sr",
        method="kfmh",
        save=True,
    )
    print(f"Calibrated a: {result['a']:.4f}, sigma: {result['sigma']:.4f}")
    hw = HullWhite.get("zc vnd ccs sr")
    result = hw.calibrate(
        "zc vnd ccs sr",
        method="kfmh",
        save=True,
    )
    print(f"Calibrated a: {result['a']:.4f}, sigma: {result['sigma']:.4f}")
    hw = HullWhite.get("zc vnd irs")
    result = hw.calibrate(
        "zc vnd irs",
        method="kfmh",
        save=True,
        as_of_range=("2016-01-01", "2025-12-31"),
    )
    print(f"Calibrated a: {result['a']:.4f}, sigma: {result['sigma']:.4f}")
