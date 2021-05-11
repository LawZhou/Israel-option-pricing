from typing import Callable, Tuple

import numpy as np


def get_future_price_mean_var(
    x: float,
    t: float,
    delta_t: float,
    lognormal: bool,  # whether dispersion is multiplied by x or not
    rate_int_func: Callable[[float], float],  # ir integral
    sigma2_int_func: Callable[[float], float],  # sigma^2 integral
) -> Tuple[float, float]:
    """
    :param x: represents underlying price at time t (= x_t)
    :param t: represents current time t
    :param delta_t: represents interval of time beyond t at which
    we want the future price, i.e., at time t + delta_t
    :param lognormal: this indicates whether dispersion func is
    multiplied by x or not (i.e., whether lognormal or normal)
    :param rate_int_func: this is ir(t) func
    :param sigma2_int_func: this is isig(t) func
    :return: mean and variance of x_{t+delta_t} if
    lognormal == True, else return mean and variance of
    log(x_{t+delta_t})
    rate_int_func is ir(t) = int_0^t r(u) du
    If lognormal == True, we have generalized GBM
    dx_t = r(t) x_t dt + sigma(t) x_t dz_t
    The solution is (denoting t + delta_t as t1):
    x_{t1} = x_t . e^{int_t^{t1} (r(u) -
     sigma^2(u)/2) du + int_t^{t1} sigma(u) dz_u}
     So, log(x_{t1}) is normal with:
    Mean[log(x_{t1})] = log(x_t) + int_t^{t1} (r(u) - sigma^2(u)/2) du
    Variance[log(x_{t1}] = int_t^{t1} sigma^2(u) du
    In the case that lognormal == True, sigma2_int_func
    = isig(t) = int_0^t sigma^2(u) du
    Therefore, in the case that lognormal == True,
    log(x_{t1}) is normal with:
    Mean[log(x_{t1})] = log(x_t) + ir(t1) - ir(t) + (isig(t) - isig(t1)) / 2
    Variance[log(x_{t1})] = isig(t1) - isig(t)
    If lognormal == False, we have generalize OU with mean-reversion to 0
    dx_t = r(t) x_t dt + sigma(t) dz_t
    The solution is (denoting t + delta_t as t1)
    x_{t1} = x_t e^{int_t^{t1} r(u) du} +
     (e^{int_0^{t1} r(u) du}) . (int_t^t1 sigma(u) e^{-int_0^u r(s) ds} d_zu)
     So, x_{t1} is normal with:
    Mean[x_{t1}] = x_t . e^{int_t^{t1} r(u) du}
    Variance[x_{t1}] = (e^{int_0^{t1} 2 r(u) du})) .
    (int_t^t1 sigma^2(u) e^{-int_0^u 2 r(s) ds} du)
    In the case that lognormal == False, sigma2_int_func
    = isig(t) = int_0^t sigma^2(u) . e^{-int_0^u 2 r(s) ds} . du
    Therefore, in the case that lognormal == False,
    x_{t1} is normal with:
    Mean[x_{t1}] = x_t . e^{ir(t1) - ir(t)}
    Variance[x_{t1}] = e^{2 ir(t1)} . (isig(t1) - isig(t))
    """
    ir_t = rate_int_func(t)
    ir_t1 = rate_int_func(t + delta_t)
    isig_t = sigma2_int_func(t)
    isig_t1 = sigma2_int_func(t + delta_t)
    ir_diff = ir_t1 - ir_t
    isig_diff = isig_t1 - isig_t

    if lognormal:
        mean = np.log(x) + ir_diff - isig_diff / 2.
        var = isig_diff
    else:
        mean = x * np.exp(ir_diff)
        var = np.exp(2. * ir_t1) * isig_diff
    return mean, var