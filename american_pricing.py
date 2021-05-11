import numpy as np
from typing import Callable, Sequence

from numpy.polynomial.laguerre import lagval

from gen_utils import memoize
from num_utils import get_future_price_mean_var


class AmericanPricing:
    """
    In the risk-neutral measure, the underlying price x_t
    follows the Ito process: dx_t = r(t) x_t dt + dispersion(t, x_t) dz_t
    spot_price is x_0
    In this module, we only allow two types of dispersion functions,
    Type 1 (a.k.a. "lognormal") : dx_t = r(t) x_t dt + sigma(t) x_t dz_t
    Type 2 (a.k.a. "normal"): dx_t = r(t) x_t dt + sigma(t) dz_t
    payoff is a function from (t, x_t) to payoff (eg: x_t - K)
    expiry is the time to expiry of american option (in years)
    lognormal is a bool that defines whether our dispersion function
    amounts to Type 1(lognormal) or Type 2(normal)
    r(t) is a function from (time t) to risk-free rate
    sigma(t) is a function from (time t) to (sigma at time t)
    We don't provide r(t) and sigma(t) as arguments
    Instead we provide their appropriate integrals as arguments
    Specifically, we provide ir(t) and isig(t) as arguments (as follows):
    ir(t) = \int_0^t r(u) du, so discount D_t = e^{- ir(t)}
    isig(t) = \int 0^t sigma^2(u) du if lognormal == True
    else \int_0^t sigma^2(u) e^{-\int_0^u 2 r(s) ds} du
    """

    def __init__(
            self,
            spot_price: float,
            payoff: Callable[[float, np.ndarray], float],
            expiry: float,
            lognormal: bool,
            ir: Callable[[float], float],
            isig: Callable[[float], float]
    ) -> None:
        self.spot_price: float = spot_price
        self.payoff: Callable[[float, np.ndarray], float] = payoff
        self.expiry: float = expiry
        self.lognormal: bool = lognormal
        self.ir: Callable[[float], float] = ir
        self.isig: Callable[[float], float] = isig

    @memoize
    def get_all_paths(self, num_paths: int, num_dt: int) -> np.ndarray:
        dt = self.expiry / num_dt
        paths = np.empty([num_paths, num_dt + 1])
        paths[:, 0] = self.spot_price
        for i in range(num_paths):
            price = self.spot_price
            for t in range(num_dt):
                m, v = get_future_price_mean_var(
                    price,
                    t,
                    dt,
                    self.lognormal,
                    self.ir,
                    self.isig
                )
                norm_draw = np.random.normal(m, np.sqrt(v))
                price = np.exp(norm_draw) if self.lognormal else norm_draw
                paths[i, t + 1] = price
        return paths


    def get_price_from_paths_and_params(
        self,
        paths: np.ndarray,
        params: np.ndarray,
        num_dt: int,
        feature_funcs: Sequence[Callable[[int, np.ndarray], float]]
    ) -> float:
        num_paths = paths.shape[0]
        prices = np.zeros(num_paths)
        dt = self.expiry / num_dt
        for path_num, path in enumerate(paths):
            step = 0
            while step <= num_dt:
                t = dt * step
                price_seq = path[:(step + 1)]
                exercise_price = self.payoff(t, price_seq)
                if step == num_dt:
                    continue_price = 0.
                else:
                    continue_price = params.dot([f(step, price_seq) for f in
                                                 feature_funcs])
                step += 1
                if exercise_price > continue_price:
                    prices[path_num] = np.exp(-self.ir(t)) * exercise_price
                    step = num_dt + 1

                # if step == num_dt + 1:
                #     print("Time = %.2f, Stock = %.3f, Exercise Price = %.3f, Continue Price = %.3f" %
                #           (t, stock_price, exercise_price, continue_price))

        # import matplotlib.pyplot as plt
        # for step in range(num_dt):
        #     t = dt * step
        #     stprcs = np.arange(100.)
        #     prsqs = [np.append(np.zeros(step), s) for s in stprcs]
        #     cp = [params.dot([f(step, prsq) for f in feature_funcs]) for prsq in prsqs]
        #     ep = [self.payoff(t, prsq) for prsq in prsqs]
        #     plt.plot(stprcs, cp, 'r', stprcs, ep, 'b')
        #     plt.show()

        return np.average(prices)


    def get_lspi_price(
        self,
        num_dt: int,
        num_paths: int,
        feature_funcs: Sequence[Callable[[int, np.ndarray], float]],
        batch_size: int,
        model_prob_draws: int
    ) -> float:
        features = len(feature_funcs)
        a_mat = np.zeros((features, features))
        b_vec = np.zeros(features)
        params = np.zeros(features)
        paths = self.get_all_paths(num_paths, num_dt + 1)
        dt = self.expiry / num_dt

        for path_num, path in enumerate(paths):

            for step in range(num_dt):
                t = step * dt
                disc = np.exp(self.ir(t) - self.ir(t + dt))
                phi_s = np.array([f(step, path[:(step + 1)]) for f in
                                  feature_funcs])
                if model_prob_draws > 1:
                    m, v = get_future_price_mean_var(
                        path[step],
                        t,
                        dt,
                        self.lognormal,
                        self.ir,
                        self.isig
                    )
                    norm_draws = np.random.normal(m, np.sqrt(v), model_prob_draws)
                    local_paths = [np.append(
                        paths[:(step + 1)],
                        nd
                    ) for nd in (np.exp(norm_draws) if self.lognormal else norm_draws)]
                else:
                    local_paths = [path[:(step + 2)]]

                all_phi_sp = np.zeros((len(local_paths), features))
                all_rewards = np.zeros(len(local_paths))

                for i, local_path in enumerate(local_paths):
                    next_payoff = self.payoff(t + dt, local_path)
                    if step == num_dt - 1:
                        next_phi = np.zeros(features)
                    else:
                        next_phi = np.array([f(step + 1, local_path)
                                             for f in feature_funcs])
                    exercise = next_payoff > params.dot(next_phi)
                    if exercise:
                        all_rewards[i] = next_payoff
                    else:
                        all_phi_sp[i, :] = next_phi

                phi_sp = np.mean(all_phi_sp, axis=0)
                reward = np.mean(all_rewards)

                a_mat += np.outer(
                    phi_s,
                    phi_s - phi_sp * disc
                )
                b_vec += reward * disc * phi_s

            if (path_num + 1) % batch_size == 0:
                params = np.linalg.inv(a_mat).dot(b_vec)
                # print(params)
                a_mat = np.zeros((features, features))
                b_vec = np.zeros(features)

        return self.get_price_from_paths_and_params(
            paths,
            params,
            num_dt,
            feature_funcs
        )

    def get_ls_price(
        self,
        num_dt: int,
        num_paths: int,
        feature_funcs: Sequence[Callable[[float, np.ndarray], float]]
    ) -> float:
        paths = self.get_all_paths(num_paths, num_dt)
        cashflow = np.array([max(self.payoff(self.expiry, paths[i, :]), 0.)
                             for i in range(num_paths)])
        dt = self.expiry / num_dt
        for t in range(num_dt - 1, 0, -1):
            """
            For each time slice t
            Step 1: collect X as features of (t, [S_0,.., S_t]) for those paths
            for which payoff(t, [S_0, ...., S_t]) > 0, and corresponding Y as
            the time-t discounted future actual cash flow on those paths.
            Step 2: Do the (X,Y) regression. Denote Y^ as regression-prediction.
            Compare Y^ versus payoff(t, [S_0, ..., S_t]). If payoff is higher,
            set cashflow at time t on that path to be the payoff, else set 
            cashflow at time t on that path to be the time-t discounted future
            actual cash flow on that path.
            """
            disc = np.exp(self.ir(t) - self.ir(t + dt))
            cashflow = cashflow * disc
            payoff = np.array([self.payoff(t, paths[i, :(t + 1)]) for
                               i in range(num_paths)])
            indices = [i for i in range(num_paths) if payoff[i] > 0]
            if len(indices) > 0:
                x_vals = np.array([[f(t, paths[i, :(t + 1)]) for f in
                                    feature_funcs] for i in indices])
                y_vals = np.array([cashflow[i] for i in indices])
                estimate = x_vals.dot(
                    np.linalg.lstsq(x_vals, y_vals, rcond=None)[0]
                )
                # plt.scatter([paths[i, t] for i in indices], y_vals, c='r')
                # plt.scatter([paths[i, t] for i in indices], estimate, c='b')
                # plt.show()

                for i, ind in enumerate(indices):
                    if payoff[ind] > estimate[i]:
                        cashflow[ind] = payoff[ind]

        return max(
            self.payoff(0., np.array([self.spot_price])),
            np.average(cashflow * np.exp(-self.ir(dt)))
        )

    def get_fqi_price(
        self,
        num_dt: int,
        num_paths: int,
        feature_funcs: Sequence[Callable[[int, np.ndarray], float]],
        batch_size: int,
        model_prob_draws: int
    ) -> float:
        features = len(feature_funcs)
        a_mat = np.zeros((features, features))
        b_vec = np.zeros(features)
        params = np.zeros(features)
        paths = self.get_all_paths(num_paths, num_dt + 1)
        dt = self.expiry / num_dt

        for path_num, path in enumerate(paths):

            for step in range(num_dt):
                t = step * dt
                disc = np.exp(self.ir(t) - self.ir(t + dt))
                phi_s = np.array([f(step, path[:(step + 1)]) for f in
                                  feature_funcs])
                if model_prob_draws > 1:
                    m, v = get_future_price_mean_var(
                        path[step],
                        t,
                        dt,
                        self.lognormal,
                        self.ir,
                        self.isig
                    )
                    norm_draws = np.random.normal(m, np.sqrt(v), model_prob_draws)
                    local_paths = [np.append(
                        paths[:(step + 1)],
                        nd
                    ) for nd in (np.exp(norm_draws) if self.lognormal else norm_draws)]
                else:
                    local_paths = [path[:(step + 2)]]

                all_max_val = np.zeros(len(local_paths))
                for i, local_path in enumerate(local_paths):
                    next_payoff = self.payoff(t + dt, local_path)
                    if step == num_dt - 1:
                        next_phi = np.zeros(features)
                    else:
                        next_phi = np.array([f(step + 1, local_path)
                                             for f in feature_funcs])
                    all_max_val[i] = max(next_payoff, params.dot(next_phi))

                max_val = np.mean(all_max_val)

                a_mat += np.outer(phi_s, phi_s)
                b_vec += phi_s * disc * max_val

            if (path_num + 1) % batch_size == 0:
                params = np.linalg.inv(a_mat).dot(b_vec)
                # print(params)
                a_mat = np.zeros((features, features))
                b_vec = np.zeros(features)

        return self.get_price_from_paths_and_params(
            paths,
            params,
            num_dt,
            feature_funcs
        )

if __name__ == '__main__':
    is_call_val = True
    spot_price_val = 60
    strike_val = 80.0
    expiry_val = 0.5
    lognormal_val = True
    r_val = 0.05
    sigma_val = 0.25
    num_dt_val = 10
    num_paths_val = 200000
    num_laguerre_val = 3
    batch_size_val = 10000
    model_prob_draws_val = 10

    # noinspection PyShadowingNames
    ir_func = lambda t, r_val=r_val: r_val * t
    # noinspection PyShadowingNames
    isig_func = lambda t, sigma_val=sigma_val: sigma_val * sigma_val * t


    def vanilla_american_payoff(_: float, x: np.ndarray) -> float:
        if is_call_val:
            ret = max(x[-1] - strike_val, 0.)
        else:
            ret = max(strike_val - x[-1], 0.)
        return ret


    # noinspection PyShadowingNames
    amp = AmericanPricing(
        spot_price=spot_price_val,
        payoff=lambda t, x: vanilla_american_payoff(t, x),
        expiry=expiry_val,
        lognormal=lognormal_val,
        ir=ir_func,
        isig=isig_func
    )

    ident = np.eye(num_laguerre_val)


    # noinspection PyShadowingNames
    def laguerre_feature_func(
            x: float,
            i: int
    ) -> float:
        # noinspection PyTypeChecker
        xp = x / strike_val
        return np.exp(-xp / 2) * lagval(xp, ident[i])


    def rl_feature_func(
            ind: int,
            x: float,
            i: int
    ) -> float:
        dt = expiry_val / num_dt_val
        t = ind * dt
        if i == 0:
            ret = 1.
        elif i < num_laguerre_val + 1:
            ret = laguerre_feature_func(x, i - 1)
        elif i == num_laguerre_val + 1:
            ret = np.sin(-t * np.pi / (2. * expiry_val) + np.pi / 2.)
        elif i == num_laguerre_val + 2:
            ret = np.log(expiry_val - t)
        else:
            rat = t / expiry_val
            ret = rat * rat
        return ret


    # lspi_price = amp.get_lspi_price(
    #     num_dt=num_dt_val,
    #     num_paths=num_paths_val,
    #     feature_funcs=[lambda t, x, i=i: rl_feature_func(t, x[-1], i) for i in
    #                    range(num_laguerre_val + 4)],
    #     batch_size=batch_size_val,
    #     model_prob_draws=model_prob_draws_val
    # )
    # print("LSPI Price = %.3f" % lspi_price)
    # fqi_price = amp.get_fqi_price(
    #     num_dt=num_dt_val,
    #     num_paths=num_paths_val,
    #     feature_funcs=[lambda t, x, i=i: rl_feature_func(t, x[-1], i) for i in
    #                    range(num_laguerre_val + 4)],
    #     batch_size=batch_size_val,
    #     model_prob_draws=model_prob_draws_val
    # )
    # print("FQI Price = %.3f" % fqi_price)

    ls_price = amp.get_ls_price(
        num_dt=num_dt_val,
        num_paths=num_paths_val,
        feature_funcs=[lambda _, x: 1.] +
                      [(lambda _, x, i=i: laguerre_feature_func(x[-1], i)) for i in
                       range(num_laguerre_val)]
    )
    print("Longstaff-Schwartz Price = %.3f" % ls_price)