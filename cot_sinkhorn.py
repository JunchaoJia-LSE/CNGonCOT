import ot
import numpy as np
from scipy import stats as ss
from scipy import optimize as opt
import warnings

from .scenario_tree import ScenarioTree


def integrate_wrt_diff(tr_M, tr_h):
    assert tr_h.T == tr_M.T, "unequal time length"
    mat_M = tr_M.get_mat_values()
    mat_h = tr_h.get_mat_values()
    return np.dot(np.diff(mat_M, axis=1), mat_h.T[:-1])


def SS_basis(tr_X, tr_Y, etol=1e-6):
    SS = []
    for i in range(tr_X.L):
        for j in range(tr_Y.N - tr_Y.L):
            tr_M = ScenarioTree(tr_X.root)
            tr_h = ScenarioTree(tr_Y.root)

            one_hot = np.full(tr_X.L, 0)
            one_hot[i] = 1
            tr_M.assign_leaf(one_hot)
            tr_M.propagate_martingale_values()

            one_hot = np.full(tr_Y.N - tr_Y.L, 0)
            one_hot[j] = 1
            tr_h.assign_nonleaf(one_hot)

            S = integrate_wrt_diff(tr_M, tr_h)
            SS.append(S)

    SS = np.asarray(SS)
    if etol is not None:
        _, ld, vh = np.linalg.svd(SS.reshape(SS.shape[0], -1), full_matrices=False)
        SS = vh[ld > etol].reshape(-1, SS[0].shape[0], SS[0].shape[1])

    return SS


def ot_extended(mu, nu, c, SS, S_coordinates, epsilon, log=False, optimizing=False):
    if SS is not None:
        cost = c + np.tensordot(SS, S_coordinates, [0, 0])
    else:
        cost = c

    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            res_ot = ot.sinkhorn(mu, nu, cost, epsilon, log=log)
        except:
            if optimizing:
                print(cost)
                raise RuntimeError("optimization failed at ot_extended")

    pi = res_ot[0] if log else res_ot
    val = np.sum(pi * cost) - epsilon * ss.entropy(pi.flatten())
    if optimizing:
        grads = np.sum(SS * pi, axis=(1, 2))
        return val, grads
    else:
        return val, res_ot


def cot(mu, nu, c, SS, epsilon, optimizing=False, **kwargs):
    def opt_func(x):
        val, grads = ot_extended(mu, nu, c, SS, x, epsilon, optimizing=True)
        return -val, -grads  # maximize instead of minimize

    res_cot = opt.minimize(opt_func, np.full(len(SS), 0), jac=True, method='SLSQP', **kwargs)

    if optimizing:
        if not res_cot.success:
            raise RuntimeError("optimization failed @ cot")
        else:
            val, res_ot = ot_extended(mu, nu, c, SS, res_cot.x, epsilon, log=True, optimizing=False)
        return val, res_ot[1]['v']
    else:
        return res_cot


def dynamic_cournot_nash(mu, nu_len, c, potential, d_potential, SS, epsilon, **kwargs):
    def opt_func(x):
        val, psi = cot(mu, x, c, SS, epsilon, optimizing=True)
        val += potential(x)
        grad = np.log(np.maximum(psi, 1e-300)) * epsilon
        grad += d_potential(x)
        grad -= np.mean(grad)
        if np.any(np.isnan(grad)):
            raise RuntimeError(f"nan @ dynamic_cournot_nash")
        return val, grad

    res = opt.minimize(opt_func, np.full(nu_len, 1. / nu_len), jac=True, method='SLSQP',
                       constraints=opt.LinearConstraint(np.r_[np.eye(nu_len), [np.full(nu_len, 1)]],
                                                        np.r_[np.full(nu_len, 0), [-np.inf]],
                                                        np.r_[np.full(nu_len, np.inf), [1]]),
                       **kwargs)
    return res


def dynamic_cournot_nash_ot(mu, nu_len, c, potential, d_potential, SS, epsilon, **kwargs):
    def opt_func(x):
        val, res_ot = ot_extended(mu, x, c, None, None, epsilon, log=True, optimizing=False)
        psi = res_ot[1]['v']
        val += potential(x)
        grad = np.log(np.maximum(psi, 1e-300)) * epsilon
        grad += d_potential(x)
        grad -= np.mean(grad)
        if np.any(np.isnan(grad)):
            raise RuntimeError(f"nan @ dynamic_cournot_nash")
        return val, grad

    res = opt.minimize(opt_func, np.full(nu_len, 1. / nu_len), jac=True, method='SLSQP',
                       constraints=opt.LinearConstraint(np.r_[np.eye(nu_len), [np.full(nu_len, 1)]],
                                                        np.r_[np.full(nu_len, 0), [-np.inf]],
                                                        np.r_[np.full(nu_len, np.inf), [1]]),
                       **kwargs)
    return res

def competitive_social_cost(res_comp, potential_competitive, potential_cooperative):
    return res_comp.fun - potential_competitive(res_comp.x) + potential_cooperative(res_comp.x)


def _cot_naive(mu, nu, c, SS, epsilon, rtol=1e-4, delta=0.1, max_nstable=1, max_iter=1000, log=False):
    k = 0
    m = 0
    last_value = np.inf

    # Here should be a deterministic initialization for the purpose of second stage optimization
    S_coordinates = np.full(SS.shape[0], 0.)

    while True:
        cost = c + np.tensordot(SS, S_coordinates, [0, 0])
        res = ot.sinkhorn(mu, nu, cost, epsilon, log=log)
        if log:
            pi = res[0]
        else:
            pi = res

        current_value = np.sum(pi * cost) - epsilon * ss.entropy(pi.flatten())

        m += 1
        if m > max_iter:
            return {
                'res': res,
                'pi': pi,
                'success': False
            }

        if np.abs(last_value / current_value - 1) < rtol:
            k += 1
            if k >= max_nstable:
                return {
                    'res': res,
                    'pi': pi,
                    'success': True
                }
        else:
            k = 0
            last_value = current_value

        grads = np.sum(SS * pi, axis=(1, 2))
        S_coordinates += delta * grads


def _dynamic_cournot_nash_naive(mu, nu_len, c, potential, d_potential, SS, epsilon, rtol=1e-4, delta=0.1, max_nstable=1,
                                max_iter=1000):
    nu = np.full(nu_len, 1. / nu_len)
    last_value = np.inf
    k = 0
    for _ in range(max_iter):
        res_cot = _cot_naive(mu, nu, c, SS, epsilon, rtol=rtol, delta=delta, max_nstable=max_nstable, max_iter=max_iter,
                             log=True)
        psi = np.log(res_cot['res'][1]['v']) * epsilon
        current_value = potential(nu) + np.sum(res_cot['pi'] * c)

        if np.abs(last_value / current_value - 1) < rtol:
            k += 1
            if k >= max_nstable:
                return {
                    'nu': nu,
                    'pi': res_cot['pi'],
                    'psi': psi,
                    'value': current_value,
                    'success': True
                }
        else:
            k = 0
            last_value = current_value

        grad_nu = psi + d_potential(nu)
        grad_nu -= np.mean(grad_nu)
        delta_now = delta
        for k in range(len(grad_nu)):
            if grad_nu[k] > 0:
                if nu[k] / grad_nu[k] < delta_now:
                    delta_now = nu[k] / grad_nu[k]

        nu -= delta_now * grad_nu

    return {
        'nu': nu,
        'pi': res_cot['pi'],
        'psi': psi,
        'success': False
    }
