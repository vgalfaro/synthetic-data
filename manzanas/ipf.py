"""Algoritmos: Sinkhorn (dominio log), IPF sobre tensor denso, IPF sobre perfiles (CPU y GPU)."""
import time

import numpy as np
import scipy.sparse as sp
from scipy.special import logsumexp


def sinkhorn(mu, nu, C, eps, n_iter=1000, tol=1e-9):
    """
    Resuelve el transporte óptimo entrópico entre mu (m,) y nu (n,)
    con matriz de costo C (m,n) y regularización eps.
    Devuelve P (m,n) como distribución de probabilidad conjunta (suma 1).
    """
    mu_ = mu / mu.sum()
    nu_ = nu / nu.sum()
    log_mu, log_nu = np.log(mu_), np.log(nu_)

    f = np.zeros_like(mu_, dtype=float)
    g = np.zeros_like(nu_, dtype=float)

    for _ in range(n_iter):
        f_prev = f.copy()
        f = eps * (log_mu - logsumexp((g[None, :] - C) / eps, axis=1))
        g = eps * (log_nu - logsumexp((f[:, None] - C) / eps, axis=0))
        if np.max(np.abs(f - f_prev)) < tol:
            break

    logP = (f[:, None] + g[None, :] - C) / eps
    return np.exp(logP)


def _reshape_to_broadcast(target, axes, ndim):
    shape = [1] * ndim
    dims = [d for d in range(ndim) if d not in axes]
    for dim, size in zip(dims, target.shape):
        shape[dim] = size
    return target.reshape(shape)


def ipf(seed, constraints, n_iter=500, tol=1e-10):
    # constraints: lista de (ejes_a_SUMAR, marginal_objetivo); NaN en el objetivo = sin restricción.
    # Para cuando el cambio máximo por celda en una pasada es < tol. Devuelve (T, iteraciones, convergió)
    T = seed.astype(float).copy()
    ndim = T.ndim
    reshaped = [(axes, _reshape_to_broadcast(t, axes, ndim)) for axes, t in constraints]
    for it in range(n_iter):
        T_prev = T.copy()
        for axes, target in reshaped:
            current = T.sum(axis=axes, keepdims=True)
            ratio = np.divide(target, current, out=np.ones_like(current), where=current > 0)
            factor = np.where(np.isnan(target), 1.0, ratio)  # NaN = sin restricción, deja la semilla intacta ahí
            T = T * factor
        delta = np.max(np.abs(T - T_prev))
        if delta < tol:
            return T, it, True
    return T, n_iter, False


def ipf_perfiles(mu_p, codigos, nus, n_iter=100000, tol=1e-10, W0=None):
    # W[p, m] >= 0 con sum_m W[p, m] = mu_p y, para cada bloque k: sum_{p: codigos[k][p] = c} W[p, m] = nus[k][c, m]
    # NaN en nus[k] = sin restricción (manzana enmascarada o categoría no publicada).
    # Para cuando el cambio máximo por celda en una pasada es < tol
    P, M = len(mu_p), nus[0].shape[1]
    nus = [np.asarray(n, dtype=float) for n in nus]
    E = [sp.csr_matrix((np.ones(P), (cod, np.arange(P))), shape=(nu_k.shape[0], P)) for cod, nu_k in zip(codigos, nus)]
    mu_col = np.asarray(mu_p, dtype=float)[:, None]
    W = np.ones((P, M)) if W0 is None else W0.astype(float).copy()
    for it in range(n_iter):
        W_prev = W.copy()
        fila = W.sum(axis=1, keepdims=True)
        W = W * np.divide(mu_col, fila, out=np.ones_like(fila), where=fila > 0)
        for cod, nu_k, E_k in zip(codigos, nus, E):
            actual = E_k @ W  # (C_k, M): personas por categoría y manzana
            ratio = np.divide(nu_k, actual, out=np.ones_like(actual), where=actual > 0)
            W = W * np.where(np.isnan(nu_k), 1.0, ratio)[cod, :]
        if np.max(np.abs(W - W_prev)) < tol:
            return W, it, True
    return W, n_iter, False


def ipf_perfiles_gpu(mu_p, codigos, nus, tol=1e-6, n_iter=200000, W0=None, cada=50, reportar=None):
    # Mismas actualizaciones que ipf_perfiles, en GPU (cupy). La suma por categoría usa cp.add.at porque el
    # entorno no trae cuBLAS ni cuSPARSE. Para cuando el error marginal máximo (personas) de todas las
    # restricciones, filas mu_p incluidas, es <= tol; se revisa cada `cada` iteraciones.
    # Devuelve (W en numpy, iteraciones, convergió, error)
    import cupy as cp

    P, M = len(mu_p), nus[0].shape[1]
    cod_g = [cp.asarray(c) for c in codigos]
    nu_g = [cp.asarray(n, dtype=cp.float64) for n in nus]
    libre_g = [cp.isnan(n) for n in nu_g]
    mu_g = cp.asarray(mu_p, dtype=cp.float64)[:, None]
    W = cp.ones((P, M)) if W0 is None else cp.asarray(W0, dtype=cp.float64)
    t0 = time.time()

    def suma_cat(cod, C):
        S = cp.zeros((C, M))
        cp.add.at(S, cod, W)
        return S

    def error_max():
        e = [float(cp.abs(W.sum(axis=1, keepdims=True) - mu_g).max())]
        e += [float(cp.nanmax(cp.abs(suma_cat(cod, n.shape[0]) - n))) for cod, n in zip(cod_g, nu_g)]
        return max(e)

    for it in range(1, n_iter + 1):
        fila = W.sum(axis=1, keepdims=True)
        W *= cp.where(fila > 0, mu_g / cp.where(fila > 0, fila, 1.0), 1.0)
        for cod, n, libre in zip(cod_g, nu_g, libre_g):
            act = suma_cat(cod, n.shape[0])
            W *= cp.where((act > 0) & ~libre, n / cp.where(act > 0, act, 1.0), 1.0)[cod, :]
        if it % cada == 0:
            err = error_max()
            if reportar and it % reportar == 0:
                print(f"    it={it:>6} | error marginal={err:.2e} | {time.time() - t0:.0f}s", flush=True)
            if err <= tol:
                return cp.asnumpy(W), it, True, err
    return cp.asnumpy(W), n_iter, False, error_max()
