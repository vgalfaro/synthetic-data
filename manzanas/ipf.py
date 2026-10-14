"""Algoritmos: Sinkhorn (dominio log), IPF sobre tensor denso, IPF sobre perfiles (CPU y GPU)."""
import os
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


_FUENTE_KERNELS = r'''
#define MAX_CAT 32
// Suma por categoría: cada hilo recorre un tramo de L perfiles para una manzana m, acumula en doble precisión
// y escribe una vez por categoría (pocas operaciones atómicas)
extern "C" __global__ void suma_categoria(const float* W, const int* cod, double* S, int P, int M, int C, int L) {
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M) return;
    double acc[MAX_CAT];
    for (int c = 0; c < C; c++) acc[c] = 0.0;
    int p0 = blockIdx.y * L, p1 = min(P, p0 + L);
    for (int p = p0; p < p1; p++) acc[cod[p]] += (double)W[(long long)p * M + m];
    for (int c = 0; c < C; c++) if (acc[c] != 0.0) atomicAdd(&S[(long long)c * M + m], acc[c]);
}
// W[p, m] *= r[cod[p], m], en el lugar
extern "C" __global__ void multiplicar(float* W, const float* r, const int* cod, long long n, int M) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { long long p = i / M; W[i] *= r[(long long)cod[p] * M + (i - p * M)]; }
}
'''
_modulo_kernels = None


def ipf_perfiles_gpu_rapido(mu_p, codigos, nus, tol=0.1, n_iter=200000, W0=None, cada=50, reportar=None, L=256,
                            tol_rel=0.0, n_per=None, log=None, checkpoint=None, cada_checkpoint=2500, it_inicial=0):
    # Mismo IPF que ipf_perfiles_gpu (mismas actualizaciones, mismo orden), con W en float32 y dos kernels propios:
    # la suma por categoría acumula en doble precisión. ~12x más rápido.
    # Convergencia: cada celda restringida cumple |error| <= max(tol, tol_rel * n_per[m]) y cada fila
    # |error| <= max(tol, tol_rel * mu_p); con tol_rel = 0 es el criterio absoluto de siempre.
    # n_iter es el tope total de iteraciones (contando las ya hechas si se retoma desde un checkpoint con it_inicial).
    # log: archivo donde se anota el avance; checkpoint: archivo .npz donde se guarda W cada `cada_checkpoint`
    # iteraciones. Devuelve (W float32, iteraciones, convergió, error marginal máximo en personas)
    import cupy as cp

    global _modulo_kernels
    if _modulo_kernels is None:
        _modulo_kernels = cp.RawModule(code=_FUENTE_KERNELS)
    k_suma, k_mult = _modulo_kernels.get_function("suma_categoria"), _modulo_kernels.get_function("multiplicar")

    P, M = len(mu_p), nus[0].shape[1]
    assert max(n.shape[0] for n in nus) <= 32, "un bloque tiene más de 32 categorías (límite del kernel)"
    cod_g = [cp.asarray(np.asarray(c, dtype=np.int32)) for c in codigos]
    nu_g = [cp.asarray(n, dtype=cp.float64) for n in nus]
    libre_g = [cp.isnan(n) for n in nu_g]
    mu_g = cp.asarray(mu_p, dtype=cp.float64)[:, None]
    W = cp.ones((P, M), dtype=cp.float32) if W0 is None else cp.asarray(W0, dtype=cp.float32)
    n_total = P * M
    grid_suma, grid_mult = ((M + 127) // 128, (P + L - 1) // L), ((n_total + 255) // 256,)
    t0 = time.time()

    def suma_cat(cod, C):
        S = cp.zeros((C, M), dtype=cp.float64)
        k_suma(grid_suma, (128,), (W, cod, S, np.int32(P), np.int32(M), np.int32(C), np.int32(L)))
        return S

    # Error permitido por celda (manzana) y por fila (perfil)
    tamaño = cp.asarray(n_per, dtype=cp.float64)[None, :] if n_per is not None else cp.zeros((1, M))
    permitido_celda = cp.maximum(tol, tol_rel * tamaño)
    permitido_fila = cp.maximum(tol, tol_rel * mu_g)

    def errores():
        # (error máximo en personas, error máximo relativo a lo permitido: <= 1 significa convergido)
        d = cp.abs(W.sum(axis=1, keepdims=True, dtype=cp.float64) - mu_g)
        e_abs, e_rel = [float(d.max())], [float((d / permitido_fila).max())]
        for cod, n in zip(cod_g, nu_g):
            d = cp.abs(suma_cat(cod, n.shape[0]) - n)  # NaN en las celdas sin restricción
            e_abs.append(float(cp.nanmax(d)))
            e_rel.append(float(cp.nanmax(d / permitido_celda)))
        return max(e_abs), max(e_rel)

    def anotar(texto, imprimir=True, escribir=True):
        if reportar and imprimir:
            print(f"    {texto}", flush=True)
        if log and escribir:
            with open(log, "a") as f:
                f.write(f"{time.strftime('%H:%M:%S')} {texto}\n")

    anotar(f"inicio: P={P} perfiles, M={M} manzanas, tol={tol}, tol_rel={tol_rel}, tope={n_iter}, desde it={it_inicial}")
    for it in range(it_inicial + 1, n_iter + 1):
        fila = W.sum(axis=1, keepdims=True, dtype=cp.float64)
        W *= cp.where(fila > 0, mu_g / cp.where(fila > 0, fila, 1.0), 1.0).astype(cp.float32)
        for cod, n, libre in zip(cod_g, nu_g, libre_g):
            S = suma_cat(cod, n.shape[0])
            r = cp.where((S > 0) & ~libre, n / cp.where(S > 0, S, 1.0), 1.0).astype(cp.float32)
            k_mult(grid_mult, (256,), (W, r, cod, np.int64(n_total), np.int32(M)))
        if it % cada == 0:
            e_abs, e_rel = errores()
            imprimir, escribir = bool(reportar) and it % reportar == 0, it % 500 == 0
            if imprimir or escribir:
                anotar(f"it={it:>6} | error marginal={e_abs:.3e} personas | relativo a lo permitido={e_rel:.3f} | "
                       f"{time.time() - t0:.0f}s", imprimir, escribir)
            if e_rel <= 1:
                anotar(f"convergió en it={it} | error marginal={e_abs:.3e} | {time.time() - t0:.0f}s")
                return cp.asnumpy(W), it, True, e_abs
        if checkpoint and it % cada_checkpoint == 0:
            tmp = checkpoint + ".tmp.npz"
            np.savez(tmp, W=cp.asnumpy(W), it=it)
            os.replace(tmp, checkpoint)
    e_abs, e_rel = errores()
    anotar(f"tope de {n_iter} iteraciones sin converger | error marginal={e_abs:.3e} | relativo={e_rel:.3f} | {time.time() - t0:.0f}s")
    return cp.asnumpy(W), n_iter, False, e_abs


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
