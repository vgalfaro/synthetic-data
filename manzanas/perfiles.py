"""Representación por perfiles observados: W[perfil, manzana], ajuste por bloques, predicción previa y persistencia."""
import time

import numpy as np
import scipy.sparse as sp

from .ipf import ipf_perfiles, ipf_perfiles_gpu, ipf_perfiles_gpu_rapido


def construir_perfiles(codigos_persona):
    # codigos_persona: dict nombre -> array (N,) de enteros >= 0. Devuelve los nombres, los perfiles únicos (P, K),
    # el perfil de cada persona (N,) y el número de personas por perfil (P,)
    nombres = list(codigos_persona)
    X = np.column_stack([np.asarray(codigos_persona[n], dtype=np.int64) for n in nombres])
    perfiles, perfil_de_persona, mu_p = np.unique(X, axis=0, return_inverse=True, return_counts=True)
    return nombres, perfiles, perfil_de_persona.reshape(-1), mu_p


def indicadora(cod, C):
    # Matriz dispersa (C, n): la fila c marca los elementos con categoría c
    return sp.csr_matrix((np.ones(len(cod)), (cod, np.arange(len(cod)))), shape=(C, len(cod)))


def codigos_bloque(inv, bloque):
    # Categoría de cada persona: 0..C-1 = columnas publicadas, C = resto (-99, NA o no publicada).
    # Marginal (C+1, M): la fila del resto es n_per - suma de publicadas (NaN si alguna está enmascarada)
    info = inv.marginales[bloque]
    cod = np.full(len(inv.per), len(info["categorias"]), dtype=np.int64)
    for c, (_, codigos) in enumerate(info["categorias"]):
        cod[inv.per[info["variable"]].isin(codigos).to_numpy()] = c
    nu_k = np.vstack([info["nu"], inv.nu - info["nu"].sum(axis=0)])
    return cod, nu_k


def _preparar(inv, bloques, previo):
    # Perfiles de `bloques` y, si hay modelo previo, la semilla heredada: su W repartido en los perfiles nuevos
    cods, nus = map(list, zip(*[codigos_bloque(inv, b) for b in bloques]))
    _, perfiles, perfil_de_persona, mu_p = construir_perfiles(dict(zip(bloques, cods)))
    W0 = None
    if previo is not None:
        padre = np.empty(len(mu_p), dtype=np.int64)
        padre[perfil_de_persona] = previo["perfil_de_persona"]
        W0 = previo["W"][padre] * (mu_p / previo["mu_p"][padre])[:, None]
    return nus, perfiles, perfil_de_persona, mu_p, W0


def _errores(bloques, perfiles, nus, W, mu_p):
    W = W.astype(np.float64, copy=False)  # si W viene en float32, se acumula en doble precisión
    errores = {}
    for j, (b, nu_k) in enumerate(zip(bloques, nus)):
        proj, ok = indicadora(perfiles[:, j], nu_k.shape[0]) @ W, ~np.isnan(nu_k)
        errores[b] = np.abs(proj[ok] - nu_k[ok]).max()
    errores["filas (mu_p)"] = np.abs(W.sum(axis=1) - mu_p).max()
    return errores


def ajustar_modelo(inv, bloques, previo=None, n_iter=200000, tol=1e-10):
    # IPF en CPU sobre los perfiles de `bloques` (criterio: cambio por celda < tol)
    nus, perfiles, perfil_de_persona, mu_p, W0 = _preparar(inv, bloques, previo)
    t0 = time.time()
    W, it, conv = ipf_perfiles(mu_p, [perfiles[:, j] for j in range(len(bloques))], nus, n_iter=n_iter, tol=tol, W0=W0)
    return {"bloques": list(bloques), "W": W, "perfiles": perfiles, "perfil_de_persona": perfil_de_persona,
            "mu_p": mu_p, "nu_m": inv.nu, "iteraciones": it, "convergio": conv, "segundos": time.time() - t0,
            "errores": _errores(bloques, perfiles, nus, W, mu_p)}


def ajustar_modelo_gpu(inv, bloques, previo=None, tol=1e-6, n_iter=200000, reportar=None, rapido=False):
    # IPF en GPU sobre los perfiles de `bloques` (criterio: error marginal máximo <= tol personas).
    # rapido=True usa ipf_perfiles_gpu_rapido (W en float32, kernels propios)
    nus, perfiles, perfil_de_persona, mu_p, W0 = _preparar(inv, bloques, previo)
    t0 = time.time()
    solver = ipf_perfiles_gpu_rapido if rapido else ipf_perfiles_gpu
    W, it, conv, _ = solver(mu_p, [perfiles[:, j] for j in range(len(bloques))], nus,
                            tol=tol, n_iter=n_iter, W0=W0, reportar=reportar)
    return {"bloques": list(bloques), "W": W, "perfiles": perfiles, "perfil_de_persona": perfil_de_persona,
            "mu_p": mu_p, "nu_m": inv.nu, "iteraciones": it, "convergio": conv, "segundos": time.time() - t0,
            "errores": _errores(bloques, perfiles, nus, W, mu_p)}


def errores_modelo(inv, modelo):
    # Error marginal máximo por bloque (y filas mu_p) de un modelo, ajustado o cargado desde disco
    nus = [codigos_bloque(inv, b)[1] for b in modelo["bloques"]]
    return _errores(modelo["bloques"], modelo["perfiles"], nus, modelo["W"], modelo["mu_p"])


def error_relativo_modelo(inv, modelo, tol, tol_rel):
    # Máximo de |error| / max(tol, tol_rel * tamaño) sobre todas las celdas restringidas (tamaño = n_per de la
    # manzana) y filas (tamaño = mu_p). <= 1 significa que el modelo cumple el criterio de convergencia
    W = modelo["W"].astype(np.float64, copy=False)
    permitido = np.maximum(tol, tol_rel * inv.nu)[None, :]
    e = [np.max(np.abs(W.sum(axis=1) - modelo["mu_p"]) / np.maximum(tol, tol_rel * modelo["mu_p"]))]
    for j, b in enumerate(modelo["bloques"]):
        nu_k = codigos_bloque(inv, b)[1]
        proj, ok = indicadora(modelo["perfiles"][:, j], nu_k.shape[0]) @ W, ~np.isnan(nu_k)
        e.append(np.max((np.abs(proj - nu_k) / permitido)[ok]))
    return float(max(e))


def predecir_bloque(inv, modelo, bloque):
    # Extensión cerrada: nu_hat[c, m] = sum_{i: x_ik = c} W[p(i), m] / mu_p(i). Devuelve (nu_hat, nu_k, cod)
    cod, nu_k = codigos_bloque(inv, bloque)
    N_cp = indicadora(cod, nu_k.shape[0]) @ indicadora(modelo["perfil_de_persona"], len(modelo["mu_p"])).T  # (C, P)
    return N_cp @ (modelo["W"] / modelo["mu_p"][:, None]), nu_k, cod


def proyectar(modelo, bloque, C):
    # Conteos ajustados por categoría del bloque y manzana: sum_{p: x_pk = c} W[p, m]  -> (C, M)
    j = modelo["bloques"].index(bloque)
    return indicadora(modelo["perfiles"][:, j], C) @ modelo["W"]


def guardar_modelo(modelo, ruta):
    # tol y tol_rel (criterio con que se ajustó) son opcionales; NaN si el modelo no los trae.
    # nu_m (total de unidades por manzana, columnas de W) es opcional; vacío si el modelo no lo trae
    nu_m = modelo.get("nu_m")
    np.savez(ruta, W=modelo["W"], perfiles=modelo["perfiles"], perfil_de_persona=modelo["perfil_de_persona"],
             mu_p=modelo["mu_p"], nu_m=np.asarray([] if nu_m is None else nu_m), bloques=np.array(modelo["bloques"]), iteraciones=modelo["iteraciones"],
             convergio=modelo["convergio"], segundos=modelo["segundos"],
             tol=modelo.get("tol", np.nan), tol_rel=modelo.get("tol_rel", np.nan))


def cargar_modelo(ruta):
    z = np.load(ruta)
    return {"bloques": z["bloques"].tolist(), "W": z["W"], "perfiles": z["perfiles"],
            "perfil_de_persona": z["perfil_de_persona"], "mu_p": z["mu_p"],
            "nu_m": z["nu_m"] if "nu_m" in z.files and z["nu_m"].size else None, "iteraciones": int(z["iteraciones"]),
            "convergio": bool(z["convergio"]), "segundos": float(z["segundos"]),
            "tol": float(z["tol"]) if "tol" in z.files else np.nan,
            "tol_rel": float(z["tol_rel"]) if "tol_rel" in z.files else np.nan}
