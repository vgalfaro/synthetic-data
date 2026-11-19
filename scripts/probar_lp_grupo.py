"""Compara variantes de Gurobi para el LP blando de un grupo (vértice para el Algoritmo 1).

Uso: python scripts/probar_lp_grupo.py <cut> <g> <variante> [--hilos 4]
Variantes: base | cbasis | perturbar | dual | perturbar_cbasis | bartol (se pueden combinar; base2 = base)
Escribe resultados/pais/redondeo/pruebas_lp/<cut>_<g>_<variante>.json (tiempos, fraccionales, desviación) y el log de
Gurobi al lado.
"""
import os
import sys

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # el script vive en scripts/; las rutas
sys.path.insert(0, RAIZ)                                             # (resultados/, manzanas/) son relativas
os.chdir(RAIZ)                                                       # a la raíz del repositorio

import argparse
import json
import time

import numpy as np

import correr_cembrano_grandes as G
from manzanas import redondeo as rd

RUTA = "resultados/pais/redondeo/pruebas_lp"


def armar(cut, g):
    datos, inv, modelo, mu, nu = G.cargar(cut)
    mu_g, grupos = G.leer_division(cut)
    cols = grupos[g]
    filas = np.nonzero(mu_g[:, g] > 0)[0]
    W_g = np.asarray(modelo["W"][np.ix_(filas, cols)], dtype=np.float64)
    mu_sub, nu_sub = mu_g[filas, g], nu[cols]
    Wa_g, _ = rd.ajustar_margenes(W_g, mu_sub, nu_sub)
    inv_g = rd.subinventario(inv, cols)
    modelo_g = {"bloques": modelo["bloques"], "perfiles": modelo["perfiles"][filas]}
    F, frac, _, _ = rd.partes(Wa_g, mu_sub, nu_sub)
    return rd.construir(Wa_g, mu_sub, nu_sub, inv_g, modelo_g, libre=frac > rd.EPS, duro=False)


def resolver(mip, variante, hilos, log, peso_cercania=1e-3):
    import gurobipy as gp
    from gurobipy import GRB
    A, rhs, n_z = mip["A"], mip["rhs"], mip["n_z"]
    n = A.shape[1]
    F_frac = np.asarray(mip["frac_libre"], dtype=np.float64)
    costo = np.r_[peso_cercania * (1 - 2 * F_frac), np.ones(n - n_z)]
    if "perturbar" in variante:   # rompe empates: perturbación aleatoria chica en el costo de las celdas
        costo[:n_z] += 1e-6 * np.random.default_rng(0).random(n_z)
    m = rd._modelo_gurobi(log=log, salida=False)
    x = m.addMVar(n, lb=0.0, ub=np.r_[np.ones(n_z), np.full(n - n_z, GRB.INFINITY)])
    m.addMConstr(A.tocsr(), x, "=", rhs)
    m.setObjective(costo @ x, GRB.MINIMIZE)
    m.Params.Threads = hilos
    m.Params.Method = 1 if variante == "dual" else 2
    m.Params.Crossover = -1
    if "bartol" in variante:   # convergencia del barrier más laxa; el crossover igual entrega un vértice exacto
        m.Params.BarConvTol = 1e-6
    if "cbasis" in variante:
        m.Params.CrossoverBasis = 1
    t0 = time.time()
    m.optimize()
    seg = time.time() - t0
    xv = np.asarray(x.X, dtype=np.float64)
    z = np.clip(xv[:n_z], 0.0, 1.0)
    return {"variante": variante, "estado": m.Status, "segundos": seg, "iter_barrier": m.BarIterCount,
            "iter_simplex": m.IterCount, "desviacion_lp": float(xv[n_z:].sum()),
            "fraccionales": int(((z > 1e-6) & (z < 1 - 1e-6)).sum()), "aristas": int(n_z)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cut", type=int)
    ap.add_argument("g", type=int)
    ap.add_argument("variante")
    ap.add_argument("--hilos", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(RUTA, exist_ok=True)
    base = os.path.join(RUTA, f"{a.cut}_{a.g}_{a.variante}")
    t0 = time.time()
    mip = armar(a.cut, a.g)
    t_armar = time.time() - t0
    r = resolver(mip, a.variante, a.hilos, base + ".log")
    r["segundos_armar"] = t_armar
    json.dump(r, open(base + ".json", "w"), indent=1)
    print(json.dumps(r), flush=True)
