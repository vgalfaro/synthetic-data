"""Redondeo iterativo de Cembrano, Correa y Verdugo (PNAS 2022) para W* (ver manzanas/redondeo.py y 09_rounding.ipynb).

Uso, desde la raíz del repo (sigue corriendo aunque se cierre la terminal):
    nohup env_censo/bin/python scripts/correr_cembrano.py 13108 > resultados/pais/redondeo/cembrano_13108.log 2>&1 &

Pasos:
    1. Punto extremo de la relajación lineal de la versión dura (todas las celdas fraccionales libres, objetivo de
       cercanía a W*), con Gurobi (barrera + crossover). Se guarda en vertice_<unidad>_<cut>.npz y se reutiliza.
    2. Algoritmo 1 (iterativo, estilo Beck-Fiala) con dos elecciones de u por parte (sum 1/(u+2) <= 1):
         A "perfiles_exactos": perfiles u = 0; manzanas y cada bloque u = 54
         B "uniforme":         las 29 partes u = 27
    3. Y = floor(W*) + z, verificación y errores por bloque -> Y_<unidad>_<cut>_cembrano_<config>.npz/.json
"""
import os
import sys

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # el script vive en scripts/; las rutas
sys.path.insert(0, RAIZ)                                             # (resultados/, manzanas/) son relativas
os.chdir(RAIZ)                                                       # a la raíz del repositorio

import json
import time

import numpy as np
import scipy.sparse as sp

from correr_redondeo import RUTA, preparar
from manzanas import redondeo as rd


SIN_LICENCIA = 75   # código de salida: la licencia de Gurobi no respondió (túnel del Mac caído); el planificador reintenta


def es_falla_de_licencia(e):
    t = str(e).lower()
    return any(k in t for k in ("token", "could not resolve", "could not connect", "license", "proxy"))


def hora():
    return time.strftime("%H:%M:%S")


def main(cut, unidad="personas", hilos=0):
    os.makedirs(RUTA, exist_ok=True)
    t_inicio = time.time()
    resumen_comuna = {"cut": cut, "unidad": unidad}
    mip, datos, inv, modelo, mu, nu = preparar(cut, unidad, tau=0.0, L=0, duro=True, objetivo="cercania")
    partes = list(dict.fromkeys(rd.partes_filas(mip)))
    print(f"[{hora()}] {cut} {unidad}: {mip['n_z']:,} aristas (celdas libres), {mip['A'].shape[0]:,} vértices "
          f"(restricciones), {len(partes)} partes", flush=True)
    resumen_comuna.update({"perfiles": mip["P"], "manzanas": mip["M"], "aristas": mip["n_z"],
                           "vertices": mip["A"].shape[0]})
    ruta_resumen = os.path.join(RUTA, f"cembrano_{unidad}_{cut}.json")

    def escribir_resumen(**extra):
        resumen_comuna.update(extra, segundos_total=time.time() - t_inicio)
        with open(ruta_resumen, "w") as f:
            json.dump(resumen_comuna, f, indent=1, default=str)

    ruta_x = os.path.join(RUTA, f"vertice_{unidad}_{cut}.npz")
    if os.path.exists(ruta_x):
        x = np.load(ruta_x)["x"]
        assert len(x) == mip["n_z"]
        print(f"[{hora()}] vértice cargado de {ruta_x}", flush=True)
    else:
        x, info = rd.vertice_lp(mip, log=os.path.join(RUTA, f"gurobi_vertice_{unidad}_{cut}.log"), hilos=hilos)
        resumen_comuna["vertice"] = info
        if x is None:
            print(f"[{hora()}] la relajación no tiene solución: {info}", flush=True)
            escribir_resumen(estado="relajacion_sin_solucion")
            return
        np.savez_compressed(ruta_x, x=x)
        print(f"[{hora()}] vértice: {info}", flush=True)
    frac = (x > 1e-9) & (x < 1 - 1e-9)
    resumen_comuna["vertice_fraccionales"] = int(frac.sum())
    print(f"[{hora()}] vértice con {frac.sum():,} aristas fraccionales; max |A x - b| = "
          f"{np.abs(mip['A'] @ x - mip['rhs']).max():.2e}", flush=True)

    configs = {
        "perfiles_exactos": {**{p: 54 for p in partes}, "perfiles": 0},
        "uniforme": {p: len(partes) - 2 for p in partes},
    }
    for nombre, u in configs.items():
        print(f"\n[{hora()}] === {nombre}: presupuesto sum 1/(u+2) = {rd.presupuesto(u):.4f}, "
              f"u = {sorted(set(u.values()))}", flush=True)
        t0 = time.time()
        z, hist = rd.redondeo_iterativo(mip, x, u, reportar=lambda s: print(f"[{hora()}]   {s}", flush=True))
        dev = rd.desviaciones_por_parte(mip, z, x)
        Y = rd.matriz_Y(mip, z)
        chequeo = rd.verificar(mip, Y, mu, nu, inv, modelo)
        eb = rd.errores_bloque(inv, modelo, Y)
        cumple = all(dev[p] <= u[p] + 1e-6 for p in partes)
        resumen = {"config": nombre, "u": u, "presupuesto": rd.presupuesto(u), "iteraciones": len(hist),
                   "segundos": time.time() - t0, "cumple_cotas_teorema": cumple, "desviacion_max_por_parte": dev,
                   "verificacion": chequeo,
                   "error_bloque_max": max(v["max"] for v in eb.values()),
                   "error_bloque_medio": float(np.mean([v["medio"] for v in eb.values()])),
                   "errores_bloque": eb, "historial": hist}
        base = os.path.join(RUTA, f"Y_{unidad}_{cut}_cembrano_{nombre}")
        sp.save_npz(base + ".npz", sp.csr_matrix(Y))
        with open(base + ".json", "w") as f:
            json.dump(resumen, f, indent=1, default=str)
        print(f"[{hora()}] {nombre}: {len(hist)} iteraciones, {resumen['segundos']:.0f} s | cumple cotas: {cumple}",
              flush=True)
        print(f"[{hora()}]   desviación máx por parte (vs vértice): perfiles {dev['perfiles']:.0f}, manzanas "
              f"{dev['manzanas']:.0f}, bloques {max(v for k, v in dev.items() if k not in ('perfiles', 'manzanas')):.0f}",
              flush=True)
        print(f"[{hora()}]   vs censo: {chequeo} | error de bloque medio {resumen['error_bloque_medio']:.3f}",
              flush=True)
        resumen_comuna[nombre] = {k: resumen[k] for k in ("iteraciones", "segundos", "cumple_cotas_teorema",
                                                          "desviacion_max_por_parte", "verificacion",
                                                          "error_bloque_max", "error_bloque_medio")}
    escribir_resumen(estado="ok")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cut", type=int)
    ap.add_argument("unidad", nargs="?", default="personas")
    ap.add_argument("--hilos", type=int, default=0, help="hilos de Gurobi (0 = todos)")
    a = ap.parse_args()
    try:
        main(a.cut, a.unidad, a.hilos)
    except Exception as e:
        if es_falla_de_licencia(e):
            print(f"[{hora()}] licencia de Gurobi no disponible: {e}", flush=True)
            raise SystemExit(SIN_LICENCIA)
        raise
