"""Redondeo controlado de W* con restricciones por bloque (MIP con HiGHS). Ver 09_rounding.ipynb y 10_redondeo_duro.ipynb.

Uso, desde la raíz del repo (sigue corriendo aunque se cierre la terminal):
    # versión dura (bloques como igualdades), objetivo 0 = solo factibilidad
    nohup env_censo/bin/python scripts/correr_redondeo.py 13108 --duro --tau 0.01 --L 3 --horas 24 \\
        > resultados/pais/redondeo/corrida_duro_13108.log 2>&1 &
    # versión blanda (min sum |d|)
    nohup env_censo/bin/python scripts/correr_redondeo.py 13108 --tau 0.01 --L 3 --horas 24 \\
        > resultados/pais/redondeo/corrida_blando_13108.log 2>&1 &
    # con Gurobi (licencia en ~/gurobi.lic), barrera paralela en el LP raíz; si es infactible escribe un IIS
    nohup env_censo/bin/python scripts/correr_redondeo.py 13108 --duro --tau 0 --L 0 --horas 24 --solver gurobi \\
        > resultados/pais/redondeo/corrida_gurobi_duro_13108.log 2>&1 &
    # solo escribir el .mps (para resolverlo con Gurobi en otro computador)
    env_censo/bin/python scripts/correr_redondeo.py 13108 --duro --tau 0.01 --L 3 --solo-mps

Escribe en resultados/pais/redondeo/, con <id> = <unidad>_<cut>_<duro|blando>_t<tau>_L<L> (rd.nombre):
    Y_<id>.npz/.json   Y entera (dispersa) de la mejor solución y su resumen; en la versión blanda se reescribe cada
                       5 minutos si mejora, así que sirve aunque se corte
    highs_<id>.log     log de HiGHS (gurobi_<id>.log con --solver gurobi)
    iis_<id>.ilp       con --solver gurobi, si el problema es infactible: restricciones en conflicto (IIS)
    redondeo_<id>.mps  el modelo (con --mps o --solo-mps); variables z0, z1, ... en el orden de las celdas libres, que
                       se recalculan igual con los mismos tau y L
"""
import os
import sys

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # el script vive en scripts/; las rutas
sys.path.insert(0, RAIZ)                                             # (resultados/, manzanas/) son relativas
os.chdir(RAIZ)                                                       # a la raíz del repositorio

import argparse
import json
import time

import scipy.sparse as sp

from manzanas import redondeo as rd
from manzanas.pais import RUTA_PAIS, rutas_unidad
from manzanas.perfiles import cargar_modelo
from manzanas.unidades import cargar_unidad

RUTA = os.path.join(RUTA_PAIS, "redondeo")


def preparar(cut, unidad, tau, L, duro, objetivo):
    # Carga W*, mu, nu y arma el MIP con las mismas celdas libres que usa el notebook
    datos, inv = cargar_unidad(cut, unidad)
    modelo = cargar_modelo(os.path.join(rutas_unidad(unidad)["modelos"], f"modelo_{cut}.npz"))
    mu = modelo["mu_p"]
    nu = modelo["nu_m"] if modelo["nu_m"] is not None else datos.nu
    F, frac, _, _ = rd.partes(modelo["W"], mu, nu)
    libre = rd.celdas_libres_duras(F, frac, inv, modelo, tau, L) if duro else rd.celdas_libres(frac, tau, L)
    mip = rd.construir(modelo["W"], mu, nu, inv, modelo, libre=libre, duro=duro, objetivo=objetivo)
    return mip, datos, inv, modelo, mu, nu


def hora():
    return time.strftime("%H:%M:%S")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cut", type=int)
    ap.add_argument("--unidad", default="personas")
    ap.add_argument("--tau", type=float, default=0.01)
    ap.add_argument("--L", type=int, default=3)
    ap.add_argument("--horas", type=float, default=24.0)
    ap.add_argument("--duro", action="store_true")
    ap.add_argument("--objetivo", default="cero", choices=["cero", "cercania"])
    ap.add_argument("--mps", action="store_true", help="escribir también el .mps")
    ap.add_argument("--solo-mps", action="store_true", help="escribir el .mps y salir")
    ap.add_argument("--solver", default="highs", choices=["highs", "gurobi"])
    ap.add_argument("--hilos", type=int, default=0, help="hilos de Gurobi (0 = todos)")
    ap.add_argument("--sin-iis", action="store_true", help="con Gurobi, no calcular IIS si es infactible")
    a = ap.parse_args()

    os.makedirs(RUTA, exist_ok=True)
    modo = "duro" if a.duro else "blando"
    base = os.path.join(RUTA, rd.nombre("{}", a.unidad, a.cut, a.duro, a.tau, a.L))
    mip, datos, inv, modelo, mu, nu = preparar(a.cut, a.unidad, a.tau, a.L, a.duro, a.objetivo)
    print(f"[{hora()}] {a.cut} {a.unidad} {modo}: binarias {mip['n_z']:,}, restricciones de bloque {mip['B']:,}, "
          f"nnz {mip['A'].nnz:,}", flush=True)
    if a.mps or a.solo_mps:
        ruta_mps = base.format("redondeo") + ".mps"
        rd.escribir_mps(mip, ruta_mps)
        print(f"[{hora()}] escrito {ruta_mps} ({os.path.getsize(ruta_mps) / 1e6:,.0f} MB)", flush=True)
        if a.solo_mps:
            return

    ruta_Y = base.format("Y") + ".npz"
    meta = {"cut": a.cut, "unidad": a.unidad, "modo": modo, "tau": a.tau, "L": a.L, "objetivo_tipo": a.objetivo,
            "solver": a.solver}

    def guardar(z, obj):
        Y = rd.matriz_Y(mip, z)
        sp.save_npz(ruta_Y, sp.csr_matrix(Y))
        chequeo = rd.verificar(mip, Y, mu, nu, inv, modelo)
        with open(ruta_Y.replace(".npz", ".json"), "w") as f:
            json.dump({**meta, "objetivo": float(obj), "verificacion": chequeo,
                       "hora": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=1)
        print(f"[{hora()}] guardada solución: objetivo {obj:,.1f} | {chequeo}", flush=True)

    z0 = None
    if not a.duro:
        z0, obj0 = rd.solucion_inicial(mip)
        print(f"[{hora()}] solución inicial: sum|d| = {obj0:,.0f}", flush=True)
        guardar(z0, obj0)
    if a.solver == "gurobi":
        z, res = rd.resolver_gurobi(mip, z0=z0, horas=a.horas, log=base.format("gurobi") + ".log", guardar=guardar,
                                    hilos=a.hilos, ruta_iis=None if a.sin_iis else base.format("iis") + ".ilp")
    else:
        z, res = rd.resolver(mip, z0=z0, horas=a.horas, log=base.format("highs") + ".log",
                             guardar=None if a.duro else guardar)
    print(f"[{hora()}] fin: {res}", flush=True)
    if z is not None:
        meta.update(res)
        guardar(z, res["objetivo"])
    else:
        print(f"[{hora()}] sin solución factible ({res['estado']})", flush=True)


if __name__ == "__main__":
    main()
