"""Trae del cluster los resultados de Cembrano de las comunas grandes y agrega sus filas a resumen_cembrano.csv.

Uso: python scripts/sincronizar_cluster.py            (una vez)
     python scripts/sincronizar_cluster.py --cada 30  (cada 30 minutos, hasta que estén todas)
Copia (rsync, sin borrar nada aquí) los json/npz de resultados/pais/redondeo y las carpetas grandes/<cut> del cluster,
y agrega al resumen las comunas cuyo cembrano_personas_<cut>.json llegó y aún no tienen fila.
"""
import os
import sys

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # el script vive en scripts/; las rutas
sys.path.insert(0, RAIZ)                                             # (resultados/, manzanas/) son relativas
os.chdir(RAIZ)                                                       # a la raíz del repositorio

import argparse
import subprocess
import time

import pandas as pd

from correr_cembrano_pais import COLUMNAS, fila_resumen

RUTA = "resultados/pais/redondeo"
REMOTO = "cluster:censo_redondeo/resultados/pais/redondeo/"


def sincronizar():
    rc = subprocess.call(["rsync", "-a", "--include=*/", "--include=cembrano_personas_*.json",
                          "--include=Y_personas_*_cembrano_*", "--include=vertice_personas_*.npz",
                          "--include=gurobi_vertice_personas_*.log", "--include=log_cembrano/*",
                          "--include=grandes/**", "--exclude=*", REMOTO, RUTA + "/"])
    if rc != 0:
        print(f"[{time.strftime('%F %T')}] rsync falló (rc {rc})", flush=True)
        return None
    resumen = pd.read_csv("resultados/pais/resumen_pais.csv").set_index("cut")
    ruta_csv = os.path.join(RUTA, "resumen_cembrano.csv")
    hechas = set(pd.read_csv(ruta_csv)["cut"])
    grandes = resumen[resumen["celdas_W"] > 45e6].index
    nuevas = []
    for cut in grandes:
        ruta_json = os.path.join(RUTA, f"cembrano_personas_{cut}.json")
        if cut not in hechas and os.path.exists(ruta_json):
            fila = resumen.loc[cut].to_dict()
            fila["cut"] = cut
            nuevas.append(fila_resumen(fila, ruta_json))
    if nuevas:
        pd.DataFrame(nuevas, columns=COLUMNAS).to_csv(ruta_csv, mode="a", header=False, index=False)
        for f in nuevas:
            print(f"[{time.strftime('%F %T')}] agregada {f['cut']} {f['comuna']}: {f.get('estado')} | A máx "
                  f"{f.get('A_error_bloque_max')}, medio {f.get('A_error_bloque_medio')} | B máx "
                  f"{f.get('B_error_bloque_max')}, medio {f.get('B_error_bloque_medio')}", flush=True)
    faltan = len([c for c in grandes if c not in hechas]) - len(nuevas)
    print(f"[{time.strftime('%F %T')}] sincronizado: {len(nuevas)} nuevas, faltan {faltan} grandes", flush=True)
    return faltan


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cada", type=float, default=0, help="minutos entre sincronizaciones (0 = una vez)")
    a = ap.parse_args()
    while True:
        faltan = sincronizar()
        if not a.cada or faltan == 0:
            break
        time.sleep(a.cada * 60)
