"""Redondeo de Cembrano et al. (configuraciones A y B) en todas las comunas, en paralelo según la memoria disponible.

Uso, desde la raíz del repo (sigue corriendo aunque se cierre la terminal):
    nohup setsid env_censo/bin/python scripts/correr_cembrano_pais.py --max-celdas 20e6 \\
        > resultados/pais/redondeo/cembrano_pais.log 2>&1 &

Cada comuna corre como un proceso aparte (correr_cembrano.py <cut> --hilos H), de la más chica a la más grande. Una
comuna se lanza solo si su memoria estimada (~3.2 GB por millón de celdas de W*, el pico medido en 13108) cabe en la
memoria disponible menos una reserva, descontando lo que aún pueden crecer las que están corriendo. Si no hay nada
corriendo y la siguiente igual no cabe, se registra como "no cabe en memoria" y se sigue con la próxima. Las comunas con
cembrano_personas_<cut>.json ya escrito se saltan, así que si se corta basta con volver a lanzarlo.

Escribe por comuna (en resultados/pais/redondeo/): log_cembrano/<cut>.log, vertice_*, Y_*_cembrano_{perfiles_exactos,
uniforme}.npz/.json y cembrano_personas_<cut>.json; y una fila por comuna en resumen_cembrano.csv.
"""
import os
import sys

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # el script vive en scripts/; las rutas
sys.path.insert(0, RAIZ)                                             # (resultados/, manzanas/) son relativas
os.chdir(RAIZ)                                                       # a la raíz del repositorio

import argparse
import json
import subprocess
import time

import pandas as pd

from manzanas.pais import RUTA_PAIS

RUTA = os.path.join(RUTA_PAIS, "redondeo")
RUTA_LOGS = os.path.join(RUTA, "log_cembrano")
GB_POR_MILLON = 3.2   # pico medido: 13108 ~3.1 GB por millón de celdas, comunas de ~3 M celdas ~2 GB por millón
COLUMNAS = ["cut", "comuna", "personas", "manzanas", "perfiles", "celdas_W", "estado", "aristas",
            "vertice_fraccionales", "segundos_vertice", "segundos_total"] + [
    f"{c}_{k}" for c in ("A", "B") for k in ("cumple", "dev_perfiles", "dev_manzanas", "dev_bloques",
                                             "error_bloque_max", "error_bloque_medio", "iteraciones")]


def memoria_disponible_gb():
    with open("/proc/meminfo") as f:
        for linea in f:
            if linea.startswith("MemAvailable"):
                return int(linea.split()[1]) / 1048576
    return 0.0


def rss_gb(pid):
    try:
        with open(f"/proc/{pid}/status") as f:
            for linea in f:
                if linea.startswith("VmRSS"):
                    return int(linea.split()[1]) / 1048576
    except FileNotFoundError:
        pass
    return 0.0


def fila_resumen(fila, ruta_json):
    out = {c: fila.get(c) for c in ("cut", "comuna", "personas", "manzanas", "perfiles", "celdas_W")}
    if not os.path.exists(ruta_json):
        out["estado"] = "error (ver log)"
        return out
    r = json.load(open(ruta_json))
    out.update({"estado": r.get("estado"), "aristas": r.get("aristas"),
                "vertice_fraccionales": r.get("vertice_fraccionales"),
                "segundos_vertice": (r.get("vertice") or {}).get("segundos"), "segundos_total": r.get("segundos_total")})
    for c, nombre in (("A", "perfiles_exactos"), ("B", "uniforme")):
        if nombre in r:
            d = r[nombre]["desviacion_max_por_parte"]
            out.update({f"{c}_cumple": r[nombre]["cumple_cotas_teorema"], f"{c}_dev_perfiles": d["perfiles"],
                        f"{c}_dev_manzanas": d["manzanas"],
                        f"{c}_dev_bloques": max(v for k, v in d.items() if k not in ("perfiles", "manzanas")),
                        f"{c}_error_bloque_max": r[nombre]["error_bloque_max"],
                        f"{c}_error_bloque_medio": r[nombre]["error_bloque_medio"],
                        f"{c}_iteraciones": r[nombre]["iteraciones"]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-celdas", type=float, default=20e6, help="solo comunas con celdas_W <= este valor")
    ap.add_argument("--max-paralelo", type=int, default=6)
    ap.add_argument("--hilos", type=int, default=8, help="hilos de Gurobi por comuna")
    ap.add_argument("--reserva-gb", type=float, default=15.0)
    a = ap.parse_args()
    os.makedirs(RUTA_LOGS, exist_ok=True)
    resumen = pd.read_csv(os.path.join(RUTA_PAIS, "resumen_pais.csv"))
    resumen = resumen[resumen["celdas_W"] <= a.max_celdas].sort_values("celdas_W")
    ruta_csv = os.path.join(RUTA, "resumen_cembrano.csv")
    hechas = set(pd.read_csv(ruta_csv)["cut"]) if os.path.exists(ruta_csv) else set()
    pendientes = [f for _, f in resumen.iterrows()
                  if f["cut"] not in hechas
                  and not os.path.exists(os.path.join(RUTA, f"cembrano_personas_{f['cut']}.json"))]
    print(f"[{time.strftime('%F %T')}] {len(pendientes)} comunas pendientes (celdas_W <= {a.max_celdas / 1e6:.0f} M), "
          f"{len(hechas)} ya en el resumen", flush=True)
    corriendo = {}   # pid -> (Popen, fila, memoria estimada, t0)
    pausa_hasta = 0.0   # sin licencia de Gurobi: no lanzar nada hasta esta hora

    def registrar(fila):
        fila_out = fila_resumen(fila, os.path.join(RUTA, f"cembrano_personas_{fila['cut']}.json"))
        pd.DataFrame([fila_out], columns=COLUMNAS).to_csv(ruta_csv, mode="a", header=not os.path.exists(ruta_csv),
                                                          index=False)
        return fila_out

    while pendientes or corriendo:
        for pid in list(corriendo):
            proc, fila, est, t0 = corriendo[pid]
            if proc.poll() is not None:
                del corriendo[pid]
                if proc.returncode == 75:   # SIN_LICENCIA: se reencola y se pausa
                    pendientes.insert(0, fila)
                    pausa_hasta = time.time() + 300
                    print(f"[{time.strftime('%F %T')}] {fila['cut']} {fila['comuna']}: sin licencia de Gurobi; "
                          f"reencolada, pausa de 5 min", flush=True)
                    continue
                out = registrar(fila)
                print(f"[{time.strftime('%F %T')}] fin {fila['cut']} {fila['comuna']} ({(time.time() - t0) / 60:.1f} min, "
                      f"rc {proc.returncode}): {out.get('estado')} | A: bloques máx {out.get('A_error_bloque_max')}, "
                      f"medio {out.get('A_error_bloque_medio')} | B: máx {out.get('B_error_bloque_max')}, "
                      f"medio {out.get('B_error_bloque_medio')}", flush=True)
        while pendientes and len(corriendo) < a.max_paralelo and time.time() >= pausa_hasta:
            fila = pendientes[0]
            est = GB_POR_MILLON * fila["celdas_W"] / 1e6 + 2
            crecimiento = sum(max(0.0, e - rss_gb(pid)) for pid, (_, _, e, _) in corriendo.items())
            if est > memoria_disponible_gb() - a.reserva_gb - crecimiento:
                if not corriendo:
                    pendientes.pop(0)
                    out = {c: fila.get(c) for c in ("cut", "comuna", "personas", "manzanas", "perfiles", "celdas_W")}
                    out["estado"] = f"no cabe en memoria (~{est:.0f} GB estimados)"
                    pd.DataFrame([out], columns=COLUMNAS).to_csv(ruta_csv, mode="a",
                                                                 header=not os.path.exists(ruta_csv), index=False)
                    print(f"[{time.strftime('%F %T')}] salto {fila['cut']} {fila['comuna']}: {out['estado']}",
                          flush=True)
                    continue
                break
            pendientes.pop(0)
            log = open(os.path.join(RUTA_LOGS, f"{fila['cut']}.log"), "w")
            proc = subprocess.Popen([sys.executable, os.path.join(RAIZ, "scripts", "correr_cembrano.py"), str(fila["cut"]), "--hilos", str(a.hilos)],
                                    stdout=log, stderr=subprocess.STDOUT)
            corriendo[proc.pid] = (proc, fila, est, time.time())
            print(f"[{time.strftime('%F %T')}] lanzo {fila['cut']} {fila['comuna']} ({fila['celdas_W'] / 1e6:.1f} M "
                  f"celdas, ~{est:.0f} GB); corriendo {len(corriendo)}, quedan {len(pendientes)}", flush=True)
        time.sleep(10)
    print(f"[{time.strftime('%F %T')}] listo", flush=True)


if __name__ == "__main__":
    main()
