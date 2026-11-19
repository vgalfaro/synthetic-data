"""Planificador de los trabajos de redondeo en el cluster (SLURM sin control de memoria: CR_CORE).

Cada minuto:
  - revisa qué trabajos nuestros terminaron (squeue) y si dejaron su archivo de salida;
  - calcula, por nodo, memoria disponible = FreeMem real - reserva - lo que nuestros trabajos en curso aún pueden
    pedir (se descuenta su estimación completa: conservador) y núcleos libres;
  - lanza, de mayor a menor memoria, los trabajos pendientes que caben, fijados al nodo con -w.
Trabajos: comunas enteras (cluster/enteras.txt), división de las gigantes, sus grupos (se agregan al terminar la
división) y el ensamblado de cada gigante (cuando están todos sus grupos).
Estado en cluster/estado.json; log en cluster/planificador.log. Se puede matar y relanzar: retoma desde las salidas.
"""
import json
import os
import subprocess
import time

import numpy as np

BASE = os.path.expanduser("~/censo_redondeo")
R = os.path.join(BASE, "resultados/pais/redondeo")
GB_POR_MILLON = 3.2
RESERVA_GB = 24
NODOS = ["n7", "n9", "n1", "n2", "n4", "n3"]          # n11 lleno, n14 chico
PARTICION = {"n7": "gpus", "n9": "gpus"}
MAX_ACTIVOS = 22                                      # cortesía: el cluster es compartido
LOG = os.path.join(BASE, "cluster/planificador.log")


def log(msg):
    with open(LOG, "a") as f:
        f.write(f"[{time.strftime('%F %T')}] {msg}\n")


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout


def nodos():
    out = {}
    for linea in sh('sinfo -N -h -o "%N %C %e"').split("\n"):
        p = linea.split()
        if len(p) == 3 and p[0] in NODOS:
            a, i, o, t = p[1].split("/")
            out[p[0]] = {"cpus": int(i), "mem": int(p[2]) / 1024}   # GB libres reales
    return out


def corriendo():
    ids = set()
    for linea in sh('squeue -u $USER -h -o "%i"').split():
        ids.add(linea.strip())
    return ids


def uso_gb(job):
    # memoria residente promedio del paso batch (sstat AveRSS): ya está descontada en FreeMem del nodo
    out = sh(f"sstat -n -j {job}.batch --format=AveRSS 2>/dev/null").strip()
    try:
        v, u = float(out[:-1]), out[-1].upper()
        return v * {"K": 1 / 1024 ** 2, "M": 1 / 1024, "G": 1.0, "T": 1024.0}[u]
    except (ValueError, KeyError, IndexError):
        return 0.0


def celdas_W():
    import csv
    return {int(r["cut"]): float(r["celdas_W"]) for r in csv.DictReader(open(os.path.join(BASE, "resultados/pais/resumen_pais.csv")))}


def salida(t):
    if t["tipo"] == "comuna":
        return os.path.join(R, f"cembrano_personas_{t['cut']}.json")
    if t["tipo"] == "dividir":
        return os.path.join(R, f"grandes/{t['cut']}/division_25M.ok")
    if t["tipo"] == "grupo":
        return os.path.join(R, f"grandes/{t['cut']}/grupo_{t['g']}.json")
    return os.path.join(R, f"cembrano_personas_{t['cut']}.json")


def exito(t):
    if t["tipo"] == "dividir":   # el trabajo imprime "fin dividir <cut> rc <código>" al final de su log
        import glob
        logs = glob.glob(os.path.join(BASE, f"logs/*_{t['job']}.out"))
        return bool(logs) and f"fin dividir {t['cut']} rc 0" in open(logs[0]).read()
    return os.path.exists(salida(t))


def comando(t):
    if t["tipo"] == "comuna":
        return f"job_comuna.sh {t['cut']}"
    if t["tipo"] == "dividir":
        return f"job_dividir.sh {t['cut']}"
    if t["tipo"] == "grupo":
        return f"job_grupo.sh {t['cut']} {t['g']}"
    return f"job_ensamblar.sh {t['cut']}"


def grupos_de(cut):
    z = np.load(os.path.join(R, f"grandes/{cut}/division.npz"))
    gm, mu_g = z["grupo_de_manzana"], z["mu_g"]
    return [(g, int((mu_g[:, g] > 0).sum()) * int((gm == g).sum())) for g in range(gm.max() + 1)]


def inicial():
    cw = celdas_W()
    tareas = []
    for l in open(os.path.join(BASE, "cluster/enteras.txt")):
        cut = int(l.split()[0])
        tareas.append({"tipo": "comuna", "cut": cut, "mem": GB_POR_MILLON * cw[cut] / 1e6 + 10, "cpus": 4})
    for l in open(os.path.join(BASE, "cluster/gigantes.txt")):
        cut = int(l.split()[0])
        tareas.append({"tipo": "dividir", "cut": cut, "mem": 0.15 * cw[cut] / 1e6 + 20, "cpus": 2})
    for t in tareas:
        t.update(estado="pendiente", job=None, nodo=None, intentos=0)
    return tareas


def main():
    ruta_estado = os.path.join(BASE, "cluster/estado.json")
    tareas = json.load(open(ruta_estado)) if os.path.exists(ruta_estado) else inicial()
    cw = celdas_W()
    for t in tareas:   # 4 hilos por trabajo: el crossover (más de la mitad del tiempo) usa casi un solo hilo, y
        if t["estado"] == "pendiente" and t["tipo"] in ("comuna", "grupo"):   # n7/n9 se llenan por núcleos
            t["cpus"] = 4
    log(f"inicio: {len(tareas)} tareas")
    while True:
        activos = corriendo()
        # 1. terminados
        for t in tareas:
            if t["estado"] == "corriendo" and t["job"] not in activos:
                if exito(t):
                    t["estado"] = "listo"
                    log(f"listo {t['tipo']} {t['cut']} {t.get('g', '')} (job {t['job']}, {t['nodo']})")
                    if t["tipo"] == "dividir":
                        for g, celdas in grupos_de(t["cut"]):
                            tareas.append({"tipo": "grupo", "cut": t["cut"], "g": g, "cpus": 4,
                                           "mem": GB_POR_MILLON * celdas / 1e6 + 10, "estado": "pendiente",
                                           "job": None, "nodo": None, "intentos": 0})
                        log(f"  {t['cut']}: {len(grupos_de(t['cut']))} grupos agregados")
                else:
                    t["intentos"] += 1
                    t["estado"] = "pendiente" if t["intentos"] < 3 else "fallido"
                    log(f"sin salida {t['tipo']} {t['cut']} {t.get('g', '')} (job {t['job']}): "
                        f"{'reintento' if t['estado'] == 'pendiente' else 'FALLIDO'}")
        # ensamblados: gigantes con todos sus grupos listos
        for cut in {t["cut"] for t in tareas if t["tipo"] == "grupo"}:
            gs = [t for t in tareas if t["tipo"] == "grupo" and t["cut"] == cut]
            if all(t["estado"] == "listo" for t in gs) and not any(
                    t["tipo"] == "ensamblar" and t["cut"] == cut for t in tareas):
                tareas.append({"tipo": "ensamblar", "cut": cut, "cpus": 2, "mem": 0.15 * cw[cut] / 1e6 + 20,
                               "estado": "pendiente", "job": None, "nodo": None, "intentos": 0})
                log(f"{cut}: todos los grupos listos, ensamblado en cola")
        # 2. capacidad por nodo
        cap = nodos()
        for t in tareas:
            if t["estado"] == "corriendo" and t["nodo"] in cap:   # lo que aún puede crecer cada trabajo nuestro
                cap[t["nodo"]]["mem"] -= max(t["mem"] - uso_gb(t["job"]), 0.0)
        n_activos = sum(t["estado"] == "corriendo" for t in tareas)
        # 3. lanzar: primero divisiones y ensamblados (baratos), luego lo más grande
        orden = sorted([t for t in tareas if t["estado"] == "pendiente"],
                       key=lambda t: (t["tipo"] not in ("dividir", "ensamblar"), -t["mem"]))
        for t in orden:
            if n_activos >= MAX_ACTIVOS:
                break
            for nodo in sorted(cap, key=lambda n: -cap[n]["mem"]):
                c = cap[nodo]
                # n7/n9 tienen mucha memoria y pocos núcleos: si no hay núcleos para 8 hilos, se lanza con 4
                cpus = t["cpus"] if c["cpus"] >= t["cpus"] or t["tipo"] not in ("comuna", "grupo") else 4
                if c["mem"] - RESERVA_GB >= t["mem"] and c["cpus"] >= cpus:
                    nombre = f"{t['tipo'][:3]}_{t['cut']}" + (f"_{t['g']}" if t["tipo"] == "grupo" else "")
                    out = sh(f"cd {BASE} && sbatch --parsable -p {PARTICION.get(nodo, '512x1024')} -w {nodo} "
                             f"-c {cpus} --mem {int(t['mem']) + 1}G -J {nombre} {comando(t)}").strip()
                    if out.isdigit():
                        t.update(estado="corriendo", job=out, nodo=nodo)
                        c["mem"] -= t["mem"]
                        c["cpus"] -= cpus
                        n_activos += 1
                        log(f"lanzo {nombre} en {nodo} (job {out}, ~{t['mem']:.0f} GB, {cpus} cpus)")
                    else:
                        log(f"sbatch falló para {nombre}: {out}")
                    break
        json.dump(tareas, open(ruta_estado + ".tmp", "w"), indent=0)
        os.replace(ruta_estado + ".tmp", ruta_estado)
        if all(t["estado"] in ("listo", "fallido") for t in tareas):
            log("todo terminado")
            break
        time.sleep(60)


if __name__ == "__main__":
    main()
