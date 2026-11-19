"""Cembrano et al. (configuraciones A y B) en comunas grandes, dividiéndolas en grupos de manzanas.

Uso, desde la raíz del repo (sigue corriendo aunque se cierre la terminal):
    nohup setsid env_censo/bin/python scripts/correr_cembrano_grandes.py --min-celdas 45e6 \\
        > resultados/pais/redondeo/cembrano_grandes.log 2>&1 &

Por comuna (archivos en resultados/pais/redondeo/grandes/<cut>/):
    dividir    grupos de manzanas contiguas con <= --max-celdas celdas no nulas; reparto de mu_p entre grupos con el
               muestreo de transporte (filas y columnas exactas) -> division.npz
    grupo g    subproblema del grupo g (filas mu_pg, columnas nu_m, bloques de sus manzanas): vértice del LP blando
               (min sum|d|, desempate por cercanía a W*) y Algoritmo 1 con A y B -> grupo_<g>.npz/.json
    ensamblar  Y completa por configuración, verificación y errores por bloque -> mismos archivos que una comuna normal
               (Y_personas_<cut>_cembrano_<config>.npz/.json, cembrano_personas_<cut>.json) y fila en
               resumen_cembrano.csv
El planificador corre los grupos en paralelo si la memoria alcanza. Lo ya hecho se salta, así que se puede relanzar.
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

import numpy as np
import pandas as pd
import scipy.sparse as sp

from correr_cembrano_pais import COLUMNAS, GB_POR_MILLON, fila_resumen, memoria_disponible_gb, rss_gb
from manzanas import redondeo as rd
from manzanas.pais import RUTA_PAIS, rutas_unidad
from manzanas.perfiles import cargar_modelo
from manzanas.unidades import cargar_unidad

RUTA = os.path.join(RUTA_PAIS, "redondeo")
U_A = 54   # perfiles 0, resto 54 (29 partes: 1/2 + 28/56 = 1)
U_B = 27   # uniforme d - 2


def hora():
    return time.strftime("%F %T")


def carpeta(cut):
    c = os.path.join(RUTA, "grandes", str(cut))
    os.makedirs(c, exist_ok=True)
    return c


def cargar(cut, unidad="personas"):
    datos, inv = cargar_unidad(cut, unidad)
    modelo = cargar_modelo(os.path.join(rutas_unidad(unidad)["modelos"], f"modelo_{cut}.npz"))
    nu = modelo["nu_m"] if modelo["nu_m"] is not None else datos.nu
    return datos, inv, modelo, modelo["mu_p"], np.asarray(nu)


def dividir(cut, max_celdas):
    datos, inv, modelo, mu, nu = cargar(cut)
    Wa, _ = rd.ajustar_margenes(modelo["W"], mu, nu)
    grupos = rd.particion_manzanas(Wa, max_celdas)
    mu_g = rd.repartir_perfiles(Wa, mu, nu, grupos)
    np.savez_compressed(os.path.join(carpeta(cut), "division.npz"), mu_g=mu_g,
                        grupo_de_manzana=np.concatenate([np.full(len(g), i) for i, g in enumerate(grupos)]))
    celdas = [int((Wa[:, g] > 0).sum()) for g in grupos]
    print(f"[{hora()}] {cut}: {len(grupos)} grupos, celdas no nulas por grupo {min(celdas):,}-{max(celdas):,}",
          flush=True)


def leer_division(cut):
    z = np.load(os.path.join(carpeta(cut), "division.npz"))
    gm = z["grupo_de_manzana"]
    return z["mu_g"], [np.nonzero(gm == i)[0] for i in range(gm.max() + 1)]


def grupo(cut, g, hilos):
    t0 = time.time()
    datos, inv, modelo, mu, nu = cargar(cut)
    mu_g, grupos = leer_division(cut)
    cols = grupos[g]
    filas = np.nonzero(mu_g[:, g] > 0)[0]
    W_g = np.asarray(modelo["W"][np.ix_(filas, cols)], dtype=np.float64)
    mu_sub, nu_sub = mu_g[filas, g], nu[cols]
    Wa_g, _ = rd.ajustar_margenes(W_g, mu_sub, nu_sub)
    inv_g = rd.subinventario(inv, cols)
    modelo_g = {"bloques": modelo["bloques"], "perfiles": modelo["perfiles"][filas]}
    F, frac, _, _ = rd.partes(Wa_g, mu_sub, nu_sub)
    mip = rd.construir(Wa_g, mu_sub, nu_sub, inv_g, modelo_g, libre=frac > rd.EPS, duro=False)
    print(f"[{hora()}] {cut} grupo {g}: {len(filas):,} perfiles x {len(cols)} manzanas, {mip['n_z']:,} aristas",
          flush=True)
    ruta_v = os.path.join(carpeta(cut), f"grupo_{g}_vertice.npz")   # se guarda apenas termina el LP (si después se
    if os.path.exists(ruta_v):                                        # cae la licencia, se reutiliza al reintentar)
        v = np.load(ruta_v, allow_pickle=True)
        z, d_lp, info = v["z"], v["d_lp"], dict(v["info"].item(), reutilizado=True)
        assert len(z) == mip["n_z"], (len(z), mip["n_z"])
    else:
        z, d_lp, info = rd.vertice_lp_blando(mip, log=os.path.join(carpeta(cut), f"gurobi_grupo_{g}.log"), hilos=hilos)
        if z is None:
            raise RuntimeError(f"LP blando del grupo {g} sin solución: {info}")
        tmp = ruta_v[:-4] + "_tmp.npz"
        np.savez(tmp, z=z, d_lp=np.asarray(d_lp if d_lp is not None else []), info=np.array(info, dtype=object))
        os.replace(tmp, ruta_v)
    print(f"[{hora()}] {cut} grupo {g}: vértice {info}", flush=True)
    vista = {**mip, "A": mip["A"][:, :mip["n_z"]], "duro": True,
             "costo": 1 - 2 * np.asarray(mip["frac_libre"], dtype=np.float64)}
    partes = list(dict.fromkeys(rd.partes_filas(vista)))
    resumen = {"cut": cut, "grupo": g, "perfiles": len(filas), "manzanas": len(cols), "aristas": mip["n_z"],
               "vertice": info}
    salida = {"filas": filas, "cols": cols}
    for nombre, u in (("perfiles_exactos", {**{p: U_A for p in partes}, "perfiles": 0}),
                      ("uniforme", {p: U_B for p in partes})):
        zc, hist = rd.redondeo_iterativo(vista, z, u, reportar=None, estricto=False)
        Y_g = rd.matriz_Y(vista, zc)
        salida[nombre] = sp.csr_matrix(Y_g)
        resumen[nombre] = {"iteraciones": len(hist), "sin_progreso": any(h.get("sin_progreso") for h in hist),
                           "desviacion_max_por_parte": rd.desviaciones_por_parte(vista, zc, z)}
    for nombre in ("perfiles_exactos", "uniforme"):
        sp.save_npz(os.path.join(carpeta(cut), f"grupo_{g}_{nombre}.npz"), salida[nombre])
    np.savez(os.path.join(carpeta(cut), f"grupo_{g}_indices.npz"), filas=filas, cols=cols)
    resumen["segundos"] = time.time() - t0
    with open(os.path.join(carpeta(cut), f"grupo_{g}.json"), "w") as f:
        json.dump(resumen, f, indent=1, default=str)
    print(f"[{hora()}] {cut} grupo {g}: listo en {resumen['segundos'] / 60:.1f} min", flush=True)


def ensamblar(cut):
    t0 = time.time()
    datos, inv, modelo, mu, nu = cargar(cut)
    mu_g, grupos = leer_division(cut)
    P, M = modelo["W"].shape
    resumen_comuna = {"cut": cut, "unidad": "personas", "perfiles": P, "manzanas": M, "grupos": len(grupos),
                      "metodo": "division_en_grupos"}
    jsons = [json.load(open(os.path.join(carpeta(cut), f"grupo_{g}.json"))) for g in range(len(grupos))]
    resumen_comuna["aristas"] = int(sum(j["aristas"] for j in jsons))
    resumen_comuna["vertice_fraccionales"] = int(sum(j["vertice"]["fraccionales"] for j in jsons))
    resumen_comuna["vertice"] = {"segundos": float(sum(j["vertice"]["segundos"] for j in jsons)),
                                 "desviacion_lp": float(sum(j["vertice"]["desviacion_lp"] for j in jsons))}
    F = np.floor(rd.ajustar_margenes(modelo["W"], mu, nu)[0])   # para verificar cotas
    for nombre, u_bloques in (("perfiles_exactos", U_A), ("uniforme", U_B)):
        Y = np.zeros((P, M), dtype=np.int64)
        for g in range(len(grupos)):
            idx = np.load(os.path.join(carpeta(cut), f"grupo_{g}_indices.npz"))
            Y[np.ix_(idx["filas"], idx["cols"])] = sp.load_npz(
                os.path.join(carpeta(cut), f"grupo_{g}_{nombre}.npz")).toarray()
        eb = rd.errores_bloque(inv, modelo, Y)
        dev_m = np.abs(Y.sum(axis=0) - nu).max()
        dev_p = np.abs(Y.sum(axis=1) - mu).max()
        dev_b = max(v["max"] for v in eb.values())
        u = {"perfiles": 0 if nombre == "perfiles_exactos" else U_B, "manzanas": u_bloques, "bloques": u_bloques}
        resumen = {"config": nombre, "u": u, "metodo": "division_en_grupos", "grupos": len(grupos),
                   "cumple_cotas_teorema": bool(dev_p <= u["perfiles"] and dev_m <= u["manzanas"]
                                                and dev_b <= u["bloques"]),
                   "desviacion_max_por_parte": {"perfiles": float(dev_p), "manzanas": float(dev_m),
                                                **{b: v["max"] for b, v in eb.items()}},
                   "verificacion": {"filas = mu_p": bool(dev_p == 0), "columnas = nu_m": bool(dev_m == 0),
                                    "floor <= Y <= floor + 1": bool(((Y >= F) & (Y <= F + 1)).all())},
                   "error_bloque_max": dev_b, "error_bloque_medio": float(np.mean([v["medio"] for v in eb.values()])),
                   "errores_bloque": eb, "iteraciones": int(max(j[nombre]["iteraciones"] for j in jsons)),
                   "grupos_sin_progreso": int(sum(j[nombre]["sin_progreso"] for j in jsons)), "segundos": None}
        base = os.path.join(RUTA, f"Y_personas_{cut}_cembrano_{nombre}")
        sp.save_npz(base + ".npz", sp.csr_matrix(Y))
        with open(base + ".json", "w") as f:
            json.dump(resumen, f, indent=1, default=str)
        resumen_comuna[nombre] = {k: resumen[k] for k in ("iteraciones", "segundos", "cumple_cotas_teorema",
                                                          "desviacion_max_por_parte", "verificacion",
                                                          "error_bloque_max", "error_bloque_medio")}
        print(f"[{hora()}] {cut} {nombre}: perfiles dev {dev_p:.0f}, manzanas dev {dev_m:.0f}, bloques máx {dev_b:.0f}, "
              f"medio {resumen['error_bloque_medio']:.3f}", flush=True)
    resumen_comuna.update(estado="ok", segundos_total=float(sum(j["segundos"] for j in jsons)) + time.time() - t0)
    with open(os.path.join(RUTA, f"cembrano_personas_{cut}.json"), "w") as f:
        json.dump(resumen_comuna, f, indent=1, default=str)


def planificar(a):
    resumen = pd.read_csv(os.path.join(RUTA_PAIS, "resumen_pais.csv"))
    resumen = resumen[resumen["celdas_W"] > a.min_celdas].sort_values("celdas_W")
    ruta_csv = os.path.join(RUTA, "resumen_cembrano.csv")
    hechas = set(pd.read_csv(ruta_csv)["cut"]) if os.path.exists(ruta_csv) else set()
    comunas = [f for _, f in resumen.iterrows() if f["cut"] not in hechas]
    print(f"[{hora()}] {len(comunas)} comunas grandes pendientes (celdas_W > {a.min_celdas / 1e6:.0f} M)", flush=True)
    corriendo = {}   # pid -> (Popen, cut, g, estimado)
    estado = {"pausa_hasta": 0.0, "fallas_licencia": 0}
    por_cut = {int(f["cut"]): f for f in comunas}
    ensambladas = set()

    def intentar_ensamblar(cut):
        if cut in ensambladas or not os.path.exists(os.path.join(carpeta(cut), "division.npz")):
            return
        _, grupos = leer_division(cut)
        if not all(os.path.exists(os.path.join(carpeta(cut), f"grupo_{g}.json")) for g in range(len(grupos))):
            return
        if any(c == cut for _, c, _, _ in corriendo.values()):
            return
        ensambladas.add(cut)
        f = por_cut[cut]
        rc = subprocess.call([sys.executable, __file__, "ensamblar", str(cut)],
                             stdout=open(os.path.join(carpeta(cut), "log.txt"), "a"), stderr=subprocess.STDOUT)
        fila = fila_resumen(f, os.path.join(RUTA, f"cembrano_personas_{cut}.json"))
        if rc != 0:
            fila["estado"] = "error al ensamblar"
        pd.DataFrame([fila], columns=COLUMNAS).to_csv(ruta_csv, mode="a", header=not os.path.exists(ruta_csv),
                                                      index=False)
        print(f"[{hora()}] ensamblada {cut} {f['comuna']}: {fila.get('estado')} | A: bloques máx "
              f"{fila.get('A_error_bloque_max')}, medio {fila.get('A_error_bloque_medio')} | B: máx "
              f"{fila.get('B_error_bloque_max')}, medio {fila.get('B_error_bloque_medio')}", flush=True)

    def lanzar(args, log, est, cut, g):
        proc = subprocess.Popen([sys.executable, __file__] + args, stdout=open(log, "a"), stderr=subprocess.STDOUT)
        corriendo[proc.pid] = (proc, cut, g, est)

    def esperar_espacio(est):
        while True:
            for pid in list(corriendo):
                proc, cut, g, _ = corriendo[pid]
                if proc.poll() is not None:
                    del corriendo[pid]
                    print(f"[{hora()}] fin {cut} grupo {g} (rc {proc.returncode})", flush=True)
                    if proc.returncode == 75:   # sin licencia: pausa; el grupo se reintenta en la próxima pasada
                        estado["pausa_hasta"] = time.time() + 300
                        estado["fallas_licencia"] += 1
                    intentar_ensamblar(cut)
            crec = sum(max(0.0, e - rss_gb(pid)) for pid, (_, _, _, e) in corriendo.items())
            libre = memoria_disponible_gb() - a.reserva_gb - crec
            if (len(corriendo) < a.max_paralelo and (est <= libre or not corriendo)
                    and time.time() >= estado["pausa_hasta"]):
                return
            time.sleep(10)

    while True:
        estado["fallas_licencia"] = 0
        for f in comunas:
            cut = int(f["cut"])
            log = os.path.join(carpeta(cut), "log.txt")
            if not os.path.exists(os.path.join(carpeta(cut), "division.npz")):
                esperar_espacio(GB_POR_MILLON * f["celdas_W"] / 1e6 * 0.6)
                print(f"[{hora()}] divido {cut} {f['comuna']} ({f['celdas_W'] / 1e6:.0f} M celdas)", flush=True)
                rc = subprocess.call([sys.executable, __file__, "dividir", str(cut), "--max-celdas", str(a.max_celdas)],
                                     stdout=open(log, "a"), stderr=subprocess.STDOUT)
                if rc != 0:
                    print(f"[{hora()}] error al dividir {cut} (ver {log})", flush=True)
                    continue
            _, grupos = leer_division(cut)
            for g in range(len(grupos)):
                if os.path.exists(os.path.join(carpeta(cut), f"grupo_{g}.json")):
                    continue
                est = GB_POR_MILLON * a.max_celdas / 1e6 + 6 + 3.2 * f["celdas_W"] / 1e9
                esperar_espacio(est)
                print(f"[{hora()}] lanzo {cut} {f['comuna']} grupo {g + 1}/{len(grupos)}; corriendo {len(corriendo) + 1}",
                      flush=True)
                lanzar(["grupo", str(cut), str(g), "--hilos", str(a.hilos)], log, est, cut, g)
        while corriendo:
            esperar_espacio(1e9)
        if estado["fallas_licencia"] == 0:
            break
        print(f"[{hora()}] hubo grupos sin licencia; nueva pasada en 5 min", flush=True)
        time.sleep(300)
    # esperar lo que queda y ensamblar las comunas completas (por si alguna quedó lista sin ensamblar)
    while corriendo:
        esperar_espacio(1e9)
    for cut in por_cut:
        intentar_ensamblar(cut)
    print(f"[{hora()}] listo", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("dividir"); p.add_argument("cut", type=int); p.add_argument("--max-celdas", type=float, default=12e6)
    p = sub.add_parser("grupo"); p.add_argument("cut", type=int); p.add_argument("g", type=int)
    p.add_argument("--hilos", type=int, default=8)
    p = sub.add_parser("ensamblar"); p.add_argument("cut", type=int)
    ap.add_argument("--min-celdas", type=float, default=45e6)
    ap.add_argument("--max-celdas", type=float, default=12e6, help="celdas no nulas por grupo")
    ap.add_argument("--max-paralelo", type=int, default=4)
    ap.add_argument("--hilos", type=int, default=12)
    ap.add_argument("--reserva-gb", type=float, default=10.0)
    a = ap.parse_args()
    if a.cmd == "dividir":
        dividir(a.cut, a.max_celdas)
    elif a.cmd == "grupo":
        from correr_cembrano import SIN_LICENCIA, es_falla_de_licencia
        try:
            grupo(a.cut, a.g, a.hilos)
        except Exception as e:
            if es_falla_de_licencia(e):
                print(f"[{hora()}] licencia de Gurobi no disponible: {e}", flush=True)
                raise SystemExit(SIN_LICENCIA)
            raise
    elif a.cmd == "ensamblar":
        ensamblar(a.cut)
    else:
        planificar(a)


if __name__ == "__main__":
    main()
