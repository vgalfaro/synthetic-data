"""Corrida nacional: procesa todas las comunas con inventario completo (ver 05_pais.ipynb, E5, y 08_hogares_viviendas.ipynb).

Uso, desde la raíz del repo (sigue corriendo aunque se cierre la terminal):
    nohup env_censo/bin/python scripts/correr_pais.py > resultados/pais/logs/corrida_pais.log 2>&1 &
    nohup env_censo/bin/python scripts/correr_pais.py viviendas > resultados/pais/viviendas/logs/corrida.log 2>&1 &
    nohup env_censo/bin/python scripts/correr_pais.py hogares > resultados/pais/hogares/logs/corrida.log 2>&1 &

El argumento es la unidad (personas si se omite). Lee el diagnóstico de la unidad (personas: resultados/pais/diagnostico.csv,
escrito por 05_pais.ipynb; hogares y viviendas: resultados/pais/<unidad>/diagnostico.csv, escrito por
08_hogares_viviendas.ipynb). Va de la comuna más chica a la más grande. Después de cada comuna agrega una fila al resumen
(personas: resultados/pais/resumen_pais.csv; si no, resultados/pais/<unidad>/resumen.csv); las comunas que ya están ahí
se saltan, así que si se corta basta con volver a lanzarlo. Si una comuna falla, se registra el error y se sigue con la
siguiente.
"""
import os
import sys

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # el script vive en scripts/; las rutas
sys.path.insert(0, RAIZ)                                             # (resultados/, manzanas/) son relativas
os.chdir(RAIZ)                                                       # a la raíz del repositorio

import time
import traceback

import pandas as pd

from manzanas.pais import RUTA_PAIS, procesar_comuna

TOL, TOL_REL, N_ITER = 0.1, 1e-4, 20000  # mismo criterio que el piloto de 05_pais.ipynb
COLUMNAS = ["cut", "comuna", "personas", "manzanas", "perfiles", "celdas_W", "iteraciones", "convergio",
            "ajustado_ahora", "segundos_ajuste", "error_max", "error_rel_max", "dif_enmascaradas",
            "dif_enmascaradas_pct", "perfiles_1_persona", "pct_1_persona_>50%_en_1_manzana",
            "manzanas_efectivas_mediana", "filas_cruces", "MB_cruces", "segundos_export", "MB_modelo",
            "segundos_total", "error"]
COLUMNAS_UNIDAD = [{"personas": "unidades", "perfiles_1_persona": "perfiles_1_unidad",
                    "pct_1_persona_>50%_en_1_manzana": "pct_1_unidad_>50%_en_1_manzana"}.get(c, c) for c in COLUMNAS]


def liberar_gpu():
    try:
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
    except ImportError:
        pass


def main(unidad="personas"):
    if unidad == "personas":
        base, ruta_resumen, columnas = RUTA_PAIS, os.path.join(RUTA_PAIS, "resumen_pais.csv"), COLUMNAS
    else:
        base = os.path.join(RUTA_PAIS, unidad)
        ruta_resumen, columnas = os.path.join(base, "resumen.csv"), COLUMNAS_UNIDAD
    diag = pd.read_csv(os.path.join(base, "diagnostico.csv"), index_col=0)
    pendientes = diag[diag["inventario"] == "ok"].sort_values("celdas_W")
    hechas = set(pd.read_csv(ruta_resumen)["cut"]) if os.path.exists(ruta_resumen) else set()
    pendientes = pendientes[~pendientes.index.isin(hechas)]
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} inicio {unidad} | pid {os.getpid()} | {len(pendientes)} comunas pendientes, "
          f"{len(hechas)} ya hechas | tol={TOL}, tol_rel={TOL_REL}, tope={N_ITER}", flush=True)
    t_inicio = time.time()
    for i, (cut, fila) in enumerate(pendientes.iterrows(), 1):
        t0 = time.time()
        try:
            r = procesar_comuna(int(cut), tol=TOL, tol_rel=TOL_REL, n_iter=N_ITER, unidad=unidad)
            r["error"] = ""
        except Exception as e:  # se registra y se sigue con la siguiente comuna
            traceback.print_exc()
            r = {"cut": int(cut), "comuna": fila["comuna"], "error": f"{type(e).__name__}: {e}"}
        r["segundos_total"] = round(time.time() - t0, 1)
        pd.DataFrame([r]).reindex(columns=columnas).to_csv(ruta_resumen, mode="a", index=False,
                                                           header=not os.path.exists(ruta_resumen))
        liberar_gpu()
        estado = (f"ERROR {r['error']}" if r["error"] else
                  f"it={r['iteraciones']} convergió={r['convergio']} error={r['error_max']:.3f}")
        print(f"{time.strftime('%H:%M:%S')} [{i}/{len(pendientes)}] {cut} {fila['comuna']} "
              f"({fila['celdas_W'] / 1e6:.1f} M celdas): {estado} | {r['segundos_total']:.0f}s "
              f"| acumulado {(time.time() - t_inicio) / 3600:.2f} h", flush=True)
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} fin | {(time.time() - t_inicio) / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:2])
