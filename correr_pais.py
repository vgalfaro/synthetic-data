"""Corrida nacional: procesa todas las comunas con inventario completo (ver 05_pais.ipynb, E5).

Uso, desde la raíz del repo (sigue corriendo aunque se cierre la terminal):
    nohup env_censo/bin/python correr_pais.py > resultados/pais/logs/corrida_pais.log 2>&1 &

Va de la comuna más chica a la más grande. Después de cada comuna agrega una fila a resultados/pais/resumen_pais.csv;
las comunas que ya están ahí se saltan, así que si se corta basta con volver a lanzarlo. Si una comuna falla,
se registra el error y se sigue con la siguiente.
"""
import os
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


def liberar_gpu():
    try:
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
    except ImportError:
        pass


def main():
    diag = pd.read_csv(os.path.join(RUTA_PAIS, "diagnostico.csv"), index_col=0)
    pendientes = diag[diag["inventario"] == "ok"].sort_values("celdas_W")
    ruta_resumen = os.path.join(RUTA_PAIS, "resumen_pais.csv")
    hechas = set(pd.read_csv(ruta_resumen)["cut"]) if os.path.exists(ruta_resumen) else set()
    pendientes = pendientes[~pendientes.index.isin(hechas)]
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} inicio | pid {os.getpid()} | {len(pendientes)} comunas pendientes, "
          f"{len(hechas)} ya hechas | tol={TOL}, tol_rel={TOL_REL}, tope={N_ITER}", flush=True)
    t_inicio = time.time()
    for i, (cut, fila) in enumerate(pendientes.iterrows(), 1):
        t0 = time.time()
        try:
            r = procesar_comuna(int(cut), tol=TOL, tol_rel=TOL_REL, n_iter=N_ITER)
            r["error"] = ""
        except Exception as e:  # se registra y se sigue con la siguiente comuna
            traceback.print_exc()
            r = {"cut": int(cut), "comuna": fila["comuna"], "error": f"{type(e).__name__}: {e}"}
        r["segundos_total"] = round(time.time() - t0, 1)
        pd.DataFrame([r]).reindex(columns=COLUMNAS).to_csv(ruta_resumen, mode="a", index=False,
                                                           header=not os.path.exists(ruta_resumen))
        liberar_gpu()
        estado = (f"ERROR {r['error']}" if r["error"] else
                  f"it={r['iteraciones']} convergió={r['convergio']} error={r['error_max']:.3f}")
        print(f"{time.strftime('%H:%M:%S')} [{i}/{len(pendientes)}] {cut} {fila['comuna']} "
              f"({fila['celdas_W'] / 1e6:.1f} M celdas): {estado} | {r['segundos_total']:.0f}s "
              f"| acumulado {(time.time() - t_inicio) / 3600:.2f} h", flush=True)
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} fin | {(time.time() - t_inicio) / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
