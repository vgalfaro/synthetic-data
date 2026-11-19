"""Generalización a todo el país: preparación de las comunas, diagnóstico y procesamiento comuna por comuna.

Cada comuna es independiente: su conjunta comunal sale de su microdato y sus restricciones de sus manzanas.
Los resultados de personas quedan en resultados/pais/: comunas/ (parquet de entrada), modelos/ (W por comuna), cruces/,
logs/ (avance de cada ajuste) y checkpoints/ (W parcial mientras se ajusta, para retomar si se corta).
Los de hogares y viviendas (manzanas/unidades.py) quedan con la misma estructura en resultados/pais/<unidad>/.
"""
import os
import time

import numpy as np
import pandas as pd

from .datos import ARCHIVOS, RUTA_DATOS, cargar_comuna, ruta_comuna
from .inventario import construir_inventario
from .ipf import ipf_perfiles_gpu_rapido
from .perfiles import (_preparar, cargar_modelo, codigos_bloque, construir_perfiles, error_relativo_modelo,
                       errores_modelo, guardar_modelo, indicadora)
from .tablas import cruces_bivariados
from .unidades import TOTAL, cargar_unidad

RUTA_PAIS = os.path.join("resultados", "pais")
RUTA_COMUNAS = os.path.join(RUTA_PAIS, "comunas")
RUTA_MODELOS = os.path.join(RUTA_PAIS, "modelos")
RUTA_CRUCES = os.path.join(RUTA_PAIS, "cruces")
RUTA_LOGS = os.path.join(RUTA_PAIS, "logs")
RUTA_CHECKPOINTS = os.path.join(RUTA_PAIS, "checkpoints")


def rutas_unidad(unidad):
    # Carpetas de salida de la unidad: personas en resultados/pais/ (como siempre), el resto en resultados/pais/<unidad>/
    base = RUTA_PAIS if unidad == "personas" else os.path.join(RUTA_PAIS, unidad)
    return {k: os.path.join(base, k) for k in ("modelos", "cruces", "logs", "checkpoints")}


def preparar_pais(ruta_datos=RUTA_DATOS, ruta_comunas=RUTA_COMUNAS, tablas=tuple(ARCHIVOS)):
    # Lee cada CSV nacional una vez (tipos inferidos sobre el país completo) y escribe un parquet por comuna
    os.makedirs(ruta_comunas, exist_ok=True)
    for tabla in tablas:
        archivo, columna = ARCHIVOS[tabla]
        t0 = time.time()
        df = pd.read_csv(os.path.join(ruta_datos, archivo), sep=";", encoding="utf-8-sig", low_memory=False)
        n = 0
        for cut, g in df.groupby(columna, sort=True):
            g.reset_index(drop=True).to_parquet(ruta_comuna(tabla, cut, ruta_comunas), index=False)
            n += 1
        print(f"  {tabla}: {len(df):,} filas -> {n} comunas ({time.time() - t0:.0f}s)", flush=True)


def diagnosticar_comuna(cut, ruta_comunas=RUTA_COMUNAS):
    # Tamaño del problema y chequeos del inventario, sin ajustar nada
    datos = cargar_comuna(cut, ruta_resultados=ruta_comunas)
    fila = {"cut": cut, "comuna": datos.manzanas["COMUNA"].iloc[0], "personas": len(datos.personas),
            "manzanas": len(datos.nu), "suma_n_per": int(datos.nu.sum()),
            "bajo_umbral": int(datos.personas["comuna_bajo_umbral"].iloc[0]),
            "personas_con_-66": int((datos.personas == -66).any(axis=1).sum())}
    try:
        inv = construir_inventario(datos)
        cods = {b: codigos_bloque(inv, b)[0] for b in inv.marginales}
        _, perfiles, _, mu_p = construir_perfiles(cods)
        fila.update({"inventario": "ok", "perfiles": len(mu_p), "celdas_W": len(mu_p) * len(datos.nu),
                     "columnas_con_*": int((inv.tabla["enmascaradas"] > 0).sum())})
    except AssertionError as e:
        fila.update({"inventario": f"falla: {e}", "perfiles": np.nan, "celdas_W": np.nan, "columnas_con_*": np.nan})
    return fila


def diagnosticar_unidad(cut, unidad, ruta_comunas=RUTA_COMUNAS):
    # Como diagnosticar_comuna, para hogares o viviendas: tamaño del problema y chequeos del inventario
    fila = {"cut": cut, "unidad": unidad}
    try:
        datos, inv = cargar_unidad(cut, unidad, ruta_comunas)
        if len(datos.tabla) == 0:  # Antártica: no hay hogares ni viviendas particulares
            fila.update({"unidades": 0, "inventario": "sin unidades", "perfiles": np.nan, "celdas_W": np.nan})
            return fila
        cods = {b: codigos_bloque(inv, b)[0] for b in inv.marginales}
        _, perfiles, _, mu_p = construir_perfiles(cods)
        fila.update({"comuna": datos.manzanas["COMUNA"].iloc[0], "unidades": len(datos.tabla),
                     "manzanas": len(datos.nu), "suma_total": int(datos.nu.sum()),
                     "bajo_umbral": int(datos.tabla["comuna_bajo_umbral"].iloc[0]), "inventario": "ok",
                     "perfiles": len(mu_p), "celdas_W": len(mu_p) * len(datos.nu),
                     "columnas_con_*": int((inv.tabla["enmascaradas"] > 0).sum())})
    except (AssertionError, IndexError) as e:
        fila.update({"inventario": f"falla: {type(e).__name__} {e}", "perfiles": np.nan, "celdas_W": np.nan})
    return fila


def procesar_comuna(cut, tol=0.1, tol_rel=1e-4, n_iter=20000, reajustar=False, reportar=None, exportar=True,
                    unidad="personas"):
    # Ajusta todos los bloques de la unidad en GPU (versión rápida), guarda W y exporta los cruces bivariados.
    # unidad: "personas" (27 bloques), "hogares" o "viviendas" (manzanas/unidades.py).
    # Convergencia: error de cada celda <= max(tol, tol_rel * total de la manzana: n_per, n_hog o n_vp); tope de n_iter
    # iteraciones (si no converge, el modelo se guarda igual y queda marcado). El avance va a logs/ajuste_<cut>.log y
    # W se guarda en checkpoints/ cada 2.500 iteraciones: si el ajuste se corta, la siguiente llamada lo retoma desde
    # ahí. Si el modelo ya existe, se ajustó con el mismo criterio (tol, tol_rel) y los mismos bloques, y reajustar=False,
    # lo carga.
    # Devuelve un resumen con los chequeos (en personas, con los nombres de columna de siempre)
    rutas = rutas_unidad(unidad)
    for ruta in rutas.values():
        os.makedirs(ruta, exist_ok=True)
    datos, inv = cargar_unidad(cut, unidad, RUTA_COMUNAS)
    bloques = list(inv.marginales)
    ruta_modelo = os.path.join(rutas["modelos"], f"modelo_{cut}.npz")
    ruta_log = os.path.join(rutas["logs"], f"ajuste_{cut}.log")
    ruta_ckpt = os.path.join(rutas["checkpoints"], f"W_{cut}.npz")

    modelo = cargar_modelo(ruta_modelo) if os.path.exists(ruta_modelo) and not reajustar else None
    if modelo is not None and (modelo["tol"], modelo["tol_rel"]) == (tol, tol_rel) and modelo["bloques"] == bloques:
        ajustado = False
    else:
        nus, perfiles, perfil_de_persona, mu_p, _ = _preparar(inv, bloques, None)
        W0, it0 = None, 0
        if os.path.exists(ruta_ckpt) and not reajustar:  # retomar un ajuste cortado
            z = np.load(ruta_ckpt)
            W0, it0 = z["W"], int(z["it"])
            assert W0.shape == (len(mu_p), len(datos.nu))
        elif os.path.exists(ruta_log):
            os.remove(ruta_log)  # ajuste nuevo: log nuevo
        t0 = time.time()
        W, it, conv, _ = ipf_perfiles_gpu_rapido(mu_p, [perfiles[:, j] for j in range(len(bloques))], nus,
                                                 tol=tol, tol_rel=tol_rel, n_per=datos.nu, n_iter=n_iter, W0=W0,
                                                 it_inicial=it0, reportar=reportar, log=ruta_log, checkpoint=ruta_ckpt)
        modelo = {"bloques": bloques, "W": W, "perfiles": perfiles, "perfil_de_persona": perfil_de_persona,
                  "mu_p": mu_p, "nu_m": datos.nu, "iteraciones": it, "convergio": conv, "segundos": time.time() - t0,
                  "tol": tol, "tol_rel": tol_rel}
        guardar_modelo(modelo, ruta_modelo)
        if os.path.exists(ruta_ckpt):
            os.remove(ruta_ckpt)
        ajustado = True
    assert modelo["bloques"] == bloques

    # Chequeos: error marginal (absoluto y relativo a lo permitido), residuo de las manzanas enmascaradas
    # (en unidades y en % del total comunal de la categoría) y concentración de los perfiles de 1 unidad
    errores = errores_modelo(inv, modelo)
    W = modelo["W"].astype(np.float64, copy=False)
    dif_masc, dif_masc_pct = 0.0, 0.0
    for j, b in enumerate(bloques):
        cod, nu_k = codigos_bloque(inv, b)
        masc = np.isnan(nu_k[0])
        if masc.any():
            total = (cod == 0).sum()
            est = (indicadora(modelo["perfiles"][:, j], nu_k.shape[0]) @ W)[0, masc].sum()
            dif = abs(est - (total - np.nansum(nu_k[0])))
            dif_masc, dif_masc_pct = max(dif_masc, dif), max(dif_masc_pct, 100 * dif / max(total, 1))
    unicos = modelo["mu_p"] == 1
    reparto = W[unicos]
    resumen = {"cut": cut, "comuna": datos.manzanas["COMUNA"].iloc[0], "unidades": len(inv.per),
               "manzanas": len(datos.nu), "perfiles": len(modelo["mu_p"]), "celdas_W": modelo["W"].size,
               "iteraciones": modelo["iteraciones"], "convergio": modelo["convergio"], "ajustado_ahora": ajustado,
               "segundos_ajuste": round(modelo["segundos"], 1), "error_max": max(errores.values()),
               "error_rel_max": error_relativo_modelo(inv, modelo, tol, tol_rel),
               "dif_enmascaradas": dif_masc, "dif_enmascaradas_pct": dif_masc_pct,
               "perfiles_1_unidad": int(unicos.sum()),
               "pct_1_unidad_>50%_en_1_manzana": float((reparto.max(axis=1) > 0.5).mean() * 100) if unicos.any() else 0.0,
               "manzanas_efectivas_mediana": float(np.median(1 / (reparto ** 2).sum(axis=1))) if unicos.any() else np.nan}

    if exportar:
        t0 = time.time()
        cruces = cruces_bivariados(inv, modelo, datos.manzanas_ids)
        ruta_cruces = os.path.join(rutas["cruces"], f"cruces_{cut}.parquet")
        cruces.to_parquet(ruta_cruces, index=False)
        resumen.update({"filas_cruces": len(cruces), "MB_cruces": round(os.path.getsize(ruta_cruces) / 1e6, 2),
                        "segundos_export": round(time.time() - t0, 1)})
    resumen["MB_modelo"] = round(os.path.getsize(ruta_modelo) / 1e6, 1)
    if unidad == "personas":  # nombres de columna de resumen_pais.csv
        resumen = {{"unidades": "personas", "perfiles_1_unidad": "perfiles_1_persona",
                    "pct_1_unidad_>50%_en_1_manzana": "pct_1_persona_>50%_en_1_manzana"}.get(k, k): v
                   for k, v in resumen.items()}
    return resumen
