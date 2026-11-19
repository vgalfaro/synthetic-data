"""Validación con verdad conocida: la provincia como unidad grande y sus comunas como "manzanas".

A nivel de comuna el microdato trae la verdad de cualquier cruce. Se repite el experimento un nivel más arriba:
la conjunta de la provincia es conocida y de cada comuna solo se usan los mismos 82 marginales univariados que el INE
publica por manzana (calculados desde el microdato). Se ajusta el mismo modelo de 27 bloques y se comparan los
cruces estimados por comuna contra los reales, junto con tres alternativas (lo mismo para hogares y viviendas con
datos_provincia_unidad y ajustar_provincia(..., unidad=)):
    independencia   el cruce sale solo de los dos marginales de la comuna: fila x columna / n
    provincial      la estructura del cruce en la provincia, escalada al tamaño de la comuna (ignora sus marginales)
    raking_2        el cruce provincial ajustado (IPF) a los dos marginales de la comuna: usa la provincia y los dos
                    marginales del par, pero no los otros 25 bloques
"""
import os
import time

import numpy as np
import pandas as pd

from .datos import DatosComuna, ruta_comuna
from .inventario import MAPEO_PERSONAS, construir_inventario, etiquetas_bloque, inventario_tabla
from .ipf import ipf_perfiles_gpu_rapido
from .pais import RUTA_COMUNAS
from .perfiles import _preparar, cargar_modelo, codigos_bloque, errores_modelo, guardar_modelo, indicadora
from .unidades import MAPEOS, TOTAL, DatosUnidad, tabla_unidad

RUTA_VALIDACION = os.path.join("resultados", "validacion")
METODOS = ["modelo", "raking_2", "independencia", "provincial"]

# Variables sin marginal por manzana que se validan con la extensión por perfil:
# nombre -> (columna del microdato, bloque relacionado para la alternativa "condicional en 1 bloque")
NIVEL_2 = {
    "país de nacimiento": ("p25_lug_nacimiento_esp", "inmigrante"),
    "pueblo específico": ("p28_pueblo_pert", "pueblo"),
    "años de escolaridad": ("escolaridad", "cine"),
    "parentesco": ("parentesco", "edad"),
}
# Lo mismo para hogares y viviendas: variables de la unidad sin marginal por manzana
NIVEL_2_UNIDADES = {
    "hogares": {
        "tipología del hogar": ("tipologia_hogar", "unipersonal"),
        "personas por hogar (tope 8)": ("n_personas_8", "unipersonal"),
    },
    "viviendas": {
        "estado de ocupación detallado": ("p3b_estado_ocupacion", "ocupacion"),
        "hacinamiento (3 niveles)": ("indice_hacinamiento", "hacinamiento"),
        "personas por vivienda (tope 8)": ("cant_per_8", "dormitorios"),
        "hogares por vivienda": ("cant_hog", "ocupacion"),
    },
}
_COLS_PERSONA = sorted({v for v, _ in MAPEO_PERSONAS.values()} - {"p37_alfabet_15"}
                       | {"p37_alfabet", "comuna", "provincia"} | {c for c, _ in NIVEL_2.values()})


def comunas_por_provincia(cuts):
    # provincia -> lista ordenada de comunas (entre `cuts`), leyendo la provincia del microdato de cada comuna
    prov = {c: int(pd.read_parquet(ruta_comuna("personas", c, RUTA_COMUNAS), columns=["provincia"])["provincia"].iloc[0])
            for c in cuts}
    out = {}
    for c, p in sorted(prov.items()):
        out.setdefault(p, []).append(c)
    return out


def datos_provincia(prov, comunas, nombres):
    # Une el microdato de las comunas de la provincia y arma, para cada comuna, los mismos 82 marginales que el INE
    # publica por manzana, calculados desde el microdato (exactos, sin '*'). Devuelve un DatosComuna donde las
    # "manzanas" son las comunas
    personas = pd.concat([pd.read_parquet(ruta_comuna("personas", c, RUTA_COMUNAS), columns=_COLS_PERSONA) for c in comunas],
                         ignore_index=True)
    alfabet_15 = personas["p37_alfabet"].where(personas["edad"] >= 15)
    grupos = personas["comuna"]
    filas = {"MANZENT": comunas, "COMUNA": [nombres[c] for c in comunas],
             "n_per": grupos.value_counts().reindex(comunas).to_numpy()}
    for var, categorias in MAPEO_PERSONAS.values():
        serie = alfabet_15 if var == "p37_alfabet_15" else personas[var]
        for col, codigos in categorias:
            filas[col] = serie.isin(codigos).groupby(grupos).sum().reindex(comunas, fill_value=0).to_numpy()
    manzanas = pd.DataFrame(filas)
    return DatosComuna(prov, personas, manzanas, np.asarray(comunas), manzanas["n_per"].to_numpy())


def datos_provincia_unidad(prov, comunas, nombres, unidad):
    # Como datos_provincia, para hogares o viviendas: une las tablas de la unidad de las comunas de la provincia y arma
    # para cada comuna los marginales de manzana de la unidad (exactos). Las comunas sin unidades se omiten
    tablas = [tabla_unidad(c, unidad, RUTA_COMUNAS) for c in comunas]
    comunas = [c for c, t in zip(comunas, tablas) if len(t)]
    tabla = pd.concat([t for t in tablas if len(t)], ignore_index=True)
    grupos = tabla["comuna"]
    filas = {"MANZENT": comunas, "COMUNA": [nombres[c] for c in comunas],
             TOTAL[unidad]: grupos.value_counts().reindex(comunas).to_numpy()}
    for var, categorias in MAPEOS[unidad].values():
        for col, codigos in categorias:
            filas[col] = tabla[var].isin(codigos).groupby(grupos).sum().reindex(comunas, fill_value=0).to_numpy()
    manzanas = pd.DataFrame(filas)
    return DatosUnidad(prov, unidad, tabla, manzanas, np.asarray(comunas), manzanas[TOTAL[unidad]].to_numpy())


def ajustar_provincia(datos, tol=0.1, tol_rel=1e-4, n_iter=20000, reajustar=False, unidad="personas"):
    # Mismo ajuste y mismo criterio que la corrida nacional (todos los bloques de la unidad, IPF rápido en GPU), con
    # las comunas como manzanas. Los modelos de hogares y viviendas van a resultados/validacion/<unidad>/modelos/
    carpeta = os.path.join(RUTA_VALIDACION, "modelos") if unidad == "personas" else os.path.join(RUTA_VALIDACION, unidad, "modelos")
    os.makedirs(carpeta, exist_ok=True)
    inv = construir_inventario(datos) if unidad == "personas" else inventario_tabla(datos.tabla, datos, MAPEOS[unidad])
    bloques = list(inv.marginales)
    ruta = os.path.join(carpeta, f"modelo_prov_{datos.cut}.npz")
    modelo = cargar_modelo(ruta) if os.path.exists(ruta) and not reajustar else None
    if modelo is None or (modelo["tol"], modelo["tol_rel"]) != (tol, tol_rel) or modelo["bloques"] != bloques:
        nus, perfiles, perfil_de_persona, mu_p, _ = _preparar(inv, bloques, None)
        t0 = time.time()
        W, it, conv, _ = ipf_perfiles_gpu_rapido(mu_p, [perfiles[:, j] for j in range(len(bloques))], nus, tol=tol,
                                                 tol_rel=tol_rel, n_per=datos.nu, n_iter=n_iter)
        modelo = {"bloques": bloques, "W": W, "perfiles": perfiles, "perfil_de_persona": perfil_de_persona,
                  "mu_p": mu_p, "nu_m": datos.nu, "iteraciones": it, "convergio": conv, "segundos": time.time() - t0,
                  "tol": tol, "tol_rel": tol_rel}
        guardar_modelo(modelo, ruta)
    return inv, modelo


def raking_2d(semilla, filas, cols, n_iter=1000, tol=1e-8):
    # IPF de una tabla semilla (Ca, Cb) a los marginales de cada comuna: filas (Ca, M) y columnas (Cb, M)
    T = np.repeat(semilla[:, :, None], filas.shape[1], axis=2).astype(float)
    for _ in range(n_iter):
        s = T.sum(axis=1)
        T *= np.divide(filas, s, out=np.ones_like(s), where=s > 0)[:, None, :]
        s = T.sum(axis=0)
        T *= np.divide(cols, s, out=np.ones_like(s), where=s > 0)[None, :, :]
        if np.abs(T.sum(axis=1) - filas).max() < tol:
            break
    return T


def _alternativas(R, n):
    # R: cruce real (Ca, Cb, M). Devuelve las tres alternativas, que solo usan sus bordes y la provincia
    filas, cols = R.sum(axis=1), R.sum(axis=0)
    prov = R.sum(axis=2) / R.sum()
    return {"independencia": filas[:, None, :] * cols[None, :, :] / n,
            "provincial": prov[:, :, None] * n,
            "raking_2": raking_2d(prov, filas, cols)}


def evaluar_cruces(datos, inv, modelo):
    # Para cada par de bloques y cada comuna: error absoluto sumado sobre las celdas del cruce, por método.
    # Personas mal ubicadas (%) = sum |estimado - real| / (2 n): los bordes son exactos, así que mide solo el interior
    bloques, P = modelo["bloques"], modelo["perfiles"]
    W = modelo["W"].astype(np.float64)
    cods = {b: codigos_bloque(inv, b)[0] for b in bloques}
    C = {b: len(etiquetas_bloque(inv, b)) for b in bloques}
    M, n = len(datos.nu), datos.nu.astype(float)
    idx = pd.Index(datos.manzanas_ids).get_indexer(datos.personas["comuna"])
    partes = []
    for ia in range(len(bloques)):
        for ib in range(ia + 1, len(bloques)):
            a, b = bloques[ia], bloques[ib]
            Ca, Cb = C[a], C[b]
            R = np.bincount((cods[a] * Cb + cods[b]) * M + idx, minlength=Ca * Cb * M).reshape(Ca, Cb, M).astype(float)
            est = {"modelo": (indicadora(P[:, ia] * Cb + P[:, ib], Ca * Cb) @ W).reshape(Ca, Cb, M), **_alternativas(R, n)}
            fila = {"var1": a, "var2": b, "comuna": datos.manzanas_ids, "n": n}
            for metodo in METODOS:
                fila[f"err_{metodo}"] = np.abs(est[metodo] - R).sum(axis=(0, 1))
            partes.append(pd.DataFrame(fila))
    return pd.concat(partes, ignore_index=True)


def celda_por_comuna(datos, inv, modelo, var1, cat1, var2, cat2):
    # Una celda de un cruce, por comuna: real (microdato), modelo y las tres alternativas
    j1, j2 = modelo["bloques"].index(var1), modelo["bloques"].index(var2)
    e1, e2 = etiquetas_bloque(inv, var1), etiquetas_bloque(inv, var2)
    i1, i2, Ca, Cb = e1.index(cat1), e2.index(cat2), len(e1), len(e2)
    M, n = len(datos.nu), datos.nu.astype(float)
    idx = pd.Index(datos.manzanas_ids).get_indexer(datos.personas["comuna"])
    R = np.bincount((codigos_bloque(inv, var1)[0] * Cb + codigos_bloque(inv, var2)[0]) * M + idx,
                    minlength=Ca * Cb * M).reshape(Ca, Cb, M).astype(float)
    P, W = modelo["perfiles"], modelo["W"].astype(np.float64)
    est = (indicadora(P[:, j1] * Cb + P[:, j2], Ca * Cb) @ W).reshape(Ca, Cb, M)
    alt = _alternativas(R, n)
    return pd.DataFrame({"comuna": datos.manzanas_ids, "nombre": datos.manzanas["COMUNA"].to_numpy(), "n_per": n,
                         "real": R[i1, i2], "modelo": est[i1, i2],
                         **{k: v[i1, i2] for k, v in alt.items()}})


def _codigo_nivel_2(personas, nombre, col):
    x = personas[col]
    if nombre == "país de nacimiento":  # los nacidos en Chile traen el código de su comuna de nacimiento
        x = x.where(x < 1000, 1)  # 1 = Chile; el resto son códigos de país; -99 no responde
    return pd.factorize(x.fillna(0))[0]  # NaN (no aplica / no pertenece) = una categoría más


def evaluar_nivel_2(datos, inv, modelo, variables=None):
    # Variables sin marginal por manzana: cada unidad hereda el reparto de su perfil (extensión del modelo) vs
    #   proporcional        n de la comuna x distribución provincial
    #   condicional_1       solo el bloque relacionado: sum_c nu[c, comuna] x P_provincia(y | c)
    # variables: nombre -> (columna, bloque relacionado); por omisión NIVEL_2 (personas)
    variables = NIVEL_2 if variables is None else variables
    M, n = len(datos.nu), datos.nu.astype(float)
    idx = pd.Index(datos.manzanas_ids).get_indexer(datos.personas["comuna"])
    reparto = modelo["W"].astype(np.float64) / modelo["mu_p"][:, None]
    filas = []
    for nombre, (col, bloque) in variables.items():
        y = _codigo_nivel_2(datos.personas, nombre, col)
        V = y.max() + 1
        real = np.bincount(y * M + idx, minlength=V * M).reshape(V, M).astype(float)
        modelo_est = indicadora(y, V) @ indicadora(modelo["perfil_de_persona"], len(modelo["mu_p"])).T @ reparto
        cod, _ = codigos_bloque(inv, bloque)
        Cb = cod.max() + 1
        nu_b = np.bincount(cod * M + idx, minlength=Cb * M).reshape(Cb, M)        # marginal del bloque por comuna
        conj = np.bincount(y * Cb + cod, minlength=V * Cb).reshape(V, Cb)         # y x bloque en la provincia
        cond = (conj / np.maximum(conj.sum(axis=0, keepdims=True), 1)) @ nu_b
        prop = real.sum(axis=1, keepdims=True) / real.sum() * n[None, :]
        for metodo, est in [("modelo", modelo_est), ("condicional_1", cond), ("proporcional", prop)]:
            filas.append(pd.DataFrame({"variable": nombre, "categorias": V, "metodo": metodo,
                                       "comuna": datos.manzanas_ids, "n": n,
                                       "err": np.abs(est - real).sum(axis=0)}))
    return pd.concat(filas, ignore_index=True)
