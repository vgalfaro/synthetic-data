"""Hogares y viviendas: el mismo modelo de perfiles que personas, con su propia unidad y sus propios marginales.

Cada unidad tiene su universo y su total por manzana (el que hace de n_per):
    viviendas   viviendas particulares (tipo_operativo = 2), total n_vp. Incluye las desocupadas: los bloques
                ocupación y tipo de vivienda se publican sobre todas las viviendas particulares, y el resto de los
                bloques sobre las ocupadas con moradores presentes, así que las demás caen en su "resto"
    hogares     hogares en viviendas particulares (tipo_operativo = 2), total n_hog

Las columnas de hogar que el INE deriva de las personas (jefatura femenina, solo personas de 60+, al menos un menor
de 15) se calculan desde el microdato de personas con edad_quinquenal, que viene también en las comunas bajo umbral
(donde edad = -66). Los tramos quinquenales cortan justo en 15 y en 60 años, así que la definición es exacta.

Fuentes: diccionario_variables_glosas_censo2024_manzana_agregados.xlsx (columnas de manzana) y
diccionario_variables_censo2024_nivel_comuna.xlsx (tablas tabla_hogares y tabla_viviendas). Cada columna se verifica
con inventario.inventario_tabla: el total comunal del microdato tiene que coincidir con la suma de las manzanas.
"""
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .datos import cargar_comuna, ruta_comuna
from .inventario import construir_inventario, inventario_tabla

RUTA_COMUNAS = os.path.join("resultados", "pais", "comunas")
UNIDADES = ["personas", "hogares", "viviendas"]
TOTAL = {"personas": "n_per", "hogares": "n_hog", "viviendas": "n_vp"}  # columna de manzanas con el total de la unidad


def _categorias(prefijo, nombres, codigos=None):
    # [(prefijo + nombre, [código])] con códigos 1..n salvo que se indiquen
    codigos = codigos or range(1, len(nombres) + 1)
    return [(f"{prefijo}{n}", [c] if np.isscalar(c) else list(c)) for n, c in zip(nombres, codigos)]


# bloque -> (variable en la tabla de hogares, [(columna en manzanas, códigos de la variable)])
MAPEO_HOGARES = {
    "unipersonal":      ("tipologia_hogar", [("n_hog_unipersonales", [1])]),
    "solo_60":          ("solo_60", [("n_hog_60", [1])]),                # derivada de personas: todas con 60+
    "con_menores":      ("con_menores", [("n_hog_menores", [1])]),       # derivada de personas: alguna de 0 a 14
    "jefatura":         ("jefe_sexo", [("n_jefatura_mujer", [2])]),      # derivada de personas: sexo del jefe/a
    "allegado":         ("allegado", [("n_hog_allegados", [1])]),        # id_hogar >= 2 (hogar no principal)
    "tenencia":         ("p12_tenencia_viv", _categorias("n_tenencia_", [
                            "propia_pagada", "propia_pagandose", "arrendada_contrato", "arrendada_sin_contrato",
                            "cedida_trabajo", "cedida_familiar", "otro"], [1, 2, 3, 4, 5, 6, (7, 8, 9)])),
    "comb_cocina":      ("p13_comb_cocina", _categorias("n_comb_cocina_", [
                            "gas", "parafina", "lena", "pellet", "carbon", "electricidad", "solar", "no_utiliza"])),
    "comb_calefaccion": ("p14_comb_calefaccion", _categorias("n_comb_calefaccion_", [
                            "gas", "parafina", "lena", "pellet", "carbon", "electricidad", "otra", "no_utiliza"])),
    **{f"serv_{k}":     (f"p15{letra}_serv_{k}", [(f"n_serv_{k}", [1])])
       for letra, k in zip("abcdef", ["tel_movil", "compu", "tablet", "internet_fija", "internet_movil",
                                      "internet_satelital"])},
    "internet":         ("internet", [("n_internet", [1])]),             # internet fija, móvil o satelital
}

# bloque -> (variable en la tabla de viviendas, [(columna en manzanas, códigos de la variable)])
MAPEO_VIVIENDAS = {
    "ocupacion":    ("p3a_estado_ocupacion", [("n_vp_ocupada", [1]), ("n_vp_desocupada", [2])]),
    "tipo_viv":     ("p2_tipo_vivienda", _categorias("n_tipo_viv_", [
                        "casa", "depto", "indigena", "pieza", "mediagua", "movil", "otro"], [1, 3, 5, 6, 7, 8, 9])),
    "dormitorios":  ("p5_num_dormitorios", _categorias("n_dormitorios_", ["1", "2", "3", "4", "5", "6_o_mas"])),
    "hacinamiento": ("indice_hacinamiento", [("n_viv_hacinadas", [2, 3])]),  # hacinamiento medio o crítico
    "irrecuperable": ("irrecuperable", [("n_viv_irrecuperables", [1])]),     # derivada: ver tabla_viviendas
    "paredes":      ("p4a_mat_paredes", _categorias("n_mat_paredes_", [
                        "hormigon", "albanileria", "tabique_forrado", "tabique_sin_forro", "artesanal", "precarios"])),
    "techo":        ("p4b_mat_techo", _categorias("n_mat_techo_", [
                        "tejas", "hormigon", "zinc", "fibrocemento", "fonolita", "paja", "precarios", "sin_cubierta"])),
    "piso":         ("p4c_mat_piso", _categorias("n_mat_piso_", [
                        "radier_con_revestimiento", "radier_sin_revestimiento", "baldosa_cemento", "capa_cemento",
                        "tierra"])),
    "fuente_agua":  ("p6_fuente_agua", _categorias("n_fuente_agua_", ["publica", "pozo", "camion", "rio"])),
    "distrib_agua": ("p7_distrib_agua", _categorias("n_distrib_agua_", ["llave", "llave_fuera", "acarreo"])),
    "serv_hig":     ("p8_serv_hig", _categorias("n_serv_hig_", [
                        "alc_dentro", "alc_fuera", "fosa", "pozo", "acequia_canal", "cajon_otro", "bano_quimico",
                        "bano_seco", "no_tiene"])),
    "fuente_elect": ("p9_fuente_elect", _categorias("n_fuente_elect_", [
                        "publica", "diesel", "solar", "eolica", "otro", "no_tiene"])),
    "basura":       ("p10_basura", _categorias("n_basura_", ["servicios", "entierra", "eriazo", "rio", "otro"])),
}

MAPEOS = {"hogares": MAPEO_HOGARES, "viviendas": MAPEO_VIVIENDAS}

# Columnas de hogares y viviendas de la base de manzanas que no entran como restricción, y por qué
COLUMNAS_SIN_USAR = {
    "n_hog": "total de la unidad hogares (hace de n_per)",
    "n_vp": "total de la unidad viviendas (hace de n_per)",
    "prom_per_hog": "promedio, no conteo",
    "n_nucleos_hacinados_allegados": "déficit habitacional: requiere identificar núcleos dentro del hogar",
    "n_viv_no_ampliables": "déficit habitacional: ninguna regla sobre el microdato da los totales (ver 08, H0)",
    "n_deficit_cuantitativo": "suma de irrecuperables, hogares allegados, núcleos hacinados allegados y no ampliables",
}

# Variables por unidad que no tienen marginal por manzana pero sí un total publicado, para validar el modelo:
# unidad -> {nombre: (columna numérica en la tabla de la unidad, columna de manzanas con el total publicado)}
TOTALES_VALIDACION = {
    "viviendas": {"hogares": ("cant_hog", "n_hog"), "personas": ("cant_per", "n_per")},
    "hogares": {"personas": ("n_personas", "n_per")},
}


def tabla_hogares(hogares, personas):
    # Hogares en viviendas particulares, con las variables derivadas de sus personas y de sus servicios
    h = hogares[hogares["tipo_operativo"] == 2].reset_index(drop=True)
    p = personas.loc[personas["tipo_operativo"] == 2, ["id_vivienda", "id_hogar", "edad_quinquenal", "parentesco", "sexo"]]
    g = p.groupby(["id_vivienda", "id_hogar"])
    jefes = p[p["parentesco"] == 1]
    assert not jefes.duplicated(["id_vivienda", "id_hogar"]).any(), "hogar con más de un jefe/a"
    agg = pd.DataFrame({"n_personas": g.size(),
                        "solo_60": (g["edad_quinquenal"].min() >= 60).astype(int),
                        "con_menores": (g["edad_quinquenal"].min() <= 10).astype(int),  # tramos 0-4, 5-9 y 10-14
                        "jefe_sexo": jefes.set_index(["id_vivienda", "id_hogar"])["sexo"]})
    h = h.join(agg, on=["id_vivienda", "id_hogar"])
    assert h["n_personas"].notna().all() and h["jefe_sexo"].notna().all(), "hogar sin personas o sin jefe/a"
    h["internet"] = h[["p15d_serv_internet_fija", "p15e_serv_internet_movil",
                       "p15f_serv_internet_satelital"]].eq(1).any(axis=1).astype(int)
    h["allegado"] = (h["id_hogar"] >= 2).astype(int)
    h["n_personas_8"] = h["n_personas"].clip(upper=8)  # para validar (provincias.NIVEL_2_UNIDADES)
    return h


def tabla_viviendas(viviendas):
    # Viviendas particulares (ocupadas y desocupadas). cant_hog es NaN en las que no tienen moradores presentes -> 0.
    # Irrecuperable (déficit habitacional): ocupada con moradores presentes y de tipo mediagua, móvil u otro, o con
    # paredes de materiales precarios, techo precario o sin cubierta, o piso de tierra; las que tienen no respuesta (-99)
    # en paredes, techo o piso no se clasifican. Reproduce n_viv_irrecuperables en las 346 comunas (08, H0b)
    v = viviendas[viviendas["tipo_operativo"] == 2].reset_index(drop=True)
    v["cant_hog"] = v["cant_hog"].fillna(0)
    mat = v[["p4a_mat_paredes", "p4b_mat_techo", "p4c_mat_piso"]]
    precaria = (v["p2_tipo_vivienda"].isin([7, 8, 9]) | (v["p4a_mat_paredes"] == 6) | v["p4b_mat_techo"].isin([7, 8])
                | (v["p4c_mat_piso"] == 5))
    v["irrecuperable"] = ((v["p3b_estado_ocupacion"] == 1) & precaria & ~mat.eq(-99).any(axis=1)).astype(int)
    v["cant_per_8"] = v["cant_per"].clip(upper=8)  # para validar (provincias.NIVEL_2_UNIDADES)
    return v


@dataclass
class DatosUnidad:
    cut: int
    unidad: str               # "hogares" o "viviendas"
    tabla: pd.DataFrame       # una fila por unidad de la comuna (índice 0..N-1)
    manzanas: pd.DataFrame    # filas del CSV de manzanas con total de la unidad > 0
    manzanas_ids: np.ndarray  # MANZENT, mismo orden que las filas de `manzanas`
    nu: np.ndarray            # total de la unidad por manzana (n_hog o n_vp)

    @property
    def personas(self):
        # Alias para las funciones escritas para personas (provincias.evaluar_cruces usa datos.personas["comuna"])
        return self.tabla


def tabla_unidad(cut, unidad, ruta_resultados=RUTA_COMUNAS):
    # Tabla de hogares o de viviendas de la comuna, desde los parquet por comuna
    leer = lambda t: pd.read_parquet(ruta_comuna(t, cut, ruta_resultados))
    if unidad == "hogares":
        return tabla_hogares(leer("hogares"), leer("personas"))
    if unidad == "viviendas":
        return tabla_viviendas(leer("viviendas"))
    raise ValueError(unidad)


def cargar_unidad(cut, unidad, ruta_resultados=RUTA_COMUNAS):
    # Devuelve (datos, inventario) de la comuna para la unidad. Personas usa cargar_comuna y construir_inventario
    if unidad == "personas":
        datos = cargar_comuna(cut, ruta_resultados=ruta_resultados)
        return datos, construir_inventario(datos)
    tabla = tabla_unidad(cut, unidad, ruta_resultados)
    manzanas = pd.read_parquet(ruta_comuna("manzanas", cut, ruta_resultados))
    total = TOTAL[unidad]
    manzanas = manzanas[pd.to_numeric(manzanas[total], errors="coerce") > 0].reset_index(drop=True)
    datos = DatosUnidad(cut, unidad, tabla, manzanas, manzanas["MANZENT"].to_numpy(),
                        manzanas[total].to_numpy().astype(np.int64))
    return datos, inventario_tabla(tabla, datos, MAPEOS[unidad])


def predecir_total(modelo, valores):
    # Extensión de una variable numérica sin marginal: cada unidad hereda el reparto de su perfil.
    # total[m] = sum_p W[p, m] * (suma de `valores` en el perfil p) / mu_p
    suma_p = np.bincount(modelo["perfil_de_persona"], weights=np.asarray(valores, dtype=float),
                         minlength=len(modelo["mu_p"]))
    return (suma_p / modelo["mu_p"]) @ modelo["W"].astype(np.float64)
