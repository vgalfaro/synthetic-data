"""Cifras del informe (docs/informe/informe_modelo_27.tex) que se calculan recorriendo las comunas con inventario completo:
composición del "resto" de cada bloque, enmascaramiento por columna y unidades del contenedor comunal."""
import numpy as np
import pandas as pd

from .datos import cargar_comuna
from .inventario import MAPEO_PERSONAS, construir_inventario
from .pais import RUTA_COMUNAS


def _codigo_texto(x):
    # Normaliza el código de una persona para contarlo: 2.0 -> "2", NaN/None -> "NA"
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "NA"
    try:
        f = float(x)
        return str(int(f)) if f == int(f) else str(f)
    except (TypeError, ValueError):
        return str(x)


def estadisticas_comunas(cuts):
    # Recorre las comunas y devuelve:
    #   columnas    una fila por columna n_*: total comunal (microdato), suma de manzanas visibles, manzanas con '*',
    #               comunas con '*' y residuo (total - visible), sumados sobre las comunas
    #   resto       personas por (bloque, código) que no caen en ninguna categoría publicada del bloque
    #   contenedor  una fila por comuna: unidades CONTENEDOR_COMUNAL = 1 con población y sus personas
    tablas, resto, contenedor = [], {}, []
    for cut in cuts:
        datos = cargar_comuna(cut, ruta_resultados=RUTA_COMUNAS)
        inv = construir_inventario(datos)
        tablas.append(inv.tabla.assign(cut=cut))
        for bloque, (var, categorias) in MAPEO_PERSONAS.items():
            x = inv.per[var]
            publicado = np.zeros(len(x), bool)
            for _, codigos in categorias:
                publicado |= x.isin(codigos).to_numpy()
            for codigo, n in x[~publicado].map(_codigo_texto).value_counts().items():
                resto[(bloque, codigo)] = resto.get((bloque, codigo), 0) + int(n)
        c = datos.manzanas[datos.manzanas["CONTENEDOR_COMUNAL"] == 1]
        contenedor.append({"cut": cut, "unidades": len(c), "personas": int(c["n_per"].sum()),
                           "ids": c["MANZENT"].tolist()})
    t = pd.concat(tablas)
    columnas = t.groupby(["bloque", "columna"], sort=False).agg(
        total=("total comuna", "sum"), visible=("suma manzanas", "sum"), manzanas_con_asterisco=("enmascaradas", "sum"),
        comunas_con_asterisco=("enmascaradas", lambda s: int((s > 0).sum())), residuo=("residuo", "sum"))
    resto = pd.Series(resto, name="personas").rename_axis(["bloque", "codigo"]).reset_index()
    return columnas, resto, pd.DataFrame(contenedor).set_index("cut")
