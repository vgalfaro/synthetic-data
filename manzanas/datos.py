"""Carga de datos por comuna: los CSV nacionales se filtran una vez y se guardan en parquet en resultados/."""
import io
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

RUTA_DATOS = "databases"
RUTA_RESULTADOS = "resultados"
RUTA_CARTOGRAFIA = os.path.join("cartografia_censal", "Cartografia_censo2024_Pais_Manzanas.parquet")

# tabla: (archivo en databases/, columna con el código de comuna)
ARCHIVOS = {
    "personas": ("personas_censo2024.csv", "comuna"),
    "hogares": ("hogares_censo2024.csv", "comuna"),
    "viviendas": ("viviendas_censo2024.csv", "comuna"),
    "manzanas": ("Base_manzana_entidad_CPV24.csv", "CUT"),
}


def ruta_comuna(tabla, cut, ruta_resultados=RUTA_RESULTADOS):
    return os.path.join(ruta_resultados, f"{tabla}_{cut}.parquet")


def _filtrar_csv(ruta, columna, valor):
    # Lee el CSV línea a línea y conserva solo las filas de la comuna; pandas infiere los tipos sobre ese subconjunto
    with open(ruta, encoding="utf-8-sig") as f:
        encabezado = f.readline()
        idx = encabezado.rstrip("\n").split(";").index(columna)
        valor = str(valor)
        filas = [encabezado] + [linea for linea in f if linea.split(";", idx + 1)[idx] == valor]
    return pd.read_csv(io.StringIO("".join(filas)), sep=";")


def preparar_comuna(cut, ruta_datos=RUTA_DATOS, ruta_resultados=RUTA_RESULTADOS, tablas=tuple(ARCHIVOS)):
    # Filtra los CSV nacionales a una comuna y guarda cada tabla en parquet (~1 min, sobre todo por personas)
    os.makedirs(ruta_resultados, exist_ok=True)
    for tabla in tablas:
        archivo, columna = ARCHIVOS[tabla]
        df = _filtrar_csv(os.path.join(ruta_datos, archivo), columna, cut)
        df.to_parquet(ruta_comuna(tabla, cut, ruta_resultados), index=False)
        print(f"  {tabla}: {len(df):,} filas -> {ruta_comuna(tabla, cut, ruta_resultados)}")


@dataclass
class DatosComuna:
    cut: int
    personas: pd.DataFrame    # microdato de personas de la comuna, en el orden del CSV (índice 0..N-1)
    manzanas: pd.DataFrame    # filas del CSV de manzanas con n_per > 0
    manzanas_ids: np.ndarray  # MANZENT, mismo orden que las filas de `manzanas`
    nu: np.ndarray            # n_per por manzana


def cargar_comuna(cut, ruta_datos=RUTA_DATOS, ruta_resultados=RUTA_RESULTADOS):
    # Carga la comuna desde parquet; si no existe, la prepara desde los CSV nacionales
    if not all(os.path.exists(ruta_comuna(t, cut, ruta_resultados)) for t in ("personas", "manzanas")):
        print(f"preparando comuna {cut} desde los CSV nacionales...")
        preparar_comuna(cut, ruta_datos, ruta_resultados)
    personas = pd.read_parquet(ruta_comuna("personas", cut, ruta_resultados))
    manzanas = pd.read_parquet(ruta_comuna("manzanas", cut, ruta_resultados))
    manzanas = manzanas[manzanas["n_per"] > 0].reset_index(drop=True)
    return DatosComuna(cut, personas, manzanas, manzanas["MANZENT"].to_numpy(), manzanas["n_per"].to_numpy())


def leer_marginal(datos, col):
    # Conteo publicado por manzana; '*' (secreto estadístico) -> NaN
    return pd.to_numeric(datos.manzanas[col].astype(str), errors="coerce").to_numpy()


def cargar_mapa(cut, ruta_resultados=RUTA_RESULTADOS):
    # Geometría de las manzanas de la comuna (cartografía oficial); se cachea en resultados/
    import geopandas as gpd

    ruta = ruta_comuna("mapa", cut, ruta_resultados)
    if not os.path.exists(ruta):
        mapa = gpd.read_parquet(RUTA_CARTOGRAFIA)
        mapa = mapa[mapa["CUT"] == cut].copy()
        mapa["MANZENT"] = mapa["MANZENT"].astype(np.int64)
        mapa.to_parquet(ruta)
    return gpd.read_parquet(ruta)
