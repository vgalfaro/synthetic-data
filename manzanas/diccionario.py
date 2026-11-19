"""Lectura de los diccionarios del INE (xlsx) sin openpyxl: el xlsx es un zip con hojas en XML.

    glosas_manzana()       una fila por columna de Base_manzana_entidad_CPV24.csv: temática, variable, descripción, universo
    codigos_microdato(t)   una fila por (variable, código) de la tabla t del microdato: "personas", "hogares" o "viviendas"
"""
import os
import re
import zipfile
import xml.etree.ElementTree as ET

import pandas as pd

from .datos import RUTA_DATOS

_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
       "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships"}
DICC_MANZANA = "diccionario_variables_glosas_censo2024_manzana_agregados.xlsx"
DICC_COMUNA = "diccionario_variables_censo2024_nivel_comuna.xlsx"


def _columna(ref):
    n = 0
    for ch in re.match(r"[A-Z]+", ref).group(0):
        n = n * 26 + ord(ch) - 64
    return n - 1


def leer_xlsx(ruta, hoja):
    # Filas de una hoja como listas de textos (None en las celdas vacías)
    z = zipfile.ZipFile(ruta)
    compartidos = []
    if "xl/sharedStrings.xml" in z.namelist():
        for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", _NS):
            compartidos.append("".join(t.text or "" for t in si.iter(f"{{{_NS['m']}}}t")))
    libro = ET.fromstring(z.read("xl/workbook.xml"))
    rels = {r.get("Id"): r.get("Target") for r in ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))}
    destino = None
    for s in libro.find("m:sheets", _NS):
        if s.get("name") == hoja:
            destino = rels[s.get(f"{{{_NS['r']}}}id")].lstrip("/")
    if destino is None:
        raise KeyError(hoja)
    destino = destino if destino.startswith("xl/") else "xl/" + destino
    filas = []
    for fila in ET.fromstring(z.read(destino)).iter(f"{{{_NS['m']}}}row"):
        celdas = {}
        for c in fila.findall("m:c", _NS):
            v = c.find("m:v", _NS)
            if c.get("t") == "s" and v is not None:
                celdas[_columna(c.get("r"))] = compartidos[int(v.text)]
            elif c.get("t") == "inlineStr":
                celdas[_columna(c.get("r"))] = "".join(x.text or "" for x in c.iter(f"{{{_NS['m']}}}t"))
            elif v is not None:
                celdas[_columna(c.get("r"))] = v.text
        if celdas:
            filas.append([celdas.get(i) for i in range(max(celdas) + 1)])
    return filas


def _tabla(filas):
    encabezado, datos = filas[0], [f + [None] * (len(filas[0]) - len(f)) for f in filas[1:] if any(f)]
    return pd.DataFrame([f[:len(encabezado)] for f in datos], columns=encabezado)


def glosas_manzana(ruta_datos=RUTA_DATOS):
    return _tabla(leer_xlsx(os.path.join(ruta_datos, DICC_MANZANA), "Dicionario"))  # sic: así se llama la hoja


def codigos_microdato(tabla, ruta_datos=RUTA_DATOS):
    df = _tabla(leer_xlsx(os.path.join(ruta_datos, DICC_COMUNA), f"tabla_{tabla}"))
    df["Valor"] = df["Valor"].astype(str).str.replace("​", "", regex=False).str.strip()  # hay códigos con espacio invisible
    return df
