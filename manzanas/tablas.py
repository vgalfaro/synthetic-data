"""Tablas por manzana a partir de un modelo ajustado: bordes = dato publicado, interior = inferido."""
import numpy as np
import pandas as pd

from .inventario import etiquetas_bloque
from .perfiles import indicadora


def cruces_bivariados(inv, modelo, manzanas_ids, umbral=1e-6):
    # Todos los cruces de a pares entre los bloques del modelo, por manzana, en formato largo:
    # MANZENT, var1, cat1, var2, cat2, conteo (personas esperadas). Se omiten las celdas con conteo <= umbral.
    # T_ab[ca, cb, m] = sum_{p: x_pa = ca, x_pb = cb} W[p, m]; las categorías incluyen el "resto" de cada bloque
    bloques, P = modelo["bloques"], modelo["perfiles"]
    W = modelo["W"].astype(np.float64, copy=False)
    etiquetas = {b: etiquetas_bloque(inv, b) for b in bloques}
    partes = []
    for ia in range(len(bloques)):
        for ib in range(ia + 1, len(bloques)):
            a, b = bloques[ia], bloques[ib]
            Ca, Cb = len(etiquetas[a]), len(etiquetas[b])
            T = indicadora(P[:, ia] * Cb + P[:, ib], Ca * Cb) @ W  # (Ca*Cb, M)
            k, m = np.nonzero(T > umbral)
            partes.append(pd.DataFrame({
                "MANZENT": manzanas_ids[m], "var1": a, "cat1": np.array(etiquetas[a])[k // Cb],
                "var2": b, "cat2": np.array(etiquetas[b])[k % Cb], "conteo": T[k, m].astype(np.float32)}))
    df = pd.concat(partes, ignore_index=True)
    for col in ["var1", "cat1", "var2", "cat2"]:
        df[col] = df[col].astype("category")
    return df
