"""Métricas de validación."""
import numpy as np


def errores_marginales(T, constraints, nombres=None):
    # Error máximo |proyección de T - objetivo| por restricción, ignorando las celdas NaN (sin restricción)
    nombres = nombres or [f"sum{axes}" for axes, _ in constraints]
    out = {}
    for (axes, target), nombre in zip(constraints, nombres):
        target = np.asarray(target, dtype=float)
        proj = T.sum(axis=axes)
        ok = ~np.isnan(target)
        out[nombre] = np.abs(proj[ok] - target[ok]).max()
    return out


def srmse(pred, real):
    # SRMSE (Voas & Williamson, 2001) sobre las celdas con dato real (NaN = enmascarada, se excluye)
    real = np.asarray(real, dtype=float)
    ok = ~np.isnan(real)
    return np.sqrt(np.mean((pred[ok] - real[ok]) ** 2)) / np.mean(real[ok])
