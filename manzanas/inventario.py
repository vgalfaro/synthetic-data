"""Inventario de marginales de personas por manzana: mapeo diccionario -> microdato, verificado contra los totales.

Fuentes: diccionario_variables_glosas_censo2024_manzana_agregados.xlsx (columnas de manzana) y
diccionario_variables_censo2024_nivel_comuna.xlsx (códigos del microdato).
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .datos import leer_marginal

EDAD_TRAMOS = [("0_5", 0, 5), ("6_13", 6, 13), ("14_17", 14, 17), ("18_24", 18, 24),
               ("25_44", 25, 44), ("45_59", 45, 59), ("60_mas", 60, 130)]

# bloque -> (variable en personas, [(columna en manzanas, códigos de la variable)])
MAPEO_PERSONAS = {
    "edad":           ("edad", [(f"n_edad_{n}", list(range(lo, hi + 1))) for n, lo, hi in EDAD_TRAMOS]),
    "sexo":           ("sexo", [("n_hombres", [1]), ("n_mujeres", [2])]),
    "inmigrante":     ("p25_lug_nacimiento_rec", [("n_inmigrantes", [2])]),
    "nacionalidad":   ("p27_nacionalidad", [("n_nacionalidad", [3])]),          # 3 = otra (excluye doble nacionalidad)
    "pueblo":         ("p28_autoid_pueblo", [("n_pueblos_orig", [1])]),
    "afro":           ("p29_afrodescendencia_rec", [("n_afrodescendencia", [1])]),
    "lengua":         ("p30_lengua_indigena_rec", [("n_lengua_indigena", [1])]),  # universo 5+
    "religion":       ("p31_religion_rec", [("n_religion", [1])]),                # universo 15+
    **{f"dif_{k}":    (c, [(f"n_dificultad_{k}", [3, 4])])                       # 3 = mucha dificultad, 4 = no puede
       for k, c in [("ver", "p32a_dificultad_ver"), ("oir", "p32b_dificultad_oir"),
                    ("mover", "p32c_dificultad_mover"), ("cogni", "p32d_dificultad_cogni"),
                    ("cuidado", "p32e_dificultad_cuidado"), ("comunic", "p32f_dificultad_comunic")]},
    "discapacidad":   ("discapacidad", [("n_discapacidad", [1])]),
    "estado_civil":   ("p23_est_civil", [("n_estcivcon_casado", [1]), ("n_estcivcon_conviviente", [2]),
                                         ("n_estcivcon_conv_civil", [3]), ("n_estcivcon_anul_sep_div", [4, 5, 6]),
                                         ("n_estcivcon_viudo", [7]), ("n_estcivcon_soltero", [8])]),
    **{f"asist_{k}":  (f"asistencia_{k}", [(f"n_asistencia_{k}", [1])]) for k in ["parv", "basica", "media", "superior"]},
    "cine":           ("cine11", [("n_cine_nunca_curso_primera_infancia", [1, 2]), ("n_cine_primaria", [3, 4, 5]),
                                  ("n_cine_secundaria", [6, 7]), ("n_cine_terciaria_maestria_doctorado", [8, 9, 10, 11]),
                                  ("n_cine_especial_diferencial", [12])]),
    "analfabetismo":  ("p37_alfabet_15", [("n_analfabet", [2])]),                # p37_alfabet restringido a 15+
    "fuerza_trabajo": ("sit_fuerza_trabajo", [("n_ocupado", [1]), ("n_desocupado", [2]),
                                              ("n_fuera_fuerza_trabajo", [3])]),
    "cise":           ("p40_cise_rec", [("n_cise_rec_independientes", [1]), ("n_cise_rec_dependientes", [2]),
                                        ("n_cise_rec_trabajador_no_remunerado", [3])]),
    "ciuo":           ("cod_ciuo", [(f"n_ciuo_{k}", [k]) for k in range(10)]),
    "caenes":         ("cod_caenes", [(f"n_caenes_{k}", [k]) for k in "ABCDEFGHIJKLMNOPQRSTU"]),
    "transporte":     ("p45_medio_transporte", [("n_transporte_auto", [1]), ("n_transporte_publico", [2]),
                                                ("n_transporte_camina", [3]), ("n_transporte_bicicleta", [4]),
                                                ("n_transporte_motocicleta", [5]), ("n_transporte_cab_lan_bote", [6]),
                                                ("n_transporte_otros", [7])]),
}


@dataclass
class Inventario:
    per: pd.DataFrame     # variables de persona que usan los bloques (+ p37_alfabet_15)
    marginales: dict      # bloque -> {"variable", "categorias", "nu" (C_k, M)}
    nu: np.ndarray        # n_per por manzana
    tabla: pd.DataFrame   # por columna: total comunal vs suma de manzanas no enmascaradas


def construir_inventario(datos, mapeo=MAPEO_PERSONAS):
    # Lee cada marginal publicado y verifica: exacto si no tiene '*', residuo >= 0 si lo tiene,
    # y las categorías publicadas no exceden n_per
    cols = sorted({v for v, _ in mapeo.values()} - {"p37_alfabet_15"} | {"p37_alfabet"})
    per = datos.personas[cols].copy()
    per["p37_alfabet_15"] = per["p37_alfabet"].where(per["edad"] >= 15)

    marginales, filas = {}, []
    for bloque, (var, categorias) in mapeo.items():
        nu_k = np.vstack([leer_marginal(datos, col) for col, _ in categorias])  # (C_k, M)
        for (col, codigos), fila_nu in zip(categorias, nu_k):
            total_comuna = int(per[var].isin(codigos).sum())
            n_masc = int(np.isnan(fila_nu).sum())
            residuo = total_comuna - np.nansum(fila_nu)
            filas.append({"bloque": bloque, "columna": col, "variable": var,
                          "códigos": str(codigos) if len(codigos) <= 6 else f"{codigos[0]}..{codigos[-1]}",
                          "total comuna": total_comuna, "suma manzanas": int(np.nansum(fila_nu)),
                          "enmascaradas": n_masc, "residuo": residuo})
            assert (residuo == 0) if n_masc == 0 else (residuo >= 0), (col, residuo)
        ok = ~np.isnan(nu_k).any(axis=0)
        assert (nu_k[:, ok].sum(axis=0) <= datos.nu[ok]).all(), bloque
        marginales[bloque] = {"variable": var, "categorias": categorias, "nu": nu_k}
    return Inventario(per, marginales, datos.nu, pd.DataFrame(filas))


def etiquetas_bloque(inv, bloque):
    # Nombre de cada categoría del bloque (sin el prefijo n_) más "resto"
    return [col.replace("n_", "", 1) for col, _ in inv.marginales[bloque]["categorias"]] + ["resto"]
