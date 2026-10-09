"""Estadísticas sintéticas a nivel manzana (Censo 2024): datos, inventario de marginales, IPF y validación.

Módulos:
    datos       carga por comuna (CSV nacionales -> parquet en resultados/), cartografía
    inventario  mapeo diccionario -> microdato de los marginales de personas, verificado
    ipf         sinkhorn, ipf (tensor denso), ipf_perfiles (CPU) e ipf_perfiles_gpu (cupy)
    perfiles    perfiles observados, ajuste por bloques, predicción previa, guardar/cargar modelos
    validacion  srmse, errores_marginales
"""
