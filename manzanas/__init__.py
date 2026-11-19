"""Estadísticas sintéticas a nivel manzana (Censo 2024): datos, inventario de marginales, IPF y validación.

Módulos:
    datos       carga por comuna (CSV nacionales -> parquet en resultados/), cartografía
    inventario  mapeo diccionario -> microdato de los marginales de personas y verificación (inventario_tabla)
    unidades    hogares y viviendas: mapeos, tablas con las variables derivadas, carga por comuna
    diccionario lectura de los diccionarios xlsx del INE (sin openpyxl)
    ipf         sinkhorn, ipf (tensor denso), ipf_perfiles (CPU), ipf_perfiles_gpu y la versión rápida (cupy)
    perfiles    perfiles observados, ajuste por bloques, predicción previa, guardar/cargar modelos
    pais        diagnóstico y procesamiento comuna por comuna de cualquier unidad (corrida nacional)
    tablas      cruces bivariados por manzana
    provincias  validación con verdad conocida (provincia -> comunas)
    informe     cifras del informe
    validacion  srmse, errores_marginales
"""
