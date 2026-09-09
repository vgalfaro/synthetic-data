# Microdatos sintéticos a nivel de manzana — Censo 2024

Generación de microdatos sintéticos a nivel de **manzana** a partir de los microdatos del Censo de Población y Vivienda 2024 (Chile), que el INE solo libera desagregados hasta nivel **comuna**.

## Motivación

El INE debe proteger la identidad de las personas y asegurar el secreto estadístico (ley 17.374) al liberar microdatos, que además son datos sensibles según la ley 21.719 de protección de datos personales. Por eso, la máxima desagregación geográfica de los microdatos es comuna; para manzana solo se publican **conteos marginales agregados** (`Base_manzana_entidad_CPV24.csv`), no la base de personas/hogares/viviendas.

Esto plantea el problema central del proyecto: a nivel comuna conocemos la distribución conjunta real de la población, pero a nivel manzana solo tenemos marginales. El objetivo es reconstruir/asignar, con algún mecanismo de privacidad, una manzana a cada fila del dataset de personas.

## Enfoque: Transporte Óptimo Entrópico

Se modela la asignación como un problema de transporte óptimo regularizado por entropía (Sinkhorn):

- **μ** — distribución de origen: la característica de interés a nivel comuna (conocida, viene de `df_personas`).
- **ν** — distribución de destino: población (u otro conteo) por manzana (conocida, viene de `df_manzanas`).
- **C** — matriz de costo: penalización de asignar cada categoría de origen a cada manzana; codifica la lógica territorial y debe construirse con información auxiliar.
- **$\epsilon$** — parámetro de regularización entrópica: controla qué tan "suave"/uniforme es la asignación resultante.

El problema tiene solución vía el algoritmo de Sinkhorn. El resultado es una matriz continua de conteos esperados, que se redondea a enteros exactos con el método de (TBD) y luego se usa para asignar, persona por persona, una manzana específica (TBD).

## Estado actual: prueba de concepto (comuna de Independencia)

El notebook `microdatos.ipynb` implementa una PoC completa para una sola comuna (Independencia, RM, CUT 13108 — 470 manzanas, 116.943 personas) y una sola variable (edad, en los mismos 7 tramos etarios que ya vienen agregados en `df_manzanas`).

Se eligió edad a propósito porque sus conteos por manzana (`n_edad_*`) ya existen en los datos públicos: se ocultan al ajustar el modelo (solo se usa `n_per`, población total por manzana, como restricción) y se comparan al final contra el resultado reconstruido — dando un ground truth real para validar el método sin inventar uno sintético.

**Resultados:**

| Modelo | SRMSE | Error absoluto promedio/celda |
|---|---|---|
| Baseline (C=0, asignación proporcional ingenua) | 0.79 | ~9.3 personas |
| Sinkhorn + costo (`C` = distancia al `prom_edad` de la manzana), ε=1.2 | 0.54 | ~6.6 personas |

- El costo se construyó con `prom_edad` (edad promedio por manzana), ya publicado en `df_manzanas` — barato y sin necesidad de datos externos.
- El modelo mejora el error en 379/469 manzanas (81%); en el resto, el proxy `prom_edad` no representa bien la composición etaria real.
- Se valida también geográficamente: cruce contra cartografía censal (`cartography/`) para mapear el error por manzana y por tramo etario (461/469 manzanas con geometría disponible — 8 quedan fuera, incluida una manzana "resto" no geolocalizable).

## Estructura de datos esperada (no versionada)

`databases/`, `cartography/` y `env_censo/` están en `.gitignore` — son locales a cada máquina. Se esperan:

```
databases/
  personas_censo2024.csv
  hogares_censo2024.csv
  viviendas_censo2024.csv
  Base_manzana_entidad_CPV24.csv
cartography/
  SHP_APC2023_R13/Manzana_Urbana.shp  (+ .dbf, .shx, etc.)
```

Todos los CSV usan `;` como separador. `Base_manzana_entidad_CPV24.csv` requiere `encoding='utf-8-sig'` (BOM en el primer header) y algunas columnas (ej. `prom_edad`) vienen con coma decimal chilena. En el shapefile, usar el campo `Mzent_TX` (texto) en vez de `MANZENT` (numérico) para el código de manzana — este último viene truncado por una limitación de ancho de campo en el `.dbf`.

## Requisitos

- Python con `cudf` (RAPIDS, requiere GPU NVIDIA) — o `pandas` como alternativa si se corre sin GPU, cambiando el import de la primera celda.
- `numpy`, `scipy`, `matplotlib`, `geopandas`, `pandas`.

## Cómo correr

Abrir `microdatos.ipynb` y ejecutar las celdas en orden. Las primeras celdas cargan y exploran las 4 bases; la sección "El Problema de Transporte Óptimo Entrópico" explica el modelo; "Ejercicio inicial: Comuna de Independencia" contiene la PoC completa (marginales → costo → Sinkhorn → validación numérica y geográfica → asignación individual).

## Limitaciones conocidas

- **`ε` no tiene una garantía formal de privacidad** — es un parámetro de sesgo-varianza estadístico (con `C=0` ni siquiera afecta el resultado), no un mecanismo de privacidad diferencial. La validación actual además usa marginales reales sin ruido, por lo que esta PoC no protege privacidad todavía, solo valida la mecánica de desagregación.
- La matriz de costo depende de la calidad del proxy elegido; con `prom_edad` el método no ayuda en ~19% de las manzanas.
- Escalar a más variables (sexo, estado civil, etc.) simultáneamente requiere o bien ampliar `μ` al producto cartesiano de categorías (mismo Sinkhorn, más filas), o bien raking multidimensional (tipo IPU) si se quiere aprovechar varios marginales de manzana a la vez — no implementado aún.

## Próximos pasos

- Extender `μ` a la distribución conjunta (edad × sexo × estado civil, etc.) y aprovechar más conteos de manzana (`n_hombres`, `n_estcivcon_*`, ...).
- Evaluar un pivote hacia métodos de datos sintéticos con privacidad diferencial formal (ej. MST / Private-PGM, [McKenna et al. 2021](https://arxiv.org/abs/2108.04978)), que dan garantías demostrables y escalan mejor a múltiples marginales simultáneos, a costa de mayor complejidad de implementación.
