# Estadísticas sintéticas a nivel de manzana — Censo 2024

Estimación de **tablas cruzadas por manzana** (por ejemplo, sexo × inmigrante × edad en cada manzana) a partir de los datos públicos del Censo de Población y Vivienda 2024 (Chile).

## Problema

El INE publica los microdatos (personas, hogares, viviendas) con desagregación geográfica máxima de **comuna**. A nivel de **manzana** solo publica conteos **univariados** (`Base_manzana_entidad_CPV24.csv`: personas por sexo, por tramo de edad, inmigrantes, ocupados, etc.), nunca cruces, y enmascara con `*` algunas celdas pequeñas por secreto estadístico (ley 17.374).

Es decir, se conoce la **conjunta comunal** de cualquier combinación de variables, pero en cada manzana solo sus **marginales**. El objetivo es reconstruir los cruces por manzana de forma consistente con todo lo publicado.

Por riesgo de reidentificación **no se asigna una manzana a cada persona**. Se estiman y publican estadísticos agregados.

## Método

**Raking multi-marginal (IPF) con perfiles observados.** Un perfil es una combinación de categorías (tramo de edad, sexo, inmigrante, pueblo, estado civil, ocupación, …) que existe en el microdato de la comuna. Se estima una matriz `W[perfil, manzana]`, el número esperado de personas de cada perfil en cada manzana, que es la más cercana en divergencia KL a una semilla y satisface a la vez:

- **las filas:** para cada perfil, la suma sobre manzanas da su conteo comunal (la conjunta comunal);
- **las columnas:** en cada manzana, la suma por categoría reproduce cada marginal publicado. Las celdas con `*` quedan sin restricción, y el residuo comunal se reparte entre ellas.

La solución es única cuando el problema es factible (Csiszár, 1975). Con 2 restricciones es el transporte óptimo entrópico (Sinkhorn); con más, IPF. A partir de `W`:

- **Cualquier cruce por manzana** sale sumando `W` sobre los perfiles de cada celda. Sus **bordes son dato** (los marginales publicados y la conjunta comunal) y su **interior es inferido**.
- **Variables sin marginal por manzana** (país de nacimiento, edad simple, parentesco, etc.): cada persona hereda el reparto espacial de su perfil. Es inferencia, no dato: supone que, dado el perfil completo, la variable no depende de la manzana.

**Validación.** Como los cruces por manzana no se publican, se valida con *leave-one-out*: se esconde un marginal publicado, se predice con el resto y se compara contra el dato, junto con un baseline proporcional.

## Estructura del repositorio

| Archivo | Contenido | Tiempo |
|---|---|---|
| `00_exploracion.ipynb` | Exploración de las cuatro bases (usa cudf sobre los CSV nacionales) | — |
| `01_poc_historica.ipynb` | Prueba de concepto histórica: transporte óptimo con `prom_edad`, sexo, inmigrante, priori de volumen de edificios. Congelado, con sus outputs | — |
| `02_espina.ipynb` | Espina edad × sexo × inmigrante × manzana con edad como restricción dura; LOO interno. Guarda `resultados/espina_<cut>.npz` | ~5 s |
| `03_perfiles_27.ipynb` | Perfiles, inventario de los 82 marginales de personas, modelo de 8 bloques y modelo de 27 bloques (GPU). Guarda `resultados/modelo_*_<cut>.npz` | ~3 min + ~15–20 min en GPU la primera vez |
| `04_tablas_y_mapas.ipynb` | Tablas por manzana y mapas, cargando los modelos guardados | segundos |
| `manzanas/` | Módulo con las funciones: `datos`, `inventario`, `ipf`, `perfiles`, `validacion` | |
| `resultados/` | Caché local (no versionada): parquet por comuna, espina y modelos | |
| `microdatos.ipynb` | Notebook original con todo el desarrollo, del que salen 00–04 | |

**Orden:** 02 → 03 → 04. Cada notebook lee de `resultados/` lo que necesita del anterior; ninguno depende del kernel de otro. El 03 no reajusta el modelo de 27 bloques si ya existe en `resultados/` (`REAJUSTAR = True` lo fuerza).

## Resultados (Independencia, CUT 13108)

116.943 personas, 469 manzanas con población, 27 bloques de variables de personas con marginal por manzana (82 columnas).

- **Espina** (edad × sexo × inmigrante): converge con errores marginales ≤ 7e-7. Las 45 manzanas con `n_inmigrantes` enmascarado absorben exactamente el residuo comunal (98 personas).
- **Modelo completo** (27 bloques): 40.946 perfiles, `W` de 19,2 M celdas; converge a 0,1 personas de error marginal en 16.100 iteraciones de GPU. Los perfiles de una sola persona quedan repartidos en ~54 manzanas efectivas: los marginales publicados no permiten ubicarlos.
- **Predicción de un bloque sin verlo** (desde el modelo de 8 bloques; mejora del SRMSE sobre el proporcional): nacionalidad 96%, asistencia básica 80%, fuerza de trabajo 68%, estado civil 31%, religión 6%. Para ocupados por manzana, el error absoluto medio baja de 8,6 a 4,4 puntos porcentuales.

## Datos esperados (no versionados)

```
databases/
  personas_censo2024.csv  hogares_censo2024.csv  viviendas_censo2024.csv
  Base_manzana_entidad_CPV24.csv
  diccionario_variables_censo2024_nivel_comuna.xlsx
  diccionario_variables_glosas_censo2024_manzana_agregados.xlsx
cartografia_censal/
  Cartografia_censo2024_Pais_Manzanas.parquet
```

Todos los CSV usan `;`. Algunas columnas de manzana traen coma decimal (`prom_edad`) o `*` (secreto estadístico). La primera vez que se carga una comuna, `manzanas.datos.cargar_comuna` la filtra de los CSV nacionales y la guarda en `resultados/` en parquet, en segundos.

## Requisitos

- `numpy`, `scipy`, `pandas`, `pyarrow`, `geopandas`, `matplotlib`.
- `cupy` para el ajuste de 27 bloques en GPU. No requiere cuBLAS ni cuSPARSE: las sumas por categoría usan `cp.add.at`.
- `cudf` solo para `00` y `01`, que leen los CSV nacionales completos.

## Limitaciones conocidas

- **Precisión del ajuste completo.** Con más de ~10 bloques el IPF converge lento: 0,1 personas se alcanza en ~15 min de GPU, y bajo ~4e-4 se estanca. Es irrelevante para tablas en enteros, pero un solver más rápido mejoraría los tiempos.
- **Validación indirecta.** El LOO valida marginales, no cruces. Pendiente: validar con verdad conocida, usando comunas como "manzanas" dentro de una provincia.
- **Escalamiento a otras comunas.** En comunas bajo el umbral (`comuna_bajo_umbral = 1`) el microdato trae `-66` (suprimido por anonimización) y los totales comunales no van a calzar con la suma de manzanas.
- **Privacidad.** Por ahora los marginales se usan sin ruido; el modelo no tiene garantía formal.

## Próximos pasos

1. Validación con comunas como manzanas (error real de los cruces).
2. Hogares y viviendas (tienen sus propios marginales por manzana), y cruces persona ↔ hogar.
3. **Fase 2 — privacidad diferencial:** ruido de Laplace sobre los marginales por manzana antes del IPF. Todo lo posterior es post-procesamiento y hereda la garantía. Las celdas ya enmascaradas no se re-ruidean, y la información pública externa (volumen de edificios) puede ayudar a corregir el ruido sin gastar presupuesto.
4. Escalar a todas las comunas: cada comuna es independiente y paralelizable.
