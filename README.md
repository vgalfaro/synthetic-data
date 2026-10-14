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
| `05_pais.ipynb` | País: preparación por comuna, diagnóstico de las 346 comunas, piloto, extrapolación, monitoreo de la corrida nacional (E5) y mapa de ejemplo de un cruce | minutos (sin la corrida) |
| `correr_pais.py` | Corrida nacional en segundo plano (ver más abajo) | ~5 h de GPU |
| `06_validacion_provincias.ipynb` | Validación con verdad conocida: provincia como unidad grande, comunas como "manzanas"; error real de los cruces, mapas real vs estimado | ~7 min |
| `manzanas/` | Módulo con las funciones: `datos`, `inventario`, `ipf`, `perfiles`, `validacion`, `pais`, `tablas` | |
| `resultados/` | Caché local (no versionada): parquet por comuna, espina, modelos y la base nacional en `resultados/pais/` | |
| `microdatos.ipynb` | Notebook original con todo el desarrollo, del que salen 00–04 | |

**Orden:** 02 → 03 → 04 para Independencia; 05 para el país. Cada notebook lee de `resultados/` lo que necesita del anterior; ninguno depende del kernel de otro. El 03 no reajusta el modelo de 27 bloques si ya existe en `resultados/` (`REAJUSTAR = True` lo fuerza).

## Base nacional

**Contenido.** Para cada comuna se ajusta el modelo de 27 bloques y se exportan los **351 cruces de a pares** entre bloques (incluida la categoría "resto" de cada uno), por manzana, en formato largo:

```
resultados/pais/cruces/cruces_<cut>.parquet    MANZENT, var1, cat1, var2, cat2, conteo
```

Los bordes de cada cruce son dato publicado (marginales de la manzana y totales comunales) y el interior es inferido. Los modelos `W` (`resultados/pais/modelos/`) son internos y no se publican, porque la mayoría de los perfiles corresponde a una sola persona. Desde ellos se puede armar cualquier otra tabla.

**Los conteos son fraccionarios.** Son valores esperados del modelo (por ejemplo, 0,17 inmigrantes hombres en una manzana) y cuadran exactamente con los marginales publicados y los totales comunales. Pasarlos a enteros es un paso pendiente (ver próximos pasos).

**Cobertura.** Entran las 295 comunas cuyo inventario de 82 marginales cuadra con el microdato: el 99,2% de las personas. Quedan pendientes las 51 comunas bajo umbral (`comuna_bajo_umbral = 1`, 0,8% de la población), cuyo microdato trae `edad = -66` para todas las personas.

**Ajuste.** IPF en GPU con `W` en float32 y kernels propios (~10× más rápido que la versión de 03). Una comuna converge cuando cada celda restringida cumple |error| ≤ máx(0,1 personas; 0,01% de la manzana). Hay un tope de 20.000 iteraciones: si una comuna no converge, se guarda igual y queda marcada. En el piloto, Puente Alto (la mayor: 541 M celdas) convergió en 4.650 iteraciones (17 min); con tolerancia absoluta de 0,1 personas no convergía tras 4,6 h.

**Cómo correrla** (retomable: si se corta, se relanza y salta las comunas hechas):

```
nohup env_censo/bin/python correr_pais.py > resultados/pais/logs/corrida_pais.log 2>&1 &
```

El avance queda en `resultados/pais/resumen_pais.csv` (una fila por comuna con iteraciones, errores y chequeos) y se ve con la celda E5 de `05_pais.ipynb`.

**Corrida del 13-10-2026:** las 295 comunas convergieron, sin fallas, en 4,1 h de GPU (mediana de 1.700 iteraciones; máximo 6.550, en La Florida). La base de cruces quedó en 520 M filas (2,9 GB) y los modelos en 31,7 GB. Error marginal máximo: 0,996 personas, dentro de lo permitido. La diferencia máxima de las manzanas enmascaradas con el residuo comunal fue 0,45% (Puerto Octay).

## Resultados (Independencia, CUT 13108)

116.943 personas, 469 manzanas con población, 27 bloques de variables de personas con marginal por manzana (82 columnas).

- **Espina** (edad × sexo × inmigrante): converge con errores marginales ≤ 7e-7. Las 45 manzanas con `n_inmigrantes` enmascarado absorben exactamente el residuo comunal (98 personas).
- **Modelo completo** (27 bloques): 40.946 perfiles, `W` de 19,2 M celdas; converge a 0,1 personas de error marginal en 16.100 iteraciones de GPU. Los perfiles de una sola persona quedan repartidos en ~54 manzanas efectivas: los marginales publicados no permiten ubicarlos.
- **Predicción de un bloque sin verlo** (desde el modelo de 8 bloques; mejora del SRMSE sobre el proporcional): nacionalidad 96%, asistencia básica 80%, fuerza de trabajo 68%, estado civil 31%, religión 6%. Para ocupados por manzana, el error absoluto medio baja de 8,6 a 4,4 puntos porcentuales.

## Validación con verdad conocida (provincias y comunas)

Los cruces por manzana no se publican, pero los cruces por comuna sí se conocen (microdato). En `06_validacion_provincias.ipynb` se repite el experimento un nivel más arriba. La conjunta conocida es la de la **provincia**, y de cada **comuna** solo se usan los 82 marginales univariados que el INE publica por manzana. Se usa el mismo modelo y el mismo criterio de la corrida nacional, en 43 provincias y 287 comunas (97% de la población).

| Estimación | Personas mal ubicadas en el interior de los cruces |
|---|---|
| **Modelo (27 bloques)** | **0,18%** |
| Raking de 2 bloques (provincia ajustada a los 2 marginales del par) | 0,25% |
| Estructura provincial | 3,5% |
| Independencia dentro de la comuna | 4,5% |

- Los otros 25 bloques mejoran 250 de 287 cruces no triviales, con una reducción mediana de 21%. La excepción son los cruces con sexo, sobre todo edad × sexo, donde el modelo es algo peor.
- El error crece al achicarse la unidad: 0,14% en comunas de más de 150 mil personas y 0,34% en las de menos de 10 mil. A nivel de manzana cabe esperar errores mayores. Estos resultados son una cota optimista.
- Para las variables sin marginal (escolaridad, parentesco, país de nacimiento, pueblo específico), la extensión por perfil supera a condicionar en un solo bloque y al reparto proporcional.

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
- `cupy` para el ajuste de 27 bloques en GPU. No requiere cuBLAS, cuSPARSE ni cuRAND: las sumas por categoría usan `cp.add.at` (03) o kernels propios compilados con NVRTC (versión rápida, 05).
- `cudf` solo para `00` y `01`, que leen los CSV nacionales completos.

## Limitaciones conocidas

- **Precisión del ajuste completo.** Con muchos bloques el IPF converge lento, y en las comunas grandes la tolerancia absoluta se estanca. Por eso se usa una tolerancia relativa al tamaño de la manzana. Un solver más rápido (por ejemplo, el dual con L-BFGS) mejoraría los tiempos.
- **Conteos fraccionarios.** La base trae valores esperados, no enteros. Redondear cada celda por separado rompería la consistencia con los marginales publicados.
- **Validación a otra escala.** La validación con verdad conocida (notebook 06) usa comunas como "manzanas". Mide el error real del mecanismo, pero en unidades mucho más grandes que una manzana. A nivel de manzana solo hay validación indirecta (LOO).
- **Comunas bajo umbral.** En las 51 comunas con `comuna_bajo_umbral = 1` el microdato trae `-66` (suprimido por anonimización) en la edad y otras variables, y su inventario no cuadra. Quedan fuera de la base por ahora.
- **Privacidad.** Por ahora los marginales se usan sin ruido; el modelo no tiene garantía formal. En comunas rurales con pocas manzanas, algunos perfiles de una sola persona quedan concentrados (en Alto Biobío, el 9% tiene más de la mitad de su masa en una manzana).

## Próximos pasos

1. **Redondeo de los conteos:** pasar la base a enteros con redondeo controlado. Cada tabla de 2 variables en una manzana se redondea al entero de arriba o de abajo en cada celda, de modo que filas y columnas sigan sumando los marginales publicados, que son enteros (para tablas de 2 dimensiones siempre existe solución; Cox, 1987). Conviene hacerlo después de la Fase 2, que también exige dejar los valores enteros y no negativos.
2. Comunas bajo umbral: repartir `edad_quinquenal` en los tramos del INE con proporciones regionales de edad simple, documentando el supuesto.
3. Hogares y viviendas (tienen sus propios marginales por manzana), y cruces persona ↔ hogar.
4. **Fase 2 — privacidad diferencial:** ruido de Laplace sobre los marginales por manzana antes del IPF. Todo lo posterior es post-procesamiento y hereda la garantía. Las celdas ya enmascaradas no se re-ruidean, y la información pública externa (volumen de edificios) puede ayudar a corregir el ruido sin gastar presupuesto. Las comunas con pocas manzanas son las prioritarias.
