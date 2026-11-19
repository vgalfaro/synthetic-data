# Estadísticas sintéticas a nivel de manzana — Censo 2024

Estimación de **tablas cruzadas por manzana** (por ejemplo, sexo × inmigrante × edad en cada manzana) a partir de los datos públicos del Censo de Población y Vivienda 2024 (Chile), para personas, hogares y viviendas.

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
| `06_validacion_provincias.ipynb` | Validación con verdad conocida: provincia como unidad grande, comunas como "manzanas"; error real de los cruces, mapas real vs estimado | ~7 min |
| `07_cifras_informe.ipynb` | Recalcula las cifras del informe que no salen de otro notebook (una celda por sección) | ~4 min |
| `08_hogares_viviendas.ipynb` | El mismo modelo para viviendas y para hogares: mapeo e inventario, ajuste en Independencia, validación con verdad conocida (provincias), diagnóstico y resumen de la corrida nacional, validación con totales no usados, chequeo de la base de cruces y mapas | ~10 min (sin la corrida) |
| `09_rounding.ipynb` | Redondeo controlado de W\* en Independencia: flujo máximo, muestreo dependiente (pizarra) y comparación de errores por bloque | minutos |
| `10_redondeo_duro.ipynb` | Redondeo con las restricciones de bloque como igualdades (MIP con HiGHS o Gurobi) y verificación de la solución | minutos (sin el MIP) |
| `scripts/` | Corridas largas en segundo plano: IPF nacional (`correr_pais.py`), redondeo (`correr_redondeo.py`, `correr_cembrano.py`, `correr_cembrano_pais.py`, `correr_cembrano_grandes.py`), pruebas de Gurobi (`probar_lp_grupo.py`) y sincronización con el clúster (`sincronizar_cluster.py`). Se corren desde la raíz: `env_censo/bin/python scripts/<script>.py` | horas |
| `cluster/` | Planificador y trabajos SLURM para el clúster de Ingeniería UC (ver `cluster/README.md`) | |
| `docs/` | `CONTEXTO.md` (estado del proyecto), `sesion_redondeo.md` (sesión de trabajo del redondeo), `informe/informe_modelo_27.tex` (informe LaTeX; su apéndice B indica de qué notebook sale cada cifra) y `referencias/` (Cembrano et al., 2022) | |
| `manzanas/` | Módulo con las funciones: `datos`, `inventario`, `ipf`, `perfiles`, `validacion`, `pais`, `tablas`, `provincias`, `informe`, `unidades` (hogares y viviendas), `diccionario` (lectura de los xlsx del INE) y `redondeo` (redondeo controlado) | |
| `resultados/` | Caché local (no versionada): parquet por comuna, espina, modelos y la base nacional en `resultados/pais/` | |
| `microdatos.ipynb` | Notebook original con todo el desarrollo, del que salen 00–04 | |

**Orden:** 02 → 03 → 04 para Independencia; 05 para el país; 08 para hogares y viviendas; 09 → 10 y `scripts/correr_cembrano*.py` para el redondeo. Cada notebook lee de `resultados/` lo que necesita del anterior; ninguno depende del kernel de otro. El 03 no reajusta el modelo de 27 bloques si ya existe en `resultados/` (`REAJUSTAR = True` lo fuerza).

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
nohup env_censo/bin/python scripts/correr_pais.py > resultados/pais/logs/corrida_pais.log 2>&1 &
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

## Hogares y viviendas

`08_hogares_viviendas.ipynb` aplica el mismo modelo a las otras dos unidades del censo. Cada una tiene su propio `W` por comuna y su propia base de cruces. Los tres modelos son independientes; los cruces entre unidades (persona × hogar) quedan pendientes.

| | Viviendas | Hogares |
|---|---|---|
| Unidad | viviendas particulares, ocupadas y desocupadas | hogares en viviendas particulares |
| Total por manzana (hace de `n_per`) | `n_vp` | `n_hog` |
| Bloques | 13 (63 columnas) | 15 (35 columnas) |
| Comunas | 345, incluidas las 50 bajo umbral con viviendas | 345 |
| `W` por comuna | `resultados/pais/viviendas/modelos/` (1,8 GB) | `resultados/pais/hogares/modelos/` (3,9 GB) |
| Cruces por manzana | `resultados/pais/viviendas/cruces/` (64 M filas, 308 MB) | `resultados/pais/hogares/cruces/` (85 M filas, 507 MB) |

Los cruces tienen las mismas columnas que los de personas (`MANZENT, var1, cat1, var2, cat2, conteo`).

**Mapeo** (`manzanas/unidades.py`). Las 98 columnas usadas cuadran exactas con el microdato en las 345 comunas.
- Tipo de vivienda y ocupación se publican sobre todas las viviendas particulares: suman `n_vp`, aunque el diccionario diga "ocupadas con moradores presentes". El resto de los bloques de vivienda se publica sobre las ocupadas con moradores presentes, y las demás caen en su categoría "resto".
- Columnas derivadas: jefatura femenina, solo personas de 60+ y al menos un menor de 15 salen del microdato de personas con `edad_quinquenal`, que viene también en las comunas bajo umbral. Además, internet es fija, móvil o satelital, y los hogares allegados son los de `id_hogar` ≥ 2.
- Viviendas irrecuperables se reconstruyó exacta: ocupada con moradores presentes, de tipo mediagua, móvil u otro, o con paredes precarias, techo precario o sin cubierta, o piso de tierra. Las que tienen no respuesta en algún material no se cuentan.
- Quedan fuera núcleos hacinados allegados, hogares en viviendas hacinadas no ampliables y el déficit cuantitativo, que no se pueden reproducir desde el microdato. El déficit es la suma de cuatro columnas.

**Corrida nacional (02-11-2026).** Todas las comunas convergieron. Viviendas tomó 12 min de GPU y hogares 4 min.

```
nohup env_censo/bin/python scripts/correr_pais.py viviendas > resultados/pais/viviendas/logs/corrida.log 2>&1 &
nohup env_censo/bin/python scripts/correr_pais.py hogares > resultados/pais/hogares/logs/corrida.log 2>&1 &
```

**Validación.**

| | Viviendas | Hogares |
|---|---|---|
| Mal ubicadas, modelo (provincias → comunas) | **0,37%** | **0,37%** |
| Mal ubicadas, raking de 2 bloques | 0,49% | 0,58% |
| Mal ubicadas, independencia | 14,8% | 3,4% |
| Pares donde el modelo mejora al raking de 2 bloques | 67 de 67 | 99 de 100 |
| Error relativo de un total no usado, por manzana real | hogares: 2,3% (5,2% proporcional) | personas: 5,9% (11,2% proporcional) |

Las variables sin marginal (estado de ocupación detallado, hogares y personas por vivienda, tipología del hogar) se estiman mejor desde `W` que condicionando en un bloque o repartiendo en proporción.

**Privacidad.** Los cruces de viviendas son los más reveladores. El 29% de los perfiles de una sola vivienda concentra más de la mitad de su masa en una manzana, contra 2,7% en hogares y 0,71% en personas. Las categorías raras de vivienda (pozo, sin electricidad, piso de tierra, mediagua) ya ubican a esas viviendas en los marginales publicados, y el cruce agrega sus otros atributos. Esta base no debería publicarse sin la Fase 2 o sin suprimir celdas chicas.

## Redondeo controlado (personas)

La base IPF trae conteos fraccionarios W\*[perfil, manzana]. El redondeo busca una matriz entera Y, con cada celda en
floor(W\*) o ceil(W\*), que respete los totales de perfiles (μ_p), de manzanas (ν_m) y, lo mejor posible, las cifras
publicadas de los 27 bloques por manzana.

- **Método:** redondeo iterativo de Cembrano, Correa y Verdugo (PNAS 2022, Teorema 5), en `manzanas/redondeo.py`.
  1. Se calcula un vértice del LP con Gurobi (barrier + crossover).
  2. Se aplica el Algoritmo 1, que garantiza una desviación máxima u por parte.
- **Dos configuraciones por comuna:**
  - A (`perfiles_exactos`): perfiles exactos (u = 0), manzanas y bloques con u = 54.
  - B (`uniforme`): u = 27 en todas las partes.
- **Corrida nacional:** `scripts/correr_cembrano_pais.py` corre las comunas de hasta 45 M celdas en el servidor.
  - Las comunas grandes corren en el clúster (`cluster/`): enteras hasta 120 M celdas; las de más de 120 M se dividen
    en grupos de manzanas (`scripts/correr_cembrano_grandes.py`).
- **Salidas** en `resultados/pais/redondeo/`:
  - `Y_personas_<cut>_cembrano_<config>.npz`: matriz Y dispersa;
  - `.json`: verificación y errores por bloque;
  - `vertice_personas_<cut>.npz`: vértice LP, para repetir el Algoritmo 1 sin resolver de nuevo el LP;
  - `resumen_cembrano.csv`: una fila por comuna.
- **Resultados** en las comunas terminadas: todas cumplen las cotas del teorema.
  - El error medio por celda (categoría × manzana) frente al censo es de ~1,1 en A y ~0,8 en B.
- **Licencia de Gurobi:**
  - en el servidor es WLS académica y necesita el túnel del Mac, porque el DNS del servidor está roto;
  - en el clúster sale de su propio servidor de tokens.

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
- **Privacidad.** Por ahora los marginales se usan sin ruido; el modelo no tiene garantía formal. En viviendas el riesgo es mucho mayor que en personas (ver "Hogares y viviendas"). En comunas rurales con pocas manzanas, algunos perfiles de una sola persona quedan concentrados (en Alto Biobío, el 9% tiene más de la mitad de su masa en una manzana).

## Próximos pasos

1. **Redondeo de los conteos (en curso para personas, ver "Redondeo controlado"):** pasar la base a enteros con redondeo controlado. Cada tabla de 2 variables en una manzana se redondea al entero de arriba o de abajo en cada celda, de modo que filas y columnas sigan sumando los marginales publicados, que son enteros (para tablas de 2 dimensiones siempre existe solución; Cox, 1987). Conviene hacerlo después de la Fase 2, que también exige dejar los valores enteros y no negativos.
2. Comunas bajo umbral: repartir `edad_quinquenal` en los tramos del INE con proporciones regionales de edad simple, documentando el supuesto.
3. Cruces entre unidades (persona × hogar, hogar × vivienda). Los modelos de hogares y viviendas ya están (ver "Hogares y viviendas"), pero son independientes del de personas.
4. **Fase 2 — privacidad diferencial:** ruido de Laplace sobre los marginales por manzana antes del IPF. Todo lo posterior es post-procesamiento y hereda la garantía. Las celdas ya enmascaradas no se re-ruidean, y la información pública externa (volumen de edificios) puede ayudar a corregir el ruido sin gastar presupuesto. Las comunas con pocas manzanas son las prioritarias.
