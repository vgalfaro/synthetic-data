# Redondeo en el clúster de Ingeniería UC

Corre el redondeo de Cembrano de las comunas grandes (W\* de más de 45 M celdas) en el clúster `cluster.ing.uc.cl`.
En el clúster la licencia de Gurobi sale de su propio servidor de tokens, así que no depende del túnel del Mac.

## Archivos

| archivo | qué hace |
|---|---|
| `planificador.py` | Lanza y vigila los trabajos. Corre con `nohup` en el nodo de login y deja su estado en `estado.json` y su log en `planificador.log`. |
| `job_comuna.sh` | Corre una comuna entera: vértice LP y Algoritmo 1 (`scripts/correr_cembrano.py`). |
| `job_dividir.sh` | Divide una comuna gigante en grupos de unos 25 M de celdas no nulas. |
| `job_grupo.sh` | Corre un grupo: LP blando y Algoritmo 1 (`scripts/correr_cembrano_grandes.py grupo`). |
| `job_ensamblar.sh` | Junta los grupos de una gigante en sus matrices Y. |
| `enteras.txt`, `gigantes.txt` | Comunas que se corren enteras (hasta 120 M) y comunas que se dividen (más de 120 M), en la corrida de noviembre de 2026. |

## Por qué hay un planificador propio

SLURM no controla la memoria en este clúster (`SelectTypeParameters=CR_CORE`): el `--mem` de un trabajo no reserva
nada. En una prueba, SLURM puso 7 comunas que sumaban unos 2,6 TB en un nodo de 512 GB.

El planificador calcula, para cada nodo, la memoria libre real (`sinfo %e`), le resta lo que todavía pueden crecer los
trabajos que ya corren (la estimación menos el uso actual que reporta `sstat`) y una reserva de 24 GB. Con eso fija cada
trabajo a un nodo con `-w`.

Otros criterios:
- La memoria estimada es de 3,2 GB por millón de celdas de W\*, el pico medido en el servidor.
- Usa 4 hilos por trabajo, porque el crossover de Gurobi, que es más de la mitad del tiempo, casi no aprovecha más.
- Tiene a lo más 22 trabajos activos, porque el clúster es compartido.
- Si un trabajo termina sin dejar su archivo de salida, lo reintenta hasta 3 veces.

## Puesta en marcha

Desde el servidor, con el alias `cluster` definido en `~/.ssh/config`:

```bash
# código y datos de las comunas a correr
rsync -a scripts manzanas cluster:censo_redondeo/
rsync -a cluster/ cluster:censo_redondeo/cluster/
# + resultados/pais/resumen_pais.csv, resultados/pais/modelos/modelo_<cut>.npz y resultados/pais/comunas/*_<cut>.parquet

# en el clúster: entorno con gurobipy 13, porque el servidor de tokens es de la versión 13
conda create -p ~/censo_redondeo/env python=3.12
~/censo_redondeo/env/bin/python -m pip install numpy scipy pandas numba pyarrow "gurobipy==13.0.*"

# lanzar el planificador
cd ~/censo_redondeo && cp cluster/job_*.sh . && mkdir -p logs
nohup env/bin/python cluster/planificador.py > cluster/planificador.out 2>&1 < /dev/null &
echo $! > cluster/planificador.pid
```

En el servidor, `scripts/sincronizar_cluster.py --cada 30` trae los resultados y agrega las filas a
`resultados/pais/redondeo/resumen_cembrano.csv`.

**Nota sobre la corrida de noviembre de 2026:** esa corrida se lanzó antes de ordenar el repositorio, así que en
`~/censo_redondeo` los scripts están en la raíz y no en `scripts/`. Por eso sus `job_*.sh` llaman a
`correr_cembrano.py` directamente.

## Seguimiento

```bash
tail -20 ~/censo_redondeo/cluster/planificador.log
squeue -u $USER
```

Para cancelar trabajos, usa los ids que aparecen en `estado.json`, no `scancel -u`.
