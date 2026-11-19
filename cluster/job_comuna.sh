#!/bin/bash
# Una comuna entera: vértice LP (Gurobi) + Algoritmo 1, configuraciones A y B.  Uso: sbatch ... job_comuna.sh <cut>
#SBATCH --job-name=cembrano
#SBATCH --output=logs/%x_%j.out
cd ~/censo_redondeo
export GRB_LICENSE_FILE=/home/GUROBI.LIC/gurobi.lic
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "$(date '+%F %T') inicio comuna $1 en $(hostname), $SLURM_CPUS_PER_TASK cpus, ${SLURM_MEM_PER_NODE} MB"
env/bin/python scripts/correr_cembrano.py "$1" --hilos "$SLURM_CPUS_PER_TASK"
echo "$(date '+%F %T') fin comuna $1 rc $?"
