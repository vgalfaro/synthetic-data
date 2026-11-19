#!/bin/bash
# Un grupo de una comuna dividida.  Uso: sbatch ... job_grupo.sh <cut> <g>
#SBATCH --job-name=grupo
#SBATCH --output=logs/%x_%j.out
cd ~/censo_redondeo
export GRB_LICENSE_FILE=/home/GUROBI.LIC/gurobi.lic
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "$(date '+%F %T') inicio $1 grupo $2 en $(hostname)"
env/bin/python scripts/correr_cembrano_grandes.py grupo "$1" "$2" --hilos "$SLURM_CPUS_PER_TASK"
echo "$(date '+%F %T') fin $1 grupo $2 rc $?"
