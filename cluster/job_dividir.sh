#!/bin/bash
# Divide una comuna gigante en grupos de ~25 M celdas no nulas.  Uso: sbatch ... job_dividir.sh <cut>
#SBATCH --job-name=dividir
#SBATCH --output=logs/%x_%j.out
cd ~/censo_redondeo
env/bin/python scripts/correr_cembrano_grandes.py dividir "$1" --max-celdas 25e6
echo "$(date '+%F %T') fin dividir $1 rc $?"
