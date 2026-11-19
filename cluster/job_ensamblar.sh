#!/bin/bash
# Ensambla una comuna gigante a partir de sus grupos.  Uso: sbatch ... job_ensamblar.sh <cut>
#SBATCH --job-name=ensamblar
#SBATCH --output=logs/%x_%j.out
cd ~/censo_redondeo
env/bin/python scripts/correr_cembrano_grandes.py ensamblar "$1"
echo "$(date '+%F %T') fin ensamblar $1 rc $?"
