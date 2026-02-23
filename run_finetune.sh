#!/bin/bash
#SBATCH --job-name=patentsbert_finetune
#SBATCH --output=finetune_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

source /ceph/home/student.aau.dk/as58zr/M4_assignments/.venv/bin/activate
cd /ceph/home/student.aau.dk/as58zr/M4_assignments
python3 finetune.py