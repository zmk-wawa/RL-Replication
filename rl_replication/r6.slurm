#!/bin/bash
#SBATCH --job=200_test
#SBATCH --partition=kshdexclu04
## SBATCH --exclude=./black_list
#SBATCH --mem=90G
#SBATCH --exclusive
#SBATCH --ntasks-per-node=1
## SBATCH --cpus-per-task=8


#SBATCH -N 1
#SBATCH --gres=dcu:4

export UCX_TLS=sm,rc_x

date


mpirun --bind-to none --mca pml ucx --mca osc ucx  -np 1 sh /public/home/ssct005t/project/zmk/main/200/test/run6.sh

date