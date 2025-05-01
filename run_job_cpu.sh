#!/bin/bash

# Instructing SLURM to locate and assign
#X number of nodes with Y number of
#cores in each node.
# X,Y are integers. Refer to table for various combinations
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=28G
#SBATCH --gres=gpu:pascal:1
#SBATCH -p ug-gpu-small
#SBATCH --qos=short
#SBATCH -t 08:00:00

#SBATCH --job-name=gpu_diffusion_sim

#SBATCH -o RB_gpu_sim.out
#SBATCH -e RB_gpu_sim.err

#SBATCH --mail-type=ALL
#SBATCH --mail-user sfbj55@durham.ac.uk

# Run the program
cd

# Run the program
./julia-1.11.2/bin/julia ./CODE/symmetrical-octo-chainsaw/dipole_shallow_water.jl