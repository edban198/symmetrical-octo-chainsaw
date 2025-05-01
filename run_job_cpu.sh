#!/bin/bash

# Instructing SLURM to locate and assign
#X number of nodes with Y number of
#cores in each node.
# X,Y are integers. Refer to table for various combinations
#SBATCH -N 1
#SBATCH -c 64
#SBATCH -p cpu
#SBATCH --qos=short
#SBATCH -t 18:00:00

#SBATCH --job-name=RB_cpu_simulation

#SBATCH -o RB_cpu_sim.out
#SBATCH -e RB_cpu_sim.err

#SBATCH --mail-type=ALL
#SBATCH --mail-user sfbj55@durham.ac.uk

# Run the program
cd

# Run the program
./julia-1.11.2/bin/julia ./symmetrical-octo-chainsaw/dipole_shallow_water.jl