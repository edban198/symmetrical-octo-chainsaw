#!/bin/bash
# This line is required to inform the Linux command line to parse the script using the bash shell

# Instructing SLURM to locate and assign X number of nodes with Y number of cores in each node.
# X,Y are integers. Refer to table for various combinations. X will almost always be 1.
#SBATCH -N 1
#SBATCH -c 8

# Governs the run time limit and resource limit for the job. 
# Please pick values from the partition and QOS tables below for various combinations
#SBATCH -p cpu
#SBATCH --qos=short
#SBATCH -t 01:00:00

#SBATCH --job-name=dipole_cpu_simulation

#SBATCH --mail-type=ALL
#SBATCH --mail-user sfbj55@durham.ac.uk

# Run the program
./julia-1.11.2/bin/julia ./dipole_shallow_water.jl