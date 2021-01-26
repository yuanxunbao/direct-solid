#!/bin/bash
#----------------------------------------------------
# Sample Slurm job script
#   for TACC Longhorn v100 nodes
#
#   *** Single Serial Job in v100 Queue ***
#
# Notes:
#
#   -- Copy/edit this script as desired.  Launch by executing
#      "sbatch sample.slurm" on a Longhorn login node.
#
#   -- Serial codes run on a single node (upper case N = 1).
#        A serial code ignores the value of lower case n,
#        but slurm needs a plausible value to schedule the job.
#----------------------------------------------------
#SBATCH -J myjob           # Job name
#SBATCH -o myjob.o%j       # Name of stdout output file
#SBATCH -e myjob.e%j       # Name of stderr error file
#SBATCH -p v100-lm            # Queue (partition) name
#SBATCH -N 1               # Total # of nodes (must be 1 for serial)
#SBATCH -n 4               # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 03:00:00        # Run time (hh:mm:ss)
#SBATCH --mail-type=all    # Send email at begin and end of job
#SBATCH --mail-user=ygqin@utexas.edu
##SBATCH -A myproject       # Allocation name (req'd if you have more than 1)

# Other commands must follow all #SBATCH directives...
conda activate mpi_py
export MY_SPECTRUM_OPTIONS="--gpu --aff on" 
module load launcher_gpu
module list
pwd
date

export LAUNCHER_PLUGIN_DIR=$LAUNCHER_DIR/plugins
export LAUNCHER_RMI=SLURM
export LAUNCHER_JOB_FILE=testjob
#ibrun -n 4 python3 DNS_NBRS.py AM_deep AM_deep.mat 100
ibrun -n 4 python3 DNS_NBRS_nogap.py Weld_shallow WD_shallow.mat 100 1
#ibrun -n 4 python3 DNS_NBRS.py AM_deep AM_deep.mat 100 6
#ibrun -n 4 python3 DNS_NBRS.py AM_shallow AM_shallow.mat 100 5
#$LAUNCHER_DIR/paramrun
