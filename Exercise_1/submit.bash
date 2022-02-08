#!/bin/bash
#SBATCH --partition=debug
#SBATCH -J exercise1
#SBATCH --time=00:02:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=%A-%a.out
#SBATCH --error=%A-%a.err
#SBATCH --array=1-10%1	#execute 10 array jobs, 1 at a time.

#Clean your env first:
conda deactivate
module purge

#Load the conda environment:
module load discovery miniconda3/2020-09

# run the python code, save all output to a log file corresponding the the current job task that is running:
python -u vector_checkpointing.py &> log.$SLURM_ARRAY_TASK_ID 
