#!/bin/bash
#SBATCH --partition=debug	#set partition to "debug"
#SBATCH -J exercise1		#set job name
#SBATCH --time=00:02:00		#set job time limit to 2 minutes
#SBATCH --nodes=1		#set node number to 1
#SBATCH --ntasks=1		#set tasks number to 1 (cpus)
#SBATCH --output=%A-%a.out      #set output filename with main job ID and task array ID
#SBATCH --error=%A-%a.err       #set error filename with main job ID and task array ID
#SBATCH --array=1-10%1		#execute 10 array jobs, 1 at a time.

#Clean your env first:
## Deactivate your existing conda environment - uncomment the below line if you have a conda environemnt automatically loaded through your ~/.bashrc
#conda deactivate
module purge

#Load the conda environment:
module load discovery miniconda3/2020-09

# run the python code, save all output to a log file corresponding the the current job task that is running:
python -u vector_checkpointing.py &> log.$SLURM_ARRAY_TASK_ID 

