#!/bin/bash
#SBATCH --partition=gpu		#Set partition to "gpu"
#SBATCH -J exercise2		#Set job name
#SBATCH --time=00:10:00		#Set job time limit
#SBATCH --nodes=1		#Set number of nodes to 1
#SBATCH --ntasks=1		#Set number of cpus to 1
#SBATCH --gres=gpu:1		#Request 1 GPU
#SBATCH --mem=10Gb		#Request 10GB of RAM memory
#SBATCH --output=%A-%a.out      #set output filename with main job ID and task array ID
#SBATCH --error=%A-%a.err       #set error filename with main job ID and task array ID
#SBATCH --array=1-10%1		#execute 10 array jobs, 1 at a time.

#Load the conda environment:
module load anaconda3/2021.05
source activate pytorch_env

##Define the number of steps based on the job id:
numOfSteps=$(( 500 * SLURM_ARRAY_TASK_ID ))

# run the python code, save all output to a log file corresponding the the current job task that is running:
python train_with_checkpoints.py $numOfSteps &> log.$SLURM_ARRAY_TASK_ID

