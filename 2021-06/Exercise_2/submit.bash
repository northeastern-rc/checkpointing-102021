#!/bin/bash

##### Use outside training ########
#SBATCH --partition=gpu
##### Use during training #########
##SBATCH --partition=reservation
##SBATCH --reservation=bootcamp2021v100
###################################
#SBATCH -J exercise2
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=10Gb
#SBATCH --output=%A-%a.out
#SBATCH --error=%A-%a.err
#SBATCH --array=1-10%1	#execute 10 array jobs, 1 at a time.

#Load the conda environment:
module load miniconda3/2020-09
source activate tf_gpu

##Define the number of steps based on the job id:
numOfSteps=$(( 500 * SLURM_ARRAY_TASK_ID ))

# run the python code, save all output to a log file corresponding the the current job task that is running:
python train_with_checkpoints.py $numOfSteps &> log.$SLURM_ARRAY_TASK_ID 

