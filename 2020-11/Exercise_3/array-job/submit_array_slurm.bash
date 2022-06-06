#!/bin/bash

##### Use outside training ########
#SBATCH --partition=debug
##### Use during training #########
##SBATCH --partition=reservation
##SBATCH --reservation=HPC-training
###################################

#SBATCH -J exercise3-array
#SBATCH --output=dmtcp_array_%A_%a.out
#SBATCH --error=dmtcp_array_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --array=1-10%1 #run 10 jobs, 1 at a time 

#Load the DMTCP module:
module load dmtcp/2.6.0

#! The variable $SLURM_ARRAY_TASK_ID contains the array index for each job (values 1-10).
#! In this example, each job will be passed its index, so each output file will contain a different value
echo "This is job" $SLURM_ARRAY_TASK_ID

#! Each array task will run in a seperate $jobDir directory:
#! Note this is optional !
#jobDir=Job_$SLURM_ARRAY_TASK_ID
#mkdir $jobDir
#cd $jobDir

#! the RESTARTSCRIPT variable holds the restart script, which only gets created when checkpointing occured:
RESTARTSCRIPT="dmtcp_restart_script.sh"
#! set level of logging output from DMTCP:
export DMTCP_QUIET=2

#! Define the command to be run by setting variable $runcmd:
runcmd="../example_array 5"
#! Define the checkpointing time interval (in seconds):
tint=30

#! Activate the dmtcp deamon (background program) with dmtcp_coordinator. Coordinates checkpoints between multiple processes. 
echo "Start coordinator"
date
#! set flags:
# --daemon : Run silently in the background
# --coord-logfile : Coordinator will dump its logs to the given file 'dmtcp_log.txt'
# --exit-after-ckpt : Kill peer processes of computation after first checkpoint is created
# --exit-on-last : Exit automatically when last client disconnects
# -i : Time in seconds between automatic checkpoints (set to $tint)
# --port-file : File to write listener port number. Useful with '--port 0', which is used to assign a random port.
# -p : Port to listen on (default: 7779) 
eval "dmtcp_coordinator --daemon --coord-logfile dmtcp_log.txt --exit-after-ckpt --exit-on-last -i "$tint" --port-file cport.txt -p 0"
## Allow for 2-second break to allow dmtcp_coordinator to start.
sleep 2
# Capture the port file and hostname for logging:
cport=$(<cport.txt)
echo "$cport"
h=`hostname`
echo $h

## Here we check if checkpointing has occured before, and use the appropriate commands accordingly:
#! If a restart script ($RESTARTSCRIPT) exists, dmtcp uses the dmtcp_restart program.
if [ -f "$RESTARTSCRIPT" ]
then
    echo "Resume the application"
    #! Restart processes from a checkpoint image 'ckpt*.dmtcp':
    # flags:
    # -p : Port where dmtcp_coordinator is run (default: 7779)
    # -i : Time in seconds between automatic checkpoints. set to $tint
    CMD="dmtcp_restart -p "$cport" -i "$tint" ckpt*.dmtcp"
    # Print command:
    echo $CMD
    # Run command:
    eval $CMD
else
    echo "Start the application"
    #! Start a process under DMTCP control. Connect to the DMTCP Coordinator and run command $runcmd.
    # flags:
    # --rm : Enable support for resource managers (Torque PBS and SLURM).
    # --no-gzip : Enable/disable compression of checkpoint images. WARNING:  gzip adds seconds.  Without gzip, ckpt is often < 1 s.
    # -h : Hostname where dmtcp_coordinator is run (default: localhost)
    # -p : Port where dmtcp_coordinator is run (default: 7779)
    CMD="dmtcp_launch --rm --no-gzip -h localhost -p "$cport" "$runcmd
    # Print command:
    echo $CMD
    # Run command:
    eval $CMD
fi

echo "Stopped program execution"
date
