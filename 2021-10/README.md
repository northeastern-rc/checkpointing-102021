# checkpointing-102021
Using checkpointing for resilient workflows on Discovery - the October 2021 session material.

## What's in this training?
* Training PowerPoint slides "Using_checkpointing_for_resilient_workflows.pptx".
* Exercises - practical examples on how to implement checkpointing using different tools and techniques:
  * Exercise 1 - User-level checkpointing example, using Python, Pickle and Slurm job arrays.
  * Exercise 2 - Application-level checkpointing, using a TensorFlow checkpointing example with Slurm job arrays.
  * Exercise 3 - System-level checkpointing, using [DMTCP](https://dmtcp.sourceforge.io/).

## Overview:
* Understand the importance of fault tolerance in HPC workloads.
* Learn about different fault tolerance approaches (Checkpointing).
* Gain practice on how to design resilient workflows using SLURM.
* Familiarize with some existing checkpointing techniques in Python.
* Introduce system-level checkpointing using DMTCP.

## Steps to use these training scripts on Discovery:
1. Login to a Discovery shell or use the [Discovery OnDemand interface](https://rc-docs.northeastern.edu/en/latest/first_steps/connect_ood.html).
2. Enter your desired directory within Discovery and download the training material. For example:
```bash
cd $HOME
git clone git@github.com:NURC-Training/checkpointing-102021.git
cd checkpointing-102021
```
3. Download the training slides to your local computer, where you have access to PowerPoint to open the slides. Follow the slides to execute the different scripts. 
4. Example:
```bash
cd Exercise_1
sbatch submit.bash 
```
