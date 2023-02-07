# Checkpointing on Discovery
This repository contains material for Checkpointing on Discovery
training modules.

The contents of this repo are described in greater details on HPC 
[Read the Docs](https://rc-docs.northeastern.edu/en/latest/best-practices/checkpointing.html).

## Repo Contents
* `Exercise_N`: `N` exercises, each containing a different type of checkpointing
  * `*checkpointing.py`: sample Python script showing how checkpointing is done.
  * `submit.bash`: a bash script to run the tutorial (i.e., Python script)
  *  `interactive_commad_list`: a list of commands to run to interactively execute the respective exercise.

Specifically, the contents are as follows:
```
training-checkpointing
├── Exercise_1                      # Python Pickle Checkpoints
│   ├── submit.bash
│   └── vector_checkpointing.py
├── Exercise_2                      # TensorFlow Model Checkpoints
│   ├── interactive_commad_list
│   ├── restore_from_checkpoint.py
│   ├── submit.bash
│   └── train_with_checkpoints.py
├── Exercise_3                      # Slurm arrays, MPI, and Serial-Job Checkpointing
│   ├── array-job
│   │   ├── example.cpp
│   │   ├── example_array
│   │   └── submit_array_slurm.bash
│   ├── mpi-job
│   │   ├── example_mpi
│   │   ├── example_mpi.cpp
│   │   └── submit_mpi_slurm.bash
│   └── serial-job
│       ├── example.cpp
│       ├── example_serial
│       └── submit_serial_slurm.bash
├── Exercise_4                      # PyTorch Model Checkpoints
│   ├── submit.bash
│   ├── train_with_checkpoints.py
└── README.md
```

## Steps to download and use the repo on Discovery
1. Login to a Discovery shell or use the [Discovery OnDemand interface](https://rc-docs.northeastern.edu/en/latest/first_steps/connect_ood.html).

2. Enter your desired directory within Discovery and download the training material. For example:
    ```bash
    cd $HOME
    git clone git@github.com:northeastern-rc/training-checkpointing.git
    cd training-checkpointing
    ```
3. Download the training slides to your local computer and open via PowerPoint. Follow the slides and execute the 
different examples.

## To Do
-[ ] Add links to each tutorial
-[ ] Reformat all code using consistent formatting
-[ ] Add README to each exercise 
-[ ] Demo for Scikit-learn
-[ ] Demo for PyTorch Lightning 
-[ ] Point to `/datasets/` for tutorial data, opposed to downloading for each user.