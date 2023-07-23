## SSVEP detection using openBCI system

This project was done as a part of the course [Acqusition Systems for Electrophysiology](https://www.etf.bg.ac.rs/fis/karton_predmeta/13E053AES-2013) at the School of Electrical Engineering, University of Belgrade.

The main goal of the project was to measure steady-state visual evoked potentials using [OpenBCI system](https://docs.openbci.com/Ganglion/GanglionLanding/) and determine the frequency of the stimulus, in order to test the possibility of using SSVEP in simple *brain-computer* interfaces.

We shown our signal processing and classification pipeline on one subject in notebook [signal_processing_and_classificaiton_example.ipynb](https://github.com/srete/ssvep-bci/blob/main/signal_processing_and_classificaiton_example.ipynb).

The script for data acquisition and recorded data can be found in the folder [data_collection](https://github.com/srete/ssvep-bci/tree/main/data_collection), and all functions that we used for signal processing and CCA in the folder [signal_processing](https://github.com/srete/ssvep-bci/tree/main/signal_processing).

### How to run the project

First, you need to clone this repository. After cloning, create a new (conda) environment with the required libraries using the command

    conda create --name <env> --file requirements.txt

After that, activate the environment, and inside the folder `ssve-bci` run the command

    pip install -e .
