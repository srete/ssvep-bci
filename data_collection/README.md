### Data collection

We used Python scipt [acqusition.py](https://github.com/srete/ssvep-bci/blob/main/data_collection/acquisition.py) for data collection. Recorded data could be found in the folder [recorded_data](https://github.com/srete/ssvep-bci/tree/main/data_collection/recorded_data).

In the folder [GUI](https://github.com/srete/ssvep-bci/tree/main/data_collection/GUI), you can find Python application that we wrote for SSVEP stimulation. Our application was not giving good results, so we decided to use web application [Quick SSVEP](https://omids.github.io/quickssvep/) during the experiment. We suppose the problem with our application could be coused by used libraries for GUI and threading, or some delay in other Python funcitons.

For recording we used [OpenBCI Ganglion Board](https://docs.openbci.com/Ganglion/GanglionLanding/). EEG electrodes were connected to four analog inputs and placed on the subject according to the 10-20 system, at positions Oz, O1, O2 and POz. These positions are located on the occipital lobe, which corresponds to the place where SSVEP usually occurs. The electrodes were placed according to the instructions available in the OpenBCI [documentation](https://docs.openbci.com/GettingStarted/Biosensing-Setups/EEGSetup/).

We done two experiment setups
- In the first setup, the stimulus was a single white charachter on a black background, which was flickering at a certain frequency. The subject was instructed to look at the stimulus and try to focus on it. The recording was done for 20 seconds.
- In the second setup, four characters were shown on the screen, each flickering at a different frequency. The subject was instructed to look an specific charachter. The recording was done for 20 seconds for one charachter, and then the subject was instructed to look at another charachter and the recording was done again. This was repeated for all four charachters.

![Subject during the experiment](https://github.com/srete/ssvep-bci/blob/main/data_collection/subject_during_experiment.jpg)

There was a limit for frequencies that we could use, because the refresh rate of the monitor was 60 Hz, so we could show only frequencies in form of 60/n, where n is an integer >= 2. We used the following frequencies for the stimulus: 7.5, 8.57, 12, 15 and 30 Hz.

The shape of recorded data is (n_blink_freqs, n_trial, n_chanels, n_samples), where n_blink_freqs is the number of frequencies used for the stimulus, n_trial is the number of trials, n_chanels is the number of EEG electrodes and n_samples is the number of samples recorded for each trial.