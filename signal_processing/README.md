### Signal processing

Signal processing pipline is shown in the diagran below and the code is available in [signal_preprocessing.py](https://github.com/srete/ssvep-bci/blob/main/signal_processing/signal_processing.py)

First, we removed the DC component from the signal and then applied notch filter to remove the power line noise. After that, we applied Chabyshev 4th order bandpass filter, to extract the signal in the frequency corresponding to the stimulus.

We used CCA method to determine the frequency of the stimulus. The implementation of CCA method is available in [cca.py](https://github.com/srete/ssvep-bci/blob/main/signal_processing/cca.py)

![Signal processing pipline](https://github.com/srete/ssvep-bci/blob/main/signal_processing/signal_processing_pipeline.png)