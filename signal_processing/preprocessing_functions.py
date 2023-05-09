# Pre-processing functions

from scipy import signal

def remove_dc_offset(data):
    return data - data.mean()

def notch_filter(freq=50.0, fs=250, Q=50):
    return signal.iirnotch(freq, freq / Q, fs=fs)
    
def butter_filter(low=5.0, high=50.0, order=4, fs=250):
    nyq = fs / 2
    return signal.butter(order, [low / nyq, high / nyq], btype='bandpass')

def cheby_filter(low=5.0, high=50.0, order=1, fs=250, rp=1):
    nyq = fs / 2
    return signal.cheby1(order, rp, [low / nyq, high / nyq], btype='bandpass')

def butter_bandpass_filter(data, lowcut, highcut, sample_rate, order):
    '''
    Returns bandpass filtered data between the frequency ranges specified in the input.
    Args:
        data (numpy.ndarray): array of samples. 
        lowcut (float): lower cutoff frequency (Hz).
        highcut (float): lower cutoff frequency (Hz).
        sample_rate (float): sampling rate (Hz).
        order (int): order of the bandpass filter.
    Returns:
        (numpy.ndarray): bandpass filtered data.
    '''
    ""
    nyq = 0.5 * sample_rate
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y