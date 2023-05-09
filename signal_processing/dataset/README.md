# Dataset

**Podaci preuzeti iz projekta https://github.com/NTX-McGill/NeuroTechX-McGill-2021/tree/main**


"
The data collection interface cued the participant to look at a specific key on the keyboard by highlighting it red. All keys would then flash at their unique frequencies and phases for five seconds. **This process iterated randomly through each key on the keyboard with a 200 ms pause in between each one, completing one block of data collection.** Data are saved to an Amazon Web Services database in the cloud, which provides easy access to the data from anywhere in the world, as well as scalability to our software.

We experimented with different frequency configurations, finding that there was a range of frequencies that produced more easily identifiable SSVEP signals. Our final configuration consists of frequencies from *6.00–12.90 Hz incremented by 0.23 Hz, with phases starting at 0 and increasing by 0.35 radians. We mapped each unique frequency-phase pair to a key in our speller, making sure that keys that were physically close to each other would flash at frequencies that are more than 1 Hz apart.
"


## Accessing the raw data

### Dependencies

* [Python](https://docs.anaconda.com/anaconda/install/index.html) 3.7.6 or later
* [NumPy](https://numpy.org/install/) 1.18.1 or later

Raw EEG data streamed from the OpenBCI GUI are available for 9 participants. EEG data and metadata are stored as Python dictionaries that have been serialized using the `pickle` module. Each `.pkl` file contains a dictionary with the following key-value pairs:

- `data`: 4-dimensional `numpy` array containing the EEG data. The array shape is: `(n_channels, n_samples, n_characters, n_blocks)`. For example, if a participant did 10 blocks of data collection using our 31-character keyboard with an 8-channel EEG device with a sampling frequency of 250 Hz, and if the stimulation duration is 5 seconds, the shape of the data will be `(8, 1250, 31, 10)`.
- `freq_type`: frequency configuration (`A`, `B`, `C`, or `D`). See `data_collection_notes.md` for details about the frequency-phase pairs used in each configuration.
- `freqs`: all frequencies (in Hz) associated with this frequency configuration, sorted in ascending order.
- `chars`: all characters associated with the frequencies in `freqs` (same order).

Each channel always corresponds to the same 10/20 electrode position:
- Channel 1: PO7
- Channel 2: PO3
- Channel 3: O1
- Channel 4: POZ
- Channel 5: OZ (best)
- Channel 6: PO4
- Channel 7: O2
- Channel 8: PO8

Here is an example showing how pickled data can be loaded in Python:

```{python}
import pickle

# this file contains all blocks collected from participant S02 with frequency configuration A
# path assumes working directory is the data directory
path_data = 'raw/S02/S02_typeA.pkl' 

with open(path_data, 'rb') as file:
    data = pickle.load(file)

print(data.keys()) # dict_keys(['data', 'freq_type', 'freqs', 'chars'])
```