### Prikupljanje podataka
Za snimanje podataka se koristi skripta [acqusition.py](https://github.com/srete/ssvep-bci/blob/main/data_collection/acquisition.py), a snimljeni podaci su u folderu [recorded_data](https://github.com/srete/ssvep-bci/tree/main/data_collection/recorded_data).

Folder [GUI](https://github.com/srete/ssvep-bci/tree/main/data_collection/GUI) sadži Python aplikaciju za SSVEP stimulaciju, ali ona nije davala rezultate,  pa je tokom snimanja korišćena web aplikacija [Quick SSVEP](https://omids.github.io/quickssvep/).

Za snimanje je korišćen OpenBCI [Ganglion Board](https://docs.openbci.com/Ganglion/GanglionLanding/). Elektrode za EEG povezane su na četiri analogna ulaza i postavljene na ispitanika po 10-20 sistemu, na pozicijama Oz, O1, O2 i POz. Ove pozicije nalaze se na okcipitalnom režnju, što odgovara mestu na kome se obično javlja SSVEP.  Elektrode su posatljane po upustvu koje je dostupno u OpenBCI [dokumentaciji](https://docs.openbci.com/GettingStarted/Biosensing-Setups/EEGSetup/).

U okviru eksperimenta ispitanicima su prikazivani stimulusi čija je frekvencija 7.5, 8.57, 12, 15 i 30 Hz.
