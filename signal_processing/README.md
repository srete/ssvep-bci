### Obrada signala

Postupak obrade signala prikazan je na dijagramu, a kod za obradu u skripti [signal_preprocessing.py](https://github.com/srete/ssvep-bci/blob/main/signal_processing/signal_processing.py)

![Postupak obrade signala](https://github.com/srete/ssvep-bci/blob/main/signal_processing/signal_processing.png)

Za određivanje frekvencije stimulusa korićena je CCA metoda koja je implementirana u skripti [cca.py](https://github.com/srete/ssvep-bci/blob/main/signal_processing/cca.py)

Folder [dataset](https://github.com/srete/ssvep-bci/tree/main/signal_processing/dataset) sadrži podatke koje nismo mi snimali, ali se mogu iskoristiti za dodatno testiranje. 
