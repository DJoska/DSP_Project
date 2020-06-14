# EEE4114F Project - Instrument Recognition Using Machine Learning

Written by JSKDAN001 and MSSRIC004

To run training:

- Place all .wav files you want to train off in the /samples folder in the working directory
- Place the corresponding examples.json file in the same /samples folder
- Run training.py

You should see the program begin to extract features from each file. Once finished, the scikit-learn KNN algorithm runs on that data.