# Generating Polyphonic Music with Adversarial Networks
Final Project for CS 236G: General Adversarial Networks. Using DCGAN's and approaches from StyleGAN to generate raw audio of EDM music. 

## Data
For the project, I trained on the Beatport EDM Dataset, which can be found here: [Beatport EDM Key Dataset | Zenodo](https://zenodo.org/record/1101082#.Yg7VIojMJEZ)
The models can be trained on any dataset however, as long the as the files are in a common sound format, like mp3 or wav. 
#### Preparing Data
To configure your dataset, split the data into train, test, and dev folders in a data/ directory.
To prepare and save the data into tensor format (for faster startup at training), run the following command for one of train, dev, or test:
```
$ python dataset.py -d [[tr]ain, [d]ev, [t]est]
```
To verify that the data is saved, you can run
```
$ python dataset.py -d [[tr]ain, [d]ev, [t]est] --load
```

## Training
```
$ python train.py
```
TODO: Add command line options for training different combinations of models

## Evaluation
```
$ python test.py
```
Running test.py will calculate and print the evaluation metrics. Open notebook.ipynb to listen to generated audio samples from the generator. 
