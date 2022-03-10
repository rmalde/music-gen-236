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
Usage: train.py [-h] [--model WaveGAN] [--epochs 20]
                [--epochs_per_save 10] [--batch_size 4]
                [--n_critic 5] [--phase_shuffle] [--spectral_norm]
                [--warmup] [--style_gan]

optional arguments:
   -h, --help           show this help message and exit
  --model MODEL         Model architecture [WaveGAN, TransGAN]. Default: WaveGAN
  --epochs EPOCHS       Training epochs. Default: 20
  --epochs_per_save     Save model every n epochs. Default: 10
  --batch_size          Batch size. Default: 4
  --n_critic            n disc updates for 1 gen update. Default: 5
  --phase_shuffle       Use phase shuffle. Default: False
  --spectral_norm       Use spectral norm. Default: False
  --warmup              Use warmup. Default: False
  --style_gan           Use AdaIN from StyleGAN. Default: False
```


## Evaluation
```
$ python test.py
```
Running test.py will calculate and print the evaluation metrics. Open notebook.ipynb to listen to generated audio samples from the generator. 
