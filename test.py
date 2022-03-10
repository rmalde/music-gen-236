import torch

import os 

from params import *
from utils import *
from dataset import SongDataset

import crepe

import numpy as np


class Tester():
    '''Testing framework to test trained model'''
    def __init__(self, generator, discriminator, dataset=None, device="cpu"):
        self.generator = generator.to(device)
        self.discriminator = discriminator
        self.dataset = dataset
        self.device=device

        self.generator.eval()
        self.discriminator.eval()


    def gen_n_samples(self, n_samples):
        noise = sample_noise(n_samples, NOISE_DIM).to(self.device)
        fake = self.generator(noise)
        return fake

    def get_pitch_sequence(self, sequence, sr):
        time, frequency, confidence, activation = crepe.predict(sequence, sr)
        return frequency

    def test(self):
        '''Prints evaluation metrics of average and std pitches'''
        samples = self.gen_n_samples(1)
        average_pitches = []
        std_pitches = []
        for sample in samples:
            sample = sample[0].cpu().detach().numpy()
            frequency = np.log10(self.get_pitch_sequence(sample, SAMPLE_RATE))
            average_pitches.append(np.mean(frequency))
            std_pitches.append(np.std(frequency))
        
        print("Average pitch:", np.mean(average_pitches))
        print("Std pitch:", np.mean(std_pitches))

if __name__ == "__main__":
    #TODO: Add argparse
    device = get_device()

    # dataset = SongDataset(load_path=os.path.join(PREPROCESSED_DATA_DIR, "train.pt"))

    generator, discriminator = load_gen_and_disc(os.path.join(MODEL_OUTPUT_DIR, "WaveGAN-5.pt"))
    print("loaded gen and disc")
    tester = Tester(generator, discriminator, device=device)
    print("testing")
    tester.test()
    






