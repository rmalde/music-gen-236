import glob
import random
import os
import sys
import argparse
from tqdm import tqdm

import torchaudio
import torch

from params import *

class Preprocessor():
    def __init__(self, raw_data_dir, preprocessed_data_dir):
        self.raw_data_dir = raw_data_dir
        self.preprocessed_data_dir = preprocessed_data_dir
        
    def preprocess(self):
        filenames = glob.glob1(self.raw_data_dir, "*.mp3")
        for i, filename in enumerate(tqdm(filenames)):
            signal = self.create_tensor()
            self.save_tensor(signal, filename)

    def create_tensor(self, filename):
        signal, sr = torchaudio.load(os.path.join(self.raw_data_dir, filename))
        if sr != SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
            signal = resampler(signal)
        signal = self.mix_down_if_necessary(signal)
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True).squeeze()
        return signal

    def save_tensor(self, signal, filename):
        torch.save(signal, os.path.join(self.preprocessed_data_dir, filename))




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", help="Dataset from: {[tr]ain, [d]ev, [te]st}", default="dev")

    args = parser.parse_args()
    print(args)

    dataset_type = None
    if args.dataset[0:2].lower() == "tr":
        dataset_type = "train"
    elif args.dataset[0].lower() == "d":
        dataset_type = "dev"
    elif args.dataset[0:2].lower() == "te":
        dataset_type = "test"
    else: sys.exit("Invalid dataset, use [tr]ain, [d]ev, or [te]st")

    raw_data_dir = os.path.join(RAW_DATA_DIR, dataset_type)
    preprocessed_data_dir = os.path.join(PREPROCESSED_DATA_DIR, dataset_type)





