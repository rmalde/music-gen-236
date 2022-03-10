import torch
from torch.utils.data import Dataset
import torchaudio

import glob
import random
import os
import sys
import argparse
from tqdm import tqdm

from params import *


class SongDataset(Dataset):
    '''
    Constructs torch.utils.data.Dataset object for use in training and testing
    '''
    def __init__(self, clip_len=SLICE_LEN, data_path= None, save_path=None, load_path=None):
        '''
        Initializes dataset object. Cmdline Options:
        --load:
            Load saved dataset from train, dev, test, as specified
        else (default):
            Construct dataset from raw audio files as specified
            Save dataset to file as specified
        '''
        if load_path:
            self.sequences = torch.load(load_path)
            self.sequences_len = self.sequences.shape[0]
            print(f"Loaded dataset from {load_path} successfully.")
        else:
            filenames = glob.glob1(data_path, "*.mp3")
            sequences = []
            for i, filename in enumerate(tqdm(filenames)):
                signal, sr = torchaudio.load(os.path.join(data_path, filename))

                signal = self._resample_if_necessary(signal, sr)
                signal = self._mix_down_if_necessary(signal)

                audio_length = signal.shape[1]
                if audio_length > clip_len:
                    for _ in range(4):  #grab 4 samples from each song
                        offset = random.randint(0, audio_length-clip_len)
                        clipped = signal[:,offset:(offset+clip_len)]
                        sequences.append(clipped)
                if (i+1) % 100 == 0:
                    sequences = [torch.cat(sequences)] #cat in batches to save memory
                    
            sequences = torch.cat(sequences)
                
            self.sequences = sequences
            self.sequences_len = sequences.shape[0]
            print("Finished constructing dataset.")
            torch.save(sequences, save_path)
            print(f"Saved dataset to {save_path}")
    
    def __len__(self):
        return self.sequences_len
    
    def __getitem__(self, index):
        return self.sequences[index:index+1] #do this to retain the dim shape, [1, 64000]

    def _resample_if_necessary(self, signal, sr):
        if sr != SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
            signal = resampler(signal)
        return signal
    
    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

#for testing
if __name__ == '__main__':
    '''
    usage: dataset.py [-h] [-d train] [--load]

    optional arguments:
        -h, --help          show this help message and exit
        -d, --dataset       Dataset from: {[tr]ain, [d]ev, [te]st}. Default: dev
        --load              load dataset. Default: False
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", help="Dataset from: {[tr]ain, [d]ev, [te]st}", default="dev")
    parser.add_argument("--load", help="load dataset", action="store_true")

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

    if args.load:
        dataset = SongDataset(
            load_path=os.path.join(PREPROCESSED_DATA_DIR, dataset_type + ".pt")
        )
    else:
        dataset = SongDataset(
            data_path=os.path.join(RAW_DATA_DIR, dataset_type),
            save_path=os.path.join(PREPROCESSED_DATA_DIR, dataset_type + ".pt")
        )

    print("Dataset info: ")
    print("Length:", len(dataset))
    print("First entry:", dataset[0])
    print("First entry shape:", dataset[0].shape)
