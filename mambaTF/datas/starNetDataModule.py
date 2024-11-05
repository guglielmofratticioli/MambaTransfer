
import os
import json
#from tkinter.tix import Tree
import numpy as np
#from typing import Any, Tuple
import soundfile as sf
import torch
#from pytorch_lightning import LightningDataModule
#from pytorch_lightning.core.sourceins import Hyperparameterssourcein
from torch.utils.data import ConcatDataset, DataLoader, Dataset
#from typing import Dict, Iterable, List, Iterator
#from rich import print
#from pytorch_lightning.utilities import rank_zero_only

import glob
import random
import argparse

def make_metadata_files():
    parser = argparse.ArgumentParser(description="Create metadata JSON files for the StarNet dataset.")
    parser.add_argument('--data_dir', type=str,default="/nas/home/gfraticcioli/datasets/starnet/audios", help='Directory containing the .wav files.')
    parser.add_argument('--output_dir', type=str,default="/nas/home/gfraticcioli/datasets/starnet/metadatas", help='Directory to save the metadata JSON files.')
    parser.add_argument('--train', type=float, default=0.8, help='Percentage of data to assign to the training set.')
    parser.add_argument('--val', type=float, default=0.1, help='Percentage of data to assign to the validation set.')
    parser.add_argument('--test', type=float, default=0.1, help='Percentage of data to assign to the test set.')
    args = parser.parse_args()

    # Check that the percentages sum to 1.0
    total_percent = args.train + args.val + args.test
    if not abs(total_percent - 1.0) < 1e-6:
        raise ValueError('Train, validation, and test percentages must sum to 1.0')

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.output_dir+'/train', exist_ok=True)
    os.makedirs(args.output_dir+'/val', exist_ok=True)
    os.makedirs(args.output_dir+'/test', exist_ok=True)

    N = 104
    files = list(range(1, N + 1))  # Replace this with your actual list of files

    train_num = int(round(N * args.train ))
    test_num = int(round(N * args.test ))
    val_num = N - train_num - test_num  # Ensure all files are assigned# Shuffle the files randomly to ensure a random distribution

    random.shuffle(files)

    # There are 6 different timbres
    timbres = [0 ,1, 2, 3, 4, 5]

    for timbre in timbres:
      
        # Find all .wav files for this timbre
        pattern = os.path.join(args.data_dir, f'*.{timbre}.wav')
        files = sorted(glob.glob(pattern))
        print(files)

        if not files:
            print(f"No files found for timbre {timbre}")
            continue

        # Split the files
        train_files = files[:train_num]
        test_files = files[train_num:train_num+test_num]
        val_files = files[train_num+test_num:]

        # Create JSON files
        train_json = os.path.join(args.output_dir+'/train', f'{timbre}.json')
        val_json = os.path.join(args.output_dir+'/val/', f'{timbre}.json')
        test_json = os.path.join(args.output_dir+'/test/', f'{timbre}.json')

        with open(train_json, 'w') as f:
            json.dump(train_files, f, indent=2)
        with open(val_json, 'w') as f:
            json.dump(val_files, f, indent=2)
        with open(test_json, 'w') as f:
            json.dump(test_files, f, indent=2)

        print(f"Created {train_json}, {val_json}, and {test_json} for timbre {timbre}")

def normalize_tensor_wav(wav_tensor, eps=1e-8, std=None):
    mean = wav_tensor.float().mean(-1, keepdim=True)
    if std is None:
        std = wav_tensor.float().std(-1, keepdim=True)
    return (wav_tensor - mean) / (std + eps)

class starNetDataset(Dataset):
    def __init__(
        self,
        json_dir: str = "",
        normalize_audio: bool = False,
        audio_only: bool = True,
        random_start: bool = True,
        sample_rate: int = 44100,
        #fps: int = 25,
        segment: float = 0.72,
        source_timbre: int = 0 ,
        target_timbre: int = 1

    ) -> None:
        super().__init__()
        self.EPS = 1e-8
        if json_dir == None:
            raise ValueError("JSON DIR is None!")
        
        self.json_dir = json_dir
        self.normalize_audio = normalize_audio
        self.audio_only = audio_only
        self.source_timbre = source_timbre
        self.target_timbre = target_timbre
        self.random_start = random_start
        self.seg_len = int(segment * sample_rate)
        #self.fps_len = int(segment * fps)


        source_json = os.path.join(json_dir, str(source_timbre)+".json")
        target_json = os.path.join(json_dir, str(target_timbre)+".json")

        with open(source_json, "r") as f:
            source_infos = json.load(f)

        with open(target_json, "r") as f:
            target_infos = json.load(f)

        self.sources = []
        self.targets = []
        for i in range(len(source_infos)):
            self.sources.append(source_infos[i])
            self.targets.append(target_infos[i])

            self.length = len(self.sources)

    def __len__(self):
        return self.length

    def __getitem__(self, index: int):

        if sf.info(self.sources[index]).frames == self.seg_len or not self.random_start:
            rand_start = 0
        else:
            rand_start = np.random.randint(0, sf.info(self.sources[index]).frames - self.seg_len)

        if not self.random_start:
            stop = None
        else:
            stop = rand_start + self.seg_len

        source, sample_rate = sf.read(
            self.sources[index], start=rand_start, stop=stop, dtype="int16"
        )
        target, sample_rate = sf.read(
            self.targets[index], start=rand_start, stop=stop, dtype="int16"
        )

        target = torch.from_numpy(target)
        source = torch.from_numpy(source)

        src_std = source.float().std(-1, keepdim=True)
        source = normalize_tensor_wav(source, eps=self.EPS, std=src_std)
        target = normalize_tensor_wav(target)

        return source, target, self.sources[index].split("/")[-1]
      #  return source, target.unsqueeze(0), self.sources[index].split("/")[-1]
    '''
        def preprocess_audio_only(self, idx: int):
            if self.n_src == 1:
                if self.source[idx][1] == self.seg_len or self.test:
                    rand_start = 0
                else:
                    rand_start = np.random.randint(0, self.source[idx][1] - self.seg_len)
                if self.test:
                    stop = None
                else:
                    stop = rand_start + self.seg_len
                # Load sourceture
                x, _ = sf.read(
                    self.source[idx][0], start=rand_start, stop=stop, dtype="int16"
                )
                # Load sources
                s, _ = sf.read(
                    self.sources[idx][0], start=rand_start, stop=stop, dtype="int16"
                )
                # torch from numpy
                target = torch.from_numpy(s)
                sourceture = torch.from_numpy(x)
                if self.normalize_audio:
                    m_std = sourceture.std(-1, keepdim=True)
                    sourceture = normalize_tensor_wav(sourceture, eps=self.EPS, std=m_std)
                    target = normalize_tensor_wav(target, eps=self.EPS, std=m_std)
                return sourceture, target.unsqueeze(0), self.source[idx][0].split("/")[-1]
            # import pdb; pdb.set_trace()
    '''

class starNetDataModule(object):
    def __init__(
        self,
        train_dir: str,
        valid_dir: str,
        test_dir: str,
        sample_rate: int = 44100,
        #fps: int = 25,
        segment: float = 0.72,
        normalize_audio: bool = False,
        batch_size: int = 1,
        num_workers: int = 0,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        audio_only: bool = True,
        source_timbre: int = 0,
        target_timbre: int = 1
    ) -> None:
        super().__init__()
        if train_dir == None or valid_dir == None or test_dir == None:
            raise ValueError("JSON DIR is None!")


        # this line allows to access init params with 'self.hparams' attribute
        self.train_dir = train_dir
        self.valid_dir = valid_dir
        self.test_dir = test_dir
        self.sample_rate = sample_rate
        #self.fps = fps
        self.segment = segment
        self.normalize_audio = normalize_audio
        self.batch_size = batch_size

        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.audio_only = audio_only

        self.source_timbre = source_timbre
        self.target_timbre = target_timbre

        self.data_train: Dataset = None
        self.data_val: Dataset = None
        self.data_test: Dataset = None

    def setup(self) -> None:
        self.data_train = starNetDataset(
            json_dir=self.train_dir,
            sample_rate=self.sample_rate,
            #fps=self.fps,
            segment=self.segment,
            normalize_audio=self.normalize_audio,
            audio_only=self.audio_only,
            source_timbre=self.source_timbre,
            target_timbre=self.target_timbre
        )
        self.data_val = starNetDataset(
            json_dir=self.valid_dir,
            sample_rate=self.sample_rate,
            #fps=self.fps,
            segment=self.segment,
            normalize_audio=self.normalize_audio,
            audio_only=self.audio_only,
            source_timbre=self.source_timbre,
            target_timbre=self.target_timbre
        )
        self.data_test = starNetDataset(
            json_dir=self.test_dir,
            sample_rate=self.sample_rate,
            #fps=self.fps,
            segment=self.segment,
            normalize_audio=self.normalize_audio,
            audio_only=self.audio_only,
            source_timbre=self.source_timbre,
            target_timbre=self.target_timbre
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_val,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_test,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    @property
    def make_loader(self):
        return self.train_dataloader(), self.val_dataloader(), self.test_dataloader()

    @property
    def make_sets(self):
        return self.data_train, self.data_val, self.data_test

if __name__ == '__main__':
    make_metadata_files()