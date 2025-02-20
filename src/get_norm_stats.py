# -*- coding: utf-8 -*-
# @Time    : 8/4/21 4:30 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : get_norm_stats.py

# this is a sample code of how to get normalization stats for input spectrogram

import torch
import numpy as np

import dataloader
import argparse




def main(datafile, label_csv, batch_size, num_workers):
    audio_conf = {'num_mel_bins': 128, 'target_length': 1100, 'freqm': 0, 'timem': 0, 'mixup': 0.0, 'skip_norm': True, 'mode': 'train', 'dataset': 'blabla'}

    train_loader = torch.utils.data.DataLoader(
        dataloader.AudiosetDataset(datafile, label_csv=label_csv, audio_conf=audio_conf), 
        batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    mean = []
    std = []
    for i, (audio_input, labels) in enumerate(train_loader):
        cur_mean = torch.mean(audio_input)
        cur_std = torch.std(audio_input)
        mean.append(cur_mean)
        std.append(cur_std)
    print(f"Mean: {np.mean(mean)}")
    print(f"Std: {np.mean(std)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get normalization stats for input spectrogram")
    parser.add_argument('--datafile', type=str, required=True, help='Path to the datafile')
    parser.add_argument('--label_csv', type=str, required=True, help='Path to the label CSV file')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for DataLoader')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for DataLoader')
    
    args = parser.parse_args()
    main(args.datafile, args.label_csv, args.batch_size, args.num_workers)