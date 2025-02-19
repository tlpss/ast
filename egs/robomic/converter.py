import torchaudio.transforms as T
import numpy as np
import scipy.signal as scpysignal
import torch
import os
import csv
from datetime import datetime
import pickle
import torchaudio

class SpectrogramCalculator:
    def __init__(self, data_directory, file_name):
        self.data_directory = data_directory
        self.file_name = file_name

        self.mel_transform = T.MelSpectrogram(sample_rate=16000, n_fft=400, win_length=400, hop_length=160, n_mels=64, normalized=False)
        data = self.load_time_signal_from_csv(data_directory=data_directory, file_name=file_name)
        resampled_data = self.resample_time_signal(data=data, f_original=self.fs, f_new=16000)
        self.data = data
        self.spectrogram = self.calc_spectrogram(resampled_data)

    def load_time_signal_from_csv(self, data_directory, file_name):
        data = []
        time_axis = []
        timestamps = []
        i = 0
        first = True
        with open(os.path.join(data_directory, f'{file_name}.csv'), 'r', newline='') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=';')
            header = next(csv_reader)
            for row in csv_reader:
                timestamps.append(row[0])
                
                timestamp = datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S.%f')
                if first:
                    first_timestamp = timestamp
                if not first:
                    time_axis += list(np.linspace((prev_timestamp - first_timestamp).total_seconds(),
                                                (timestamp - first_timestamp).total_seconds(), len(row) - 1, endpoint=False))
                i += 1
                data.append(np.array(row[1:]).astype('float'))
                prev_timestamp = timestamp
                first = False
        data = np.array(data[:-1]).reshape((1, np.size(data[:-1]))).flatten()  # reject last data row because no timestamps are estimated for it
        time1 = datetime.strptime(timestamps[0], '%Y-%m-%d %H:%M:%S.%f')
        time2 = datetime.strptime(timestamps[-1], '%Y-%m-%d %H:%M:%S.%f')
        time_difference = (time2 - time1).total_seconds()
        self.fs = len(data)/time_difference 


        self.data = data
        self.header = header
        self.timestamps = timestamps
        self.time_axis = time_axis

        return data
    
    def resample_time_signal(self, data, f_original, f_new):
        num_samples = int(len(data) * f_new / f_original)  # Compute new number of samples
        resampled_data = scpysignal.resample(data, num_samples)
        self.resampled_data = resampled_data
        return resampled_data

    def calc_spectrogram(self, signal):
        signal = np.array(signal) - np.mean(np.array(signal))
        signal_tensor = torch.as_tensor(signal, dtype=torch.float32).unsqueeze(0)
        mel_spectrogram = self.mel_transform(signal_tensor).squeeze().numpy()
        mel_spectrogram_db = 10 * np.log10(mel_spectrogram + np.finfo(float).eps)
        return mel_spectrogram_db
    
    def get_spectrogram(self):
        return self.spectrogram
    
    def pickle_spectrogram(self, directory, filename, spectrogram):
        if not directory:
            directory = self.data_directory
        if not filename:
            filename = self.file_name
        if not spectrogram:
            spectrogram = self.spectrogram
        recorded_file = os.path.join(directory, f'spectrogram_{filename}') + ".pkl"

        with open(recorded_file, "wb") as f:
            pickle.dump(spectrogram, f)

    def encode_as_wav(self):
        waveform = torch.tensor(self.data).unsqueeze(0).float()
        
        # normalize waveform to -1, 1
        waveform = (waveform - waveform.min()) / (waveform.max() - waveform.min())
        waveform = waveform * 2 - 1
        # save as wav
        torchaudio.save(os.path.join(self.data_directory, f'{self.file_name}.wav'), waveform, int(self.fs))

