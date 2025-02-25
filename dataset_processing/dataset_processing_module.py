import os
import pandas as pd
import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from scipy import signal
from scipy.signal import lfilter
import time
import torch

segment_length = 1024


# generate Training Dataset and Testing Dataset
def get_all_segments_and_labels(args):
    '''
    use data_load() to generate the all segments and the corresponding labels.
    '''
    all_segments, all_labels = [], []
    if args.dataset_name == 'EccGear':
        t = time.time()
        num_classes = 11
        file_name = 'G1_T1_S1200_Sensor05.mat'
        filename_and_its_path = os.path.join(args.data_dir, file_name)
        datafile = h5py.File(filename_and_its_path)
        sampling_freq = 51200
        down_samp_rate = 8
        cutting_freq = sampling_freq / down_samp_rate / 2
        data_in_the_file = datafile['signal_data'][
            :,
            0:int(args.length_dataset*sampling_freq)
            ]
        data_downsampled = np.zeros((
            num_classes,
            int(args.length_dataset * sampling_freq / down_samp_rate)
            ))
        for i_class in range(num_classes):
            data_filtered = anti_aliasing_filter(
                data_in_the_file[i_class, :],
                fs=sampling_freq,
                fcut=cutting_freq,
                order=10
                )
            data_downsampled[i_class, :] = data_filtered[::down_samp_rate]
    elif args.dataset_name == 'XJTU_Spurgear':
        t = time.time()
        num_classes = 5
        sampling_freq = 10000
        down_samp_rate = 2
        cutting_freq = sampling_freq / down_samp_rate / 2
        num_samples = int(
            args.length_dataset * sampling_freq / down_samp_rate
            / segment_length
            )
        ideal_signal_len = num_samples*down_samp_rate*segment_length
        file_name_list = [
            r'XJTU_Spurgear\20Hz\spurgear00.txt',
            r'XJTU_Spurgear\20Hz\spurgear02.txt',
            r'XJTU_Spurgear\20Hz\spurgear06.txt',
            r'XJTU_Spurgear\20Hz\spurgear10.txt',
            r'XJTU_Spurgear\20Hz\spurgear14.txt'
            ]
        data_downsampled = np.zeros((
                num_classes,
                int(ideal_signal_len / down_samp_rate)
                ))
        for i_class in range(num_classes):
            args.data_dir
            filename_and_its_path = os.path.join(
                args.data_dir, file_name_list[i_class]
                )
            data_in_the_file = np.loadtxt(
                filename_and_its_path, usecols=(1,)
                )[0:ideal_signal_len]
            data_filtered = anti_aliasing_filter(
                data_in_the_file,
                fs=sampling_freq,
                fcut=cutting_freq,
                order=10
                )
            data_downsampled[i_class, :] = data_filtered[::down_samp_rate]
            pass
    else:
        raise Exception("wrong dataset_name")
    print('Loading and handeling the data file from ' +
          args.dataset_name +
          ' cost: {:.4f} s'
          .format(time.time() - t))
    for i_class in range(num_classes):
        start, end = 0, segment_length
        data_1_class = data_downsampled[i_class, :]
        if args.raw_signal_or_FFT == 'raw_signal':
            while end <= data_1_class.shape[0]:
                all_segments.append(data_1_class[start:end])
                all_labels.append(i_class)
                start += segment_length
                end += segment_length
        elif args.raw_signal_or_FFT == 'FFT':
            while end <= data_1_class.shape[0]:
                x = data_1_class[start:end]
                x = x - np.mean(x)
                x = np.fft.fft(x)
                x = np.abs(x) / len(x) * 2
                x = x[range(int(x.shape[0] / 2))]
                all_segments.append(x)
                all_labels.append(i_class)
                start += segment_length
                end += segment_length
        else:
            raise Exception("wrong data_name_and_processing_method")
    return [all_segments, all_labels]


def anti_aliasing_filter(x_signal, fs=51200, fcut=12800, order=10):
    '''
    https://www.cnblogs.com/LXP-Never/p/10886622.html
    fs: sampling frequency
    fcut: cut off frequency
    '''
    # design Butterworth filter and return filter coefficents
    b, a = signal.butter(order, fcut, btype='lowpass', analog=False, fs=fs)
    y = lfilter(b, a, x_signal)
    return y


class final_output_dataset():

    def __init__(self, args):
        self.args = args

    def data_prepare(self):
        '''
        load files, get signal data, cut segments,
        split training & test sets, normalize.
        '''
        all_segments_and_labels = get_all_segments_and_labels(self.args)
        data_pd = pd.DataFrame({
            "data": all_segments_and_labels[0],
            "label": all_segments_and_labels[1]
            })
        train_pd, val_pd = train_test_split(data_pd, test_size=0.2,
                                            random_state=40,
                                            stratify=data_pd["label"])

        # normalization
        processing_method = Compose([
            Reshape(),
            Normalize(self.args.normlizetype),
            Retype()
            ])

        train_dataset = dataset_processed(
            list_data=train_pd,
            processing=processing_method
            )
        val_dataset = dataset_processed(
            list_data=val_pd,
            processing=processing_method
            )

        return train_dataset, val_dataset


class dataset_processed(Dataset):  # normalization

    def __init__(self, list_data, processing=None):
        self.seq_data = list_data['data'].tolist()
        self.labels = list_data['label'].tolist()
        if processing is None:
            self.processing = Compose([Reshape()])
        else:
            self.processing = processing

    def __len__(self):
        return len(self.seq_data)

    def __getitem__(self, item):
        seq = self.seq_data[item]
        label = self.labels[item]
        seq = self.processing(seq)
        return seq, label


class Compose(object):
    def __init__(self, processing):
        self.processing = processing

    def __call__(self, seq):
        for t in self.processing:
            seq = t(seq)
        return seq


class Reshape(object):
    def __call__(self, seq):
        return seq.transpose()


class Retype(object):
    def __call__(self, seq):
        return seq.astype(np.float32)


class Normalize(object):
    def __init__(self, type="0-1"):
        self.type = type

    def __call__(self, seq):
        if self.type == "0-1":
            seq = (seq-seq.min())/(seq.max()-seq.min())
        elif self.type == "1-1":
            seq = 2*(seq-seq.min())/(seq.max()-seq.min()) + -1
        elif self.type == "mean-std":
            seq = (seq-seq.mean())/seq.std()
        else:
            raise NameError('This normalization is not included!')
        return seq


def add_noise_perSNR(x, snr, noise_type='white'):
    p_signal = torch.sum(x ** 2) / len(x)
    p_noise = p_signal / 10 ** (snr / 10)
    noise = torch.randn(x.shape[0]) * torch.sqrt(p_noise)
    if noise_type == 'white':
        pass
    elif noise_type == 'pink':
        noise = torch.linspace(
            1.1, 0.9, steps=len(x)) * (2.02/2) * noise
    elif noise_type == 'Laplacian':
        miu = 0
        b = 1
        noise = torch.tensor(np.random.laplace(miu, b, len(x)))
        p_noise_before = torch.sum(noise ** 2) / len(noise)
        noise = noise * torch.sqrt(p_noise / p_noise_before)
    else:
        raise NameError('This noise type is not included!')
    x_noised = x + noise
    return x_noised
