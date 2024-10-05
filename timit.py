import os
import torchaudio
import pandas as pd
import torch
from torchaudio.transforms import MFCC
import glob
import librosa
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset
from loguru import logger

SAMPLE_RATE = 16000
FRAME_LENGTH = 0.025    # ms
FRAME_STEP = 0.01       # ms


def extract_frames(audio, frame_length=0.025, frame_step=0.010, sampling_rate=16000):
    # Extract frames from audio using librosa
    frames = librosa.util.frame(audio, frame_length=int(sampling_rate * frame_length), hop_length=int(sampling_rate * frame_step))
    return frames.T  # Transpose to have frames in the first dimension

def load_phoneme_labels(phn_path):
    phonemes = []
    with open(phn_path, 'r') as file:
        for line in file:
            start, end, phoneme = line.strip().split()
            phonemes.append((int(start), int(end), phoneme))
    return phonemes

def frame_phoneme_labels(phonemes, num_frames, wav_range):
    """Create phoneme labels based on frames"""
    frame_shift = SAMPLE_RATE * FRAME_STEP
    frame_size = SAMPLE_RATE * FRAME_LENGTH
    frame_labels = []
    for frame_idx in range(num_frames):
        frame_start = frame_idx * frame_shift + wav_range[0]
        frame_end = frame_start + frame_size
        frame_phonemes = [ph for ph in phonemes if ph[0] < frame_end and ph[1] > frame_start]
        if frame_phonemes:
            label = max(frame_phonemes, key=lambda ph: min(frame_end, ph[1]) - max(frame_start, ph[0]))[2]
        else:
            logger.error(f'frame_phonemes label not found.')
            exit(0)
        frame_labels.append(label)
    return frame_labels

def process_timit_file(wav_path, phn_path):
    """process a single timit file"""
    waveform, sample_rate = torchaudio.load(wav_path)
    assert sample_rate == SAMPLE_RATE, f"Expected sample rate {SAMPLE_RATE}, but got {sample_rate}"

    phonemes = load_phoneme_labels(phn_path)
    # get label range based on phonemes
    wav_range = [phonemes[0][0], phonemes[-1][1]]
    audio = waveform[0][wav_range[0]:wav_range[1]]

    # frames.shape -> (num_frames, frame_len)
    frames = extract_frames(audio, frame_length=FRAME_LENGTH, frame_step=FRAME_STEP, sampling_rate=SAMPLE_RATE)
    num_frames = frames.shape[0]

    # Generate phoneme labels on frame-based level
    frame_phonemes = frame_phoneme_labels(phonemes, num_frames, wav_range)

    speaker_ids = [os.path.basename(os.path.dirname(wav_path))] * num_frames

    return frames, frame_phonemes, speaker_ids

def process_timit_dataset(data_dir, data_type='train', split=False, val_ratio=0.1):
    all_frames = []
    all_phonemes = []
    all_speaker_ids = []

    wav_files = sorted(glob.glob(os.path.join(data_dir, data_type, '*', '*', '*.wav')))

    for wav_path in tqdm(wav_files, desc=f'Loading "{data_type}" data'):
        phn_path = wav_path.replace('.wav', '.phn')
        if os.path.exists(phn_path):
            frames, frame_phonemes, speaker_ids = process_timit_file(wav_path, phn_path)
            all_frames.extend(frames)
            all_phonemes.extend(frame_phonemes)
            all_speaker_ids.extend(speaker_ids)
    all_frames = np.array(all_frames)
    all_speaker_ids = np.array(all_speaker_ids)

    phoneme_to_id = {ph: id for id, ph in enumerate(np.sort(np.unique(all_phonemes)))}
    id_to_phoneme = {id: ph for ph, id in phoneme_to_id.items()}

    # convert phoneme to id
    all_labels = np.array([phoneme_to_id[ph] for ph in all_phonemes])

    # return all_frames, all_labels
    if split:
        train_frames = []
        train_labels = []
        val_frames = []
        val_labels = []
        unique_speaker_ids = np.unique(all_speaker_ids)
        for id in tqdm(unique_speaker_ids, desc='Split data by speakers'):
            select_idxs = np.where(all_speaker_ids == id)[0]
            speaker_frames = all_frames[select_idxs]
            speaker_labels = all_labels[select_idxs]
            # shuffle the data
            indices = np.arange(len(speaker_frames))
            np.random.shuffle(indices)
            speaker_frames = speaker_frames[indices]
            speaker_labels = speaker_labels[indices]
            
            # split the data by the given ratio
            train_frames.extend(speaker_frames[:int(len(speaker_frames)*(1-val_ratio))])
            train_labels.extend(speaker_labels[:int(len(speaker_frames)*(1-val_ratio))])
            val_frames.extend(speaker_frames[int(len(speaker_frames)*(1-val_ratio)):])
            val_labels.extend(speaker_labels[int(len(speaker_frames)*(1-val_ratio)):])
        train_frames = np.array(train_frames)
        train_labels = np.array(train_labels)
        val_frames = np.array(val_frames)
        val_labels = np.array(val_labels)
        return train_frames, train_labels, val_frames, val_labels
    else:
        return all_frames, all_labels

class TimitDataset(Dataset):
    def __init__(self, timit_data=None) -> None:
        super().__init__()

        self.frames = timit_data['frames']
        self.labels = timit_data['labels']

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, index: int):
        frame = self.frames[index]
        label = self.labels[index]
        return frame, label

def main():
    timit_root = '/raid/yixu/Projects/Speech/proj/timit/timit'

    train_frames, train_labels, val_frames, val_labels = process_timit_dataset(timit_root, 'train', split=True, val_ratio=0.1)
    test_frames, test_labels = process_timit_dataset(timit_root, 'test', split=False)

    # Save the preprocessed dataset as npz file
    np.savez('/raid/yixu/Projects/Speech/proj/dataset_train.npz', frames=train_frames, labels=train_labels)
    np.savez('/raid/yixu/Projects/Speech/proj/dataset_val.npz', frames=val_frames, labels=val_labels)
    np.savez('/raid/yixu/Projects/Speech/proj/dataset_test.npz', frames=test_frames, labels=test_labels)

if __name__ == '__main__':
    main()
