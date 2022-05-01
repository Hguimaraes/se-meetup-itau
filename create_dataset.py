import os
import torch
import torchaudio
import numpy as np
import pandas as pd
from glob import glob
from typing import List
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import librosa
from tqdm import tqdm
from scipy.io import wavfile

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


# Prepare LibriSpeech dataset
SAMPLE_RATE = 16000
NUM_MAX_SEGS = 10
trn_folder = "dataset/LibriSpeech/train-clean-100"
dev_folder = "dataset/LibriSpeech/dev-clean/"

NOISE_PATH = "dataset/FSD50K.dev_audio"
NOISES_TRN = [
    'Crackle', 'Mechanical_fan', 'Microwave_oven', 'Computer_keyboard',
    'Telephone', 'Drawer_open_or_close', 'Laughter', 'Printer', 
    'Keys_jangling', 'Scissors'
]
NOISES_DEV = [
    'Alarm', 'Mechanical_fan', 'Cupboard_open_or_close', 'Finger_snapping',
     'Knock', 'Telephone', 'Computer_keyboard', 'Writing'
]


def transcript_parser(transcript):
    with open(f'{transcript}', 'r') as f:
        lines = f.readlines()
    
    parsed = []
    for l in lines:
        splited = l.split(" ")
        fname = splited[0]
        text = " ".join(splited[1:]).replace("\n", "")

        parsed.append({'fname': fname, 'text': text})
    
    return pd.DataFrame.from_dict(parsed)


def select_audios(audio_list:List[str]) -> pd.DataFrame:
    filtered_audios = list(filter(
        lambda x: torchaudio.info(x).num_frames <= NUM_MAX_SEGS*SAMPLE_RATE, audio_list
    ))

    fnames = [f.split('/').pop()[:-5] for f in filtered_audios]
    return pd.DataFrame({'fname': fnames, 'audio_path': filtered_audios})


def get_librispeech_data(folder):
    transcripts = glob(f"{folder}/**/*.txt", recursive=True)
    transcripts_df = pd.concat([transcript_parser(t) for t in transcripts])
    
    audio_files = glob(f"{folder}/**/*.flac", recursive=True)
    audio_df = select_audios(audio_files)

    return audio_df.merge(transcripts_df, how="inner")


def insert_controlled_noise(signal, noise, desired_snr_db=5):
    if signal.shape[0] != noise.shape[0]:
        raise ValueError("Incompatible shape between noise and signal")

    # Calculate the power of signal and noise
    n = signal.shape[0]
    S_signal = signal.dot(signal) / n
    S_noise = noise.dot(noise) / n

    # Proportion factor
    K = (S_signal / S_noise) * (10 ** (-desired_snr_db / 10))

    # Rescale the noise
    new_noise = np.sqrt(K) * noise

    return new_noise + signal


def shape_noise(noise, n_samples):
    m = noise.shape[0]

    # If the audio signal is longer then the noise
    if n_samples >= m:
        # Tile the noise signal
        factor = int(np.ceil(n_samples / m))
        noise = np.tile(noise, factor)

    return noise[:n_samples]


def create_experiment_folder(
    folder_name: str,
    ls_dataset,
    noise_dataset
):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    csv_info = []

    for _id, row in tqdm(ls_dataset.iterrows(), total=ls_dataset.shape[0]):
        # Get transcription
        text = row['text']

        # Sample a noise and read
        sample_noise_row = noise_dataset.sample().squeeze()
        fn_noise = f"{NOISE_PATH}/{sample_noise_row.fname}.wav"
        noise, _ = librosa.load(fn_noise, sr=SAMPLE_RATE)

        # Read the signal
        clean_signal, _ = librosa.load(row['audio_path'], sr=SAMPLE_RATE)
        n = clean_signal.shape[0]

        # Randomly select a SNR value
        snr = np.random.uniform(low=2.5, high=10)

        # Check noise size and combine then
        reshaped_noise = shape_noise(noise, n_samples=n)
        noisy_signal =  insert_controlled_noise(
            clean_signal,
            reshaped_noise,
            desired_snr_db=snr
        )

        # Save to disk
        csv_info.append({
            'fname': row.fname,
            'text': row.text,
            'clean_fname': row.audio_path,
            'noisy_fname': sample_noise_row.fname,
            'noisy_labels': sample_noise_row.labels
        })
        
        # Clean wav file
        wavfile.write(
            f"{folder_name}/{row.fname}_clean.wav",
            rate=SAMPLE_RATE, 
            data=clean_signal
        )

        # Noisy wav file
        wavfile.write(
            f"{folder_name}/{row.fname}_noisy.wav",
            rate=SAMPLE_RATE,
            data=noisy_signal
        )

        with open(f"{folder_name}/{row.fname}.txt", "w") as f:
            f.write(text)

    csv_info = pd.DataFrame.from_dict(csv_info)
    csv_info.to_csv(f"{folder_name}/metadata.csv", sep="|")


if __name__ == "__main__":
    train_librispeech = get_librispeech_data(trn_folder)
    dev_librispeech = get_librispeech_data(dev_folder)

    # Reading the Noise database (FSD50K)
    fsd = pd.read_csv("dataset/dev.csv")
    labels = fsd.labels.str.split(",").tolist()
    mlb = MultiLabelBinarizer()

    encoded = mlb.fit_transform(labels)
    classes = mlb.classes_

    encoded = pd.DataFrame(data=encoded, columns=classes)
    df_noises = pd.concat([fsd, encoded], axis=1)
    df_noises = df_noises[df_noises[set(NOISES_TRN + NOISES_DEV)].sum(axis=1) >= 1]
    trn_subset, dev_subset = train_test_split(df_noises, test_size=0.4)

    trn_subset = trn_subset[trn_subset[NOISES_TRN].sum(axis=1) >= 1]
    dev_subset = dev_subset[dev_subset[NOISES_DEV].sum(axis=1) >= 1]


    # Apply noises and create final experiment folder
    create_experiment_folder(
        "dataset/se_itau_train", train_librispeech, trn_subset
    )
    create_experiment_folder(
        "dataset/se_itau_dev", dev_librispeech, dev_subset
    )
