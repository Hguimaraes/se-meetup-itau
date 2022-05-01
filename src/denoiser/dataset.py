import torch
import speechbrain as sb
from speechbrain.dataio.dataset import DynamicItemDataset

import re
import glob
import json
import torchaudio
import numpy as np
from tqdm import tqdm
from typing import List
from pathlib import Path
from speechbrain.utils.data_utils import get_all_files


def read_audio(wav_files):
    # Read the waveform
    predictor = sb.dataio.dataio.read_audio(wav_files['predictors'])
    predictor = torch.unsqueeze(predictor, 0)
    predictor = predictor.transpose(0, 1)

    target = sb.dataio.dataio.read_audio(wav_files['wave_target'])
    target = torch.unsqueeze(target, 0)
    target = target.transpose(0, 1)

    return predictor, target


def sample(
    predictor: torch.Tensor,
    target: torch.Tensor, 
    max_size: int
):
    samples = predictor.shape[0]
    if samples > max_size:
        offset = torch.randint(low=0, high=max_size-1, size=(1,))
        target = target[offset:(offset+max_size), :]
        predictor = predictor[offset:(offset+max_size), :]
    
    return predictor, target


def create_datasets(hparams):
    datasets = {}

    @sb.utils.data_pipeline.takes("wav_files")
    @sb.utils.data_pipeline.provides("predictor", "target")
    def audio_pipeline(wav_files):
        predictor, target = read_audio(wav_files)

        # Sampling procedure
        if hparams['train_sampling']:
            max_size = hparams['max_train_sample_size']
            predictor, target = sample(predictor, target, max_size)

        return predictor, target


    @sb.utils.data_pipeline.takes("wav_files")
    @sb.utils.data_pipeline.provides("predictor", "target")
    def audio_pipeline_valid(wav_files):
        predictor, target = read_audio(wav_files)
        return predictor, target

    
    dynamic_items_map = {
        'train': [audio_pipeline],
        'valid': [audio_pipeline_valid],
    }

    for set_ in ['train', 'valid']:
        output_keys = ["id", "predictor", "length", "target", "transcript"]
        dynamic_items = dynamic_items_map[set_]
        
        # Construct the dynamic item dataset
        datasets[set_] = DynamicItemDataset.from_json(
            json_path=hparams[f"{set_}_annotation"],
            replacements={"data_root": hparams["data_folder"]},
            dynamic_items=dynamic_items,
            output_keys=output_keys,
        ).filtered_sorted(sort_key="length", reverse=False)

    return datasets



def  prep_librispeech(
    data_folder:str,
    save_json_train:str,
    save_json_valid:str,
    train_folder:list,
    valid_folder:str,
):
    # File extension to look for
    extension = ["_clean.wav"]

    # Parameters to search and save
    files = [
        [train_folder, extension, save_json_train],
        [valid_folder, extension, save_json_valid],
    ]

    for folder, exts, save_json in files:
        a_files = get_all_files(folder, match_and=exts)

        # Create Json for dataio
        create_json(a_files, save_json)


def create_json(
    a_wav_list:List[str],
    json_file:str
):
    # Processing all the wav files in the list
    json_dict = {}

    for utterance in tqdm(a_wav_list):
        # Manipulate paths and get info about the 4 files
        utt_id = utterance.split('/').pop()[:-10]

        clean_utt = utterance
        noisy_utt = utterance.replace("_clean", "_noisy")
        transcript_utt = utterance.replace("_clean.wav", ".txt")

        with open(transcript_utt) as f:
            transcript = f.readline().replace('\n', '')

        # Construct Json structure
        audio_len = torchaudio.info(clean_utt).num_frames
        json_dict[utt_id] = {
            "wav_files": {
                "predictors": {
                    "file": noisy_utt,
                    "start": 0,
                    "stop": audio_len
                },
                "wave_target": {
                    "file": clean_utt,
                    "start": 0,
                    "stop": audio_len
                },
            },
            "length": audio_len,
            "transcript": transcript
        }

    # Writing the dictionary to the json file
    with open(json_file, mode="w") as json_f:
        json.dump(json_dict, json_f, indent=2)
