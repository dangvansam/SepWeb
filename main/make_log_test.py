import os
import json
import glob
import argparse
import random
import audiofile as af

import librosa
import numpy as np
import numpy.typing as npt


parser = argparse.ArgumentParser()
parser.add_argument('--folder_dataset', type=str, required=True, default="/home/SpeechSeparation/dataset/data_tt")
parser.add_argument('--log_path_test', type=str, required=True, default="/home/SpeechSeparation/dataset/mix_2_spk_tt.txt")
parser.add_argument('--meta_test', type=str, required=True, default="/home/SpeechSeparation/dataset/test_meta.json")
parser.add_argument('--max_mixture_audio_test', type=int, required=True, default=5000)


def load_wav(
        path: str
    ) -> npt.ArrayLike:

    signal, sr = librosa.load(path)
    return signal


def write_txt( data: list, path: str) -> None:
    with open(path, mode="w", encoding="utf8") as fp:
        fp.writelines(data)


def signal_to_noise(
            wav_array: npt.ArrayLike,
            axis=0, 
            ddof=0
        ) -> npt.ArrayLike:

    wav_array = np.asanyarray(wav_array)
    mean = wav_array.mean(axis)
    std = wav_array.std(axis = axis, ddof = ddof)

    return np.where(std == 0, 0, mean / std)


def find_gender(spk_name: str) -> str:
    gender = "unk"

    if "-F" in spk_name:
        gender = "F"
    elif "-M" in spk_name:
        gender = "M"
    else:
        print("Missing gender: {}".format(spk_name))
    
    return gender


def save_json(
        data: dict,
        path: str
    ) -> None:

    with open(path, mode="w", encoding="utf8") as fp:
        json.dump(data, fp)


def update_data_meta(
            data: dict,
            spk1_name: str,
            spk2_name: str,
            gender_spk1: str, 
            gender_spk2: str,
    ) -> None:

    pair_name = sorted([spk1_name, spk2_name])
    pair_name = '_'.join(pair_name)
    pair_gender = sorted([gender_spk1, gender_spk2])
    pair_gender = '_'.join(pair_gender)

    if pair_name not in data:
        data[pair_name] = { "num_pair": 1, "gen_type": pair_gender}
    else:
        data[pair_name]['num_pair'] += 1

    if pair_gender not in data:
        data[pair_gender] = { "count": 1}
    else:
        data[pair_gender]['count'] += 1


if __name__ == "__main__":
    args_input = parser.parse_args()
    folder_dataset = args_input.folder_dataset
    max_mixture_audio = args_input.max_mixture_audio_test
    log_path = args_input.log_path_test
    meta_test = args_input.meta_test

    data_meta = dict()

    data_path = folder_dataset
    spk_folders = os.listdir(data_path)
    num_spk = len(spk_folders)
    output_data = list()

    max_pair_per_spk = int(max_mixture_audio/num_spk)

    for spk_name in spk_folders:
        num_pair = 0
        
        spk1_file_paths = glob.glob(os.path.join(data_path, spk_name, "*.wav"))
        spk1_file_paths = [path for path in spk1_file_paths if 15 > af.duration(path) > 0.5]
        gender_spk1 = find_gender(spk_name)

        if not spk1_file_paths:
            continue

        while num_pair < max_pair_per_spk:

            wav_spk1_path = random.sample(spk1_file_paths, k= 1)[0]
            s1_wav = load_wav(wav_spk1_path)
            s1_snr = signal_to_noise(s1_wav)

            for other_spk_name in spk_folders:

                if num_pair > max_pair_per_spk:
                    break

                if other_spk_name != spk_name:
                    gender_spk2 = find_gender(other_spk_name)

                    spk2_file_paths = glob.glob(os.path.join(data_path, other_spk_name, "*.wav"))
                    spk2_file_paths = [path for path in spk2_file_paths if 15 > af.duration(path) > 0.5]

                    if not spk2_file_paths:
                        continue

                    wav_spk2_path = random.sample(spk2_file_paths, k= 1)[0]
                    s2_wav = load_wav(wav_spk2_path)
                    s2_snr = signal_to_noise(s2_wav)

                    output_line = "{} {} {} {}\n".format(wav_spk1_path, s1_snr, wav_spk2_path, s2_snr)
                    print(output_line)
                    output_data.append(output_line)
                    update_data_meta(data_meta, spk_name, other_spk_name, gender_spk1, gender_spk2)

                    num_pair += 1
    
    save_json(data_meta, meta_test)
    write_txt(output_data, log_path)

