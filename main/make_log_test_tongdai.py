import os
import json
import glob
import argparse

import librosa
import numpy as np
import numpy.typing as npt


parser = argparse.ArgumentParser()
parser.add_argument('--folder_dataset', type=str, required=True, default="/home/SpeechSeparation/dataset/data_tongdai")
parser.add_argument('--log_path_test', type=str, required=True, default="/home/SpeechSeparation/dataset/mix_2_spk_tongdai.txt")
parser.add_argument('--meta_test', type=str, required=True, default="/home/SpeechSeparation/dataset/test_meta_tongdai.json")
parser.add_argument('--max_mixture_audio_test', type=int, required=True, default=5000)


"""
python3.7 make_log_test_tongdai.py \
    --folder_dataset /home/SpeechSeparation/dataset/data_tongdai \
    --log_path_test /home/SpeechSeparation/dataset/mix_2_spk_tongdai.txt \
    --meta_test /home/SpeechSeparation/dataset/test_meta_tongdai.json \
    --max_mixture_audio_test 5000
"""



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
            gender_spk1: str, 
            gender_spk2: str,
    ) -> None:

    pair_gender = sorted([gender_spk1, gender_spk2])
    pair_gender = '_'.join(pair_gender)

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

    pair_spk_folders = os.listdir(folder_dataset)
    output_data = list()

    for sub_folder in pair_spk_folders:

        if not os.path.isdir(os.path.join(folder_dataset, sub_folder)):
            continue

        spk1_file_paths = glob.glob(os.path.join(folder_dataset, sub_folder, "spk1*" , "*.wav"))
        spk2_file_paths = glob.glob(os.path.join(folder_dataset, sub_folder, "spk2*" , "*.wav"))

        for wav_spk1_path in spk1_file_paths:
            gender_spk1 = find_gender(wav_spk1_path)
            s1_wav = load_wav(wav_spk1_path)
            s1_snr = signal_to_noise(s1_wav)

            for wav_spk2_path in spk2_file_paths:
                gender_spk2 = find_gender(wav_spk2_path)

                s2_wav = load_wav(wav_spk2_path)
                s2_snr = signal_to_noise(s2_wav)

                output_line = "{} {} {} {}\n".format(wav_spk1_path, s1_snr, wav_spk2_path, s2_snr)
                print(output_line)

                output_data.append(output_line)
                update_data_meta(data_meta, gender_spk1, gender_spk2)

    save_json(data_meta, meta_test)
    write_txt(output_data, log_path)

