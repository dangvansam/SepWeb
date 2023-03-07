
Speech Separation
========================================

The task of extracting all overlapping speech sources in a given mixed speech signal refers to the Speech Separation. Speech Separation is a special scenario of source separation problem, where the focus is only on the overlapping speech signal sources and other interferences such as music or noise signals are not the main concern of the study.

## Requirements

```bash

python version >= 3.7

sudo apt-get install -y gnuplot-x11
sudo apt-get install -y gnuplot
sudo apt install octave ffmpeg

```

```bash
torch==1.7.1+cu110
torchaudio==0.7.2
torchvision==0.8.2+cu110
torchmetrics==0.4.1
requests==2.25.1
librosa==0.8.1
tqdm==4.61.2
mir-eval==0.6
oct2py==5.2.0
octave-kernel==0.32.0
flask
```


## Data Preprocess

#### Step 1: Prepare struct project


```bash

|___/home/SpeechSeparation
    |___SpeechSeparation
        |___main
        |    |___config
        |    |___hparams
        |    |___meta
        |    |___speechbrain
        |    |___static
        |    |___templates
        |    |___dynamic_mixing.py
        |    |___make_log.py
        |    |___prepare_data.py
        |    |___mps_storage.py
        |    |___server.py
        |    |___train.py
        |    |___eval.py
        |    |___utils.py
        |___README.md
        |___requirements.txt
    |___dataset
        |___data_tr
        |    |___train_spk1
        |        |___train_spk1_0.wav
        |        |___train_spk1_1.wav
        |        ...
        |    |___train_spk2
        |        |___train_spk2_0.wav
        |        |___train_spk2_1.wav
        |        ...
        |    ...
        |___data_vd
        |    |___dev_spk1
        |        |___dev_spk1_0.wav
        |        |___dev_spk1_1.wav
        |        ...
        |    |___dev_spk2
        |        |___dev_spk2_0.wav
        |        |___dev_spk2_1.wav
        |        ...
        |___data_tt
        |    |___test_spk1
        |        |___test_spk1_0.wav
        |        |___test_spk1_1.wav
        |        ...
        |    |___test_spk2
        |        |___test_spk2_0.wav
        |        |___test_spk2_1.wav
        |        ...
        |    ...
        |___output_folder
        |    |___save
                 |___log

```



#### Step 2: Make log input

```bash

cd /home/SpeechSeparation/SpeechSeparation/main

python make_log.py \
    --folder_dataset /home/SpeechSeparation/dataset \
    --log_path_train /home/SpeechSeparation/dataset/mix_2_spk_tr.txt \
    --log_path_valid /home/SpeechSeparation/dataset/mix_2_spk_vd.txt \
    --max_mixture_audio_train 20000 \
    --max_mixture_audio_valid 600

```

#### Step 3: Prepare input training

```bash

cd /home/SpeechSeparation/SpeechSeparation/main

python prepare_data.py --data_type tr,vd

```

## Training Sepformer model

```bash

cd /home/SpeechSeparation/SpeechSeparation/main

python train.py hparams/sepformer.yaml


```


## Evaluation model on test set


#### Step 1: Make log testing

```bash

cd /home/SpeechSeparation/SpeechSeparation/main

python make_log_test.py \
    --folder_dataset /home/SpeechSeparation/dataset/data_tt \
    --log_path_test /home/SpeechSeparation/dataset/mix_2_spk_tt.txt \
    --meta_test /home/SpeechSeparation/dataset/test_meta.json \
    --max_mixture_audio_test 5000

```

#### Step 2: Prepare input testing

```bash

cd /home/SpeechSeparation/SpeechSeparation/main

python prepare_data.py --data_type tt

```

#### Step 2: Prepare input testing with noise

```bash

cd /home/SpeechSeparation/SpeechSeparation/main

python prepare_data_noise.py \
    --input_csv /home/SpeechSeparation/dataset/output_folder/save/data_tt.csv 
    --output_csv /home/SpeechSeparation/dataset/output_folder/save/data_tt_noise.csv 
    --data_type tt 
    --noise_folder /home/SpeechSeparation/dataset/noise 
    --output_mix_noise_folder wav16k_tt_noise 

```

#### Step 3: Evaluation

Eval clean test set:

```bash

cd /home/SpeechSeparation/SpeechSeparation/main

python eval.py hparams/eval.yaml


```

Eval noisy test set: 

Change in config `eval.yaml`:
  - `test_data` to `data_tt_noise.csv`
  - `skip_prep` to `True`




## API 

Download [model][model-url] and copy to `/home/SpeechSeparation/SpeechSeparation/dataset/output_folder/save`

```bash

cd /home/SpeechSeparation/SpeechSeparation/main

python server.py hparams/run.yaml


```