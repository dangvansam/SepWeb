import utils 
import argparse

import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from train import Separation

global model 

model = None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hparams_dir', action='store', type=str, default='')
    parser.add_argument('--wavfile', action='store', type=str, default='')

    args = parser.parse_args()
    hparams_dir = [args.hparams_dir]
    wavfile = args.wavfile

    hparams_file, run_opts, overrides = sb.parse_arguments(hparams_dir)
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)
    checkpoint = hparams["checkpointer"]
    checkpoint.recover_if_possible()
    model = Separation(modules=hparams["modules"],
                        hparams=hparams,
                        run_opts=run_opts,
                        checkpointer=checkpoint,)
    separated = utils._process(wavfile, model)