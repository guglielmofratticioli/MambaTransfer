import argparse
import json
import mambaTF.datas
import mambaTF.models
import mambaTF.system
import mambaTF.losses
import mambaTF.metrics
import mambaTF.utils
from torchsummary import summary
from mambaTF.system import make_optimizer

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, RichProgressBar
from pytorch_lightning.callbacks.progress.rich_progress import *

import torch
import os
import sys

from pytorch_lightning.loggers import TensorBoardLogger



parser = argparse.ArgumentParser()
parser.add_argument(
    "--conf_dir",
    default="./configs/mambaTF-starNet.yml",
    help="Full path to save best validation model",
)
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"

def main(config):
    print("Instantiating AudioNet <{}>".format(config["audionet"]["audionet_name"]))
    model = getattr(mambaTF.models, config["audionet"]["audionet_name"])(
        sample_rate=config["datamodule"]["data_config"]["sample_rate"],
        **config["audionet"]["audionet_config"],
    )
    summary(model, (32000,1))
    while(True): pass

if __name__ == "__main__":
    import yaml
    from pprint import pprint
    from mambaTF.utils.parser_utils import (
        prepare_parser_from_dict,
        parse_args_as_dict,
    )

    args = parser.parse_args()
    with open(args.conf_dir) as f:
        def_conf = yaml.safe_load(f)
    parser = prepare_parser_from_dict(def_conf, parser=parser)

    arg_dic, plain_args = parse_args_as_dict(parser, return_plain_args=True)
    # pprint(arg_dic)
    main(arg_dic)