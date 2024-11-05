import os
import sys
import torch
from torch import Tensor
import argparse
import json
import mambaTF.datas
import mambaTF.models
import mambaTF.system
import mambaTF.losses
import mambaTF.metrics
import mambaTF.utils
from mambaTF.system import make_optimizer
#from dataclasses import dataclass
#from torch.optim.lr_scheduler import ReduceLROnPlateau
#from torch.utils.data import DataLoader
#import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, RichProgressBar
from pytorch_lightning.callbacks.progress.rich_progress import *
#from rich.console import Console
from pytorch_lightning.loggers import TensorBoardLogger

# # # # # # ssh -NfL 6006:localhost:6006 gfraticcioli@10.79.251.109    (Hack)
#### Run on SSH shell ->
# # # # # # tensorboard --logdir=/nas/home/gfraticcioli/projects/MambaTransfer/Experiments/tensorboard_logs/MambaTF-StarNet --bind_all

from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy
#from rich import print, reconfigure
#from collections.abc import MutableMapping
from mambaTF.utils import MyRichProgressBar, RichProgressBarTheme

import warnings

parser = argparse.ArgumentParser()
parser.add_argument(
    "--conf_dir",
    default="./configs/mambaTF-starNet.yml",
    help="Full path to save best validation model",
)


def main(config):
    print("Instantiating datamodule <{}>".format(config["datamodule"]["data_name"]))
    datamodule: object = getattr(mambaTF.datas, config["datamodule"]["data_name"])(
        **config["datamodule"]["data_config"]
    )
    datamodule.setup()

    train_loader, val_loader, test_loader = datamodule.make_loader

    # Define model and optimizer
    print("Instantiating AudioNet <{}>".format(config["audionet"]["audionet_name"]))
    model = getattr(mambaTF.models, config["audionet"]["audionet_name"])(
        sample_rate=config["datamodule"]["data_config"]["sample_rate"],
        **config["audionet"]["audionet_config"],
    )
    # import pdb; pdb.set_trace()
    print("Instantiating Optimizer <{}>".format(config["optimizer"]["optim_name"]))
    optimizer = make_optimizer(model.parameters(), **config["optimizer"])

    # Define scheduler
    scheduler = None
    if config["scheduler"]["sche_name"]:
        print("Instantiating Scheduler <{}>".format(config["scheduler"]["sche_name"]))
        if config["scheduler"]["sche_name"] != "DPTNetScheduler":
            scheduler = getattr(torch.optim.lr_scheduler, config["scheduler"]["sche_name"])(
                optimizer=optimizer, **config["scheduler"]["sche_config"]
            )
        else:
            scheduler = {
                "scheduler": getattr(mambaTF.system.schedulers, config["scheduler"]["sche_name"])(
                    optimizer, len(train_loader) // config["datamodule"]["data_config"]["batch_size"], 64
                ),
                "interval": "step",
            }

    # Just after instantiating, save the args. Easy loading in the future.
    config["main_args"]["exp_dir"] = os.path.join(
        os.getcwd(), "Experiments", "checkpoint", config["exp"]["exp_name"]
    )
    exp_dir = config["main_args"]["exp_dir"]
    os.makedirs(exp_dir, exist_ok=True)
    conf_path = os.path.join(exp_dir, "conf.yml")
    with open(conf_path, "w") as outfile:
        yaml.safe_dump(config, outfile)

    # Define Loss function.
    print("Instantiating Loss, Train <{}>, Val <{}>".format(config["loss"]["train"], config["loss"]["val"])
    )
    loss_func = {
        "train": getattr(mambaTF.losses, config["loss"]["train"]["loss_func"])(
            #**config["loss"]["train"]["config"],
        ),
        "val": getattr(mambaTF.losses, config["loss"]["val"]["loss_func"])(
            #**config["loss"]["val"]["config"],
        ),
    }

    print("Instantiating System <{}>".format(config["training"]["system"]))
    system = getattr(mambaTF.system, config["training"]["system"])( #AudioLightningModule
        audio_model=model,
        loss_func=loss_func,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        scheduler=scheduler,
        config=config,
    )

    # Define callbacks
    print("Instantiating ModelCheckpoint")
    callbacks = []
    checkpoint_dir = os.path.join(exp_dir)
    checkpoint = ModelCheckpoint(
        checkpoint_dir,
        filename="{epoch}",
        monitor="val_loss/dataloader_idx_0",
        mode="min",
        save_top_k=5,
        verbose=True,
        save_last=True,
    )
    callbacks.append(checkpoint)

    if config["training"]["early_stop"]:
        print("Instantiating EarlyStopping")
        callbacks.append(EarlyStopping(**config["training"]["early_stop"]))
    callbacks.append(MyRichProgressBar(theme=RichProgressBarTheme()))

    # Don't ask GPU if they are not available.
    gpus = config["training"]["gpus"] if torch.cuda.is_available() else None
    distributed_backend = "cuda" if torch.cuda.is_available() else None

    # default logger used by trainer
    logger_dir = os.path.join(os.getcwd(), "Experiments", "tensorboard_logs")
    os.makedirs(os.path.join(logger_dir, config["exp"]["exp_name"]), exist_ok=True)
    logger = TensorBoardLogger(logger_dir, name=config["exp"]["exp_name"])
    #comet_logger = WandbLogger(
    #    name=config["exp"]["exp_name"],
    #    save_dir=os.path.join(logger_dir, config["exp"]["exp_name"]),
    #    project="Real-work-dataset",
    #    # offline=True
    #)

    trainer = pl.Trainer(
        precision="bf16-mixed",
        max_epochs=config["training"]["epochs"],
        callbacks=callbacks,
        default_root_dir=exp_dir,
        devices=gpus,
        accelerator=distributed_backend,
        strategy=DDPStrategy(find_unused_parameters=True),
        limit_train_batches=1.0,  # Useful for fast experiment
        gradient_clip_val=5.0,
        logger=logger,
        sync_batchnorm=True,
        # num_sanity_val_steps=0,
        # sync_batchnorm=True,
        # fast_dev_run=True,
    )
    trainer.fit(system)
    print("Finished Training")
    best_k = {k: v.item() for k, v in checkpoint.best_k_models.items()}
    with open(os.path.join(exp_dir, "best_k_models.json"), "w") as f:
        json.dump(best_k, f, indent=0)

    state_dict = torch.load(checkpoint.best_model_path)
    system.load_state_dict(state_dict=state_dict["state_dict"])
    system.cpu()

    to_save = system.audio_model.serialize()
    torch.save(to_save, os.path.join(exp_dir, "best_model.pth"))
    
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