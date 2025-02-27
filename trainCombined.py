import os
import torch

import argparse
import mambaTF.datas
import mambaTF.models
import mambaTF.system
import mambaTF.losses
import mambaTF.metrics
import mambaTF.utils
from mambaTF.system import make_optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, RichProgressBar
from pytorch_lightning.callbacks.progress.rich_progress import *
from pytorch_lightning.loggers import TensorBoardLogger

# # # # # # ssh -NfL 6006:localhost:6006 gfraticcioli@10.79.23.8  (Hack)
# # # # # # ssh -NfL 6006:localhost:6006 gfraticcioli@10.79.0.8  (Euler)
#### Run on SSH shell ->
# # # # # # tensorboard --logdir=/nas/home/gfraticcioli/projects/MambaTransfer/Experiments/tensorboard_logs/MambaTF-StarNet --bind_all

from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy
from mambaTF.utils import MyRichProgressBar, RichProgressBarTheme

import torch.profiler

parser = argparse.ArgumentParser()
parser.add_argument(
    "--conf_dir",
    default="configs/MambaCoder-starNet.yml",
    help="Full path to save best validation model",
)
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
torch.backends.nnpack.enabled = False

def main(config):
    
    train_loader, val_loader, test_loader = make_datamodule(config)
    model, SnakeNet = make_models(config)
    scheduler, optimizer = make_scheduler(config, model, train_loader)
    logger, exp_dir, checkpoint_dir, conf_path = make_loggers(config)
    loss_func = make_lossFn(config)
    callbacks = define_callbacks(config)

    # Don't ask GPU if they are not available.
    gpus = config["training"]["gpus"] if torch.cuda.is_available() else None
    distributed_backend = "cuda" if torch.cuda.is_available() else None

    print("Instantiating System <{}>".format(config["training"]["system"]))
    system = mambaTF.system.CombinedLightningModule(
        frozen_model=model,
        new_network=SnakeNet,
        loss_func=loss_func,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        scheduler=scheduler,
        config=config,
        sr = config["datamodule"]["data_config"]["sample_rate"],
    )

    trainer = pl.Trainer(
        precision="bf16",
        #precision="32-true", #Euler
        max_epochs=config["training"]["epochs"],
        callbacks=callbacks,
        enable_checkpointing=True,
        default_root_dir=exp_dir,
        devices=gpus,
        accelerator=distributed_backend,
        strategy=DDPStrategy(find_unused_parameters=True),
        limit_train_batches=1.0,  # Useful for fast experiment
        gradient_clip_val=5.0,
        logger=logger,
        sync_batchnorm=True,
        # profiler='simple'
        # num_sanity_val_steps=0,
        # sync_batchnorm=True,
        # fast_dev_run=True,
    )

    trainer.fit(system)

    trainer.save_checkpoint(checkpoint_dir+"/final_model.ckpt")

    print("Finished Training")

def define_callbacks(config):
    callbacks = []

    if config["training"]["early_stop"]:
        print("Instantiating EarlyStopping")
        callbacks.append(EarlyStopping(**config["training"]["early_stop"]))
    callbacks.append(MyRichProgressBar(theme=RichProgressBarTheme()))

    # ModelCheckpoint to save the best model every 100 epochs

    # Add this callback to your existing code
    checkpoint100_callback = ModelCheckpoint(
        every_n_epochs=200,
        save_weights_only=True,  # Only save model weights (not optimizer states etc.)
        verbose=False  # Disable logging messages
    )
    callbacks.append(checkpoint100_callback) 
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss/dataloader_idx_0",  # Replace with the metric you want to track
        mode="min",  # "min" for loss, "max" for accuracy
        save_top_k=1,  # Keep only the best checkpoint
        save_weights_only=True,  # Save only model weights
        verbose=False  # Disable logging messages
    )
    callbacks.append(checkpoint_callback) # Add the callback to the callbacks list
    return callbacks

def make_datamodule(config):
    torch.backends.nnpack.enabled = False
    print("Instantiating datamodule <{}>".format(config["datamodule"]["data_name"]))
    datamodule: object = getattr(mambaTF.datas, config["datamodule"]["data_name"])(
        **config["datamodule"]["data_config"]
    )
    datamodule.setup()

    train_loader, val_loader, test_loader = datamodule.make_loader
    return train_loader, val_loader, test_loader

def make_models(config):
    print("Instantiating AudioNet <{}>".format(config["audionet"]["audionet_name"]))
    model = getattr(mambaTF.models, config["audionet"]["audionet_name"])(
        **config["audionet"]["audionet_config"],
    )
    SnakeNet = mambaTF.models.SnakeNet()
    return model, SnakeNet

def make_scheduler(config,model, train_loader):
    # Define scheduler
    print("Instantiating Optimizer <{}>".format(config["optimizer"]["optim_name"]))
    optimizer = make_optimizer(model.parameters(), **config["optimizer"])
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
        return scheduler, optimizer

def make_loggers(config):
    # Just after instantiating, save the args. Easy loading in the future.
    config["main_args"]["exp_dir"] = os.path.join(
        os.getcwd(), "Experiments", "checkpoint", config["exp"]["exp_name"]
    )
    exp_dir = config["main_args"]["exp_dir"]
    checkpoint_dir = os.path.join(exp_dir)
    os.makedirs(exp_dir, exist_ok=True)
    conf_path = os.path.join(exp_dir, "conf.yml")
    
    # Save the config file
    with open(conf_path, "w") as outfile:
        yaml.safe_dump(config, outfile)
    

    # default logger used by trainer
    logger_dir = os.path.join(os.getcwd(), "Experiments", "tensorboard_logs")
    os.makedirs(os.path.join(logger_dir, config["exp"]["exp_name"]), exist_ok=True)
    logger = TensorBoardLogger(logger_dir, name=config["exp"]["exp_name"])

    return logger, exp_dir, checkpoint_dir, conf_path



def make_lossFn(config): 
    # Define Loss function.
    print("Instantiating Loss, Train <{}>, Val <{}>".format(config["loss"]["train"], config["loss"]["val"])
    )
    loss_func = {
        "train": getattr(mambaTF.losses, config["loss"]["train"]["loss_func"])(
            **config["loss"]["train"]["config"],
        ),
        "val": getattr(mambaTF.losses, config["loss"]["val"]["loss_func"])(
            **config["loss"]["val"]["config"],
        ),
    }
    return loss_func

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