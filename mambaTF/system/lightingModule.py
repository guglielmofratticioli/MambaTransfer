
import torch
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau
from collections.abc import MutableMapping


#from speechbrain.processing.speech_augmentation import SpeedPerturb


def flatten_dict(d, parent_key="", sep="_"):
    """Flattens a dictionary into a single-level dictionary while preserving
    parent keys. Taken from
    `SO <https://stackoverflow.com/questions/6027558/flatten-nested-dictionaries-compressing-keys>`_

    Args:
        d (MutableMapping): Dictionary to be flattened.
        parent_key (str): String to use as a prefix to all subsequent keys.
        sep (str): String to use as a separator between two key levels.

    Returns:
        dict: Single-level dictionary, flattened.
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


class AudioLightningModule(pl.LightningModule):
    def __init__(
            self,
            audio_model=None,
            video_model=None,
            optimizer=None,
            loss_func=None,
            train_loader=None,
            val_loader=None,
            test_loader=None,
            scheduler=None,
            config=None,
            sr=44100,
    ):
        super().__init__()
        self.audio_model = audio_model
        self.video_model = video_model
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.train_loader = train_loader
        self.learning_rate = 0.00001
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.scheduler = scheduler
        self.config = {} if config is None else config
        # Speed Aug
        #self.speedperturb = SpeedPerturb(
        #    self.config["datamodule"]["data_config"]["sample_rate"],
        #    speeds=[95, 100, 105],
        #    perturb_prob=1.0
        #)
        # Save lightning"s AttributeDict under self.hparams
        self.default_monitor = "val_loss/dataloader_idx_0"
        self.save_hyperparameters(self.config_to_hparams(self.config))
        # self.print(self.audio_model)
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.sr = sr

    def forward(self, wav, mouth=None):
        """Applies forward pass of the model.
        
        Returns:
            :class:`torch.Tensor`
        """

        return self.audio_model(wav)

    def log_dataset_statistics(self, dataloader, logger, dataset_name):
        """Logs basic statistics about a dataset."""
        data_stds = []
        sample_count = 0

        for batch in dataloader:
            sources, targets, _ = batch
            sample_count += sources.size(0)

            # Calculate mean and std for sources
            data_stds.append(sources.std().item())

    def setup(self, stage=None):
        # Log training dataset statistics
        #self.log_dataset_statistics(self.train_loader, self.logger, dataset_name="train")

        # Log validation dataset statistics
        #self.log_dataset_statistics(self.val_loader, self.logger, dataset_name="val")

        # Log test dataset statistics
        #self.log_dataset_statistics(self.test_loader, self.logger, dataset_name="test")
        pass

    def training_step(self, batch, batch_nb):
        sources, targets, _ = batch

        new_targets = []
        min_len = -1
        if self.config["training"]["SpeedAug"] == True:
            with torch.no_grad():
                for i in range(targets.shape[1]):
                    new_target = self.speedperturb(targets[:, i, :])
                    new_targets.append(new_target)
                    if i == 0:
                        min_len = new_target.shape[-1]
                    else:
                        if new_target.shape[-1] < min_len:
                            min_len = new_target.shape[-1]

                targets = torch.zeros(
                    targets.shape[0],
                    targets.shape[1],
                    min_len,
                    device=targets.device,
                    dtype=torch.float,
                )
                for i, new_target in enumerate(new_targets):
                    targets[:, i, :] = new_targets[i][:, 0:min_len]

                sources = targets.sum(1)
        # print(mixtures.shape)
        est_targets= self(sources)
        loss = self.loss_func["train"](est_targets, targets)

        if batch_nb % 10 == 0:
            self.log(
                "train_loss",
                loss,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
                logger=True,
            )

        ## Log gradients
        #for name, param in self.audio_model.named_parameters():
        #    if param.grad is not None:
        #        self.logger.experiment.add_histogram(f"gradients/{name}", param.grad, self.current_epoch)

        return {"loss": loss}

    def validation_step(self, batch, batch_nb, dataloader_idx):
        with torch.no_grad():
        # cal val loss
            if dataloader_idx == 0:
                sources, targets, _ = batch
                # print(mixtures.shape)
                est_targets = self(sources)
                loss = self.loss_func["val"](est_targets, targets)
                
                self.log(
                    "val_loss",
                    loss,
                    on_epoch=True,
                    prog_bar=True,
                    sync_dist=True,
                    logger=True,
                )
                self.validation_step_outputs.append(loss)

                if batch_nb % 20 == 0:
                    input_audio = sources[0]
                    target_audio = targets[0]
                    output_audio = est_targets[0]
                    
                    self.logger.experiment.add_audio(
                        tag=f'{"val"}/input/batch_{batch_nb}/sample_/channel_0',
                        snd_tensor=input_audio[:],
                        global_step=self.current_epoch,
                        sample_rate=self.sr
                    )
                    self.logger.experiment.add_audio(
                        tag=f'{"val"}/output/batch_{batch_nb}/sample_/channel_0',
                        snd_tensor=output_audio[:],
                        global_step=self.current_epoch,
                        sample_rate=self.sr
                    )
                    self.logger.experiment.add_audio(
                        tag=f'{"val"}/target/batch_{batch_nb}/sample_/channel_0',
                        snd_tensor=target_audio[:],
                        global_step=self.current_epoch,
                        sample_rate=self.sr
                    )

                return {"val_loss": loss}

            ## cal test loss
            if (self.trainer.current_epoch) % 10 == 0 and dataloader_idx == 1:
                sources, targets, _ = batch
                # print(mixtures.shape)
                est_targets = self(sources)
                tloss = self.loss_func["val"](est_targets, targets)
                self.log(
                    "test_loss",
                    tloss,
                    on_epoch=True,
                    prog_bar=True,
                    sync_dist=True,
                    logger=True,
                )
                self.test_step_outputs.append(tloss)
                return {"test_loss": tloss}


    def test_step(self, batch, batch_nb, dataloader_idx):
        with torch.no_grad():
            sources, targets, _ = batch
            # print(mixtures.shape)
            est_targets = self(sources)

            input_audio = sources[0]
            target_audio = targets[0]
            output_audio = est_targets[0]
                
            self.logger.experiment.add_audio(
                tag=f'{"test"}/input/batch_{batch_nb}/sample_/channel_0',
                snd_tensor=input_audio[:],
                global_step=self.current_epoch,
                sample_rate=self.sr
            )
            self.logger.experiment.add_audio(
                tag=f'{"test"}/output/batch_{batch_nb}/sample_/channel_0',
                snd_tensor=output_audio[:],
                global_step=self.current_epoch,
                sample_rate=self.sr
            )
            return 

            ## cal test loss
            if (self.trainer.current_epoch) % 10 == 0 and dataloader_idx == 1:
                sources, targets, _ = batch
                # print(mixtures.shape)
                est_targets = self(sources)
                tloss = self.loss_func["val"](est_targets, targets)
                self.log(
                    "test_loss",
                    tloss,
                    on_epoch=True,
                    prog_bar=True,
                    sync_dist=True,
                    logger=True,
                )
                self.test_step_outputs.append(tloss)
                return {"test_loss": tloss}
        pass
    
    def on_train_epoch_end(self):
        #self.log_waveforms(self.train_loader(), split='train')
        pass

    def on_test_epoch_end(self):
        #self.log_waveforms(self.test_loader_loader, split='test')
        pass

    def on_validation_epoch_end(self): 
        self.log(
           "lr",
           self.optimizer.param_groups[0]["lr"],
           on_epoch=True,
           prog_bar=True,
           sync_dist=True,
        )
        self.logger.experiment.add_scalar(
           "learning_rate", self.optimizer.param_groups[0]["lr"], self.current_epoch
        )

    def configure_optimizers(self):
        """Initialize optimizers, batch-wise and epoch-wise schedulers."""
        if self.scheduler is None:
            return self.optimizer

        if not isinstance(self.scheduler, (list, tuple)):
            self.scheduler = [self.scheduler]  # support multiple schedulers

        epoch_schedulers = []
        for sched in self.scheduler:
            if not isinstance(sched, dict):
                if isinstance(sched, ReduceLROnPlateau):
                    sched = {"scheduler": sched, "monitor": self.default_monitor}
                epoch_schedulers.append(sched)
            else:
                sched.setdefault("monitor", self.default_monitor)
                sched.setdefault("frequency", 1)
                # Backward compat
                if sched["interval"] == "batch":
                    sched["interval"] = "step"
                assert sched["interval"] in [
                    "epoch",
                    "step",
                ], "Scheduler interval should be either step or epoch"
                epoch_schedulers.append(sched)
        return [self.optimizer], epoch_schedulers

    # def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
    #     if metric is None:
    #         scheduler.step()
    #     else:
    #         scheduler.step(metric)

    def train_dataloader(self):
        """Training dataloader"""
        return self.train_loader

    def val_dataloader(self):
        """Validation dataloader"""
        return [self.val_loader, self.test_loader]

    # def on_save_checkpoint(self, checkpoint):
    #     """Overwrite if you want to save more things in the checkpoint."""
    #     #checkpoint["training_config"] = self.config
    #     #return checkpoint
    #     return 
    @staticmethod
    def config_to_hparams(dic):
        """Sanitizes the config dict to be handled correctly by torch
        SummaryWriter. It flatten the config dict, converts ``None`` to
        ``"None"`` and any list and tuple into torch.Tensors.

        Args:
            dic (dict): Dictionary to be transformed.

        Returns:
            dict: Transformed dictionary.
        """
        dic = flatten_dict(dic)
        for k, v in dic.items():
            if v is None:
                dic[k] = str(v)
            elif isinstance(v, (list, tuple)):
                dic[k] = torch.tensor(v)
        return dic
    

class CombinedLightningModule(pl.LightningModule):
    def __init__(
        self,
        frozen_model: torch.nn.Module,
        new_network: torch.nn.Module,
        loss_func: dict,
        optimizer_config: dict,
        scheduler_config: dict = None,
        train_loader = None,
        val_loader = None,
        test_loader = None,
        sr: int = 44100,
        config: dict = None,
    ):
        super().__init__()
        # Model components
        self.frozen_model = frozen_model
        self.new_network = new_network
        
        # Freeze the pretrained model
        for param in self.frozen_model.parameters():
            param.requires_grad = False
        self.frozen_model.eval()

        # Training components
        self.loss_func = loss_func
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        # Configuration
        self.sr = sr
        self.config = config or {}
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, x):
        """Forward pass through both models"""
        with torch.no_grad():
            frozen_out = self.frozen_model(x)
        return self.new_network(frozen_out), frozen_out

    def training_step(self, batch, batch_idx):
        sources, targets, _ = batch
        
        # Apply speed augmentation if configured
        if self.config.get("SpeedAug", False):
            sources, targets = self._apply_speed_aug(sources, targets)
        
        # Forward pass
        new_out, frozen_out = self(sources)
        loss = self.loss_func['train'](new_out, targets)

        # Logging
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        
        # Optional: Log training audio samples periodically
        if batch_idx % 100 == 0:
            self._log_audio_samples(sources, new_out, frozen_out, targets, "train", batch_idx)
            
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        sources, targets, _ = batch
        new_out, frozen_out = self(sources)
        
        # Calculate losses
        val_loss = self.loss_func['val'](new_out, targets)
        frozen_loss = self.loss_func['val'](frozen_out, targets)

        # Logging
        self.log("val/new_loss", val_loss, prog_bar=True, sync_dist=True)
        self.log("val/frozen_loss", frozen_loss, sync_dist=True)
        
        # Log audio samples for first validation loader
        if dataloader_idx == 0 and batch_idx % 10 == 0:
            self._log_audio_samples(sources, new_out, frozen_out, targets, "val", batch_idx)

        return {"val_loss": val_loss, "frozen_loss": frozen_loss}

    def test_step(self, batch, batch_idx):
        sources, targets, _ = batch
        new_out, frozen_out = self(sources)
        
        # Calculate and log losses
        test_loss = self.loss_func['val'](new_out, targets)
        frozen_test_loss = self.loss_func['val'](frozen_out, targets)
        
        self.log("test/new_loss", test_loss, sync_dist=True)
        self.log("test/frozen_loss", frozen_test_loss, sync_dist=True)
        
        # Log audio samples
        if batch_idx % 10 == 0:
            self._log_audio_samples(sources, new_out, frozen_out, targets, "test", batch_idx)
            
        return {"test_loss": test_loss, "frozen_test_loss": frozen_test_loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.new_network.parameters(), 
            **self.optimizer_config
        )
        
        if not self.scheduler_config:
            return optimizer
            
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            **self.scheduler_config
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/new_loss",
                "interval": "epoch",
                "frequency": 1
            }
        }

    def _log_audio_samples(self, sources, new_out, frozen_out, targets, split, batch_idx):
        """Helper method to log audio samples from all stages"""
        idx = 0  # Log first sample in batch
        
        # Log original input
        self.logger.experiment.add_audio(
            f"{split}/input_{batch_idx}", 
            sources[idx], 
            self.current_epoch, 
            sample_rate=self.sr
        )
        
        # Log frozen model output
        self.logger.experiment.add_audio(
            f"{split}/frozen_out_{batch_idx}", 
            frozen_out[idx], 
            self.current_epoch, 
            sample_rate=self.sr
        )
        
        # Log new network output
        self.logger.experiment.add_audio(
            f"{split}/new_out_{batch_idx}", 
            new_out[idx], 
            self.current_epoch, 
            sample_rate=self.sr
        )
        
        # Log target
        self.logger.experiment.add_audio(
            f"{split}/target_{batch_idx}", 
            targets[idx], 
            self.current_epoch, 
            sample_rate=self.sr
        )

    def _apply_speed_aug(self, sources, targets):
        """Speed augmentation implementation"""
        # Add your speed perturbation implementation here
        # Return augmented sources and targets
        return sources, targets

    # DataLoader methods
    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return [self.val_loader, self.test_loader]  # Maintain multiple val loaders

    def test_dataloader(self):
        return self.test_loader
        # Clear stored outputs after epoch
        self.frozen_outputs.clear()
        super().on_test_epoch_end()
        """Helper method for logging audio examples"""
        input_audio = sources[0]
        target_audio = targets[0]
        output_audio = est_targets[0]
        
        self.logger.experiment.add_audio(
            f'{split}/input/batch_{batch_nb}/channel_0',
            input_audio,
            self.current_epoch,
            sample_rate=self.sr
        )
        self.logger.experiment.add_audio(
            f'{split}/output/batch_{batch_nb}/channel_0',
            output_audio,
            self.current_epoch,
            sample_rate=self.sr
        )
        self.logger.experiment.add_audio(
            f'{split}/target/batch_{batch_nb}/channel_0',
            target_audio,
            self.current_epoch,
            sample_rate=self.sr
        )