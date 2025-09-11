from argparse import ArgumentParser
from models.zones_segmentation_model import ZonesUNet
from models.front_segmentation_model import FrontUNet
from data_processing.glacier_zones_data import GlacierZonesDataModule
from data_processing.glacier_front_data import GlacierFrontDataModule
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
import sys
from torchsummary import summary
from pytorch_lightning.callbacks import ModelCheckpoint
import time
import torch
import os
from pytorch_lightning.callbacks import LearningRateMonitor, Callback
#from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import builtins
import matplotlib.pyplot as plt


class LossHistory(Callback):
    """Callback to record and plot loss curves."""

    def __init__(self):
        self.train_losses = []
        self.val_losses = []

    def on_train_epoch_end(self, trainer, pl_module):
        loss = trainer.callback_metrics.get('avg_train_loss')
        if loss is not None:
            self.train_losses.append(loss.detach().cpu().item())

    def on_validation_epoch_end(self, trainer, pl_module):
        loss = trainer.callback_metrics.get('avg_loss_validation')
        if loss is not None:
            self.val_losses.append(loss.detach().cpu().item())

    def on_fit_end(self, trainer, pl_module):
        if not self.train_losses:
            return
        epochs = range(1, len(self.train_losses) + 1)
        plt.figure()
        plt.plot(epochs, self.train_losses, label='Train Loss')
        if self.val_losses:
            plt.plot(epochs[:len(self.val_losses)], self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.xticks(epochs)
        plt.legend()
        plt.title('Loss over Epochs')
        log_dir = getattr(trainer.logger, 'log_dir', None)
        if log_dir:
            plt.savefig(os.path.join(log_dir, 'loss.png'))
        plt.close()


def main(hparams, run_number, running_mode):
    if hparams.target_masks == "zones":
        checkpoint_dir = os.path.join('checkpoints', 'zones_segmentation', 'run_' + str(run_number))
        tb_logs_dir = os.path.join('tb_logs', 'zones_segmentation', 'run_' + str(run_number))
        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename='{epoch:02d}-{avg_train_loss:.2f}',
            monitor='avg_train_loss',
            mode='max',
            save_top_k=-1,
            save_last=True,
        )
        clip_norm = 1.0
    else:
        checkpoint_dir = os.path.join('checkpoints', 'fronts_segmentation', 'run_' + str(run_number))
        tb_logs_dir = os.path.join('tb_logs', 'fronts_segmentation', 'run_' + str(run_number))
        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename='{epoch:02d}-{avg_train_loss:.2f}',
            monitor='avg_train_loss',
            mode='max',
            save_top_k=-1,
            save_last=True,
        )
        clip_norm = 1.0

    logger = TensorBoardLogger(tb_logs_dir, name="log", default_hp_metric=False)
    lr_monitor = LearningRateMonitor(logging_interval='step')

    loss_history = LossHistory()
    callbacks = [checkpoint_callback, lr_monitor, loss_history]


    # if we have already trained this model take up the training at the last checkpoint
    latest_ckpt_path = None
    ckpt_files = [f for f in os.listdir(checkpoint_dir) if f.startswith("epoch=") and f.endswith(".ckpt")]
    if ckpt_files:
        def _epoch_from_name(name: str) -> int:
            try:
                return int(name.split('-')[0].split('=')[1])
            except Exception:
                return -1
        latest_ckpt = max(ckpt_files, key=_epoch_from_name)
        latest_ckpt_path = os.path.join(checkpoint_dir, latest_ckpt)

    if latest_ckpt_path:
        print(f"Taking up the training where it was left ({os.path.basename(latest_ckpt_path)})")
        # When resuming from a checkpoint, `max_epochs` is treated as the
        # total number of epochs to run. To allow continuing training for
        # additional epochs, we extend `max_epochs` by the number of epochs
        # already completed.
        ckpt = torch.load(latest_ckpt_path, map_location="cpu")
        new_max_epochs = ckpt.get("epoch", 0) + hparams.epochs
        trainer = Trainer(
            resume_from_checkpoint=latest_ckpt_path,
            callbacks=callbacks,
            deterministic=True,
            gpus=1,  # Train on gpu
            gradient_clip_val=clip_norm,
            logger=logger,
            max_epochs=new_max_epochs,
            num_sanity_val_steps=0,
        )
    else:
        if running_mode == "batch_overfit":
            # Try to overfit on some batches
            trainer = Trainer.from_argparse_args(
                hparams,
                callbacks=callbacks,
                deterministic=True,
                fast_dev_run=False,
                flush_logs_every_n_steps=100,
                gpus=1,
                log_every_n_steps=1,
                logger=logger,
                max_epochs=1000,
                overfit_batches=1,
                num_sanity_val_steps=0,
            )
        elif running_mode == "debugging":
            # Debugging mode
            trainer = Trainer.from_argparse_args(
                hparams,
                callbacks=callbacks,
                deterministic=True,
                fast_dev_run=3,
                flush_logs_every_n_steps=100,
                gpus=1,
                log_every_n_steps=1,
                logger=logger,
                num_sanity_val_steps=0,
            )
        elif running_mode == "training":
            # Training mode
            trainer = Trainer.from_argparse_args(
                hparams,
                callbacks=callbacks,
                deterministic=True,
                gpus=1,  # Train on gpu
                gradient_clip_val=clip_norm,
                logger=logger,
                max_epochs=hparams.epochs,
                num_sanity_val_steps=0,
            )
        else:
            print("Running mode not recognized")
            sys.exit()

    if hparams.target_masks == "zones":
        datamodule = GlacierZonesDataModule(batch_size=hparams.batch_size,
                                            augmentation=running_mode != "batch_overfit",
                                            parent_dir=hparams.parent_dir,
                                            bright=hparams.bright,
                                            wrap=hparams.wrap,
                                            noise=hparams.noise,
                                            rotate=hparams.rotate,
                                            flip=hparams.flip,
                                            load_val=False)  # Flase = skip validation, True otherwise
        model = ZonesUNet(vars(hparams))

    else:
        datamodule = GlacierFrontDataModule(batch_size=hparams.batch_size,
                                            augmentation=running_mode != "batch_overfit",
                                            parent_dir=hparams.parent_dir,
                                            bright=hparams.bright,
                                            wrap=hparams.wrap,
                                            noise=hparams.noise,
                                            rotate=hparams.rotate,
                                            flip=hparams.flip,
                                            load_val=False) # Flase = skip validation, True otherwise
        model = FrontUNet(vars(hparams))

    summary(model.cuda(), (1, 256, 256))
    print(model.eval())

    trainer.fit(model, datamodule=datamodule)


if __name__ == '__main__':
    seed_everything(42)
    torch.multiprocessing.set_start_method('spawn')  # needed if ddp mode in Trainer

    parent_parser = ArgumentParser(add_help=False)
    # add PROGRAM level args
    parent_parser.add_argument('--parent_dir', default="../",
                               help="The directory in which the data directory lies. "
                                    "Default is '.' - this is where the data_preprocessing script has produced it.")
    parent_parser.add_argument('--training_mode', default="training",
                               help="Either 'training', 'debugging' or 'batch_overfit'.")
    parent_parser.add_argument('--epochs', type=int, default=15, help="The number of epochs the model shall be trained."
                                                                       "The weights after the last training epoch will "
                                                                       "be stored in temporary.ckpt."
                                                                       "Train.py will resume training from a "
                                                                       "temporary.ckpt if one is available for this "
                                                                       "be stored in files named 'epoch=...ckpt' inside the checkpoint directory."
                                                                       "Train.py will resume training from the latest such checkpoint if one is available for this "
                                                                       "run.")
    parent_parser.add_argument('--run_number', type=int, default=0,
                               help="The number how often this model was already trained. "
                                    "If you run train.py twice with the same run_number, "
                                    "the second run will pick up training from the most recent 'epoch=...ckpt' file.")
    parent_parser.add_argument('--target_masks', default="fronts", help="Either 'fronts' or 'zones'. "
                                                                        "This decides which model will be trained.")

    tmp = parent_parser.parse_args()
    if tmp.target_masks == "fronts":
        parser = FrontUNet.add_model_specific_args(parent_parser)
    else:
        parser = ZonesUNet.add_model_specific_args(parent_parser)
    parser = Trainer.add_argparse_args(parser)
    hparams = parser.parse_args()

    if hparams.target_masks == "zones":
        log_dir = os.path.join('checkpoints', 'zones_segmentation', 'run_' + str(hparams.run_number))
    else:
        log_dir = os.path.join('checkpoints', 'fronts_segmentation', 'run_' + str(hparams.run_number))
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, "log.txt")
    log_file = open(log_file_path, "w")
    original_print = builtins.print

    def print_and_log(*args, **kwargs):
        original_print(*args, **kwargs)
        original_print(*args, file=log_file, **kwargs)
        log_file.flush()

    builtins.print = print_and_log

    try:
        print(vars(hparams))

        assert hparams.target_masks == "fronts" or hparams.target_masks == "zones", \
            "Please set --target_masks correctly. Either 'fronts' or 'zones'."
        assert hparams.training_mode == "training" or hparams.training_mode == "debugging" or hparams.training_mode == "batch_overfit", \
            "Please set --training_mode correctly. Either 'training' or 'debugging' or 'batch_overfit'."

        start_save = time.time()
        main(hparams, hparams.run_number, hparams.training_mode)
        end_save = time.time()
        print(f"Total time for training: {(end_save - start_save)/60:.2f} minutes")
    finally:
        builtins.print = original_print
        log_file.close()

    # run the following if you want to open tensorboard
    # tensorboard --logdir tb_logs


