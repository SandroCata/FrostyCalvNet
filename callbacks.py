from pytorch_lightning.callbacks import ModelCheckpoint, Callback

# Progress bar callback changed locations across Lightning versions.
# Try importing the modern `TQDMProgressBar` and fall back to the legacy
# `ProgressBar` if necessary so training works with older releases like 1.3.x.
try:  # pragma: no cover - depends on installed Lightning version
    from pytorch_lightning.callbacks import TQDMProgressBar as _BaseProgressBar
except Exception:  # older Lightning (<1.4) exposes only `ProgressBar`
    try:  # pragma: no cover - legacy import path
        from pytorch_lightning.callbacks import ProgressBar as _BaseProgressBar
    except Exception:  # pragma: no cover - very old versions
        try:
            from pytorch_lightning.callbacks.progress import TQDMProgressBar as _BaseProgressBar
        except Exception:  # pragma: no cover
            try:
                from pytorch_lightning.callbacks.progress import ProgressBar as _BaseProgressBar
            except Exception:  # pragma: no cover - give up
                _BaseProgressBar = None

import os
import matplotlib.pyplot as plt


class OneBasedModelCheckpoint(ModelCheckpoint):
    """ModelCheckpoint that labels epochs starting from 1 instead of 0."""

    def format_checkpoint_name(self, metrics, filename=None, **kwargs):
        metrics = {**metrics}
        if "epoch" in metrics:
            try:
                metrics["epoch"] = int(metrics["epoch"]) + 1
            except Exception:  # pragma: no cover - handle non-int values
                pass
        return super().format_checkpoint_name(metrics, filename, **kwargs)


if _BaseProgressBar is not None:
    class OneBasedProgressBar(_BaseProgressBar):
        """Progress bar that displays epochs starting from 1."""

        def on_train_epoch_start(self, trainer, pl_module):
            super().on_train_epoch_start(trainer, pl_module)
            bar = getattr(self, "main_progress_bar", None)
            if bar is not None:
                try:
                    bar.set_description(f"Epoch {trainer.current_epoch + 1}")
                except Exception:  # pragma: no cover - tqdm internals
                    pass

        def get_metrics(self, trainer, model):  # pragma: no cover - only for newer Lightning
            base_metrics = getattr(super(), "get_metrics", None)
            if base_metrics is None:
                return {}
            items = base_metrics(trainer, model)
            if "epoch" in items:
                items["epoch"] = trainer.current_epoch + 1
            return items
else:  # pragma: no cover - Lightning without progress bar support
    OneBasedProgressBar = None


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
        # plt.xticks(epochs)
        plt.legend()
        plt.title('Loss over Epochs')
        log_dir = getattr(trainer.logger, 'log_dir', None)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            plt.savefig(os.path.join(log_dir, 'loss.png'))
        plt.close()