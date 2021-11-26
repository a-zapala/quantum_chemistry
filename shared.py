import pytorch_lightning as pl
import numpy as np
import torch

from torch_geometric.datasets import QM9


class QM9MainDatamodule(pl.LightningDataModule):
    def __init__(self, root="./datasets", batch_size=32, num_workers=10):
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        dataset = QM9(self.root)

        # DimeNet uses the atomization energy for targets U0, U, H, and G.
        idx = torch.tensor([0, 1, 2, 3, 4, 5, 6, 12, 13, 14, 15, 11])
        dataset.data.y = dataset.data.y[:, idx]
        dataset.data.y = (dataset.data.y - dataset.data.y.mean(axis=0)) / dataset.data.y.std(axis=0)

        # Use the same random seed as the official DimeNet` implementation.
        random_state = np.random.RandomState(seed=42)
        perm = torch.from_numpy(
            random_state.permutation(np.arange(len(dataset))))

        train_idx = perm[:110000]
        val_idx = perm[110000:120000]
        test_idx = perm[120000:]

        self.train_set = dataset[train_idx]
        self.val_set = dataset[val_idx]
        self.test_set = dataset[test_idx]
    
    def predict_dataloader(self):
        return self.test_dataloader()

TARGETS_NAMES = ["mu", "alpha", "homo", "lumo", "gap", "r2", "ZPVE", "U0", "U", "H", "G", "Cv"]

def calculate_metrics(outputs, ys):
    # return mea and std mea, respect to every target and means of these metrics
    maes = (ys - outputs).abs().mean(axis=0)
    stds = ys.std(axis=0)
    std_maes = maes / stds
    return maes.squeeze().cpu().numpy(), std_maes.squeeze().cpu().numpy(), maes.mean().cpu().item(), std_maes.mean().cpu().item()


class Module(pl.LightningModule):
    def __init__(self, lr=1e-3):
        super().__init__()
        self.lr = lr
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        loss, _, _ = self.step(batch, batch_idx)
        self.log("loss/train", loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, _, _ = self.step_validation(batch, batch_idx)
        self.log("loss/val", loss, on_step=True, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, output, y = self.step(batch, batch_idx)
        self.log("loss/test", loss, on_step=True, on_epoch=True)
        maes, std_maes, mae, std_mae = calculate_metrics(output, y)

        for n, m in zip(TARGETS_NAMES, maes):
            self.log(f"maes/{n}", m)

        for n, m in zip(TARGETS_NAMES, std_maes):
            self.log(f"std_maes/{n}", m)

        self.log("mae", mae)
        self.log("std_mae", std_mae)
        return loss
    
    def prediction_step(self, batch, batch_idx):
        _, output, _ = self.step(batch, batch_idx)
        return output

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
