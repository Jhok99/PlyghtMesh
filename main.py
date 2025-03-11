import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning import seed_everything

from model import Adapt_gen_pl
from datasets.ModelNet40Ply2048 import ModelNet40Ply2048DataModule
import open3d as o3d
import os
import torch
import numpy as np

from pytorch_lightning.loggers import TensorBoardLogger

@hydra.main(config_path=".", config_name="config", version_base=None)
def main(cfg: DictConfig):
    seed_everything(cfg.experiment.seed)

    ply_data_module = ModelNet40Ply2048DataModule(
        data_dir="datasets/ply_shapenet",
        batch_size=cfg.train.batch_size,
        drop_last=True,
        num_workers=cfg.train.num_workers,
    )

    ply_data_module.setup()

    model = Adapt_gen_pl(
        cfg,
        embed_dim=cfg.model.embed_dim,
        n_points=ply_data_module.num_points,
        n_blocks=cfg.model.n_blocks,
        groups=cfg.model.groups,
    )

    callbacks = [
        ModelCheckpoint(
            dirpath="checkpoints",
            filename="norepulsion-convonet-lowmasknotransl-{epoch:02d}-{val_loss:.5f}",
            save_top_k=1,
            verbose=True,
            monitor="val_loss",
            mode="min",
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    logger = TensorBoardLogger("tb_logs", name=cfg.experiment.name)

    trainer = pl.Trainer(
        max_epochs=cfg.train.epochs,
        accelerator="gpu" if cfg.cuda else "cpu",
        devices=1,
        callbacks=callbacks
    )

    trainer.fit(
        model,
        train_dataloaders=ply_data_module.train_dataloader(),
        val_dataloaders=ply_data_module.val_dataloader(),
        #ckpt_path="checkpoints/convonet-lowmasktransl-epoch=07-val_loss=1.03661.ckpt"
    )


if __name__ == "__main__":
    main()
