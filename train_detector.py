import argparse
from functools import partial
from pathlib import Path
from pprint import pformat

import pytorch_lightning as pl
import torch
import torchvision.transforms as tf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from twitter_ai_faces.data import RealFakeDataModule
from twitter_ai_faces.image import JPEG
from twitter_ai_faces.models import ResNet50

torch.set_float32_matmul_precision("high")


def main(args):
    pl.seed_everything(args.seed)

    # create output directory
    output_dir = Path("results/train_detector") / args.run_name
    output_dir.mkdir(parents=True)

    # save configuration
    with open(output_dir / "config.txt", "w") as f:
        f.write(pformat(vars(args), sort_dicts=False))

    # define transforms
    train_transform = tf.Compose(
        [
            tf.ToTensor(),
            tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            tf.RandomCrop(size=224, pad_if_needed=True),
        ]
    )
    test_transform = tf.Compose(
        [
            tf.ToTensor(),
            tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            tf.CenterCrop(size=224),
        ]
    )
    augmentation = tf.Compose(
        [
            tf.RandomApply(
                [tf.GaussianBlur(kernel_size=9, sigma=(0.5, 5.0))], p=args.aug_p
            ),
            tf.RandomApply([JPEG(quality=(30, 100))], p=args.aug_p),
            tf.RandomApply(
                [
                    tf.RandomResizedCrop(
                        size=224,
                        scale=(0.25, 0.75),
                        ratio=(0.8, 1 / 0.8),
                        antialias=True,
                    )
                ],
                p=args.aug_p,
            ),
        ]
    )

    # set up datamodule
    dm = RealFakeDataModule(
        real_dirs=args.real_dirs,
        fake_dirs=args.fake_dirs,
        train_transform=train_transform,
        test_transform=test_transform,
        augmentation=augmentation,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # set up optimization
    optimizer = partial(torch.optim.Adam, lr=0.0001, betas=(0.9, 0.999))
    lr_scheduler = partial(
        torch.optim.lr_scheduler.ReduceLROnPlateau,
        factor=0.1,
        patience=5,
        threshold=0.001,
        threshold_mode="abs",
    )

    # set up model
    model = ResNet50(optimizer=optimizer, lr_scheduler=lr_scheduler)

    # set up trainer
    trainer = pl.Trainer(
        logger=CSVLogger(save_dir=output_dir / "logs"),
        callbacks=ModelCheckpoint(
            dirpath=output_dir / "checkpoints", monitor="val_loss"
        ),
        deterministic=True,
        max_epochs=-1,
        devices=1,
    )

    # train and test
    trainer.fit(model, datamodule=dm)
    trainer.test(model, datamodule=dm)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", default="default", help="Name of the run")
    parser.add_argument(
        "--real-dirs",
        nargs="+",
        type=Path,
        help="One or multiple paths to directories containing real images",
    )
    parser.add_argument(
        "--fake-dirs",
        nargs="+",
        type=Path,
        help="One or multiple paths to directories containing fake images",
    )
    parser.add_argument(
        "--aug-p", type=float, default=0.0, help="Probability for data augmentation"
    )

    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=8, help="Number of workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
