import warnings
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple, Union

import pandas as pd
import pytorch_lightning as pl
import torch
import torchvision.transforms as tf
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchsampler import ImbalancedDatasetSampler

IMG_EXTENSIONS = [".jpg", ".jpeg", ".jpe", ".jfif", ".png"]


def get_file_list(path: Union[str, Path]):
    return sorted(Path(path).rglob("*"))


class ImageFolder(Dataset):
    """Read images from directory or from specified paths."""

    def __init__(
        self,
        path: Union[List[Union[str, Path]], Union[str, Path]],
        label: Optional[int] = None,
        transform: Optional[Callable] = None,
    ) -> None:
        self.path = path
        if isinstance(path, (Path, str)):
            path = [Path(path)]

        self.img_paths = []
        for p in path:
            p = Path(p)
            if p.is_dir():
                for file in get_file_list(p):
                    if file.suffix in IMG_EXTENSIONS:
                        self.img_paths.append(file)
            else:
                self.img_paths.append(p)
        self.label = label
        self.transform = transform

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Union[str, int]]:
        with warnings.catch_warnings():  # silence RGBA warning
            warnings.simplefilter("ignore")
            img = Image.open(self.img_paths[idx]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        if self.label is None:
            return img, str(self.img_paths[idx])
        else:
            return img, self.label

    def __repr__(self) -> str:
        return "\n".join(
            [
                "ImageFolder",
                f"Path: {self.path}",
                f"Label: {self.label}",
                f"Length: {self.__len__()}",
                f"Transform: {self.transform}",
            ]
        )


def downcast_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast DataFrame to smaller data types."""
    fcols = df.select_dtypes("float").columns
    icols = df.select_dtypes("integer").columns
    ucols = icols[(~df[icols] < 0).any()]
    icols = icols.difference(ucols)
    df[fcols] = df[fcols].apply(pd.to_numeric, downcast="float")
    df[icols] = df[icols].apply(pd.to_numeric, downcast="integer")
    df[ucols] = df[ucols].apply(pd.to_numeric, downcast="unsigned")
    return df


class RealFakeDataModule(pl.LightningDataModule):
    """PL DataModule for real-vs-fake classification."""

    def __init__(
        self,
        real_dirs: Union[Path, Iterable[Path]],
        fake_dirs: Union[Path, Iterable[str]],
        train_transform: Optional[Callable] = None,
        test_transform: Optional[Callable] = None,
        augmentation: Optional[Callable] = None,
        batch_size: int = 1,
        num_workers: int = 0,
    ) -> None:
        super().__init__()
        self.real_dirs = [real_dirs] if isinstance(real_dirs, Path) else real_dirs
        self.fake_dirs = [fake_dirs] if isinstance(fake_dirs, Path) else fake_dirs
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.augmentation = augmentation
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str) -> None:
        if stage == "fit":
            # augmentation before train_transform
            if self.augmentation is None:
                train_transform = self.train_transform
            else:
                train_transform = tf.Compose([self.augmentation, self.train_transform])

            ds_train_real = [
                ImageFolder(path=dir / "train", label=0, transform=train_transform)
                for dir in self.real_dirs
            ]
            ds_train_fake = [
                ImageFolder(path=dir / "train", label=1, transform=train_transform)
                for dir in self.fake_dirs
            ]
            self.ds_train = ConcatDataset(ds_train_real + ds_train_fake)
            self.ds_train_labels = [0] * sum(len(ds) for ds in ds_train_real) + [
                1
            ] * sum(len(ds) for ds in ds_train_fake)

            ds_val_real = [
                ImageFolder(path=dir / "val", label=0, transform=self.test_transform)
                for dir in self.real_dirs
            ]
            ds_val_fake = [
                ImageFolder(path=dir / "val", label=1, transform=self.test_transform)
                for dir in self.fake_dirs
            ]
            self.ds_val = ConcatDataset(ds_val_real + ds_val_fake)

        if stage == "test":
            ds_test_real = [
                ImageFolder(path=dir / "test", label=0, transform=self.test_transform)
                for dir in self.real_dirs
            ]
            ds_test_fake = [
                ImageFolder(path=dir / "test", label=1, transform=self.test_transform)
                for dir in self.fake_dirs
            ]
            self.ds_test = ConcatDataset(ds_test_real + ds_test_fake)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.ds_train,
            sampler=ImbalancedDatasetSampler(
                self.ds_train, labels=self.ds_train_labels
            ),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.ds_val, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.ds_test, batch_size=self.batch_size, num_workers=self.num_workers
        )
