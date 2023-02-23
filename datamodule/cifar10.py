import os
from typing import Optional, Tuple, Union

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.datasets import CIFAR10


class CIFAR10DataModule(LightningDataModule):
    """
    With reference from
    https://github.com/Lightning-AI/lightning-bolts/blob/master/pl_bolts/datamodules/vision_datamodule.py
    """

    name: str = "cifar10"
    dataset_cls: type = CIFAR10
    dims: tuple[int, ...] = (3, 32, 32)

    def __init__(
            self,
            data_dir: Optional[str] = None,
            val_split: Union[int, float] = 0.2,
            num_workers: int = 0,
            batch_size: int = 32,
            seed: int = 42,
            shuffle: bool = True,
            pin_memory: bool = True,
            drop_last: bool = False,
            crop_size: int = 224,
            resize_size: int = 232,
            img_mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
            img_std: Tuple[float, ...] = (0.229, 0.224, 0.225),
    ) -> None:
        super().__init__()
        self.data_dir = data_dir if data_dir is not None else os.getcwd()
        self.val_split = val_split
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.crop_size = crop_size
        self.resize_size = resize_size
        self.img_mean = img_mean
        self.img_std = img_std
        self.transform = transforms.Compose(
            [
                transforms.Resize(self.resize_size),
                transforms.CenterCrop(self.crop_size),
                transforms.ToTensor(),
                transforms.ConvertImageDtype(torch.float),
                transforms.Normalize(mean=self.img_mean, std=self.img_std),
            ]
        )

    def prepare_data(self) -> None:
        """Saves files to data_dir."""
        self.dataset_cls(self.data_dir, train=True, download=True)
        self.dataset_cls(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None) -> None:
        """Creates train, val, and test dataset."""
        if stage == "fit" or stage is None:
            dataset = self.dataset_cls(
                self.data_dir, train=True, transform=self.transform
            )

            # Split
            self.dataset_train, self.dataset_val = self._split_dataset(dataset)

        if stage == "test" or stage is None:
            self.dataset_test = self.dataset_cls(
                self.data_dir, train=False, transform=self.transform
            )

    @property
    def num_classes(self) -> int:
        return 10

    def _get_splits(self, len_dataset: int) -> list[int]:
        """Computes split lengths for train and validation set."""
        if isinstance(self.val_split, int):
            train_len = len_dataset - self.val_split
            val_len = self.val_split
        elif isinstance(self.val_split, float):
            val_len = int(self.val_split * len_dataset)
            train_len = len_dataset - val_len
        else:
            raise ValueError(f"Unsupported type {type(self.val_split)}")

        return [train_len, val_len]

    def _split_dataset(self, dataset: Dataset) -> tuple[Dataset, Dataset]:
        """Splits the dataset into train and validation set."""
        len_dataset = len(dataset)
        splits = self._get_splits(len_dataset)

        train_dataset, val_dataset = random_split(
            dataset, splits, generator=torch.Generator().manual_seed(self.seed)
        )

        return train_dataset, val_dataset

    def train_dataloader(self) -> DataLoader:
        return self._data_loader(self.dataset_train, shuffle=self.shuffle)

    def val_dataloader(self) -> DataLoader:
        return self._data_loader(self.dataset_val)

    def test_dataloader(self) -> DataLoader:
        return self._data_loader(self.dataset_test)

    def _data_loader(self, dataset: Dataset, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )
