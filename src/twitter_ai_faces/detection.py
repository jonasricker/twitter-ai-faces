import abc
from pathlib import Path
from typing import Callable, List, Optional, Union

import pandas as pd
import torch
import torchvision.transforms as tf
from torch.utils.data import DataLoader
from tqdm import tqdm

from twitter_ai_faces.data import ImageFolder, downcast_dataframe
from twitter_ai_faces.misc import get_device
from twitter_ai_faces.models import ResNet50
from twitter_ai_faces.optimization import SimpleMemory


def compute_predictions(
    path: Union[Path, List[Path]],
    weights_path: Path,
    crop_size: Optional[int] = None,
    augmentation: Optional[Callable] = None,
    batch_size: int = 1,
    num_workers: int = 1,
) -> pd.DataFrame:
    """Compute scores for images using detector."""
    device = get_device()

    # set up detector
    DETECTORS = {
        "resnet": ResNetDetector,
    }
    for name, cls in DETECTORS.items():
        if name in str(weights_path):
            detector = cls(weights_path)
            break
    else:
        raise NotImplementedError
    detector.setup()
    detector.model.eval()
    if torch.cuda.device_count() > 1:
        detector.model = torch.nn.DataParallel(detector.model)
    detector.model.to(device)

    # set up transform
    transforms = [tf.ToTensor(), detector.normalizer]
    if detector.additional_transforms is not None:
        transforms.extend(detector.additional_transforms)
    elif crop_size is not None:
        transforms.append(tf.CenterCrop(crop_size))
    transform = tf.Compose(transforms)

    # set up augmentation
    if augmentation is not None:
        transform = tf.Compose([augmentation, transform])

    ds = ImageFolder(path, transform=transform)
    dl = DataLoader(ds, batch_size=batch_size, num_workers=num_workers)
    scores, files = [], []
    for image_batch, file_batch in tqdm(
        dl, desc=f"Classifying {str(path):.50} with {detector}"
    ):
        with torch.no_grad():
            scores.extend(detector.predict(image_batch.to(device)).tolist())
        files.extend(file_batch)

    data = {
        "file": files,
        "score": scores,
    }
    df = pd.DataFrame(data)
    return downcast_dataframe(df)


compute_predictions = SimpleMemory(
    location=".cache", backend="pandas", verbose=0
).cache(compute_predictions, ignore=["batch_size", "num_workers"])


class Detector(abc.ABC):
    normalizer = tf.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    additional_transforms: Optional[List[Callable]] = None

    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path

    @abc.abstractmethod
    def setup(self) -> None:
        pass

    @abc.abstractmethod
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.model_path})"


class ResNetDetector(Detector):
    def setup(self) -> None:
        self.model = ResNet50.load_from_checkpoint(self.model_path)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).sigmoid().squeeze(1)
