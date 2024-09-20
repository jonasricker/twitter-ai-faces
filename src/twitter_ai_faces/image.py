# ruff: noqa: E402
import copy
import functools
import numbers
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import List, Union

import imagehash
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision

torchvision.disable_beta_transforms_warning()
import torchvision.transforms as tf
import torchvision.transforms.v2 as tf2
from lpips import LPIPS
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader
from torchvision.io import decode_jpeg, encode_jpeg
from torchvision.transforms.functional import (
    _is_pil_image,
    convert_image_dtype,
    to_pil_image,
    to_tensor,
)
from tqdm import tqdm

from third_party.blazeface.blazeface import BlazeFace
from twitter_ai_faces.data import ImageFolder, downcast_dataframe
from twitter_ai_faces.misc import get_device
from twitter_ai_faces.optimization import SimpleMemory

sys.path.append("src/third_party/stylegan2")
import dnnlib
import legacy


def detect_faces(
    path: Union[Path, List[Path]],
    back_model: bool = False,
    batch_size: int = 1,
    num_workers: int = 1,
) -> pd.DataFrame:
    """Compute facial landmarks using the BlazeFace face detector."""
    device = get_device()

    net = BlazeFace(back_model=back_model).to(device)
    if back_model:
        net.load_weights("weights/blazeface/blazefaceback.pth")
        net.load_anchors("weights/blazeface/anchorsback.npy")
        size = (256, 256)
    else:
        net.load_weights("weights/blazeface/blazeface.pth")
        net.load_anchors("weights/blazeface/anchors.npy")
        size = (128, 128)

    ds = ImageFolder(
        path, transform=tf2.Compose([tf2.Resize(size=size), tf2.ToImageTensor()])
    )
    dl = DataLoader(ds, batch_size=batch_size, num_workers=num_workers)

    all_faces, all_files = [], []
    for images, labels in tqdm(dl, desc=f"Detecting faces in {str(path):.50}"):
        all_files.extend(labels)
        pred = net.predict_on_batch(images)
        for faces in pred:
            if len(faces) == 0:
                all_faces.append(np.array([None] * 17))
            elif len(faces) == 1:
                all_faces.append(faces[0].cpu().numpy())
            else:
                faces = faces.cpu().numpy()
                areas = np.apply_along_axis(
                    lambda bbox: (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]),
                    1,
                    faces[:, :4],
                )
                all_faces.append(faces[np.argsort(areas)[-1]])
    columns = [
        "ymin",
        "xmin",
        "ymax",
        "xmax",
        "right_eye_x",
        "right_eye_y",
        "left_eye_x",
        "left_eye_y",
        "nose_x",
        "nose_y",
        "mouth_x",
        "mouth_y",
        "right_ear_x",
        "right_ear_y",
        "left_ear_x",
        "left_ear_y",
        "score",
    ]
    df = pd.DataFrame(all_faces, columns=columns)
    df.insert(0, "file", all_files)
    return downcast_dataframe(df)


detect_faces = SimpleMemory(location=".cache", backend="pandas", verbose=0).cache(
    detect_faces, ignore=["batch_size", "num_workers"]
)


def detect_faces_post(df: pd.DataFrame) -> pd.DataFrame:
    """Post-process facial landmarks."""
    # remove score column
    df = df.iloc[:, :-1]

    # add eye distance
    df["eye_dist"] = np.sqrt(
        (df["left_eye_x"] - df["right_eye_x"]) ** 2
        + (df["left_eye_y"] - df["right_eye_y"]) ** 2
    )

    return df


def faces_valid(
    df: pd.DataFrame,
    min_eye_dist=0.10,
) -> pd.DataFrame:
    """For each image, determine whether it contains a valid face."""
    df["valid"] = df["eye_dist"] >= min_eye_dist
    return df


def faces_aligned(
    df: pd.DataFrame, reference: pd.DataFrame, num_stds: int
) -> pd.DataFrame:
    """For each image, determine whether the face is aligned according to reference."""
    description = reference.describe()
    minimum = description.loc["mean"] - num_stds * description.loc["std"]
    maximum = description.loc["mean"] + num_stds * description.loc["std"]
    df["aligned"] = functools.reduce(
        lambda a, b: (a & b),
        [
            (df[col] > minimum[col]) & (df[col] < maximum[col])
            for col in description.columns
            if col.endswith(("_x", "_y"))
        ],
    )
    return df


class JPEG:
    def __init__(self, quality: Union[int, Sequence[int, int]] = (30, 95)) -> None:
        if isinstance(quality, numbers.Number):
            if quality <= 0 or quality > 100:
                raise ValueError(
                    "If quality is a single number, it must be between 1 and 100."
                )
            quality = (quality, quality)
        elif isinstance(quality, Sequence) and len(quality) == 2:
            if not 0 < quality[0] <= quality[1] <= 100:
                raise ValueError(
                    "quality values should be positive and of the form (min, max)."
                )
        else:
            raise ValueError(
                "quality should be a single number or a list/tuple with length 2."
            )

        self.quality = quality

    def get_params(self, quality_min: int, quality_max: int) -> int:
        return torch.randint(low=quality_min, high=quality_max + 1, size=())

    def __call__(
        self, img: Union[torch.Tensor, Image.Image]
    ) -> Union[torch.Tensor, Image.Image]:
        quality = self.get_params(
            quality_min=self.quality[0], quality_max=self.quality[1]
        )
        t_img = img
        if not isinstance(img, torch.Tensor):
            if not _is_pil_image(img):
                raise TypeError(f"img should be PIL Image or Tensor. Got {type(img)}")
            t_img = to_tensor(img)
        t_img = convert_image_dtype(t_img, torch.uint8)

        output = decode_jpeg(encode_jpeg(t_img, quality=quality))
        output = convert_image_dtype(output)

        if not isinstance(img, torch.Tensor):
            output = to_pil_image(output, mode=img.mode)
        return output


class UnNormalize(tf.Normalize):
    def __init__(self, mean, std, *args, **kwargs):
        new_mean = [-m / s for m, s in zip(mean, std)]
        new_std = [1 / s for s in std]
        super().__init__(new_mean, new_std, *args, **kwargs)


def compute_reconstruction(
    path: Union[Path, List[Path]],
    output_dir: Path,
    seed: int = 42,
    num_steps: int = 1000,
) -> List[Path]:
    """Compute reconstructions using StyleGAN2 projection and make side-by-side view."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    output_dir.mkdir(exist_ok=True, parents=True)

    # Load networks.
    device = torch.device("cuda")
    with dnnlib.util.open_url(
        "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl"
    ) as fp:
        G = legacy.load_network_pkl(fp)["G_ema"].requires_grad_(False).to(device)

    rec_paths = []
    for file in tqdm(path, desc=f"Computing reconstruction {str(path):.50}"):
        file = Path(file)
        output_path = output_dir / (file.stem + ".png")
        if not output_path.exists():
            # Load target image.
            target_pil = Image.open(file).convert("RGB")
            w, h = target_pil.size
            s = min(w, h)
            target_pil = target_pil.crop(
                ((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2)
            )
            target_pil = target_pil.resize(
                (G.img_resolution, G.img_resolution), Image.LANCZOS
            )
            target_uint8 = np.array(target_pil, dtype=np.uint8)

            # Optimize projection.
            lpips_dist, projected_w = modified_stylegan2_project(
                G,
                target=torch.tensor(target_uint8.transpose([2, 0, 1]), device=device),
                num_steps=num_steps,
                device=device,
            )
            synth_image = G.synthesis(projected_w.unsqueeze(0), noise_mode="const")
            synth_image = (synth_image + 1) * (255 / 2)
            synth_image = (
                synth_image.permute(0, 2, 3, 1)
                .clamp(0, 255)
                .to(torch.uint8)[0]
                .cpu()
                .numpy()
            )
            mse = ((synth_image - target_uint8) ** 2).mean()
            synth_image = Image.fromarray(synth_image)

            # create side-by-side image
            sidebyside = Image.new("RGB", (800, 400))
            sidebyside.paste(target_pil.resize((400, 400)), (0, 0))
            sidebyside.paste(synth_image.resize((400, 400)), (400, 0))
            draw = ImageDraw.Draw(sidebyside)
            text = f"LPIPS: {lpips_dist:.3}, MSE: {mse:.3}"
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                size=12,
            )
            bbox = draw.textbbox((0, -2), text, font=font)
            draw.rectangle(bbox, fill="white")
            draw.text((0, 0), text, font=font, fill="black")
            sidebyside.save(output_path)
        rec_paths.append(output_path)
    return rec_paths


def modified_stylegan2_project(
    G,
    target: torch.Tensor,  # [C,H,W] and dynamic range [0,255], W & H must match G output resolution
    *,
    num_steps=1000,
    w_avg_samples=10000,
    initial_learning_rate=0.1,
    initial_noise_factor=0.05,
    lr_rampdown_length=0.25,
    lr_rampup_length=0.05,
    noise_ramp_length=0.75,
    regularize_noise_weight=1e5,
    verbose=False,
    device: torch.device,
):
    """Modified projection function which returns the LPIPS distance and the reconstruction."""
    assert target.shape == (G.img_channels, G.img_resolution, G.img_resolution)

    def logprint(*args):
        if verbose:
            print(*args)

    G = copy.deepcopy(G).eval().requires_grad_(False).to(device)  # type: ignore

    # Compute w stats.
    logprint(f"Computing W midpoint and stddev using {w_avg_samples} samples...")
    z_samples = np.random.RandomState(123).randn(w_avg_samples, G.z_dim)
    w_samples = G.mapping(torch.from_numpy(z_samples).to(device), None)  # [N, L, C]
    w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)  # [N, 1, C]
    w_avg = np.mean(w_samples, axis=0, keepdims=True)  # [1, 1, C]
    w_std = (np.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5

    # Setup noise inputs.
    noise_bufs = {
        name: buf
        for (name, buf) in G.synthesis.named_buffers()
        if "noise_const" in name
    }

    # Load VGG16 feature detector.
    url = "https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt"
    with dnnlib.util.open_url(url) as f:
        vgg16 = torch.jit.load(f).eval().to(device)

    # Features for target image.
    target_images = target.unsqueeze(0).to(device).to(torch.float32)
    if target_images.shape[2] > 256:
        target_images = F.interpolate(target_images, size=(256, 256), mode="area")
    target_features = vgg16(target_images, resize_images=False, return_lpips=True)

    w_opt = torch.tensor(w_avg, dtype=torch.float32, device=device, requires_grad=True)  # pylint: disable=not-callable
    w_out = torch.zeros(
        [num_steps] + list(w_opt.shape[1:]), dtype=torch.float32, device=device
    )
    optimizer = torch.optim.Adam(
        [w_opt] + list(noise_bufs.values()),
        betas=(0.9, 0.999),
        lr=initial_learning_rate,
    )

    # Init noise.
    for buf in noise_bufs.values():
        buf[:] = torch.randn_like(buf)
        buf.requires_grad = True

    for step in range(num_steps):
        # Learning rate schedule.
        t = step / num_steps
        w_noise_scale = (
            w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
        )
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Synth images from opt_w.
        w_noise = torch.randn_like(w_opt) * w_noise_scale
        ws = (w_opt + w_noise).repeat([1, G.mapping.num_ws, 1])
        synth_images = G.synthesis(ws, noise_mode="const")

        # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
        synth_images = (synth_images + 1) * (255 / 2)
        if synth_images.shape[2] > 256:
            synth_images = F.interpolate(synth_images, size=(256, 256), mode="area")

        # Features for synth images.
        synth_features = vgg16(synth_images, resize_images=False, return_lpips=True)
        dist = (target_features - synth_features).square().sum()

        # Noise regularization.
        reg_loss = 0.0
        for v in noise_bufs.values():
            noise = v[None, None, :, :]  # must be [1,1,H,W] for F.avg_pool2d()
            while True:
                reg_loss += (noise * torch.roll(noise, shifts=1, dims=3)).mean() ** 2
                reg_loss += (noise * torch.roll(noise, shifts=1, dims=2)).mean() ** 2
                if noise.shape[2] <= 8:
                    break
                noise = F.avg_pool2d(noise, kernel_size=2)
        loss = dist + reg_loss * regularize_noise_weight

        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        logprint(
            f"step {step+1:>4d}/{num_steps}: dist {dist:<4.2f} loss {float(loss):<5.2f}"
        )

        # Save projected W for each optimization step.
        w_out[step] = w_opt.detach()[0]

        # Normalize noise.
        with torch.no_grad():
            for buf in noise_bufs.values():
                buf -= buf.mean()
                buf *= buf.square().mean().rsqrt()

    # return w_out.repeat([1, G.mapping.num_ws, 1])
    return dist.item(), w_out.repeat([1, G.mapping.num_ws, 1])[-1]


def compute_hash(files: List[Path], hash_function: str = "phash") -> pd.DataFrame:
    hashes = []
    for file in tqdm(files):
        if hash_function == "phash":
            hash = imagehash.phash(image=Image.open(file))
        else:
            raise NotImplementedError
        hashes.append(str(hash))

    return pd.DataFrame({hash_function: hashes, "file": files})


compute_hash = SimpleMemory(location=".cache").cache(compute_hash)


def compute_lpips(original_dir: Union[str, Path], rec_dir: Union[str, Path]):
    device = get_device()
    lpips = LPIPS(net="vgg").to(device)

    original_ds = ImageFolder(original_dir, transform=tf.ToTensor())
    rec_ds = ImageFolder(rec_dir, transform=tf.ToTensor())

    distances = []
    for (original_img, original_file), (rec_img, rec_file) in tqdm(
        zip(DataLoader(original_ds), DataLoader(rec_ds))
    ):
        assert Path(original_file[0]).stem == Path(rec_file[0]).stem
        rec_img = rec_img[:, :, :, 400:]  # only use right half of parallel view
        distances.append(
            lpips(original_img.to(device), rec_img.to(device), normalize=True).item()
        )
    return distances


compute_lpips = SimpleMemory(location=".cache").cache(compute_lpips)
