import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode

from alexnet import (
    format_duration,
    get_device,
    print_realtime_progress,
    save_accuracy_plot,
    save_loss_plot,
    set_seed,
)


@dataclass
class SegConfig:
    dataset: str
    data_dir: str
    imagenets_dir: str
    imagenets_split: str
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    seed: int
    num_workers: int
    val_ratio: float
    log_interval: int
    realtime_progress: bool
    image_size: int
    num_classes: int
    ignore_index: int
    save_path: str
    best_save_path: str
    scheduler: str
    step_size: int
    gamma: float
    min_lr: float
    tb_logdir: str
    no_tensorboard: bool


class MNISTSegmentation(Dataset):
    def __init__(self, root: str, train: bool, image_size: int) -> None:
        self.base = datasets.MNIST(root=root, train=train, download=True)
        self.image_size = image_size
        self.normalize = transforms.Normalize((0.1307,), (0.3081,))

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, _ = self.base[idx]
        mask = (np.array(image, dtype=np.uint8) > 0).astype(np.uint8) * 255
        mask = Image.fromarray(mask, mode="L")

        image = TF.resize(image, (self.image_size, self.image_size), interpolation=InterpolationMode.BILINEAR)
        mask = TF.resize(mask, (self.image_size, self.image_size), interpolation=InterpolationMode.NEAREST)

        image = self.normalize(TF.to_tensor(image))
        mask_tensor = torch.from_numpy(np.array(mask, dtype=np.uint8)).unsqueeze(0).float() / 255.0
        return image, mask_tensor


def _find_matching_image(image_dir: Path, relative_mask_path: Path) -> Optional[Path]:
    image_stem = relative_mask_path.with_suffix("")
    candidates = [
        image_dir / image_stem.with_suffix(".JPEG"),
        image_dir / image_stem.with_suffix(".jpeg"),
        image_dir / image_stem.with_suffix(".jpg"),
        image_dir / image_stem.with_suffix(".JPG"),
        image_dir / image_stem.with_suffix(".png"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


class ImageNetSSegmentationDataset(Dataset):
    def __init__(
        self,
        image_dir: Path,
        mask_dir: Path,
        image_size: int,
        num_classes: int,
        ignore_index: int,
    ) -> None:
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not self.mask_dir.exists():
            raise FileNotFoundError(f"Mask directory not found: {self.mask_dir}")

        mask_files = sorted(self.mask_dir.rglob("*.png"))
        if not mask_files:
            raise FileNotFoundError(f"No PNG mask files found in {self.mask_dir}")

        pairs: list[tuple[Path, Path]] = []
        missing_images: list[str] = []
        for mask_path in mask_files:
            relative_mask = mask_path.relative_to(self.mask_dir)
            image_path = _find_matching_image(self.image_dir, relative_mask)
            if image_path is None:
                missing_images.append(str(relative_mask))
                continue
            pairs.append((image_path, mask_path))

        if missing_images:
            sample_missing = "\n".join(missing_images[:5])
            raise FileNotFoundError(
                f"Could not find matching images for {len(missing_images)} masks in {self.image_dir}.\n"
                f"Examples:\n{sample_missing}"
            )

        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, mask_path = self.pairs[idx]
        with Image.open(image_path) as img:
            image = img.convert("RGB")
        with Image.open(mask_path) as msk:
            mask = msk.convert("RGB")

        image = TF.resize(image, (self.image_size, self.image_size), interpolation=InterpolationMode.BILINEAR)
        mask = TF.resize(mask, (self.image_size, self.image_size), interpolation=InterpolationMode.NEAREST)

        image_tensor = self.normalize(TF.to_tensor(image))
        mask_rgb = np.array(mask, dtype=np.uint16)
        mask_ids = mask_rgb[:, :, 0] + (mask_rgb[:, :, 1] * 256)

        # ImageNet-S masks use id 0 for background/other, 1..N for classes, and 1000 as ignore.
        mapped = np.full(mask_ids.shape, self.ignore_index, dtype=np.int64)
        mapped[mask_ids == 0] = 0
        valid_fg = (mask_ids >= 1) & (mask_ids <= (self.num_classes - 1))
        mapped[valid_fg] = mask_ids[valid_fg]
        mask_tensor = torch.from_numpy(mapped)
        return image_tensor, mask_tensor


class ConvBNReLU(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)


class SimpleBackbone(nn.Module):
    def __init__(self, in_channels: int = 1) -> None:
        super().__init__()
        self.stem = ConvBNReLU(in_channels, 32, kernel_size=3, stride=2, padding=1)
        self.layer1 = nn.Sequential(
            ConvBNReLU(32, 64, kernel_size=3, stride=2, padding=1),
            ConvBNReLU(64, 64, kernel_size=3, padding=1),
        )
        self.layer2 = nn.Sequential(
            ConvBNReLU(64, 128, kernel_size=3, stride=2, padding=1),
            ConvBNReLU(128, 128, kernel_size=3, padding=1),
        )
        self.layer3 = nn.Sequential(
            ConvBNReLU(128, 256, kernel_size=3, stride=2, padding=1),
            ConvBNReLU(256, 256, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        low_level = self.layer1(x)
        x = self.layer2(low_level)
        high_level = self.layer3(x)
        return low_level, high_level


class ASPP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, atrous_rates: list[int]) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            [
                ConvBNReLU(in_channels, out_channels, kernel_size=1),
                *[
                    ConvBNReLU(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        padding=rate,
                        dilation=rate,
                    )
                    for rate in atrous_rates
                ],
            ]
        )
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
        )
        total_branches = len(self.branches) + 1
        self.project = ConvBNReLU(out_channels * total_branches, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        outputs = [branch(x) for branch in self.branches]
        pooled = self.global_pool(x)
        pooled = F.interpolate(pooled, size=size, mode="bilinear", align_corners=False)
        outputs.append(pooled)
        x = torch.cat(outputs, dim=1)
        return self.project(x)


class DeepLabV3Plus(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 1) -> None:
        super().__init__()
        self.backbone = SimpleBackbone(in_channels=in_channels)
        self.aspp = ASPP(in_channels=256, out_channels=256, atrous_rates=[6, 12, 18])
        self.low_level_proj = ConvBNReLU(64, 48, kernel_size=1)
        self.decoder = nn.Sequential(
            ConvBNReLU(256 + 48, 256, kernel_size=3, padding=1),
            ConvBNReLU(256, 256, kernel_size=3, padding=1),
            nn.Conv2d(256, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        low_level, high_level = self.backbone(x)
        x = self.aspp(high_level)
        x = F.interpolate(x, size=low_level.shape[-2:], mode="bilinear", align_corners=False)
        low_level = self.low_level_proj(low_level)
        x = torch.cat([x, low_level], dim=1)
        x = self.decoder(x)
        x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)
        return x


def build_mnist_loaders(cfg: SegConfig) -> tuple[DataLoader, DataLoader]:
    dataset = MNISTSegmentation(root=cfg.data_dir, train=True, image_size=cfg.image_size)
    val_size = int(len(dataset) * cfg.val_ratio)
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=generator)

    loader_kwargs = {
        "batch_size": cfg.batch_size,
        "num_workers": cfg.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)
    print(f"MNIST seg split | train: {len(train_set)} | val: {len(val_set)}")
    return train_loader, val_loader


def build_imagenets_loaders(cfg: SegConfig) -> tuple[DataLoader, DataLoader]:
    split = cfg.imagenets_split
    root = Path(cfg.imagenets_dir) / f"ImageNetS{split}"
    train_image_dir = root / "train-semi"
    train_mask_dir = root / "train-semi-segmentation"
    val_image_dir = root / "validation"
    val_mask_dir = root / "validation-segmentation"

    missing_paths = [path for path in [train_image_dir, train_mask_dir, val_image_dir, val_mask_dir] if not path.exists()]
    if missing_paths:
        missing_lines = "\n".join(str(path) for path in missing_paths)
        raise FileNotFoundError(
            f"ImageNet-S files are missing:\n{missing_lines}\n\n"
            "You need ImageNet-1K train/val images and ImageNet-S annotations.\n"
            "After preparing data, expected layout is:\n"
            f"{root}/train-semi, {root}/train-semi-segmentation, {root}/validation, {root}/validation-segmentation"
        )

    train_dataset = ImageNetSSegmentationDataset(
        image_dir=train_image_dir,
        mask_dir=train_mask_dir,
        image_size=cfg.image_size,
        num_classes=cfg.num_classes,
        ignore_index=cfg.ignore_index,
    )
    val_dataset = ImageNetSSegmentationDataset(
        image_dir=val_image_dir,
        mask_dir=val_mask_dir,
        image_size=cfg.image_size,
        num_classes=cfg.num_classes,
        ignore_index=cfg.ignore_index,
    )

    loader_kwargs = {
        "batch_size": cfg.batch_size,
        "num_workers": cfg.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    print(f"ImageNet-S-{split} seg split | train: {len(train_dataset)} | val: {len(val_dataset)}")
    return train_loader, val_loader


def build_loaders(cfg: SegConfig) -> tuple[DataLoader, DataLoader]:
    if cfg.dataset == "mnist":
        return build_mnist_loaders(cfg)
    return build_imagenets_loaders(cfg)


def _mean_iou_from_inter_union(intersection: torch.Tensor, union: torch.Tensor, num_classes: int) -> float:
    if num_classes == 1:
        if union.numel() == 0 or union[0].item() == 0:
            return 0.0
        return (intersection[0].float() / union[0].float()).item()

    # For ImageNet-S, foreground quality is more informative than including background.
    fg_inter = intersection[1:]
    fg_union = union[1:]
    valid = fg_union > 0
    if not torch.any(valid):
        return 0.0
    return (fg_inter[valid].float() / fg_union[valid].float()).mean().item()


def _update_inter_union(
    outputs: torch.Tensor,
    masks: torch.Tensor,
    num_classes: int,
    ignore_index: Optional[int],
    intersection: torch.Tensor,
    union: torch.Tensor,
) -> None:
    if num_classes == 1:
        preds = outputs.sigmoid() > 0.5
        targets = masks > 0.5
        if preds.ndim == 4 and preds.shape[1] == 1:
            preds = preds[:, 0]
        if targets.ndim == 4 and targets.shape[1] == 1:
            targets = targets[:, 0]
        preds = preds.bool()
        targets = targets.bool()
        inter = (preds & targets).sum().item()
        uni = (preds | targets).sum().item()
        intersection[0] += int(inter)
        union[0] += int(uni)
        return

    preds = outputs.argmax(dim=1)
    valid_mask = masks != ignore_index if ignore_index is not None else torch.ones_like(masks, dtype=torch.bool)
    preds = preds[valid_mask]
    targets = masks[valid_mask]
    if preds.numel() == 0:
        return

    pred_hist = torch.bincount(preds, minlength=num_classes).to(dtype=torch.long, device="cpu")
    target_hist = torch.bincount(targets, minlength=num_classes).to(dtype=torch.long, device="cpu")
    inter_hist = torch.bincount(targets[preds == targets], minlength=num_classes).to(dtype=torch.long, device="cpu")
    intersection += inter_hist
    union += pred_hist + target_hist - inter_hist


def build_scheduler(cfg: SegConfig, optimizer: optim.Optimizer) -> Optional[optim.lr_scheduler.LRScheduler]:
    if cfg.scheduler == "none":
        return None
    if cfg.scheduler == "step":
        return optim.lr_scheduler.StepLR(optimizer, step_size=cfg.step_size, gamma=cfg.gamma)
    if cfg.scheduler == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.min_lr)
    raise ValueError(f"Unsupported scheduler: {cfg.scheduler}")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    log_interval: int,
    realtime_progress: bool,
    num_classes: int,
    ignore_index: Optional[int],
) -> tuple[float, float, float]:
    model.train()
    running_loss = 0.0
    running_loss_weight = 0
    running_correct = 0
    running_pixels = 0
    running_samples = 0
    intersection = torch.zeros(1 if num_classes == 1 else num_classes, dtype=torch.long)
    union = torch.zeros_like(intersection)
    epoch_start = time.perf_counter()
    total_steps = len(loader)

    for batch_idx, (inputs, masks) in enumerate(loader, start=1):
        inputs, masks = inputs.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        batch_size = masks.size(0)
        if num_classes == 1:
            loss_weight = masks.numel()
        else:
            if ignore_index is not None:
                loss_weight = (masks != ignore_index).sum().item()
            else:
                loss_weight = masks.numel()
        running_loss += loss.item() * loss_weight
        running_loss_weight += loss_weight
        if num_classes == 1:
            running_correct += (outputs.sigmoid() > 0.5).eq(masks > 0.5).sum().item()
            running_pixels += masks.numel()
        else:
            preds = outputs.argmax(dim=1)
            valid_mask = masks != ignore_index if ignore_index is not None else torch.ones_like(masks, dtype=torch.bool)
            running_correct += preds.eq(masks).logical_and(valid_mask).sum().item()
            running_pixels += valid_mask.sum().item()
        _update_inter_union(outputs, masks, num_classes, ignore_index, intersection, union)
        running_samples += batch_size

        avg_loss = running_loss / max(running_loss_weight, 1)
        avg_acc = running_correct / max(running_pixels, 1)
        if realtime_progress:
            print_realtime_progress("Training", batch_idx, total_steps, avg_loss, avg_acc, epoch_start)
        else:
            if log_interval > 0 and (batch_idx % log_interval == 0 or batch_idx == total_steps):
                progress = batch_idx / max(total_steps, 1)
                elapsed = time.perf_counter() - epoch_start
                eta = elapsed / max(progress, 1e-8) - elapsed
                print(
                    f"Epoch {epoch} | Step {batch_idx}/{total_steps} ({progress * 100:5.1f}%) | "
                    f"Loss {avg_loss:.4f} | Acc {avg_acc:.4f} | "
                    f"Elapsed {format_duration(elapsed)} | ETA {format_duration(eta)}"
                )

    epoch_loss = running_loss / max(running_loss_weight, 1)
    epoch_acc = running_correct / max(running_pixels, 1)
    epoch_miou = _mean_iou_from_inter_union(intersection, union, num_classes)
    return epoch_loss, epoch_acc, epoch_miou


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    ignore_index: Optional[int],
) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    total_loss_weight = 0
    total_correct = 0
    total_pixels = 0
    intersection = torch.zeros(1 if num_classes == 1 else num_classes, dtype=torch.long)
    union = torch.zeros_like(intersection)

    with torch.no_grad():
        for inputs, masks in loader:
            inputs, masks = inputs.to(device), masks.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, masks)

            if num_classes == 1:
                loss_weight = masks.numel()
            else:
                if ignore_index is not None:
                    loss_weight = (masks != ignore_index).sum().item()
                else:
                    loss_weight = masks.numel()
            total_loss += loss.item() * loss_weight
            total_loss_weight += loss_weight
            if num_classes == 1:
                total_correct += (outputs.sigmoid() > 0.5).eq(masks > 0.5).sum().item()
                total_pixels += masks.numel()
            else:
                preds = outputs.argmax(dim=1)
                valid_mask = masks != ignore_index if ignore_index is not None else torch.ones_like(masks, dtype=torch.bool)
                total_correct += preds.eq(masks).logical_and(valid_mask).sum().item()
                total_pixels += valid_mask.sum().item()
            _update_inter_union(outputs, masks, num_classes, ignore_index, intersection, union)

    avg_loss = total_loss / max(total_loss_weight, 1)
    avg_acc = total_correct / max(total_pixels, 1)
    avg_miou = _mean_iou_from_inter_union(intersection, union, num_classes)
    return avg_loss, avg_acc, avg_miou


def parse_args() -> tuple[SegConfig, str, str, bool]:
    parser = argparse.ArgumentParser(
        description="Train DeepLabV3+ on MNIST segmentation or ImageNet-S."
    )
    parser.add_argument("--dataset", choices=["mnist", "imagenets"], default="mnist")
    parser.add_argument("--data-dir", default="./data", help="MNIST root directory")
    parser.add_argument("--imagenets-dir", default="./imagenet-s", help="ImageNet-S root directory")
    parser.add_argument("--imagenets-split", choices=["50", "300", "919"], default="50")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--ignore-index", type=int, default=255, help="Ignored label index for segmentation masks")
    parser.add_argument(
        "--no-realtime-progress",
        action="store_true",
        help="disable one-line real-time batch progress output",
    )
    parser.add_argument("--save-path", default="./deeplabv3plus_last.pth", help="model checkpoint path")
    parser.add_argument("--best-save-path", default="./deeplabv3plus_best.pth", help="best model checkpoint path (by val mIoU)")
    parser.add_argument("--scheduler", choices=["none", "step", "cosine"], default="cosine")
    parser.add_argument("--step-size", type=int, default=30, help="step scheduler step size")
    parser.add_argument("--gamma", type=float, default=0.1, help="step scheduler gamma")
    parser.add_argument("--min-lr", type=float, default=1e-5, help="cosine scheduler minimum lr")
    parser.add_argument("--tb-logdir", default="./runs/deeplabv3plus_exp", help="TensorBoard log directory")
    parser.add_argument("--no-tensorboard", action="store_true", help="disable TensorBoard logging")
    parser.add_argument("--plot-path", default="./deeplabv3plus_acc_compare.png", help="train-vs-val accuracy graph")
    parser.add_argument(
        "--loss-plot-path",
        default="./deeplabv3plus_loss_compare.png",
        help="train-vs-val loss graph",
    )
    parser.add_argument("--no-plot", action="store_true", help="disable saving graphs")
    args = parser.parse_args()

    dataset = args.dataset
    image_size = args.image_size
    if image_size is None:
        image_size = 224
    num_classes = args.num_classes
    expected_imagenets_classes = int(args.imagenets_split) + 1
    if num_classes is None:
        if dataset == "mnist":
            num_classes = 1
        else:
            num_classes = expected_imagenets_classes

    if dataset == "mnist" and num_classes != 1:
        raise ValueError("MNIST segmentation requires --num-classes 1.")
    if dataset == "imagenets" and num_classes != expected_imagenets_classes:
        raise ValueError(
            f"ImageNet-S-{args.imagenets_split} requires --num-classes {expected_imagenets_classes} "
            "(background + foreground classes)."
        )

    cfg = SegConfig(
        dataset=dataset,
        data_dir=args.data_dir,
        imagenets_dir=args.imagenets_dir,
        imagenets_split=args.imagenets_split,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        num_workers=args.num_workers,
        val_ratio=args.val_ratio,
        log_interval=args.log_interval,
        realtime_progress=not args.no_realtime_progress,
        image_size=image_size,
        num_classes=num_classes,
        ignore_index=args.ignore_index,
        save_path=args.save_path,
        best_save_path=args.best_save_path,
        scheduler=args.scheduler,
        step_size=args.step_size,
        gamma=args.gamma,
        min_lr=args.min_lr,
        tb_logdir=args.tb_logdir,
        no_tensorboard=args.no_tensorboard,
    )
    return cfg, args.plot_path, args.loss_plot_path, args.no_plot


def main() -> None:
    cfg, plot_path, loss_plot_path, no_plot = parse_args()
    set_seed(cfg.seed)

    os.makedirs(cfg.data_dir, exist_ok=True)
    device = get_device()
    print(f"Dataset: {cfg.dataset} | Device: {device}")

    train_loader, val_loader = build_loaders(cfg)
    in_channels = 1 if cfg.dataset == "mnist" else 3
    model = DeepLabV3Plus(in_channels=in_channels, num_classes=cfg.num_classes).to(device)

    ignore_index = cfg.ignore_index if cfg.dataset == "imagenets" else None
    criterion = (
        nn.BCEWithLogitsLoss()
        if cfg.num_classes == 1
        else nn.CrossEntropyLoss(ignore_index=cfg.ignore_index if ignore_index is not None else -100)
    )
    optimizer = optim.SGD(
        model.parameters(),
        lr=cfg.lr,
        momentum=0.9,
        weight_decay=cfg.weight_decay,
    )
    scheduler = build_scheduler(cfg, optimizer)
    train_start = time.perf_counter()
    best_val_miou = -1.0
    writer: Optional[SummaryWriter] = None
    if not cfg.no_tensorboard:
        os.makedirs(cfg.tb_logdir, exist_ok=True)
        writer = SummaryWriter(log_dir=cfg.tb_logdir)
        try:
            dummy = torch.randn(1, in_channels, cfg.image_size, cfg.image_size, device=device)
            writer.add_graph(model, dummy)
        except Exception as exc:
            print(f"TensorBoard graph logging skipped: {exc}")
        print(f"TensorBoard logging to: {cfg.tb_logdir}")

    history = {"train_loss": [], "train_acc": [], "train_miou": [], "val_loss": [], "val_acc": [], "val_miou": []}

    for epoch in range(1, cfg.epochs + 1):
        epoch_start = time.perf_counter()
        train_loss, train_acc, train_miou = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch=epoch,
            log_interval=cfg.log_interval,
            realtime_progress=cfg.realtime_progress,
            num_classes=cfg.num_classes,
            ignore_index=ignore_index,
        )
        val_loss, val_acc, val_miou = evaluate(
            model,
            val_loader,
            criterion,
            device,
            num_classes=cfg.num_classes,
            ignore_index=ignore_index,
        )
        current_lr = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["train_miou"].append(train_miou)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_miou"].append(val_miou)
        if writer is not None:
            writer.add_scalar("epoch/train_loss", train_loss, epoch)
            writer.add_scalar("epoch/train_acc", train_acc, epoch)
            writer.add_scalar("epoch/train_miou", train_miou, epoch)
            writer.add_scalar("epoch/val_loss", val_loss, epoch)
            writer.add_scalar("epoch/val_acc", val_acc, epoch)
            writer.add_scalar("epoch/val_miou", val_miou, epoch)
            writer.add_scalar("epoch/lr", current_lr, epoch)

        if cfg.best_save_path and val_miou > best_val_miou:
            best_val_miou = val_miou
            torch.save(model.state_dict(), cfg.best_save_path)
            print(f"Saved best checkpoint to {cfg.best_save_path} (val mIoU: {val_miou:.4f})")

        epoch_elapsed = time.perf_counter() - epoch_start
        total_elapsed = time.perf_counter() - train_start
        overall_progress = epoch / max(cfg.epochs, 1)
        avg_epoch_time = total_elapsed / max(epoch, 1)
        total_eta = avg_epoch_time * (cfg.epochs - epoch)
        print(
            f"Epoch {epoch}/{cfg.epochs} complete ({overall_progress * 100:5.1f}%) | "
            f"Train Loss {train_loss:.4f} | Train Acc {train_acc:.4f} | Train mIoU {train_miou:.4f} | "
            f"Val Loss {val_loss:.4f} | Val Acc {val_acc:.4f} | Val mIoU {val_miou:.4f} | "
            f"LR {current_lr:.6f} | "
            f"Epoch Time {format_duration(epoch_elapsed)} | "
            f"Total Elapsed {format_duration(total_elapsed)} | "
            f"Total ETA {format_duration(total_eta)}"
        )

    if cfg.save_path:
        torch.save(model.state_dict(), cfg.save_path)
        print(f"Saved checkpoint to {cfg.save_path}")
    if not no_plot:
        save_accuracy_plot(history, plot_path)
        save_loss_plot(history, loss_plot_path)
    if writer is not None:
        writer.close()

    print("Finished training.")


if __name__ == "__main__":
    main()
