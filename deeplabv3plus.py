import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, random_split
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
    data_dir: str
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
    save_path: str


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
            ConvBNReLU(in_channels, out_channels, kernel_size=1),
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


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    log_interval: int,
    realtime_progress: bool,
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    running_correct = 0
    running_pixels = 0
    running_samples = 0
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
        running_loss += loss.item() * batch_size
        running_correct += (outputs.sigmoid() > 0.5).eq(masks > 0.5).sum().item()
        running_pixels += masks.numel()
        running_samples += batch_size

        avg_loss = running_loss / max(running_samples, 1)
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

    epoch_loss = running_loss / max(len(loader.dataset), 1)
    epoch_acc = running_correct / max(running_pixels, 1)
    return epoch_loss, epoch_acc


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_pixels = 0

    with torch.no_grad():
        for inputs, masks in loader:
            inputs, masks = inputs.to(device), masks.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, masks)

            total_loss += loss.item() * masks.size(0)
            total_correct += (outputs.sigmoid() > 0.5).eq(masks > 0.5).sum().item()
            total_pixels += masks.numel()

    avg_loss = total_loss / max(len(loader.dataset), 1)
    avg_acc = total_correct / max(total_pixels, 1)
    return avg_loss, avg_acc


def parse_args() -> tuple[SegConfig, str, str, bool]:
    parser = argparse.ArgumentParser(description="Train DeepLabV3+ on MNIST segmentation")
    parser.add_argument("--data-dir", default="./data", help="MNIST root directory")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--no-realtime-progress",
        action="store_true",
        help="disable one-line real-time batch progress output",
    )
    parser.add_argument("--save-path", default="./deeplabv3plus_last.pth", help="model checkpoint path")
    parser.add_argument("--plot-path", default="./deeplabv3plus_acc_compare.png", help="train-vs-val accuracy graph")
    parser.add_argument(
        "--loss-plot-path",
        default="./deeplabv3plus_loss_compare.png",
        help="train-vs-val loss graph",
    )
    parser.add_argument("--no-plot", action="store_true", help="disable saving graphs")
    args = parser.parse_args()

    cfg = SegConfig(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        num_workers=args.num_workers,
        val_ratio=args.val_ratio,
        log_interval=args.log_interval,
        realtime_progress=not args.no_realtime_progress,
        image_size=args.image_size,
        save_path=args.save_path,
    )
    return cfg, args.plot_path, args.loss_plot_path, args.no_plot


def main() -> None:
    cfg, plot_path, loss_plot_path, no_plot = parse_args()
    set_seed(cfg.seed)

    os.makedirs(cfg.data_dir, exist_ok=True)
    device = get_device()
    print(f"Dataset: MNIST (seg) | Device: {device}")

    train_loader, val_loader = build_mnist_loaders(cfg)
    model = DeepLabV3Plus(in_channels=1, num_classes=1).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=cfg.lr,
        momentum=0.9,
        weight_decay=cfg.weight_decay,
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
    train_start = time.perf_counter()

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, cfg.epochs + 1):
        epoch_start = time.perf_counter()
        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch=epoch,
            log_interval=cfg.log_interval,
            realtime_progress=cfg.realtime_progress,
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        epoch_elapsed = time.perf_counter() - epoch_start
        total_elapsed = time.perf_counter() - train_start
        overall_progress = epoch / max(cfg.epochs, 1)
        avg_epoch_time = total_elapsed / max(epoch, 1)
        total_eta = avg_epoch_time * (cfg.epochs - epoch)
        print(
            f"Epoch {epoch}/{cfg.epochs} complete ({overall_progress * 100:5.1f}%) | "
            f"Train Loss {train_loss:.4f} | Train Acc {train_acc:.4f} | "
            f"Val Loss {val_loss:.4f} | Val Acc {val_acc:.4f} | "
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

    print("Finished training.")


if __name__ == "__main__":
    main()
