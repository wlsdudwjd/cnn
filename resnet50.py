import argparse
import os
import time
from typing import Optional

import torch
from torch import nn, optim

from alexnet import (
    TrainConfig,
    build_loaders,
    evaluate,
    format_duration,
    get_device,
    set_seed,
    train_one_epoch,
)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(
            out_channels,
            out_channels * self.expansion,
            kernel_size=1,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, layers: list[int], in_channels: int = 3, num_classes: int = 1000,) -> None:
        super().__init__()
        self.inplanes = 64

        self.conv1 = nn.Conv2d(in_channels,self.inplanes,kernel_size=7,stride=2,padding=3,bias=False,
        )
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, layers[0], stride=1)
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def _make_layer(self, out_channels: int, blocks: int, stride: int) -> nn.Sequential:
        downsample = None
        out_planes = out_channels * Bottleneck.expansion
        if stride != 1 or self.inplanes != out_planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes),
            )

        layers: list[nn.Module] = [Bottleneck(self.inplanes, out_channels, stride, downsample)]
        self.inplanes = out_planes
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def build_model(cfg: TrainConfig) -> ResNet:
    if cfg.dataset == "mnist":
        return ResNet(layers=[3, 4, 6, 3], in_channels=1, num_classes=10)
    return ResNet(layers=[3, 4, 6, 3], in_channels=3, num_classes=1000)


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train ResNet50 on MNIST or ImageNet64")
    parser.add_argument("--dataset", choices=["mnist", "imagenet", "imagenet64"], default="mnist")
    parser.add_argument("--data-dir", default="./data", help="MNIST root directory")
    parser.add_argument("--imagenet-dir", default="./Imagenet", help="ImageNet64 batch directory")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument(
        "--no-realtime-progress",
        action="store_true",
        help="disable one-line real-time batch progress output",
    )
    parser.add_argument(
        "--imagenet-cache-size",
        type=int,
        default=1,
        help="Number of ImageNet64 batch files to keep in RAM cache",
    )
    parser.add_argument("--save-path", default="./resnet50_last.pth", help="model checkpoint path")
    args = parser.parse_args()

    dataset = "imagenet64" if args.dataset == "imagenet" else args.dataset
    return TrainConfig(
        dataset=dataset,
        data_dir=args.data_dir,
        imagenet_dir=args.imagenet_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        num_workers=args.num_workers,
        val_ratio=args.val_ratio,
        log_interval=args.log_interval,
        realtime_progress=not args.no_realtime_progress,
        imagenet_cache_size=args.imagenet_cache_size,
        save_path=args.save_path,
    )


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)

    os.makedirs(cfg.data_dir, exist_ok=True)
    device = get_device()
    print(f"Dataset: {cfg.dataset} | Device: {device}")

    train_loader, val_loader = build_loaders(cfg)
    model = build_model(cfg).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=cfg.lr,
        momentum=0.9,
        weight_decay=cfg.weight_decay,
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
    train_start = time.perf_counter()

    for epoch in range(1, cfg.epochs + 1):
        epoch_start = time.perf_counter()
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

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

    print("Finished training.")


if __name__ == "__main__":
    main()
