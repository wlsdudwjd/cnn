import argparse
import os
import time
from pathlib import Path

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


def make_vgg_features(in_channels: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    channels = in_channels
    cfg = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"]
    for value in cfg:
        if value == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            out_channels = int(value)
            layers.append(nn.Conv2d(channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
            channels = out_channels
    return nn.Sequential(*layers)


class VGG16(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 1000) -> None:
        super().__init__()
        self.features = make_vgg_features(in_channels=in_channels)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def build_model(cfg: TrainConfig) -> VGG16:
    if cfg.dataset == "mnist":
        return VGG16(in_channels=1, num_classes=10)
    return VGG16(in_channels=3, num_classes=1000)


def save_accuracy_plot(history: dict[str, list[float]], plot_path: str) -> None:
    if not history["train_acc"]:
        return

    os.environ.setdefault("MPLCONFIGDIR", str((Path(".") / ".matplotlib").resolve()))
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Could not save plot: {exc}")
        return

    epochs = list(range(1, len(history["train_acc"]) + 1))
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(epochs, history["train_acc"], marker="o", linewidth=2, label="Train Acc")
    ax.plot(epochs, history["val_acc"], marker="o", linewidth=2, label="Val Acc")
    ax.set_title("Train vs Val Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    plot_file = Path(plot_path)
    plot_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_file, dpi=150)
    plt.close(fig)
    print(f"Saved accuracy graph to {plot_file}")


def save_loss_plot(history: dict[str, list[float]], plot_path: str) -> None:
    if not history["train_loss"]:
        return

    os.environ.setdefault("MPLCONFIGDIR", str((Path(".") / ".matplotlib").resolve()))
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Could not save plot: {exc}")
        return

    epochs = list(range(1, len(history["train_loss"]) + 1))
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(epochs, history["train_loss"], marker="o", linewidth=2, label="Train Loss")
    ax.plot(epochs, history["val_loss"], marker="o", linewidth=2, label="Val Loss")
    ax.set_title("Train vs Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    plot_file = Path(plot_path)
    plot_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_file, dpi=150)
    plt.close(fig)
    print(f"Saved loss graph to {plot_file}")


def parse_args() -> tuple[TrainConfig, str, str, bool]:
    parser = argparse.ArgumentParser(description="Train VGG16 on MNIST or ImageNet64")
    parser.add_argument("--dataset", choices=["mnist", "imagenet", "imagenet64"], default="mnist")
    parser.add_argument("--data-dir", default="./data", help="MNIST root directory")
    parser.add_argument("--imagenet-dir", default="./Imagenet", help="ImageNet64 batch directory")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
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
    parser.add_argument("--save-path", default="./vgg16_last.pth", help="model checkpoint path")
    parser.add_argument("--plot-path", default="./vgg16_acc_compare.png", help="train-vs-val accuracy graph path")
    parser.add_argument("--loss-plot-path", default="./vgg16_loss_compare.png", help="train-vs-val loss graph path")
    parser.add_argument("--no-plot", action="store_true", help="disable saving graphs")
    args = parser.parse_args()

    dataset = "imagenet64" if args.dataset == "imagenet" else args.dataset
    cfg = TrainConfig(
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
    return cfg, args.plot_path, args.loss_plot_path, args.no_plot


def main() -> None:
    cfg, plot_path, loss_plot_path, no_plot = parse_args()
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
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
    scheduler = None
    train_start = time.perf_counter()
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

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
        # scheduler.step()
        if scheduler is not None:
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
